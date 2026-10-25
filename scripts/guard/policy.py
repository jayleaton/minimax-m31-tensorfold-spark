"""Durable temperature / memory state machine; pauses and STOPs survive profile and process changes.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48), without its operator ladder. The temperature
limits are the device's (common.read_device_limits: /etc/spark/thermal.env, the same file on both nodes) and the
device's spark-thermal-guard publishes its state (normal, throttled, emergency). A sample is valid only when it is
fresh, its device limits file has the same SHA-256 as the head's (the file the guard loaded) and its device state is one
of the three words and fresh; any other sample fails closed (admission blocked, the hot episode reset, the reason in
status).

- Pause admission when a valid sample's device is throttled or in emergency, or reads at or above pause_c; resume only
  when both nodes are valid, normal and at or below resume_c.
- Durable hard STOP (only an explicit operator profile change clears it, docs/OPERATIONS.md): a node's device in
  emergency (from its fresh state file, even when its limits file differs), a rank whose valid readings stay at or above
  kill_c for kill_hold_seconds (defence in depth beside the device's own emergency), or a fresh reading under the memory
  floor.
"""

import json
import math
import threading
import time
from pathlib import Path

from common import (
    DEVICE_STATES,
    MEMORY_MARGIN_GIB,
    POLICY,
    RANKS,
    atomic,
    kill_condition,
    state_stale_seconds,
)

# A rank's hot episode needs readings this close together (the sampler writes every 2 s; samples go stale at 6 s).
EPISODE_GAP_SECONDS = 6


class Policy:
    def __init__(self, path, events_path, *, limits, initial_profile=None, now=None):
        """`limits`: the device's effective limits (common.read_device_limits), the source of every temperature."""
        self.path, self.events_path = Path(path), Path(events_path)
        self.limits = dict(limits)
        self.kill_condition = kill_condition(self.limits)
        self.invalid = {}
        self.lock = threading.Condition()
        self.inflight = 0
        self.active = {}
        self.request_metadata = {}
        self.prefill_request = None
        self.prefill_completed_epoch = 0
        self.prefill_sample_epochs = {}
        self.memory_reservations = {}
        self.admission_pause_c = self.admission_resume_c = None
        self.transient_reserve_gib = 0.5
        self.unresolved_requests = set()
        self.admission_waiters = {}
        self.fresh = self.owned = False
        self.samples = {}
        now = time.time() if now is None else now
        if self.path.exists():
            self.data = json.loads(self.path.read_bytes())
            if self.data.get("version") != 1 or self.data.get("policy") != POLICY:
                raise ValueError("invalid durable policy; no automatic reset")
        else:
            self.data = {
                "version": 1,
                "policy": POLICY,
                "profile": initial_profile,
                "paused": False,
                "terminal_stop": False,
                "hard_stop": False,
                "pending_action": None,
                "episodes": {"0": None, "1": None},
                "history": [],
                "bootstrap_profile": None,
                "last_update_epoch": now,
                "lease_mode": "unknown",
            }
        if self.data.get("device_limits") != self.limits:
            if "device_limits" in self.data:
                self.event("DEVICE_LIMITS_CHANGED", now, before=self.data["device_limits"], after=self.limits)
            # The effective limits stay visible in the durable state and in status.
            self.data["device_limits"] = self.limits
            self.save()

    def configure_admission(self, limits):
        with self.lock:
            self.admission_pause_c = limits.get("prefill_admission_pause_c")
            self.admission_resume_c = limits.get("prefill_admission_resume_c")
            if self.admission_pause_c is not None:
                # A pre-feature/restarted state without a latch must first prove cool.
                self.data.setdefault("admission_cooldown", True)
                self.update_admission_cooldown(time.time())
                self.save()

    def update_admission_cooldown(self, now):
        if self.admission_pause_c is None:
            return
        before = self.data.get("admission_cooldown", True)
        after = before
        if any(row["max_zone_c"] >= self.admission_pause_c for row in self.samples.values()):
            after = True
        elif (self.fresh and set(self.samples) == {"0", "1"}
              and all(-1 <= now - row["epoch"] <= 6
                      and row["max_zone_c"] <= self.admission_resume_c for row in self.samples.values())):
            after = False
        self.data["admission_cooldown"] = after
        if before != after:
            self.event("ADMISSION_COOLDOWN_ENTER" if after else "ADMISSION_COOLDOWN_EXIT", now,
                       samples=self.samples, pause_c=self.admission_pause_c, resume_c=self.admission_resume_c)

    def thermal_wait_reason(self, now=None):
        if self.admission_pause_c is None:
            return None
        now = time.time() if now is None else now
        if not self.fresh or set(self.samples) != {"0", "1"} or any(
            not -1 <= now - row["epoch"] <= 6 for row in self.samples.values()
        ):
            return "admission_telemetry_stale_or_missing"
        if self.data.get("admission_cooldown", True):
            return "admission_thermal_cooldown"
        return None

    def save(self):
        atomic(self.path, self.data)

    def event(self, kind, now, **fields):
        row = {
            "utc_epoch": now,
            "event": kind,
            "profile": self.data["profile"],
            **fields,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path.exists() and self.events_path.stat().st_size >= 4 * 1024 * 1024:
            for index in range(3, 0, -1):
                older = Path(str(self.events_path) + "." + str(index))
                newer = Path(str(self.events_path) + "." + str(index + 1))
                if older.exists():
                    older.replace(newer)
            self.events_path.replace(Path(str(self.events_path) + ".1"))
        with self.events_path.open("a") as output:
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
        return row

    def abort(self):
        for connection in list(self.active.values()):
            try:
                if connection is not None:
                    connection.shutdown(2)
            except OSError:
                pass

    def terminal(self, reason, now, hard=False, stop_evidence=None):
        if self.data["terminal_stop"] and (not hard or self.data["hard_stop"]):
            return
        self.data.update(terminal_stop=True, paused=True, hard_stop=hard or self.data["hard_stop"])
        self.data["stop_reason"] = reason
        if hard:
            self.data["pending_action"] = "stop_owned"
        if stop_evidence is not None and "stop_evidence" not in self.data:
            self.data["stop_evidence"] = stop_evidence
        self.save()  # Durable latch precedes cancellation; never authorize watchdog restart after reboot.
        if hard:
            self.abort()
        evidence = {"stop_evidence": self.data["stop_evidence"]} if "stop_evidence" in self.data else {}
        self.event("HARD_STOP" if hard else "SOFT_STOP", now, reason=reason, inflight=self.inflight, **evidence)
        self.lock.notify_all()

    def device_problem(self, sample):
        """Why a fresh reading's device evidence fails closed (None: it is valid)."""
        sha = sample.get("device_limits_sha256")
        if not isinstance(sha, str):
            error = str(sample.get("device_limits_error", "no sha256"))[:200]
            return "device limits unreadable on this node (" + error + ")"
        if sha != self.limits["sha256"]:
            return ("device limits differ from the head's: " + self.limits["path"] + " sha256 " + sha[:12] + " here, "
                    + self.limits["sha256"][:12] + " loaded by the guard")
        return self.state_problem(sample)

    def state_problem(self, sample):
        state, age = sample.get("device_state"), sample.get("device_state_age_s")
        if not isinstance(state, str):
            return ("device state unreadable (" + str(sample.get("device_state_error", "none reported"))[:200]
                    + "): spark-thermal-guard not running?")
        if state not in DEVICE_STATES:
            return ("device state " + repr(state[:40]) + " is not " + ", ".join(DEVICE_STATES)
                    + ": spark-thermal-guard not running?")
        try:
            age = float(age)
        except (TypeError, ValueError):
            age = math.nan
        stale = state_stale_seconds(self.limits)
        if not -1 <= age <= stale:
            return ("device state " + str(age) + " s old (over " + str(stale) + " s): spark-thermal-guard not running?")
        return None

    def update(self, samples, owned, lease_mode, now=None, errors=None):
        """`samples`: rank -> the sampler's row; `errors`: rank -> why the sampler has none (shown in status)."""
        now = time.time() if now is None else now
        with self.lock:
            old_update = self.data["last_update_epoch"]
            self.data.update(last_update_epoch=now, lease_mode=lease_mode)
            self.owned = owned
            valid, readable, invalid = {}, {}, {}
            for rank, sample in samples.items():
                if rank not in RANKS:
                    continue
                try:
                    age = now - float(sample["epoch"])
                    temp, memory = float(sample["max_zone_c"]), float(sample["available_gib"])
                except (ValueError, TypeError, KeyError):
                    invalid[rank] = "malformed sample"
                    continue
                if not (-1 <= age <= 6 and all(map(math.isfinite, (temp, memory))) and 0 < temp < 150):
                    invalid[rank] = "stale or out-of-range reading"
                    continue
                readable[rank] = {**sample, "max_zone_c": temp, "available_gib": memory}
                problem = self.device_problem(sample)
                if problem:
                    invalid[rank] = problem
                else:
                    valid[rank] = readable[rank]
            for rank in RANKS:
                if rank not in samples:
                    invalid[rank] = "no sample: " + str((errors or {}).get(rank, "missing"))[:300]
            self.invalid = invalid
            self.samples = valid
            self.fresh = set(valid) == {"0", "1"}
            if not owned:
                self.data["episodes"] = {"0": None, "1": None}
                self.data["fresh"] = self.fresh
                self.update_admission_cooldown(now)
                self.save()
                self.lock.notify_all()
                return
            if now - old_update > EPISODE_GAP_SECONDS or now < old_update:
                self.data["episodes"] = {"0": None, "1": None}  # The guard itself had a gap: no continuous proof.
            fired = []
            for rank in RANKS:
                # The device's own emergency, from a fresh reading and a fresh state file: stop at once.
                row = readable.get(rank)
                if row is not None and row.get("device_state") == "emergency" and self.state_problem(row) is None:
                    fired.append({"condition": "device emergency on rank " + rank, "rank": rank, "value": "emergency",
                                  "device_state_age_s": row.get("device_state_age_s")})
            for rank in RANKS:
                # Per rank, the hot episode runs from its first valid reading >= kill_c to its latest, measured by the
                # rank's own sample times; a cooler, missing or invalid sample or a gap ends it.
                row, episode = valid.get(rank), self.data["episodes"][rank]
                if row is None or row["max_zone_c"] < self.limits["kill_c"]:
                    episode = None
                else:
                    epoch = float(row["epoch"])
                    if episode is None or not 0 <= epoch - episode["last_epoch"] <= EPISODE_GAP_SECONDS:
                        episode = {"start_epoch": epoch, "last_epoch": epoch}
                    episode["last_epoch"] = epoch
                    if epoch - episode["start_epoch"] >= self.limits["kill_hold_seconds"]:
                        fired.append({"condition": self.kill_condition, "rank": rank, "value": row["max_zone_c"],
                                      "episode_start": episode["start_epoch"]})
                self.data["episodes"][rank] = episode
                # The memory floor needs only a fresh reading: a failed device check never hides it.
                row = readable.get(rank)
                if row is not None and row["available_gib"] < POLICY["memory_floor_gib"]:
                    fired.append({"condition": "memory<4GiB", "rank": rank, "value": row["available_gib"]})
            if fired:
                # Telemetry can include rejected NaN/Infinity values. Retain those raw
                # values as explicit strings so evidence cannot prevent the durable STOP.
                evidence = json.loads(
                    json.dumps({
                        "epoch": now,
                        "raw_samples": samples,
                        "validated_samples": valid,
                        "invalid_samples": invalid,
                        "device_limits": self.limits,
                        "fired": fired,
                        "inflight": self.inflight,
                        "request_metadata": self.request_metadata,
                    }),
                    parse_constant=str,
                )
                reason = "; ".join(dict.fromkeys(row["condition"] for row in fired))
                self.terminal(reason, now, hard=True, stop_evidence=evidence)
            if not self.data["terminal_stop"]:
                # Pause on any valid sample whose device is throttled or in emergency, or that reads >= pause_c; resume
                # only when both nodes are valid, normal and <= resume_c.
                hot = {rank: row for rank, row in valid.items()
                       if row["device_state"] != "normal" or row["max_zone_c"] >= self.limits["pause_c"]}
                if hot and not self.data["paused"]:
                    self.data["paused"] = True
                    self.event("PAUSE", now, samples=valid, hot_ranks=sorted(hot), device_limits=self.limits)
                if (
                    self.fresh
                    and self.data["paused"]
                    and all(row["device_state"] == "normal" and row["max_zone_c"] <= self.limits["resume_c"]
                            for row in valid.values())
                ):
                    self.data["paused"] = False
                    self.event("RESUME", now, samples=valid)
            self.data["fresh"] = self.fresh
            # Hard STOP processing above precedes optional pacing/event I/O.
            self.update_admission_cooldown(now)
            self.save()
            self.lock.notify_all()

    def ready(self):
        return (
            self.owned
            and self.fresh
            and not self.data["paused"]
            and not self.data["terminal_stop"]
            and self.data["bootstrap_profile"] is None
            and not self.data.get("operator_hold", False)
        )

    def memory_wait_reason(self, reserve=0.5, now=None):
        """Conservative per-request budget, retained until completion; not native workspace proof."""
        now = time.time() if now is None else now
        if not self.fresh or set(self.samples) != {"0", "1"} or any(
            not -1 <= now - row["epoch"] <= 6 for row in self.samples.values()
        ):
            return "telemetry_stale_or_missing"
        if self.thermal_wait_reason(now):
            return self.thermal_wait_reason(now)
        if not self.ready():
            return "thermal_or_ownership_hold"
        if any(row["epoch"] <= max(self.prefill_completed_epoch, self.prefill_sample_epochs.get(rank, 0))
               for rank, row in self.samples.items()):
            return "waiting_for_post_token_sample"
        if min(row["available_gib"] for row in self.samples.values()) < (
            MEMORY_MARGIN_GIB + sum(self.memory_reservations.values()) + reserve
        ):
            return "memory_margin_and_retained_reservations"
        return None

    def memory_ready(self, reserve=0.5, now=None):
        return self.memory_wait_reason(reserve, now) is None

    def release_prefill(self, request_id):
        # Positive generated delta, or transport cleanup; reservations remain until completion.
        with self.lock:
            if self.prefill_request == request_id:
                self.prefill_completed_epoch = time.time()
                self.prefill_sample_epochs = {rank: row["epoch"] for rank, row in self.samples.items()}
                self.prefill_request = None
                self.lock.notify_all()

    def snapshot(self):
        with self.lock:
            return {
                **json.loads(json.dumps(self.data)),
                "thermal_wait_reason": self.thermal_wait_reason(),
                "admission_cooldown": (self.admission_pause_c is not None
                                       and self.data.get("admission_cooldown", True)),
                "prefill_admission_pause_c": self.admission_pause_c,
                "prefill_admission_resume_c": self.admission_resume_c,
                "inflight": self.inflight,
                "prefill_request": self.prefill_request,
                "admission_waiters": dict(self.admission_waiters),
                "native_completion_unverified": sorted(self.unresolved_requests),
                "memory_reservations_gib": dict(self.memory_reservations),
                "memory_reserved_total_gib": sum(self.memory_reservations.values()),
                "memory_transient_reserve_gib": self.transient_reserve_gib,
                "memory_margin_gib": MEMORY_MARGIN_GIB,
                "memory_wait_reason": self.memory_wait_reason(self.transient_reserve_gib),
                "prefill_completed_epoch": self.prefill_completed_epoch,
                "owned": self.owned,
                "fresh": self.fresh,
                "samples": dict(self.samples),
                "invalid_samples": dict(self.invalid),
                "device_state_stale_seconds": state_stale_seconds(self.limits),
                "ready": self.ready(),
                "active_upstreams": len(self.active),
            }
