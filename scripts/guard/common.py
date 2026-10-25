"""Bounded durable storage and strict parameterized deployment configuration.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48, branch feat/flash17-stability-guard),
with model-agnostic naming. The engine setting names below are the names this recipe plans for the MiniMax family;
the family does not exist yet, so confirm them against the engine on release day (docs/RELEASE-DAY.md).
"""

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

# The guard's own fixed policy: only the memory floor. The temperature limits are the device's (THERMAL_LIMITS below),
# and a config must restate this exactly.
POLICY = {"memory_floor_gib": 4}
# The device's thermal limits: one KEY=value file on every node (the same file on both), the source of truth for the
# device's spark-thermal-guard service and for every job guard. The recipe reads them and never defines its own.
DEVICE_LIMITS = "/etc/spark/thermal.env"
# spark-thermal-guard rewrites this every SPARK_THERMAL_INTERVAL_S seconds: normal, throttled (at or above
# GUARD_PAUSE_C, until at or below GUARD_RESUME_C) or emergency (GUARD_KILL_C held GUARD_KILL_HOLD_S seconds). It never
# stops a job: a job guard pauses on throttled and stops its own job on emergency.
DEVICE_STATE = "/run/spark-thermal/state"
DEVICE_STATES = ("normal", "throttled", "emergency")
# The device file's keys this guard reads, and their names in the effective limits.
THERMAL_LIMITS = {
    "GUARD_PAUSE_C": "pause_c",
    "GUARD_RESUME_C": "resume_c",
    "GUARD_KILL_C": "kill_c",
    "GUARD_KILL_HOLD_S": "kill_hold_seconds",
}
INTERVAL_KEY = "SPARK_THERMAL_INTERVAL_S"
# Sparks start to degrade at this reading: a device file whose stop limit is not below it is refused.
DEGRADE_C = 94
DEVICE_HOME = "the thermal limits live on the device (" + DEVICE_LIMITS + "), not in the recipe config"
# An operator profile transition waits for this (control.py retries any error that starts with it).
TRANSITION_HOLD_PREFIX = "operator transition requires both fresh<="
ENV_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)")
ENV_QUOTED = re.compile(r"\"([^\"]*)\"\s*(?:#.*)?|'([^']*)'\s*(?:#.*)?")
# Admission keeps this much above the 4 GiB hard floor before any per-request reservation.
MEMORY_MARGIN_GIB = 4.25
# Engine settings the guard records in a profile and compares across ranks: the minimax_m3_vl family's names
# (docs/rank-contract.md) and the recipe's own record of the serve flags (start.sh puts them in MM31_SERVE_PROFILE, which
# the engine does not read, so that a CONTEXT, PARALLEL or KV change changes the profile's identity).
ENGINE_ENV_PREFIXES = ("TF_MM3_", "TF_ROCE", "TENSORFOLD_")
POOL_KEY = "MM31_SERVE_PROFILE"
# The engine's reserve (what it leaves free when it sizes the KV pool): a profile under the 4 GiB hard floor can plan a STOP.
FLOOR_KEY = "TENSORFOLD_MEMORY_RESERVE_GIB"
# A host-CPU pin for a plan thread (the DeepSeek engine's TF_DSV41_PLAN_PIN). The minimax_m3_vl family reads none; the check
# stays so that a profile which carries one keeps it inside the rank's CPU set.
PLAN_KEY = "MM31_PLAN_PIN"
LAUNCH_STAMP_KEY = "MM31_LAUNCH_T0"  # gitleaks:allow (an environment variable name)
# TensorFold's host wait mode. Not confirmed for v1.1.2's native server; config/prod.env.example leaves it unset.
SCHEDULE_KEY = "TF_CUDA_SCHED"
EXTRA_ENV_KEYS = (SCHEDULE_KEY, POOL_KEY, PLAN_KEY)
SCHEDULES = (None, "auto", "spin", "yield", "blocking")
CPU_SET = re.compile(r"(0|[1-9][0-9]{0,3})(-(0|[1-9][0-9]{0,3}))?(,(0|[1-9][0-9]{0,3})(-(0|[1-9][0-9]{0,3}))?)*")
RANKS = ("0", "1")


def inference_env(pairs):
    """The engine settings of a container's `KEY=value` list that a profile pins."""
    env = {}
    for value in pairs:
        if "=" not in value:
            continue
        key, item = value.split("=", 1)
        if key.startswith(ENGINE_ENV_PREFIXES) or key in EXTRA_ENV_KEYS:
            env[key] = item
    env.pop(LAUNCH_STAMP_KEY, None)
    return env


def cpu_set_members(text):
    members = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        low, high = int(low), int(high or low)
        if low > high or high > 4095:
            raise ValueError("invalid host CPU set range")
        members.update(range(low, high + 1))
    return members


def validate_log_regex(value, name, groups):
    """An engine log pattern from the config: a compiling regex with exactly `groups` capture groups."""
    if not isinstance(value, str) or not value or value.startswith("<"):
        raise ValueError(name + " must be set (it is engine-specific: read it from rank 0's log on the first start)")
    try:
        compiled = re.compile(value)
    except re.error:
        raise ValueError(name + " is not a valid regular expression") from None
    if compiled.groups != groups:
        raise ValueError(name + " must have exactly " + str(groups) + " capture group(s)")
    return value


def validate_profiles(profiles):
    """Each snapshotted profile: exact per-rank identity, rank-equal qualified knobs, floor not under the hard floor."""
    if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 32:
        raise ValueError("1..32 exact profiles required")
    for sha, profile in profiles.items():
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("exact profile config SHA required")
        knobs = []
        sets = profile.get("cpu_sets", {})
        if not isinstance(sets, dict) or not set(sets) <= set(RANKS) or any(type(v) is not str for v in sets.values()):
            raise ValueError("host CPU sets must map rank 0/1 to cpuset strings")
        for rank in RANKS:
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", profile["image_ids"][rank]):
                raise ValueError("exact per-node image ID required")
            if not profile["mounts"][rank].get("/model") or not profile["inference_environment"][rank]:
                raise ValueError("exact mounts and full inference environment required")
            env = profile["inference_environment"][rank]
            floor, schedule, plan = env.get(FLOOR_KEY), env.get(SCHEDULE_KEY), env.get(PLAN_KEY)
            if floor is not None:
                try:
                    value = float(floor)
                except ValueError:
                    raise ValueError("workspace floor must be a number") from None
                if not math.isfinite(value) or value < POLICY["memory_floor_gib"]:
                    raise ValueError("workspace floor below the 4 GiB hard floor can plan a STOP")
            if schedule not in SCHEDULES:
                raise ValueError("TF_CUDA_SCHED must be auto, spin, yield or blocking")
            cpus = sets.get(rank, "")
            if cpus and not CPU_SET.fullmatch(cpus):
                raise ValueError("host CPU set must be canonical cpuset syntax")
            if cpus:
                cpu_set_members(cpus)  # Reversed or out-of-range ranges are not canonical either.
            if plan not in (None, "auto"):
                if not isinstance(plan, str) or not re.fullmatch(r"0|[1-9][0-9]{0,3}", plan) or int(plan) > 4095:
                    raise ValueError("plan CPU must be auto or a canonical local CPU index 0..4095")
                if cpus and int(plan) not in cpu_set_members(cpus):
                    raise ValueError("plan CPU must lie inside the rank's host CPU set")
            knobs.append((env.get(POOL_KEY), floor, schedule))
        if knobs[0] != knobs[1]:
            raise ValueError("pool, workspace floor and host scheduling must match on both ranks")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, separators=(",", ":"), allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def env_values(text):
    """KEY=value lines as scripts/config.sh reads them (read, never run): quotes stripped, # comments dropped,
    the first assignment of a key wins."""
    values = {}
    for line in text.splitlines():
        match = ENV_LINE.fullmatch(line)
        if not match:
            continue
        key, value = match.groups()
        quoted = ENV_QUOTED.fullmatch(value)
        if quoted:
            value = quoted[1] if quoted[1] is not None else quoted[2]
        else:
            value = value.split("#", 1)[0].rstrip()
        values.setdefault(key, value)
    return values


def env_file_values(path):
    return env_values(Path(path).read_text())


def kill_condition(limits):
    return "temperature>={kill_c} for {kill_hold_seconds} s".format(**limits)


def transition_hold(limits):
    return TRANSITION_HOLD_PREFIX + str(limits["resume_c"]) + " and no local work"


def state_stale_seconds(limits):
    """A device state file older than this means spark-thermal-guard is not rewriting it."""
    return max(10, 5 * limits["interval_s"])


def read_device_limits(path):
    """The device's thermal limits from its KEY=value file (read, never run), with the file's SHA-256.

    The four GUARD_* keys must each be present once, as integers, with resume < pause <= kill < DEGRADE_C and a hold
    of at least 1 s; SPARK_THERMAL_INTERVAL_S (how often the device rewrites its state) must be an integer 1..10.
    The device's other settings are not the guard's business."""
    try:
        raw = Path(path).read_bytes()
        text = raw.decode()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("device limits " + str(path) + " unreadable (" + type(exc).__name__
                         + "): install the device's thermal limits on both nodes first") from None
    values = env_values(text)
    assigned = [match[1] for match in map(ENV_LINE.fullmatch, text.splitlines()) if match]
    limits = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    for key, name in (*THERMAL_LIMITS.items(), (INTERVAL_KEY, "interval_s")):
        if key not in values:
            raise ValueError("device limits " + str(path) + " lack " + key)
        if assigned.count(key) != 1:
            raise ValueError("device limits " + str(path) + " set " + key + " more than once")
        if not re.fullmatch(r"[0-9]{1,4}", values[key]):
            raise ValueError("device limits " + str(path) + ": " + key + "=" + values[key] + " is not an integer")
        limits[name] = int(values[key])
    if not limits["resume_c"] < limits["pause_c"] <= limits["kill_c"] < DEGRADE_C:
        raise ValueError("device limits " + str(path) + " out of order: GUARD_RESUME_C < GUARD_PAUSE_C"
                         + " <= GUARD_KILL_C < " + str(DEGRADE_C) + " required")
    if limits["kill_hold_seconds"] < 1:
        raise ValueError("device limits " + str(path) + ": GUARD_KILL_HOLD_S must be at least 1")
    if not 1 <= limits["interval_s"] <= 10:
        raise ValueError("device limits " + str(path) + ": " + INTERVAL_KEY + " must be 1..10 s")
    return limits


def check_production_config(path):
    """The production config must not carry thermal limits of its own: they are the device's."""
    try:
        values = env_file_values(path)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("production_config unreadable: " + type(exc).__name__) from None
    found = [key for key in THERMAL_LIMITS if key in values]
    if found:
        raise ValueError("production_config sets " + ", ".join(found) + ": " + DEVICE_HOME)


def load_config(path):
    config = json.loads(Path(path).read_bytes())
    if config.get("enabled") is not True or config.get("mode") != "production" or config.get("policy") != POLICY:
        raise ValueError("explicitly enabled exact production policy required")
    config.setdefault("device_limits", DEVICE_LIMITS)
    config.setdefault("device_state", DEVICE_STATE)
    for name in ("state_path", "samples_path", "events_path", "lease_path", "production_config", "control_socket",
                 "device_limits", "device_state"):
        value = Path(config[name])
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("absolute canonical path required: " + name)
    units = config.get("watchdog_units", [])
    if not isinstance(units, list) or any(not re.fullmatch(r"[A-Za-z0-9@_.-]+\.timer", u) for u in units):
        raise ValueError("watchdog units must be systemd timer names")
    upstream = urlsplit(config["upstream"])
    if upstream.scheme != "http" or upstream.hostname != "127.0.0.1" or not upstream.port:
        raise ValueError("native upstream must be loopback http with an explicit port")
    if config.get("listen_host") != "127.0.0.1" or type(config.get("listen_port")) is not int:
        raise ValueError("front must listen on loopback with an explicit port")
    if config["listen_port"] == upstream.port:
        raise ValueError("front and native ports must differ")
    for rank in RANKS:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["containers"][rank]):
            raise ValueError("invalid container name")
    validate_log_regex(config.get("boot_floor_log_regex"), "boot_floor_log_regex", 1)
    if config.get("resume_log_regex") is not None:
        validate_log_regex(config["resume_log_regex"], "resume_log_regex", 3)
    validate_profiles(config["profiles"])
    check_production_config(config["production_config"])
    thermal = read_device_limits(config["device_limits"])
    limits = config["limits"]
    for name, lower, upper in [
        ("max_inflight", 1, 4),
        ("max_pending", 1, 16),
        ("max_connections", 8, 64),
        ("max_body_bytes", 1, 64 * 1024 * 1024),
        ("body_buffers", 1, 4),
    ]:
        if type(limits[name]) is not int or not lower <= limits[name] <= upper:
            raise ValueError("invalid bounded limit: " + name)
    reserve = limits.get("memory_transient_reserve_gib", 0.5)
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or not 0.5 <= reserve <= 4:
        raise ValueError("transient reserve must be finite 0.5..4 GiB above 4.25 GiB")
    pause = limits.get("prefill_admission_pause_c")
    resume = limits.get("prefill_admission_resume_c")
    if "prefill_admission_pause_c" in limits or "prefill_admission_resume_c" in limits:
        # An optional pre-prefill throttle inside the device's pause band: never under its resume limit (too low), never
        # over its pause limit.
        if (type(pause) is not int or type(resume) is not int or not 1 <= resume < pause
                or not thermal["resume_c"] <= pause <= thermal["pause_c"]):
            raise ValueError("admission temperatures require integers 1<=resume<pause and "
                             + str(thermal["resume_c"]) + "<=pause<=" + str(thermal["pause_c"]) + "; both required")
    seconds = limits.get("queue_seconds", 1200)
    if type(seconds) is not int or not 1 <= seconds <= 1200:
        raise ValueError("queue timeout must be 1..1200 s")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", config["worker_ssh"]) or config["worker_ssh"].startswith("-"):
        raise ValueError("invalid worker SSH target")
    config["_thermal_limits"] = thermal  # The effective limits, read from the device (never from this config).
    return config


def daemon_lock(state_path):
    path = Path(str(state_path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream
