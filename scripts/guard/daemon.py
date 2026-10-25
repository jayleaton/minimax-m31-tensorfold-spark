"""Permanent guard daemon: front on the public port, durable policy and authenticated local operator control.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48), without its operator ladder.
"""

import argparse
import json
import os
import socket
import socketserver
import struct
import subprocess
import threading
import time
from pathlib import Path

from backend import Backend
from common import (
    POLICY,
    RANKS,
    atomic,
    daemon_lock,
    load_config,
    read_device_limits,
    transition_hold,
)
from front import Front, Server
from policy import Policy


class Manager:
    def __init__(self, config, backend=None, state=None):
        self.config = config
        self.backend = backend or Backend(config)
        self.closed = threading.Event()
        sha, _ = self.backend.profile()
        new = not Path(config["state_path"]).exists()
        if state is None and new:
            owned, mode = self.backend.ownership()
            if not owned:
                raise ValueError("new durable state only while the guard owns the pair: " + mode)
        if state is None:
            # The device's limits as load_config read them (a config dict that skipped it reads the device file).
            limits = config.get("_thermal_limits") or read_device_limits(config["device_limits"])
            state = Policy(config["state_path"], config["events_path"], limits=limits, initial_profile=sha)
        self.policy = state
        self.policy.configure_admission(config.get("limits", {}))
        self.last_action = 0
        self.boot_lock = threading.Lock()

    def tick(self):
        owned, mode = self.backend.ownership()
        current, _ = self.backend.profile()
        with self.policy.lock:
            expected = self.policy.data["bootstrap_profile"] or self.policy.data["profile"]
        if current != expected:
            owned, mode = False, "profile_mismatch"
        try:
            written = json.loads(Path(self.config["samples_path"]).read_bytes())
            samples, errors = written["samples"], written.get("errors") or {}
        except (OSError, ValueError, KeyError, AttributeError):
            samples, errors = {}, {"0": "no samples file", "1": "no samples file"}
        self.policy.update(samples, owned, mode, errors=errors if isinstance(errors, dict) else {})

    def act(self):
        state = self.policy.snapshot()
        if not state["owned"] or time.time() - self.last_action < 2:
            return
        expected = state["bootstrap_profile"] or state["profile"]
        if state["pending_action"] == "stop_owned":
            self.last_action = time.time()
            records = self.backend.stop_owned(expected)
            try:
                context = self.backend.stop_request_context(self.policy.data.get("stop_evidence", {}))
            except Exception as exc:
                context = {"error": type(exc).__name__}
            with self.policy.lock:
                if self.policy.data.get("stop_evidence") is not None:
                    self.policy.data["stop_evidence"]["native_request_context"] = context
                    self.policy.save()
                    self.policy.event("STOP_REQUEST_CONTEXT", time.time(), context=context)
                self.policy.event("HARD_STOP_ACTION", time.time(), records=records)
                if all(row["stopped"] for row in records):
                    self.policy.data["pending_action"] = None
                    self.policy.save()
        elif state["terminal_stop"]:
            self.last_action = time.time()
            self.backend.hold_heals(expected)

    def monitor(self):
        while not self.closed.is_set():
            try:
                self.tick()
            except Exception as exc:
                with self.policy.lock:
                    self.policy.fresh = False
                    self.policy.data["fresh"] = False
                    self.policy.data["monitor_error"] = str(exc)
                    self.policy.save()
                    self.policy.lock.notify_all()
            self.closed.wait(0.5)

    def actions(self):
        while not self.closed.is_set():
            try:
                self.act()
            except Exception as exc:
                with self.policy.lock:
                    self.policy.event("ACTION_REFUSED", time.time(), reason=str(exc))
            self.closed.wait(0.5)

    def verify_generation(self):
        state = self.policy.snapshot()
        if not state["ready"]:
            raise ValueError("guard state changed before upstream")
        identities = {rank: self.backend.identity(rank, state["profile"]) for rank in RANKS}
        current_ids = {rank: value["id"] for rank, value in identities.items()}
        with self.boot_lock:
            if self.policy.data.get("boot_verified_ids") != current_ids:
                proof = self.backend.boot_proof(state["profile"])
                with self.policy.lock:
                    self.policy.data["boot_verified_ids"] = current_ids
                    self.policy.save()
                    self.policy.event("NATIVE_BOOT_VERIFIED", time.time(), proof=proof)
        return identities

    def cool_idle_operator(self, target=None):
        state = self.policy.snapshot()
        limits = self.policy.limits
        owned, mode = self.backend.ownership()
        if not owned:
            raise ValueError("explicit profile transition requires guard ownership: " + mode)
        if (
            not state["fresh"]
            or state["inflight"]
            or len(state["samples"]) != 2
            or max(row["max_zone_c"] for row in state["samples"].values()) > limits["resume_c"]
            or any(row.get("device_state") != "normal" for row in state["samples"].values())
            or any(time.time() - row["epoch"] > 6 for row in state["samples"].values())
        ):
            raise ValueError(transition_hold(limits))
        try:
            return self.backend.idle()
        except Exception:
            for rank in RANKS:
                # No running native container may be mistaken for an idle/absent backend.
                try:
                    output = self.backend.run(rank, ["docker", "inspect", self.config["containers"][rank]])
                    if json.loads(output)[0]["State"]["Running"]:
                        raise ValueError("running native container without idle proof")
                except subprocess.CalledProcessError as exc:
                    if "no such" not in (exc.stderr or "").lower():
                        raise
            return {"native_absent_and_both_containers_stopped": True}

    def control(self, operation, target):
        if target not in self.config["profiles"]:
            raise ValueError("unapproved target profile")
        if operation == "prepare-profile":
            with self.policy.lock:
                same = target == self.policy.data["profile"]
                if same and self.policy.data["terminal_stop"]:
                    raise ValueError("same-profile STOP cannot be reset")
                if self.policy.data["bootstrap_profile"] is not None:
                    if self.policy.data["bootstrap_profile"] == target and not self.policy.data["hard_stop"]:
                        return self.policy.snapshot()
                    raise ValueError("another pending profile exists")
                self.policy.data.update(paused=True, operator_hold=True)
                self.policy.save()
            proof = self.cool_idle_operator(target)
            with self.policy.lock:
                archive = Path(self.config["state_path"]).parent / "archives"
                archive.mkdir(exist_ok=True)
                if len(list(archive.glob("*.json"))) >= 128:
                    raise ValueError("archive bound reached; never silently delete STOP history")
                atomic(archive / (str(time.time_ns()) + ".json"), self.policy.snapshot())
                self.policy.data.update(
                    bootstrap_profile=target,
                    terminal_stop=not same,
                    hard_stop=False,
                    pending_action=None,
                    operator_hold=False,
                )
                self.policy.data["episodes"] = {"0": None, "1": None}
                self.policy.data.pop("stop_evidence", None)
                self.policy.save()
                self.policy.event(
                    "PROFILE_BOOTSTRAP_PREPARED",
                    time.time(),
                    target=target,
                    proof=proof,
                    same_healthy_profile=same,
                )
        elif operation == "authorize-profile":
            self.cool_idle_operator(target)
            with self.policy.lock:
                if self.policy.data["bootstrap_profile"] != target:
                    if self.policy.data["profile"] == target and not self.policy.data["terminal_stop"]:
                        return self.policy.snapshot()
                    raise ValueError("pending profile mismatch")
                if self.policy.data["hard_stop"]:
                    raise ValueError("new hard STOP prevents authorization")
            if self.backend.profile()[0] != target:
                raise ValueError("profile mismatch; no reset")
            ids = {rank: self.backend.identity(rank, target) for rank in RANKS}
            self.backend.idle()
            proof = self.backend.boot_proof(target)
            with self.policy.lock:
                if self.policy.data["hard_stop"] or not self.policy.fresh:
                    raise ValueError("hard STOP/stale telemetry during native boot proof")
                if any(row["available_gib"] < POLICY["memory_floor_gib"] for row in self.policy.samples.values()):
                    raise ValueError("fresh available memory below4GiB")
                self.policy.data["boot_verified_ids"] = {rank: row["id"] for rank, row in ids.items()}
                self.policy.data.update(
                    profile=target,
                    bootstrap_profile=None,
                    terminal_stop=False,
                    hard_stop=False,
                    paused=True,
                    pending_action=None,
                )
                self.policy.save()
                self.policy.event(
                    "PROFILE_AUTHORIZED",
                    time.time(),
                    target=target,
                    identities=ids,
                    boot_proof=proof,
                )
                self.policy.lock.notify_all()
        else:
            raise ValueError("unknown control operation")
        return self.policy.snapshot()


class Control(socketserver.StreamRequestHandler):
    def handle(self):
        uid = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        if uid not in (0, os.getuid()):
            return
        self.request.settimeout(10)
        raw = self.rfile.readline(16385)
        if len(raw) > 16384:
            return
        try:
            request = json.loads(raw)
            result = {
                "ok": True,
                "state": self.server.manager.control(request["operation"], request["profile"]),
            }
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(result).encode() + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except Exception as exc:
        raise SystemExit(78) from exc
    config["_path"] = args.config
    lock = daemon_lock(config["state_path"])
    manager = Manager(config)
    server = Server((config["listen_host"], config["listen_port"]), Front)
    server.configure(config, manager)
    control_path = Path(config["control_socket"])
    if control_path.exists():
        control_path.unlink()  # Exclusive daemon flock prevents removing another live instance's socket.
    control = socketserver.UnixStreamServer(str(control_path), Control)
    os.chmod(control_path, 0o600)
    control.manager = manager
    for target in (manager.monitor, manager.actions, control.serve_forever):
        threading.Thread(target=target, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        manager.closed.set()
        control.shutdown()
        control.server_close()
        server.server_close()
        control_path.unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    main()
