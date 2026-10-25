"""Ownership-qualified native identity and hard-stop actions; no broad process killing.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48). The engine log lines it reads (the boot
memory forecast, RAM session resumes) are engine-specific, so their patterns come from the guard config.
"""

import hashlib
import http.client
import json
import os
import re
import shlex
import stat
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from common import RANKS, inference_env

def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("nonregular ownership evidence")
    return info


class Backend:
    def __init__(self, config):
        self.config = config
        self.upstream = urlsplit(config["upstream"])

    def run(self, rank, args, timeout=6):
        command = (
            args
            if rank == "0"
            else [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=3",
                self.config["worker_ssh"],
                shlex.join(args),
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
        return result.stdout + result.stderr if args[:2] == ["docker", "logs"] else result.stdout

    def profile(self):
        path = Path(self.config["production_config"])
        regular(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in self.config["profiles"]:
            raise ValueError("unapproved prod config SHA")
        return sha, self.config["profiles"][sha]

    def ownership(self):
        """A fresh operator maintenance lease makes the guard stand aside: no STOP actions, no admissions."""
        lease = Path(self.config["lease_path"])
        if os.path.lexists(lease):
            info = regular(lease)
            if time.time() - info.st_mtime < self.config.get("lease_minutes", 20) * 60:
                return False, "operator_lease"
        return True, "owned"

    def identity(self, rank, expected_sha, require_running=True):
        owned, mode = self.ownership()
        sha, profile = self.profile()
        if not owned or sha != expected_sha:
            raise ValueError("ownership/profile changed: " + mode)
        item = json.loads(self.run(rank, ["docker", "inspect", self.config["containers"][rank]]))[0]
        if item["Image"] != profile["image_ids"][rank] or item["Name"] != "/" + self.config["containers"][rank]:
            raise ValueError("container image/name mismatch")
        if require_running and item["State"]["Running"] is not True:
            raise ValueError("owned native container not running")
        host = item.get("HostConfig")
        if not isinstance(host, dict) or host.get("CpusetCpus") != profile.get("cpu_sets", {}).get(rank, ""):
            raise ValueError("container host CPU set changed or unavailable")
        mounts = {m["Destination"]: m["Source"] for m in item["Mounts"]}
        if mounts != profile["mounts"][rank]:
            raise ValueError("container mounts changed")
        model = [m for m in item["Mounts"] if m["Destination"] == "/model"]
        if len(model) != 1 or model[0]["RW"] is not False:
            raise ValueError("model mount must be read only")
        env = inference_env(item["Config"]["Env"])
        if env != profile["inference_environment"][rank]:
            raise ValueError("inference environment changed")
        return {"id": item["Id"], "image": item["Image"], "profile": sha, "lease_mode": mode}

    def request_token_count(self, body, path):
        if path.split("?", 1)[0] not in ("/v1/chat/completions", "/v1/completions"):
            return {"count_source": "unsupported_route"}
        document = json.loads(body)
        # Avoid running image preparation twice merely for STOP metadata.
        if any(isinstance(m.get("content"), list) for m in document.get("messages", [])):
            return {"count_source": "multimodal_not_retokenized"}
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=3)
        try:
            connection.request("POST", "/tokenize", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("native tokenize failed")
            value = json.loads(response.read(16 * 1024 * 1024))
            count = value["count"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid native token count")
            return {"prompt_tokens": count, "count_source": "native_tokenize"}
        finally:
            connection.close()

    def stop_request_context(self, evidence):
        requests = evidence.get("request_metadata", {})
        if not requests:
            return {"requests": {}, "resume_observations": []}
        since = min(row["epoch"] for row in requests.values())
        logs = self.run("0", ["docker", "logs", "--timestamps", "--since", str(since),
                              "--tail", "400", self.config["containers"]["0"]], timeout=4)
        observations = []
        # (slot, resumed tokens, prompt tokens); without a configured pattern nothing is matched (stays unknown).
        pattern = self.config.get("resume_log_regex")
        for line in logs.splitlines() if pattern else ():
            match = re.search(pattern, line)
            if match:
                observations.append({"slot": int(match[1]), "resumed_tokens": int(match[2]),
                                     "prompt_tokens": int(match[3]), "native_log": line})
        enriched = json.loads(json.dumps(requests))
        for observation in observations:
            matches = [key for key, row in enriched.items()
                       if row.get("prompt_tokens") == observation["prompt_tokens"]]
            same_count = [row for row in observations
                          if row["prompt_tokens"] == observation["prompt_tokens"]]
            if len(matches) == 1 and len(same_count) == 1:
                enriched[matches[0]].update(resumed_tokens=observation["resumed_tokens"],
                                           resumed_source="native_session_log")
        return {"requests": enriched, "resume_observations": observations}

    def health(self):
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("native health not200")
            return json.loads(response.read(1048576))
        finally:
            connection.close()

    def idle(self):
        health = self.health()
        if health.get("ok") is not True or any(health.get(key, 0) != 0 for key in ("requests_running", "queued")):
            raise ValueError("native not healthy/idle")
        if health.get("inflight") != 0:
            raise ValueError("explicit native inflight0 required")
        return health

    def boot_proof(self, expected_sha):
        identities = {rank: self.identity(rank, expected_sha) for rank in RANKS}
        health = self.idle()
        if health.get("streams", {}).get("max") != self.config["limits"]["max_inflight"]:
            raise ValueError("native slots must equal the guard's max_inflight")
        records = {}
        for rank in RANKS:
            logs = self.run(rank, ["docker", "logs", "--tail", "400", self.config["containers"][rank]])
            # The engine's boot memory forecast: one capture group, the worst serving point's free GiB.
            pattern = self.config["boot_floor_log_regex"]
            lines = [line for line in logs.splitlines() if re.search(pattern, line)]
            line = lines[-1] if lines else ""
            match = re.search(pattern, line)
            if match is None or float(match[1]) < 4:
                raise ValueError("current-container native boot floor proof missing/failed")
            records[rank] = {"identity": identities[rank], "native_boot_line": line, "worst_gib": float(match[1])}
        return {"native_health": health, "ranks": records, "epoch": time.time()}

    def hold_heals(self, expected_sha):
        owned, _ = self.ownership()
        if not owned or self.profile()[0] != expected_sha:
            raise ValueError("ownership changed before watchdog hold")
        units = self.config.get("watchdog_units", [])
        if not units:
            return
        subprocess.run(
            ["systemctl", "--user", "stop", *units],
            capture_output=True,
            timeout=5,
            check=True,
        )

    def stop_owned(self, expected_sha):
        records = []
        self.hold_heals(expected_sha)
        for rank in RANKS:
            try:
                before = self.identity(rank, expected_sha, require_running=False)
                if self.identity(rank, expected_sha, require_running=False)["id"] != before["id"]:
                    raise ValueError("container identity changed before hard stop")
                output = self.run(rank, ["docker", "stop", "--time", "0", before["id"]], timeout=8)
                records.append({"rank": rank, "id": before["id"], "stopped": True, "output": output})
            except Exception as exc:
                records.append({"rank": rank, "stopped": False, "error": str(exc)})
        return records
