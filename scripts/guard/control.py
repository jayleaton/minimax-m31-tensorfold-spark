"""Explicit same-UID profile control; automatic bounded cooling/drain wait, no STOP reset bypass.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48).
"""

import argparse
import json
import socket
import time

from common import TRANSITION_HOLD_PREFIX, load_config

HOLD_ERRORS = {
    "running native container without idle proof",
}


def hold(error):
    """A known wait: cooling (the device's resume limit is in the text) or a drain."""
    return error in HOLD_ERRORS or (isinstance(error, str) and error.startswith(TRANSITION_HOLD_PREFIX))


def transaction(config, operation, sha):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(120)
    try:
        client.connect(config["control_socket"])
        client.sendall(json.dumps({"operation": operation, "profile": sha}).encode() + b"\n")
        with client.makefile("rb") as response:
            raw = response.readline(2 * 1024 * 1024)
        return json.loads(raw)
    finally:
        client.close()


def wait_control(config, operation, sha, wait_seconds=1200, monotonic=time.monotonic, sleep=time.sleep):
    deadline = monotonic() + wait_seconds
    while True:
        result = transaction(config, operation, sha)
        if result.get("ok") is True or not hold(result.get("error")):
            return result
        if monotonic() >= deadline:
            return {
                "ok": False,
                "error": "automatic cooling/drain wait deadline",
                "last_hold": result,
            }
        sleep(min(2, max(0, deadline - monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("operation", choices=["prepare-profile", "authorize-profile"])
    parser.add_argument("--sha", required=True)
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 1800:
        parser.error("wait seconds must be0..1800")
    result = wait_control(load_config(args.config), args.operation, args.sha, args.wait_seconds)
    print(json.dumps(result))
    raise SystemExit(0 if result.get("ok") is True else 78)


if __name__ == "__main__":
    main()
