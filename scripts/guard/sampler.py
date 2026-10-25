"""Bounded atomic two-node telemetry; missing worker remains explicitly missing/fail-closed.

Each node's reading (`max_zone_c`, the name kept from the original) is the hottest of every valid ACPI thermal zone and
the GPU temperature; `acpi_c` and `gpu_c` record both. A node with no valid zone or no valid GPU reading has no sample:
missing telemetry blocks admission and ends a hot episode (policy.py).

Each sample also carries the node's device evidence, read as it is (policy.py judges it): `device_limits_sha256` (the
SHA-256 of the node's device limits file, /etc/spark/thermal.env, or None with `device_limits_error`), and
`device_state` / `device_state_age_s` (spark-thermal-guard's /run/spark-thermal/state, stripped, and its age from its
mtime, or None with `device_state_error`). A failed device read never hides the memory reading.

Ported from the guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48).
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from common import DEVICE_LIMITS, DEVICE_STATE, atomic, load_config

THERMAL_GLOB = "/sys/class/thermal/thermal_zone*/temp"
GPU_QUERY = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
GPU_TIMEOUT = 1.5
# The worker's copy of local_sample (python3 -c over ssh; argv[1..3] name the zone glob, the device limits file and the
# device state file). Keep the two alike.
NODE_SAMPLE = r"""
import glob,hashlib,json,math,os,re,subprocess,sys,time
from pathlib import Path
def ok(n):return math.isfinite(n) and 0<n<150
a=sys.argv[1:]+['']*3
zones=a[0] or '/sys/class/thermal/thermal_zone*/temp'
limits=a[1] or '/etc/spark/thermal.env';state=a[2] or '/run/spark-thermal/state'
mem=Path('/proc/meminfo').read_text();available=int(re.search(r'^MemAvailable:\s+(\d+) kB$',mem,re.M)[1])/1048576
acpi=[]
for f in glob.glob(zones):
 try:
  n=float(Path(f).read_text())/1000
  if ok(n):acpi.append(n)
 except (OSError,ValueError):pass
if not acpi:raise SystemExit('no valid thermal zones')
out=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=1.5,check=True).stdout
try:gpu=[float(x) for x in out.split()]
except ValueError:gpu=[]
if not gpu or not all(map(ok,gpu)):raise SystemExit('no valid GPU temperature')
row={'epoch':time.time(),'available_gib':available,'max_zone_c':max(acpi+gpu),'gpu_c':max(gpu),'acpi_c':max(acpi)}
try:row['device_limits_sha256']=hashlib.sha256(Path(limits).read_bytes()).hexdigest()
except OSError as e:row.update(device_limits_sha256=None,device_limits_error=str(e))
try:
 with open(state,'rb') as f:m=os.fstat(f.fileno()).st_mtime;text=f.read(64).decode(errors='replace').strip()
 row.update(device_state=text,device_state_age_s=time.time()-m)
except OSError as e:row.update(device_state=None,device_state_age_s=None,device_state_error=str(e))
print(json.dumps(row))
"""


def valid_temperature(value):
    return math.isfinite(value) and 0 < value < 150


def acpi_temperatures(pattern=THERMAL_GLOB):
    values = []
    for name in glob.glob(pattern):
        try:
            value = float(Path(name).read_text()) / 1000
            if valid_temperature(value):
                values.append(value)
        except (OSError, ValueError):
            pass
    if not values:
        raise ValueError("no valid thermal zones")
    return values


def gpu_temperatures(command=GPU_QUERY, timeout=GPU_TIMEOUT):
    """Every GPU's temperature; one unreadable GPU (e.g. "[N/A]") fails the sample rather than hiding it."""
    output = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True).stdout
    try:
        values = [float(item) for item in output.split()]
    except ValueError:
        values = []
    if not values or not all(map(valid_temperature, values)):
        raise ValueError("no valid GPU temperature")
    return values


def device_evidence(device_limits=DEVICE_LIMITS, device_state=DEVICE_STATE):
    """The node's device limits SHA-256 and spark-thermal-guard state, as read (None plus an error when unreadable)."""
    row = {}
    try:
        row["device_limits_sha256"] = hashlib.sha256(Path(device_limits).read_bytes()).hexdigest()
    except OSError as exc:
        row.update(device_limits_sha256=None, device_limits_error=str(exc))
    try:
        with open(device_state, "rb") as stream:
            mtime = os.fstat(stream.fileno()).st_mtime
            text = stream.read(64).decode(errors="replace").strip()
        row.update(device_state=text, device_state_age_s=time.time() - mtime)
    except OSError as exc:
        row.update(device_state=None, device_state_age_s=None, device_state_error=str(exc))
    return row


def local_sample(thermal_glob=THERMAL_GLOB, meminfo="/proc/meminfo", device_limits=DEVICE_LIMITS,
                 device_state=DEVICE_STATE):
    memory = Path(meminfo).read_text()
    available = int(re.search(r"^MemAvailable:\s+(\d+) kB$", memory, re.MULTILINE)[1]) / 1048576
    acpi = acpi_temperatures(thermal_glob)
    gpu = gpu_temperatures()
    return {"epoch": time.time(), "available_gib": available, "max_zone_c": max(acpi + gpu),
            "gpu_c": max(gpu), "acpi_c": max(acpi), **device_evidence(device_limits, device_state)}


def worker_sample(config):
    # The worker reads its own device files at the same paths as the head.
    paths = [THERMAL_GLOB, config.get("device_limits", DEVICE_LIMITS), config.get("device_state", DEVICE_STATE)]
    result = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=2",
            config["worker_ssh"],
            "python3 -c " + shlex.quote(NODE_SAMPLE) + " " + shlex.join(paths),
        ],
        capture_output=True,
        text=True,
        timeout=4,  # Bounds the whole remote sample, nvidia-smi included.
        check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            started = time.monotonic()
            result = {"samples": {}, "errors": {}}
            jobs = {
                "0": executor.submit(local_sample, device_limits=config["device_limits"],
                                     device_state=config["device_state"]),
                "1": executor.submit(worker_sample, config),
            }
            for rank, job in jobs.items():
                try:
                    result["samples"][rank] = job.result(timeout=5)
                except Exception as exc:
                    result["errors"][rank] = str(exc)
            atomic(config["samples_path"], result)
            time.sleep(max(0.1, 2 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
