"""The sampler's reading: the hottest of the ACPI thermal zones and the GPU, failing closed when either is missing.

A fake nvidia-smi on PATH, a fake thermal glob root and a fake device (limits file and state file); the real nvidia-smi
and the real device files are never reached (PATH holds only the fake; every sample is given the fake paths). Both the
head's local_sample and the worker's NODE_SAMPLE (run here with this Python) are checked alike.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sampler
from guard_fixtures import write_device

FAILED = (ValueError, OSError, subprocess.SubprocessError)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bin, self.zones = root / "bin", root / "thermal"
        self.bin.mkdir()
        self.zones.mkdir()
        self.pattern = str(self.zones / "thermal_zone*" / "temp")
        self.meminfo = root / "meminfo"
        self.meminfo.write_text(
            "MemTotal:       131072000 kB\nMemAvailable:    8388608 kB\n"
        )
        self.args = root / "nvidia-smi.args"
        self.limits, self.state = write_device(root)
        self.device = [str(self.limits), str(self.state)]

    def zone_values(self, *values):
        for index, value in enumerate(values):
            zone = self.zones / ("thermal_zone" + str(index))
            zone.mkdir()
            (zone / "temp").write_text(value + "\n")

    def gpu(self, body):
        script = self.bin / "nvidia-smi"
        script.write_text(f'#!/bin/sh\necho "$@" > {self.args}\n{body}\n')
        script.chmod(0o755)

    def local(self):
        with patch.dict(os.environ, {"PATH": str(self.bin)}):
            return sampler.local_sample(self.pattern, self.meminfo, *self.device)

    def node(self):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                sampler.NODE_SAMPLE,
                self.pattern,
                *self.device,
            ],
            env={"PATH": str(self.bin)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "node sample failed")
        return json.loads(result.stdout)

    def both(self):
        return {"local": self.local, "node": self.node}.items()

    def test_reading_is_the_hottest_of_gpu_and_acpi(self):
        for zones, gpu, expected in (
            (
                ("85000", "70000", "bogus", "200000", "0"),
                "printf '88\\n91\\n'",
                (91, 91, 85),
            ),
            (("93500", "40000"), "echo 80", (93.5, 80, 93.5)),
            (("86000",), "echo ' 86'", (86, 86, 86)),
        ):
            for name, sample in self.both():
                with self.subTest(zones=zones, gpu=gpu, sampler=name):
                    self.setUp()
                    self.zone_values(*zones)
                    self.gpu(gpu)
                    row = sample()
                    self.assertEqual(
                        (row["max_zone_c"], row["gpu_c"], row["acpi_c"]), expected
                    )
                    self.assertEqual(
                        self.args.read_text().split(), sampler.GPU_QUERY[1:]
                    )
                    if name == "local":
                        self.assertEqual(row["available_gib"], 8)
                    self.assertLessEqual(abs(row["epoch"] - time.time()), 5)

    def test_fails_closed_without_both_sources(self):
        for zones, gpu in (
            (("bogus", "0", "200000"), "echo 80"),  # no valid ACPI zone
            ((), "echo 80"),  # no zone at all
            (("80000",), "exit 1"),  # nvidia-smi fails
            (("80000",), "echo '[N/A]'"),  # no GPU reading
            (("80000",), "true"),  # empty output
            (("80000",), "printf '80\\n[N/A]\\n'"),  # one GPU unreadable hides nothing
            (("80000",), "echo 150"),  # out of range
            (("80000",), "echo nan"),
            (("80000",), None),  # no nvidia-smi on PATH
        ):
            for name, sample in self.both():
                with self.subTest(zones=zones, gpu=gpu, sampler=name):
                    self.setUp()
                    self.zone_values(*zones)
                    if gpu is not None:
                        self.gpu(gpu)
                    with self.assertRaises(FAILED):
                        sample()

    def test_gpu_query_is_bounded(self):
        self.gpu("exec /bin/sleep 5")
        with (
            patch.dict(os.environ, {"PATH": str(self.bin)}),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            sampler.gpu_temperatures(timeout=0.2)
        self.zone_values("80000")
        with patch(
            "sampler.subprocess.run", return_value=SimpleNamespace(stdout="80\n")
        ) as run:
            sampler.local_sample(self.pattern, self.meminfo, *self.device)
        self.assertEqual(run.call_args.args[0], sampler.GPU_QUERY)
        self.assertEqual(run.call_args.kwargs["timeout"], 1.5)
        self.assertIn("timeout=1.5", sampler.NODE_SAMPLE)

    def test_sample_reports_the_device_state_its_age_and_the_limits_sha(self):
        sha = hashlib.sha256(self.limits.read_bytes()).hexdigest()
        for name, sample in self.both():
            with self.subTest(sampler=name):
                self.setUp()
                self.zone_values("80000")
                self.gpu("echo 80")
                self.state.write_text("throttled\n")
                old = time.time() - 30
                os.utime(self.state, (old, old))
                row = sample()
                self.assertEqual(
                    (row["device_state"], row["device_limits_sha256"]),
                    ("throttled", sha),
                )
                self.assertAlmostEqual(row["device_state_age_s"], 30, delta=5)
                self.state.write_bytes(
                    b"\xff garbage that is longer than sixty-four bytes " + b"x" * 80
                )
                self.assertTrue(
                    sample()["device_state"].endswith("x"),
                    "garbage is reported, bounded, never raised",
                )
                self.assertLessEqual(len(sample()["device_state"]), 64)

    def test_unreadable_device_files_are_reported_not_hidden(self):
        for name, sample in self.both():
            with self.subTest(sampler=name):
                self.setUp()
                self.zone_values("80000")
                self.gpu("echo 80")
                self.state.unlink()
                self.limits.unlink()
                row = sample()
                self.assertEqual(
                    row["max_zone_c"], 80, "the memory and temperature reading stays"
                )
                self.assertIsNone(row["device_state"])
                self.assertIsNone(row["device_state_age_s"])
                self.assertIsNone(row["device_limits_sha256"])
                self.assertIn("No such file", row["device_state_error"])
                self.assertIn("No such file", row["device_limits_error"])

    def test_worker_reads_its_own_device_files_at_the_configured_paths(self):
        config = {
            "worker_ssh": "worker",
            "device_limits": "/srv/a/thermal.env",
            "device_state": "/srv/b/state",
        }
        with patch(
            "sampler.subprocess.run", return_value=SimpleNamespace(stdout="{}")
        ) as run:
            sampler.worker_sample(config)
        remote = run.call_args.args[0][-1]
        paths = [sampler.THERMAL_GLOB, "/srv/a/thermal.env", "/srv/b/state"]
        self.assertTrue(remote.endswith(" " + shlex.join(paths)))
        with patch(
            "sampler.subprocess.run", return_value=SimpleNamespace(stdout="{}")
        ) as run:
            sampler.worker_sample({"worker_ssh": "worker"})
        self.assertTrue(
            run.call_args.args[0][-1].endswith(
                "/etc/spark/thermal.env /run/spark-thermal/state"
            )
        )

    def test_worker_ssh_timeout_bounds_the_whole_sample(self):
        row = {
            "epoch": 1,
            "available_gib": 8,
            "max_zone_c": 80,
            "gpu_c": 80,
            "acpi_c": 70,
        }
        with patch(
            "sampler.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps(row)),
        ) as run:
            self.assertEqual(sampler.worker_sample({"worker_ssh": "worker"}), row)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["ssh", "-n"])
        self.assertIn("ConnectTimeout=2", command)
        self.assertTrue(command[-1].startswith("python3 -c "))
        self.assertGreater(run.call_args.kwargs["timeout"], 1.5)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 4)
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
