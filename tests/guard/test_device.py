"""The device's limits and state in the guard: same limits file on both nodes, a fresh spark-thermal-guard state.

The limits come from a fake device file (guard_fixtures.DEVICE_ENV unless a test writes others) and every sample's
device evidence is synthetic; the real /etc/spark and /run/spark-thermal are never read.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import load_config, read_device_limits
from control import hold
from daemon import Manager
from guard_fixtures import (
    DEVICE_ENV,
    LIMITS,
    make_policy,
    production_config,
    row,
    write_config,
    write_device,
)

# Another valid device file: every limit differs from DEVICE_ENV's, and the state goes stale after max(10, 5 * 4) s.
OTHER_ENV = "GUARD_PAUSE_C=88\nGUARD_RESUME_C=80\nGUARD_KILL_C=90\nGUARD_KILL_HOLD_S=5\nSPARK_THERMAL_INTERVAL_S=4\n"


class DeviceFixture(unittest.TestCase):
    limits = LIMITS

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = make_policy(
            self.root, limits=self.limits, initial_profile="current", now=1000
        )
        self.now = 1000
        self.aborted = False
        self.p.active["a"] = SimpleNamespace(
            shutdown=lambda *_: setattr(self, "aborted", True)
        )

    def sample(self, a=80, b=80, owned=True, **rank1):
        """One-second step; keyword arguments override rank 1's device evidence."""
        self.now += 1
        sha = self.limits["sha256"]
        rows = {
            "0": row(self.now, a, device_limits_sha256=sha),
            "1": row(self.now, b, **{"device_limits_sha256": sha, **rank1}),
        }
        self.p.update(rows, owned, "owned" if owned else "operator_lease", self.now)

    def events(self):
        return [
            json.loads(line)["event"]
            for line in (self.root / "events").read_text().splitlines()
        ]


class DeviceEvidenceTests(DeviceFixture):
    def test_worker_limits_sha_mismatch_blocks_and_resets_the_hold(self):
        self.sample()
        self.assertTrue(self.p.ready())
        for _ in range(20):
            self.sample(80, 93)
        self.sample(80, 93, device_limits_sha256="0" * 64)
        self.assertFalse(self.p.fresh)
        self.assertFalse(self.p.ready(), "admission blocked")
        self.assertIsNone(self.p.data["episodes"]["1"], "the hot episode ends")
        reason = self.p.snapshot()["invalid_samples"]["1"]
        self.assertIn("device limits differ from the head's", reason)
        self.assertIn("000000000000 here, " + LIMITS["sha256"][:12], reason)
        for _ in range(30):
            self.sample(80, 93)
        self.assertFalse(
            self.p.data["hard_stop"], "the 30 s hold restarted after the invalid sample"
        )
        self.sample(80, 93)
        self.assertTrue(self.p.data["hard_stop"])

    def test_unreadable_worker_limits_block(self):
        self.sample(
            device_limits_sha256=None,
            device_limits_error="[Errno 2] No such file or directory",
        )
        self.assertFalse(self.p.ready())
        self.assertIn(
            "device limits unreadable on this node ([Errno 2]",
            self.p.snapshot()["invalid_samples"]["1"],
        )
        self.sample()
        self.assertTrue(self.p.ready())
        self.assertEqual(self.p.snapshot()["invalid_samples"], {})

    def test_missing_stale_or_garbage_state_blocks(self):
        for name, evidence in (
            (
                "missing",
                {
                    "device_state": None,
                    "device_state_age_s": None,
                    "device_state_error": "No such file",
                },
            ),
            ("stale", {"device_state_age_s": 10.5}),
            ("no age", {"device_state_age_s": None}),
            ("garbage", {"device_state": "hot"}),
            ("empty", {"device_state": ""}),
            ("cased", {"device_state": "Normal"}),
        ):
            with self.subTest(state=name):
                self.setUp()
                self.sample()
                self.assertTrue(self.p.ready())
                self.sample(**evidence)
                self.assertFalse(self.p.ready())
                self.assertIn(
                    "spark-thermal-guard not running?",
                    self.p.snapshot()["invalid_samples"]["1"],
                )
                self.assertFalse(self.p.data["hard_stop"])
        self.setUp()
        self.sample(device_state_age_s=10)
        self.assertTrue(self.p.ready(), "max(10, 5 * 2) s is still fresh")
        self.assertEqual(self.p.snapshot()["device_state_stale_seconds"], 10)

    def test_sampler_errors_show_in_status(self):
        self.now += 1
        self.p.update(
            {"0": row(self.now)},
            True,
            "owned",
            self.now,
            errors={"1": "no valid GPU temperature"},
        )
        self.assertEqual(
            self.p.snapshot()["invalid_samples"],
            {"1": "no sample: no valid GPU temperature"},
        )


class DeviceStateTests(DeviceFixture):
    def test_throttled_pauses_and_blocks_resume(self):
        self.sample(80, 80, device_state="throttled")
        self.assertTrue(self.p.data["paused"], "throttled pauses at any reading")
        self.assertFalse(self.p.ready())
        self.sample(70, 70, device_state="throttled")
        self.assertTrue(
            self.p.data["paused"], "a throttled node blocks resume even when cool"
        )
        self.sample(70, 70)
        self.assertFalse(self.p.data["paused"])
        self.assertTrue(self.p.ready())
        self.sample(92, 80)
        self.assertTrue(
            self.p.data["paused"],
            "a reading at the pause limit pauses with the device still normal",
        )
        self.sample(80, 80)
        self.assertFalse(self.p.data["paused"])
        self.assertEqual(self.events(), ["PAUSE", "RESUME", "PAUSE", "RESUME"])
        self.assertFalse(self.p.data["terminal_stop"])

    def test_device_emergency_is_an_immediate_hard_stop_naming_the_rank(self):
        self.sample()
        self.sample(80, 80, device_state="emergency", device_state_age_s=1.5)
        self.assertTrue(self.p.data["hard_stop"])
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertTrue(self.aborted, "in-flight work cancelled")
        self.assertEqual(
            self.p.data["pending_action"], "stop_owned", "both owned containers stop"
        )
        self.assertEqual(self.p.data["stop_reason"], "device emergency on rank 1")
        state = json.loads((self.root / "state").read_text())
        evidence = state["stop_evidence"]
        self.assertEqual(
            evidence["fired"],
            [
                {
                    "condition": "device emergency on rank 1",
                    "rank": "1",
                    "value": "emergency",
                    "device_state_age_s": 1.5,
                }
            ],
        )
        self.assertEqual(evidence["device_limits"], LIMITS)
        self.assertEqual(evidence["raw_samples"]["1"]["device_state"], "emergency")
        self.assertEqual(self.events()[-1], "HARD_STOP")

    def test_emergency_stops_even_when_the_limits_file_differs(self):
        self.sample(80, 80, device_state="emergency", device_limits_sha256="f" * 64)
        self.assertEqual(
            self.p.data["stop_evidence"]["fired"][0]["condition"],
            "device emergency on rank 1",
        )
        self.assertIn("1", self.p.data["stop_evidence"]["invalid_samples"])

    def test_emergency_needs_a_fresh_state_and_ownership(self):
        self.sample(80, 80, device_state="emergency", device_state_age_s=60)
        self.assertFalse(
            self.p.data["hard_stop"], "a stale state is unknown, not an emergency"
        )
        self.assertFalse(self.p.ready())
        self.sample(80, 80, owned=False, device_state="emergency")
        self.assertFalse(self.p.data["hard_stop"], "an operator lease never stops")
        self.now += 1
        self.p.update({"0": row(self.now, state="emergency")}, True, "owned", self.now)
        self.assertEqual(self.p.data["stop_reason"], "device emergency on rank 0")


class OtherLimitsTests(DeviceFixture):
    """A device file with other valid values changes behaviour: the limits are read, not hardcoded."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path, _ = write_device(directory.name, OTHER_ENV)
        self.limits = read_device_limits(path)
        super().setUp()

    def test_pause_and_resume_follow_the_file(self):
        self.sample(87, 80)
        self.assertFalse(self.p.data["paused"])
        self.sample(88, 80)
        self.assertTrue(self.p.data["paused"])
        self.sample(81, 81)
        self.assertTrue(self.p.data["paused"], "86 is not this device's resume limit")
        self.sample(80, 80)
        self.assertFalse(self.p.data["paused"])

    def test_kill_hold_follows_the_file(self):
        for _ in range(5):
            self.sample(90, 80)
        self.assertFalse(self.p.data["hard_stop"])
        self.sample(90, 80)
        self.assertTrue(self.p.data["hard_stop"])
        self.assertEqual(self.p.data["stop_reason"], "temperature>=90 for 5 s")
        self.assertEqual(self.p.data["stop_evidence"]["device_limits"]["kill_c"], 90)

    def test_state_staleness_follows_the_interval(self):
        self.sample(device_state_age_s=19.5)
        self.assertTrue(self.p.ready(), "5 * 4 s")
        self.sample(device_state_age_s=20.5)
        self.assertFalse(self.p.ready())


class ManagerDeviceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = production_config(self.root)
        self.sha = next(iter(self.config["profiles"]))
        self.backend = SimpleNamespace(
            profile=lambda: (self.sha, {}), ownership=lambda: (True, "owned")
        )

    def manager(self, text=DEVICE_ENV):
        Path(self.config["device_limits"]).write_text(text)
        return Manager(
            load_config(write_config(self.root, self.config)), backend=self.backend
        )

    def test_the_daemon_uses_the_device_files_limits(self):
        manager = self.manager(OTHER_ENV)
        self.assertEqual(
            (manager.policy.limits["pause_c"], manager.policy.limits["path"]),
            (88, self.config["device_limits"]),
        )
        self.assertEqual(
            manager.policy.snapshot()["device_limits"], manager.policy.limits
        )

    def test_tick_passes_sampler_errors_to_status(self):
        manager = self.manager()
        Path(self.config["samples_path"]).write_text(
            json.dumps(
                {
                    "samples": {"0": row(time.time())},
                    "errors": {"1": "ssh: connect timed out"},
                }
            )
        )
        manager.tick()
        self.assertEqual(
            manager.policy.snapshot()["invalid_samples"],
            {"1": "no sample: ssh: connect timed out"},
        )
        self.assertFalse(manager.policy.ready())

    def test_operator_transition_hold_uses_the_device_resume_limit(self):
        manager = self.manager(OTHER_ENV)
        sha = manager.policy.limits["sha256"]
        for temps, state, held in (
            ((84, 84), "normal", True),
            ((80, 80), "throttled", True),
            ((80, 80), "normal", False),
        ):
            with self.subTest(temps=temps, state=state):
                now = time.time()
                rows = {
                    rank: row(now, t, state=state, device_limits_sha256=sha)
                    for rank, t in zip("01", temps)
                }
                manager.policy.update(rows, True, "owned", now)
                manager.backend.idle = lambda: {"idle": True}
                if held:
                    with self.assertRaisesRegex(
                        ValueError, "^operator transition requires both fresh<=80 and"
                    ) as error:
                        manager.cool_idle_operator()
                    self.assertTrue(
                        hold(str(error.exception)), "control.py waits on it"
                    )
                else:
                    self.assertEqual(manager.cool_idle_operator(), {"idle": True})


if __name__ == "__main__":
    unittest.main()
