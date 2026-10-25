"""Snapshotted-profile, production-config and device-limits validation rules.

Ported from tests/guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48).
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from backend import Backend
from common import DEVICE_HOME, FLOOR_KEY, PLAN_KEY, POLICY, POOL_KEY, SCHEDULE_KEY, load_config, validate_profiles
from guard_fixtures import (
    ADMISSION,
    CPU_SETS,
    DEVICE_ENV,
    LIMITS,
    make_profile,
    make_profiles,
    production_config,
    synthetic_sha,
    write_config,
    write_device,
)


class ProfileTests(unittest.TestCase):
    def check(self, profile):
        validate_profiles({synthetic_sha("checked"): profile})

    def rejected(self, change, pattern=None):
        profile = make_profile(cpu_sets=CPU_SETS)
        change(profile)
        with self.assertRaisesRegex(ValueError, pattern or ""):
            self.check(profile)

    def test_valid_profile_accepted_with_and_without_cpu_sets(self):
        self.check(make_profile())
        self.check(make_profile(cpu_sets=CPU_SETS))
        self.check(make_profile(**{FLOOR_KEY: "4"}))
        validate_profiles(make_profiles(*map(str, range(32))))

    def test_profile_count_and_key_shape(self):
        for profiles in ({}, make_profiles(*map(str, range(33))), [], None,
                         {"A" * 64: make_profile()}, {"a" * 63: make_profile()}):
            with self.subTest(count=len(profiles or ())), self.assertRaises(ValueError):
                validate_profiles(profiles)

    def test_per_rank_identity_required(self):
        self.rejected(lambda p: p["image_ids"].update({"1": "sha256:" + "A" * 64}), "image ID")
        self.rejected(lambda p: p["image_ids"].update({"0": synthetic_sha("bare")}), "image ID")
        self.rejected(lambda p: p["mounts"]["0"].pop("/model"), "mounts")
        self.rejected(lambda p: p["inference_environment"].update({"1": {}}), "environment")
        # Per-rank images and mounts may legitimately differ (one host per rank).
        profile = make_profile()
        profile["image_ids"]["1"] = "sha256:" + synthetic_sha("other-image")
        profile["mounts"]["1"]["/model"] = "/srv/elsewhere"
        self.check(profile)

    def test_workspace_floor_below_hard_floor_rejected(self):
        for floor in ("3.99", "3.5", "0", "-4", "nan", "inf", "-inf", "four"):
            with self.subTest(floor=floor):
                self.rejected(lambda p, value=floor: [env.update({FLOOR_KEY: value})
                                                      for env in p["inference_environment"].values()], "floor")

    def test_rank_mismatched_knobs_rejected(self):
        for key, value in ((POOL_KEY, "1500000"), (FLOOR_KEY, "5.0"), (SCHEDULE_KEY, "spin")):
            with self.subTest(key=key):
                self.rejected(lambda p, k=key, v=value: p["inference_environment"]["1"].update({k: v}),
                              "both ranks")
                self.rejected(lambda p, k=key: p["inference_environment"]["1"].pop(k), "both ranks")

    def test_bad_cuda_schedule_rejected(self):
        for value in ("blocking ", "BLOCKING", "busy", ""):
            with self.subTest(value=value):
                self.rejected(lambda p, v=value: [env.update({SCHEDULE_KEY: v})
                                                  for env in p["inference_environment"].values()], "TF_CUDA_SCHED")
        for value in ("auto", "spin", "yield", "blocking"):
            with self.subTest(value=value):
                self.check(make_profile(**{SCHEDULE_KEY: value}))

    def test_plan_pin_outside_cpu_set_rejected(self):
        for value in ("0", "1", "6", "4095"):
            with self.subTest(value=value):
                self.rejected(lambda p, v=value: p["inference_environment"]["1"].update({PLAN_KEY: v}),
                              "inside the rank")
        for value in ("2", "5"):
            self.check(make_profile(cpu_sets=CPU_SETS, **{PLAN_KEY: value}))
        # Without a host set the pin only has to be a canonical local index.
        self.check(make_profile(**{PLAN_KEY: "4095"}))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = production_config(self.root)

    def load(self, config):
        return load_config(write_config(self.root, config))

    def refused(self, change):
        config = copy.deepcopy(self.config)
        change(config)
        with self.assertRaises((ValueError, KeyError)):
            self.load(config)

    def test_valid_config_loads_and_registers_exact_profiles(self):
        loaded = self.load(self.config)
        self.assertEqual(loaded["profiles"], self.config["profiles"])
        for units in ([], None):
            config = copy.deepcopy(self.config)
            if units is None:
                config.pop("watchdog_units")
            else:
                config["watchdog_units"] = units
            self.load(config)
        backend = Backend(loaded)
        Path(loaded["production_config"]).write_text("unregistered bytes")
        with self.assertRaisesRegex(ValueError, "unapproved"):
            backend.profile()

    def test_policy_must_be_enabled_and_exact(self):
        self.refused(lambda c: c.update(enabled=False))
        self.refused(lambda c: c.update(mode="shadow"))
        self.refused(lambda c: c["policy"].update(kill_c=95))
        self.refused(lambda c: c["policy"].update(extra_c=80))
        self.refused(lambda c: c["policy"].update(sustain_c=90, sustain_seconds=600))
        self.assertNotIn("extra_c", POLICY)

    def env_refused(self, text, pattern):
        Path(self.config["production_config"]).write_text(text)
        with self.assertRaisesRegex(ValueError, pattern):
            self.load(self.config)

    def device(self, text):
        Path(self.config["device_limits"]).write_text(text)

    def device_refused(self, text, pattern):
        self.device(text)
        with self.assertRaisesRegex(ValueError, pattern):
            self.load(self.config)

    def test_production_config_must_not_carry_thermal_limits(self):
        for line in ("GUARD_PAUSE_C=92", "GUARD_RESUME_C=86", "GUARD_KILL_C=93", "export GUARD_KILL_HOLD_S='30'"):
            with self.subTest(line=line):
                self.env_refused("WORKER=w\n" + line + "\n", "production_config sets GUARD_.*: "
                                 + DEVICE_HOME.replace("(", r"\(").replace(")", r"\)"))
        # A comment, the guard's own config path and the device's keys elsewhere are fine.
        Path(self.config["production_config"]).write_text("# GUARD_PAUSE_C=92\nGUARD_CONFIG=/srv/guard.json\n")
        self.load(self.config)
        Path(self.config["production_config"]).unlink()
        with self.assertRaisesRegex(ValueError, "production_config unreadable"):
            self.load(self.config)

    def test_limits_are_read_from_the_device_file(self):
        loaded = self.load(self.config)
        self.assertEqual(loaded["_thermal_limits"], dict(LIMITS, path=self.config["device_limits"]))
        self.assertEqual(loaded["policy"], POLICY)
        text = ("GUARD_PAUSE_C=90\nGUARD_RESUME_C='80'\nexport GUARD_KILL_C=\"91\"\nGUARD_KILL_HOLD_S=10\n"
                "SPARK_THERMAL_INTERVAL_S=3\n")
        self.device(text)
        self.assertEqual(self.load(self.config)["_thermal_limits"], {
            "path": self.config["device_limits"], "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "pause_c": 90, "resume_c": 80, "kill_c": 91, "kill_hold_seconds": 10, "interval_s": 3})

    def test_missing_invalid_or_out_of_order_device_file_refused(self):
        lines = [line for line in DEVICE_ENV.splitlines() if line.startswith(("GUARD_", "SPARK_THERMAL"))]
        for index, line in enumerate(lines):
            key = line.split("=")[0]
            with self.subTest(missing=key):
                self.device_refused("\n".join(lines[:index] + lines[index + 1:]) + "\n# " + line + "\n", "lack " + key)
        for value in ("92.0", "hot", "", "-1", "0x5c", "92 93"):
            with self.subTest(value=value):
                self.device_refused(DEVICE_ENV.replace("GUARD_PAUSE_C=92", "GUARD_PAUSE_C=" + value), "not an integer")
        for pause, resume, kill in ((86, 86, 93), (92, 93, 93), (93, 86, 92), (92, 86, 94), (95, 86, 99)):
            with self.subTest(pause=pause, resume=resume, kill=kill):
                text = DEVICE_ENV.replace("PAUSE_C=92", "PAUSE_C=" + str(pause))
                text = text.replace("RESUME_C=86", "RESUME_C=" + str(resume))
                text = text.replace("KILL_C=93", "KILL_C=" + str(kill))
                self.device_refused(text, "out of order")
        self.device_refused(DEVICE_ENV.replace("HOLD_S=30", "HOLD_S=0"), "at least 1")
        for interval in ("0", "11"):
            self.device_refused(DEVICE_ENV.replace("INTERVAL_S=2", "INTERVAL_S=" + interval), "1..10")
        self.device_refused(DEVICE_ENV + "GUARD_PAUSE_C=90\n", "more than once")
        self.device(DEVICE_ENV.replace("PAUSE_C=92", "PAUSE_C=93"))
        self.assertEqual(self.load(self.config)["_thermal_limits"]["pause_c"], 93, "pause may equal kill")
        Path(self.config["device_limits"]).unlink()
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.load(self.config)
        self.refused(lambda c: c.update(device_limits="device/thermal.env"))
        self.refused(lambda c: c.update(device_state="run/state"))

    def test_engine_log_patterns_and_units(self):
        self.refused(lambda c: c.pop("boot_floor_log_regex"))
        self.refused(lambda c: c.update(boot_floor_log_regex="<fill on release: engine boot forecast line>"))
        self.refused(lambda c: c.update(boot_floor_log_regex="no group GiB"))
        self.refused(lambda c: c.update(boot_floor_log_regex="(unbalanced"))
        self.refused(lambda c: c.update(resume_log_regex="slot (\\d+) only"))
        config = copy.deepcopy(self.config)
        config.pop("resume_log_regex")
        self.assertNotIn("resume_log_regex", self.load(config))
        self.refused(lambda c: c.update(watchdog_units=["native-watch.service"]))
        self.refused(lambda c: c.update(watchdog_units="native-watch.timer"))

    def test_endpoints_and_names(self):
        self.refused(lambda c: c.update(upstream="http://127.0.0.1"))
        self.refused(lambda c: c.update(upstream="https://127.0.0.1:8001"))
        self.refused(lambda c: c.update(upstream="http://localhost:8001"))
        self.refused(lambda c: c.update(listen_host="0.0.0.0"))
        self.refused(lambda c: c.update(listen_port="8000"))
        self.refused(lambda c: c.update(listen_port=8001))
        self.refused(lambda c: c["containers"].update({"1": "-flag"}))
        self.refused(lambda c: c["containers"].pop("1"))
        self.refused(lambda c: c.update(worker_ssh="-oProxyCommand=x"))
        self.refused(lambda c: c.update(state_path="relative/state"))

    def test_bounded_limits(self):
        for key, value in (("max_inflight", 0), ("max_inflight", 5), ("max_inflight", True), ("max_pending", 17),
                           ("max_connections", 7), ("body_buffers", 5), ("memory_transient_reserve_gib", 0.49),
                           ("memory_transient_reserve_gib", 4.01), ("memory_transient_reserve_gib", True),
                           ("queue_seconds", 1201), ("queue_seconds", 0)):
            with self.subTest(key=key, value=value):
                self.refused(lambda c, k=key, v=value: c["limits"].update({k: v}))
        for inflight in (1, 2, 4):
            config = copy.deepcopy(self.config)
            config["limits"]["max_inflight"] = inflight
            self.assertEqual(self.load(config)["limits"]["max_inflight"], inflight)

    def test_invalid_profile_fails_the_whole_config(self):
        self.refused(lambda c: next(iter(c["profiles"].values()))["inference_environment"]["0"]
                     .update({FLOOR_KEY: "3.5"}))
        self.refused(lambda c: c.update(profiles={}))

    def test_example_config_needs_only_a_snapshotted_profile(self):
        example = Path(__file__).resolve().parents[2] / "config" / "guard.example.json"
        if not example.exists():
            self.skipTest("no config/guard.example.json")
        config = json.loads(example.read_text())
        with self.assertRaisesRegex(ValueError, "boot_floor_log_regex"):
            self.load(config)
        config["boot_floor_log_regex"] = r"worst serving point ([0-9.]+) GiB"
        with self.assertRaisesRegex(ValueError, "profiles"):
            self.load(config)
        config["profiles"] = make_profiles("snapshot")
        self.assertEqual(config["policy"], POLICY)
        self.assertEqual((config["device_limits"], config["device_state"]),
                         ("/etc/spark/thermal.env", "/run/spark-thermal/state"))
        self.assertNotIn("prefill_admission_pause_c", config["limits"])
        config["production_config"] = str(self.root / "prod.env")
        Path(config["production_config"]).write_text("WORKER=user@rank1.example\n")
        config["device_limits"], config["device_state"] = map(str, write_device(self.root / "example"))
        self.assertEqual(self.load(config)["profiles"], config["profiles"])

    def test_admission_band_is_the_device_files(self):
        # 88 sits inside the fake device's 86..92 band; a device file with an 84..87 band refuses it.
        self.device(DEVICE_ENV.replace("PAUSE_C=92", "PAUSE_C=87").replace("RESUME_C=86", "RESUME_C=84"))
        with self.assertRaisesRegex(ValueError, "84<=pause<=87"):
            self.load(self.config)

    def test_config_requires_integer_admission_pair_inside_the_pause_band(self):
        self.assertEqual(self.load(self.config)["limits"]["prefill_admission_pause_c"],
                         ADMISSION["prefill_admission_pause_c"])
        for pause, resume in ((86, 1), (92, 86), (90, 89)):
            config = copy.deepcopy(self.config)
            config["limits"].update(prefill_admission_pause_c=pause, prefill_admission_resume_c=resume)
            self.assertEqual(self.load(config)["limits"]["prefill_admission_pause_c"], pause)
        for pause, resume in ((88.0, 80), (88, 80.0), (True, 1), (78, 70), (85, 80), (93, 80), (88, 88), (88, 0),
                              (None, 80), (88, None), (None, None)):
            config = copy.deepcopy(self.config)
            config["limits"].update(prefill_admission_pause_c=pause, prefill_admission_resume_c=resume)
            with self.subTest(pause=pause, resume=resume), self.assertRaises(ValueError):
                self.load(config)
        config = copy.deepcopy(self.config)
        for key in ("prefill_admission_pause_c", "prefill_admission_resume_c"):
            config["limits"].pop(key)
        self.assertNotIn("prefill_admission_pause_c", self.load(config)["limits"])
        self.assertEqual(json.loads((self.root / "config.json").read_text())["limits"], config["limits"])


if __name__ == "__main__":
    unittest.main()
