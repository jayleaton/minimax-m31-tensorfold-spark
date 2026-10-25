"""CPU-only durable-policy and actual loopback HTTP transport contracts.

Ported from tests/guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48), on the device's limits as the fake device
file states them (guard_fixtures.DEVICE_ENV): pause at 92, resume at 86 with both nodes fresh, hard STOP after a rank's
readings stay at or above 93 for 30 s. test_device.py covers the device state and the limits file itself.
"""

import concurrent.futures
import http.client
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import Backend
from common import POLICY, kill_condition, transition_hold
from control import wait_control
from daemon import Manager
from guard_fixtures import LIMITS, Native, TransportFixture, backend_config, make_policy, row, synthetic_sha
from lifecycle import decision

KILL_CONDITION = kill_condition(LIMITS)
TRANSITION_HOLD = transition_hold(LIMITS)

CURRENT, NEXT = synthetic_sha("current"), synthetic_sha("next")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = make_policy(self.root, initial_profile=CURRENT, now=1000)
        self.now = 1000

    def sample(self, a=80, b=80, memory=8, owned=True, missing=False, step=1):
        self.now += step
        rows = {str(i): row(self.now, t, memory) for i, t in enumerate((a, b))}
        if missing:
            rows.pop("1")
        self.p.update(rows, owned, "owned" if owned else "operator_lease", self.now)

    def hot(self, seconds, a=93, b=80):
        """`seconds` more one-second samples at these readings."""
        for _ in range(seconds):
            self.sample(a, b)

    def events(self):
        return [json.loads(line)["event"] for line in (self.root / "events").read_text().splitlines()]

    def test_policy_keeps_only_the_memory_floor_and_records_the_device_limits(self):
        self.assertEqual(POLICY, {"memory_floor_gib": 4})
        self.assertEqual(KILL_CONDITION, "temperature>=93 for 30 s")
        self.assertIn("fresh<=86", TRANSITION_HOLD)
        self.assertEqual(self.p.data["device_limits"], LIMITS)
        self.assertEqual(self.p.snapshot()["device_limits"], LIMITS)

    def test_pause_at_92_and_resume_at_86_only_with_both_fresh(self):
        self.sample(91.9, 91.9)
        self.assertTrue(self.p.ready())
        self.sample(80, 92)
        self.assertTrue(self.p.data["paused"])
        self.assertFalse(self.p.ready())
        self.sample(86, 86.1)
        self.assertTrue(self.p.data["paused"])
        self.sample(86, missing=True)
        self.assertTrue(self.p.data["paused"], "one fresh node cannot resume")
        self.sample(86, 86)
        self.assertFalse(self.p.data["paused"])
        self.assertTrue(self.p.ready())
        self.sample(92, missing=True)
        self.assertTrue(self.p.data["paused"], "any fresh reading at 92 pauses, even with the other node missing")
        self.assertEqual(self.events(), ["PAUSE", "RESUME", "PAUSE"])
        self.assertFalse(self.p.data["terminal_stop"])

    def test_repeated_pauses_never_stop(self):
        for _ in range(10):
            self.sample(92, 92)
            self.sample(86, 86)
        self.assertTrue(self.p.ready())
        self.assertFalse(self.p.data["terminal_stop"])
        self.assertNotIn("global_pauses", self.p.data)

    def test_pause_survives_reload(self):
        self.sample(92)
        self.assertTrue(make_policy(self.root).data["paused"])

    def test_worker_missing_fail_closed(self):
        self.sample(missing=True)
        self.assertFalse(self.p.ready())
        self.sample()
        self.assertTrue(self.p.ready())

    def test_hold_29_seconds_no_stop_30_seconds_stop(self):
        sock = SimpleNamespace(shutdown=lambda *_: setattr(self, "aborted", True))
        self.p.active["a"] = sock
        self.hot(1)
        start = self.now
        self.hot(29)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertEqual(self.p.data["episodes"]["0"], {"start_epoch": start, "last_epoch": start + 29})
        self.hot(1)
        self.assertTrue(self.p.data["hard_stop"])
        self.assertTrue(self.aborted)
        self.assertEqual(self.p.data["pending_action"], "stop_owned")
        self.assertEqual(self.p.data["stop_reason"], "temperature>=93 for 30 s")
        evidence = self.p.data["stop_evidence"]
        self.assertEqual(evidence["device_limits"], LIMITS)
        self.assertEqual(evidence["fired"], [{"condition": "temperature>=93 for 30 s", "rank": "0", "value": 93,
                                              "episode_start": start}])
        self.assertEqual(evidence["validated_samples"]["0"]["max_zone_c"], 93)
        self.assertEqual(self.events()[-1], "HARD_STOP")

    def test_a_dip_below_93_resets_the_hold(self):
        self.hot(20)
        self.sample(92.9)
        self.assertIsNone(self.p.data["episodes"]["0"])
        self.hot(30)
        self.assertFalse(self.p.data["hard_stop"], "the hold restarts at the first reading back at 93")
        self.hot(1)
        self.assertTrue(self.p.data["hard_stop"])

    def test_a_missing_or_invalid_sample_resets_the_hold(self):
        for name, temperature, age in (("missing", None, 0), ("stale", 99, 7), ("nan", float("nan"), 0)):
            with self.subTest(sample=name):
                self.setUp()
                self.hot(20, 80, 93)
                self.now += 1
                rows = {"0": row(self.now)}
                if temperature is not None:
                    rows["1"] = row(self.now - age, temperature)
                self.p.update(rows, True, "owned", self.now)
                self.assertIsNone(self.p.data["episodes"]["1"])
                self.hot(30, 80, 93)
                self.assertFalse(self.p.data["hard_stop"])
                self.hot(1, 80, 93)
                self.assertTrue(self.p.data["hard_stop"])

    def test_an_update_gap_resets_the_hold(self):
        self.hot(20)
        self.sample(93, step=7)
        self.assertEqual(self.p.data["episodes"]["0"]["start_epoch"], self.now)
        self.hot(29)
        self.assertFalse(self.p.data["hard_stop"])
        self.hot(1)
        self.assertTrue(self.p.data["hard_stop"])

    def test_each_rank_holds_separately(self):
        self.hot(10, 93, 80)
        self.hot(10, 93, 93)
        rank1 = self.now - 9
        self.hot(20, 80, 93)
        self.assertFalse(self.p.data["hard_stop"], "rank 0's 20 s and rank 1's do not add up")
        self.assertIsNone(self.p.data["episodes"]["0"])
        self.hot(1, 80, 93)
        self.assertTrue(self.p.data["hard_stop"])
        self.assertEqual(self.p.data["stop_evidence"]["fired"],
                         [{"condition": KILL_CONDITION, "rank": "1", "value": 93, "episode_start": rank1}])

    def test_hold_continues_across_a_quick_restart(self):
        self.hot(20)
        self.p = make_policy(self.root)
        self.hot(10)
        self.assertFalse(self.p.data["hard_stop"])
        self.hot(1)
        self.assertTrue(self.p.data["hard_stop"])

    def test_memory_floor_is_immediate_on_a_single_fresh_node(self):
        sock = SimpleNamespace(shutdown=lambda *_: setattr(self, "aborted", True))
        self.p.active["a"] = sock
        self.sample(memory=3.99, missing=True)
        self.assertTrue(self.aborted)
        self.assertEqual(self.p.data["pending_action"], "stop_owned")
        self.assertTrue(self.p.data["hard_stop"])

    def test_stop_evidence_durable_before_abort_and_immutable_after_reload(self):
        metadata = {"request-a": {"epoch": 1000, "body_sha256": "abc", "prompt_tokens": None,
                                  "resumed_tokens": None, "token_source": None}}
        self.p.request_metadata = metadata
        self.p.inflight = 1
        cancelled = []

        def shutdown(_):
            cancelled.append(json.loads((self.root / "state").read_text()))
            metadata["request-a"]["prompt_tokens"] = 123

        self.p.active["request-a"] = SimpleNamespace(shutdown=shutdown)
        rows = {
            "0": row(1001, "80", "3.5"),
            "1": row(1000, 80, 9),
        }
        self.p.update(rows, True, "owned", 1001)
        evidence = cancelled[0]["stop_evidence"]
        self.assertTrue(cancelled[0]["hard_stop"])
        self.assertEqual(cancelled[0]["pending_action"], "stop_owned")
        self.assertEqual(evidence["epoch"], 1001)
        self.assertEqual(evidence["inflight"], 1)
        self.assertEqual(evidence["raw_samples"], rows)
        self.assertEqual(evidence["validated_samples"]["0"]["available_gib"], 3.5)
        self.assertIsNone(evidence["request_metadata"]["request-a"]["prompt_tokens"])
        events = [json.loads(line) for line in (self.root / "events").read_text().splitlines()]
        self.assertEqual(events[-1]["stop_evidence"], evidence)
        rows["0"]["max_zone_c"] = 100
        self.p.update(rows, True, "owned", 1002)
        reboot = make_policy(self.root)
        self.assertEqual(reboot.request_metadata, {})
        reboot.update({"1": row(1003, 99, 1)}, True, "owned", 1003)
        self.assertEqual(reboot.data["stop_evidence"], evidence)
        self.assertEqual(json.loads((self.root / "state").read_text())["stop_evidence"], evidence)
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(len((self.root / "events").read_text().splitlines()), 1)

    def test_stop_evidence_distinguishes_conditions_and_ranks(self):
        hold = KILL_CONDITION
        for temperatures, memories, expected in (
            ((93, 80), (4, 8), [(hold, "0", 93, 1001)]),
            ((80, 80), (8, 3.99), [("memory<4GiB", "1", 3.99, None)]),
            ((95, 80), (8, 3.99), [("memory<4GiB", "1", 3.99, None)]),
            ((95, 94), (8, 8), [(hold, "0", 95, 1001), (hold, "1", 94, 1001)]),
        ):
            with self.subTest(expected=expected):
                root = self.root / str(temperatures) / str(memories)
                policy = make_policy(root, now=1000)
                for now in range(1001, 1032):
                    rows = {str(i): row(now, temperatures[i], memories[i]) for i in range(2)}
                    policy.update(rows, True, "owned", now)
                evidence = policy.data["stop_evidence"]
                self.assertEqual([(row["condition"], row["rank"], row["value"], row.get("episode_start"))
                                  for row in evidence["fired"]], expected)
                self.assertEqual(evidence["epoch"], 1031 if expected[0][0] == hold else 1001)
                self.assertEqual(evidence["request_metadata"], {})

    def test_stop_evidence_excludes_invalid_samples_from_triggers(self):
        rows = {
            "0": row(1001, 80, 3.9),
            "1": row(990, float("nan"), 1),
        }
        self.p.update(rows, True, "owned", 1001)
        evidence = self.p.data["stop_evidence"]
        self.assertEqual(evidence["raw_samples"]["1"]["max_zone_c"], "NaN")
        self.assertEqual(set(evidence["validated_samples"]), {"0"})
        self.assertEqual(evidence["fired"], [{"condition": "memory<4GiB", "rank": "0", "value": 3.9}])

    def test_hard_stop_threshold_boundaries_and_ownership_unchanged(self):
        for _ in range(40):
            self.sample(92.99, 92.99, memory=4)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertNotIn("stop_evidence", self.p.data)
        self.sample(100, memory=1, owned=False)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertNotIn("stop_evidence", self.p.data)

    def test_lease_interrupts_the_hold(self):
        self.hot(20)
        self.sample(93, owned=False)
        self.assertEqual(self.p.data["episodes"], {"0": None, "1": None})
        self.hot(30)
        self.assertFalse(self.p.data["hard_stop"])
        self.hot(1)
        self.assertTrue(self.p.data["hard_stop"])

    def test_operator_lease_never_stops_and_never_admits(self):
        self.sample(100, memory=1, owned=False)
        self.assertFalse(self.p.data["hard_stop"])
        self.assertEqual(self.p.data["lease_mode"], "operator_lease")
        self.assertFalse(self.p.ready())

    def test_lifecycle_stop_and_explicit_restore(self):
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (CURRENT, {}))
        state = dict(version=1, profile=CURRENT, hard_stop=False, fresh=True, last_update_epoch=1000,
                     terminal_stop=True, paused=True)
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 78)
        state["bootstrap_profile"] = CURRENT
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 0)
        self.assertEqual(decision({}, state, "watch", backend, 1000)[0], 78)
        state["hard_stop"] = True
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 78)

    def test_boot_proof_current_logs_floor(self):
        backend = Backend(backend_config())
        backend.identity = lambda *_: {"id": "exact"}
        backend.idle = lambda: {"streams": {"max": 4}}
        backend.run = lambda *_: "boot forecast: worst serving point 4.00 GiB"
        self.assertEqual(len(backend.boot_proof("sha")["ranks"]), 2)
        for logs in ("boot forecast: worst serving point 3.99 GiB", "no forecast line"):
            backend.run = lambda *_, text=logs: text
            with self.assertRaises(ValueError):
                backend.boot_proof("sha")

    def test_boot_proof_requires_native_slots_equal_guard_inflight(self):
        backend = Backend(backend_config(limits={"max_inflight": 2}))
        backend.identity = lambda *_: {"id": "exact"}
        backend.run = lambda *_: "boot forecast: worst serving point 6.00 GiB"
        for streams in ({"max": 4}, {"max": 1}, {}, None):
            backend.idle = lambda value=streams: {} if value is None else {"streams": value}
            with self.subTest(streams=streams), self.assertRaisesRegex(ValueError, "max_inflight"):
                backend.boot_proof("sha")
        backend.idle = lambda: {"streams": {"max": 2}}
        self.assertEqual(len(backend.boot_proof("sha")["ranks"]), 2)

    def test_control_waits_only_known_hold_and_never_retries_stop(self):
        ticks = iter(range(10))
        hold = {"ok": False, "error": TRANSITION_HOLD}
        with patch("control.transaction", side_effect=[hold, {"ok": True}]) as transaction:
            result = wait_control({}, "prepare-profile", NEXT, 10, lambda: next(ticks), lambda _: None)
            self.assertTrue(result["ok"])
            self.assertEqual(transaction.call_count, 2)
        with patch("control.transaction",
                   return_value={"ok": False, "error": "same-profile STOP cannot be reset"}) as call:
            self.assertFalse(wait_control({}, "prepare-profile", CURRENT)["ok"])
            self.assertEqual(call.call_count, 1)

    def test_profile_bootstrap_archives_stop_and_boot_proof(self):
        self.sample(92)
        self.sample(86, 86)
        with self.p.lock:
            self.p.terminal("current final", self.now)
        manager = object.__new__(Manager)
        manager.policy = self.p
        manager.config = {"profiles": {CURRENT: {}, NEXT: {}}, "state_path": str(self.root / "state")}
        manager.backend = SimpleNamespace(profile=lambda: (NEXT, {}),
                                          identity=lambda rank, _: {"id": rank}, idle=lambda: {"inflight": 0},
                                          boot_proof=lambda _: {"real_boot": True})
        manager.cool_idle_operator = lambda *_: {"idle": True}
        manager.control("prepare-profile", NEXT)
        self.assertTrue(self.p.data["terminal_stop"])
        self.assertEqual(self.p.data["bootstrap_profile"], NEXT)
        manager.control("authorize-profile", NEXT)
        self.assertFalse(self.p.data["terminal_stop"])
        archives = list((self.root / "archives").glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertTrue(json.loads(archives[0].read_text())["terminal_stop"])
        with self.p.lock:
            self.p.terminal("next final", self.now)
        with self.assertRaisesRegex(ValueError, "same-profile STOP"):
            manager.control("prepare-profile", NEXT)
        with self.assertRaisesRegex(ValueError, "unapproved target"):
            manager.control("prepare-profile", synthetic_sha("unregistered"))

    def test_absent_lowercase_docker_objects_allow_bootstrap_but_running_refused(self):
        self.sample()
        manager = object.__new__(Manager)
        manager.policy = self.p
        manager.config = {"containers": {"0": "native-r0", "1": "native-r1"}}
        manager.backend = SimpleNamespace(
            ownership=lambda **_: (True, "owned"),
            idle=lambda: (_ for _ in ()).throw(OSError("backend absent")),
            run=lambda *_: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, "docker inspect", stderr="error: no such object: native-r0\n")),
        )
        with patch("daemon.time.time", return_value=self.now):
            self.assertTrue(manager.cool_idle_operator()["native_absent_and_both_containers_stopped"])
            manager.backend.run = lambda *_: json.dumps([{"State": {"Running": True}}])
            with self.assertRaisesRegex(ValueError, "running native"):
                manager.cool_idle_operator()

    def test_pending_healthy_bootstrap_start_allowed_watch_held(self):
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (CURRENT, {}))
        state = dict(version=1, profile=CURRENT, bootstrap_profile=CURRENT, hard_stop=False, terminal_stop=False,
                     paused=False, fresh=True, last_update_epoch=1000)
        self.assertEqual(decision({}, state, "start", backend, 1000)[0], 0)
        self.assertEqual(decision({}, state, "watch", backend, 1000)[0], 75)

    def test_docker_logs_includes_stderr_only_for_logs(self):
        backend = Backend(backend_config())
        with patch("backend.subprocess.run", return_value=SimpleNamespace(stdout="out", stderr="err")):
            self.assertEqual(backend.run("0", ["docker", "logs", "x"]), "outerr")
            self.assertEqual(backend.run("0", ["docker", "inspect", "x"]), "out")


class TransportTests(TransportFixture, unittest.TestCase):
    def test_four_parallel_sse_and_unchanged_auth_body(self):
        # Hold native decode until all four arrive; real-time sleeps can undercount overlap.
        Native.finish_barrier = threading.Barrier(4)
        self.addCleanup(setattr, Native, "finish_barrier", None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.request(), range(4)))
        self.assertEqual(Native.maximum, 4)
        self.assertEqual(results, [(200, Native.payload)] * 4)
        self.assertEqual(Native.received, [("Bearer exact", b'{"image":"raw","tools":[1]}')] * 4)
        self.wait_idle()
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(len(self.p.active), 0)

    def test_health_passes_paused_and_terminal_blocks_generation(self):
        self.p.data["paused"] = True
        self.assertEqual(self.request("GET"), (200, b"{}"))
        self.p.data["terminal_stop"] = True
        self.assertEqual(self.request()[0], 503)
        self.assertFalse(Native.received)

    def test_oversize_not_forwarded(self):
        self.assertEqual(self.request(body=b"x" * 1025)[0], 502)
        self.assertFalse(Native.received)

    def test_status_endpoint_reports_and_no_http_mutation_api(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_port, timeout=2)
        try:
            connection.request("GET", "/__guard/status")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            status = json.loads(response.read())
            self.assertEqual(status["policy"], POLICY)
            self.assertEqual(status["device_limits"], LIMITS)
            connection.request("POST", "/__guard/reset", b"{}")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
        finally:
            connection.close()
        self.assertFalse(Native.received)


if __name__ == "__main__":
    unittest.main()
