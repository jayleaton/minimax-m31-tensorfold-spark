"""Durable pause and state, prefill admission and generated-token release contracts.

Ported from tests/guard in jayleaton/deepseek-v41-tensorfold-spark (PR #48).
"""

import concurrent.futures
import http.client
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from front import GeneratedTokenProbe
from guard_fixtures import Native, PolicyFixture, TransportFixture, row
from lifecycle import decision


class StabilityTests(PolicyFixture, unittest.TestCase):
    def hot_pause(self, rows=("0", "1")):
        now = time.time()
        self.p.update({r: row(now, 92) for r in rows}, True, "owned", now)

    def test_pause_survives_reload_and_resumes_only_when_both_cool(self):
        self.hot_pause(rows=("1",))
        self.assertTrue(self.p.data["paused"])
        self.p = self.reload()
        self.assertTrue(self.p.data["paused"])
        self.temperatures(86, 86.5)
        self.assertFalse(self.p.ready())
        self.samples()
        self.assertTrue(self.p.ready())
        self.assertFalse(self.p.data["terminal_stop"])

    def test_durable_policy_mismatch_refused_without_reset(self):
        self.hot_pause()
        state = json.loads((self.root / "state").read_text())
        state["policy"] = dict(state["policy"], kill_c=99)
        (self.root / "state").write_text(json.dumps(state))
        before = (self.root / "state").read_bytes()
        with self.assertRaisesRegex(ValueError, "no automatic reset"):
            self.reload()
        self.assertEqual((self.root / "state").read_bytes(), before)

    def test_lifecycle_never_clears_hold_or_stop(self):
        self.samples()
        backend = SimpleNamespace(ownership=lambda: (True, "owned"), profile=lambda: (self.sha, {}))
        self.assertEqual(decision({}, self.p.data, "watch", backend)[0], 0)
        self.p.data["paused"] = True
        self.assertEqual(decision({}, self.p.data, "watch", backend)[0], 75)
        with self.p.lock:
            self.p.terminal("test", time.time(), hard=True)
        self.assertEqual(decision({}, self.p.data, "start", backend)[0], 78)
        self.assertTrue(self.reload().data["hard_stop"])

    def test_memory_floor_stale_and_native_health_close_admission(self):
        server, health = self.server()
        for memory in (4.25, 4.749):
            self.samples(memory)
            self.assertFalse(server.admit("low"))
            self.assertEqual(self.p.inflight, 0)
            self.assertIsNone(self.p.prefill_request)
        self.samples(4.75)
        self.assertTrue(server.admit("boundary"))
        self.p.release_prefill("boundary")
        self.p.inflight -= 1
        self.samples(epoch=time.time() - 7)
        self.assertFalse(server.admit("stale"))
        self.samples()
        for result in ({"ok": True}, {"ok": True, "queued": 1}, {"ok": True, "queued": False}):
            health.return_value = result
            self.assertFalse(server.admit("native"))
            self.assertIsNone(self.p.prefill_request)
        health.side_effect = OSError("unavailable")
        self.assertFalse(server.admit("unavailable"))

    def test_lane_atomic_fresh_after_body_and_decode_overlap(self):
        server, health = self.server()
        server.config["limits"]["queue_seconds"] = 1
        self.samples()
        self.assertTrue(server.admit("first"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a, b = [pool.submit(server.admit, name) for name in ("a", "b")]
            time.sleep(0.05)
            self.assertEqual(self.p.inflight, 1)
            self.p.release_prefill("first")
            time.sleep(0.05)
            self.assertEqual(self.p.inflight, 1, "cached pre-body samples must not permit another prefill")
            self.samples()
            until = time.monotonic() + 0.5
            while self.p.inflight < 2 and time.monotonic() < until:
                time.sleep(0.005)
            self.assertEqual(self.p.inflight, 2, "decode may overlap next prefill")
            self.assertEqual(sum(f.done() for f in (a, b)), 1)
            with self.p.lock:
                self.p.terminal("cancel queued", time.time(), hard=True)
            self.assertEqual(sorted([a.result(), b.result()]), [False, True])

    def test_stop_during_native_probe_never_increments_inflight(self):
        server, health = self.server()
        self.samples()

        def stop():
            with self.p.lock:
                self.p.terminal("stop during probe", time.time(), hard=True)
            return {"ok": True, "queued": 0, "inflight": 0, "requests_running": 0}

        health.side_effect = stop
        self.assertFalse(server.admit("racing-stop"))
        self.assertEqual(self.p.inflight, 0)
        self.assertIsNone(self.p.prefill_request)

    def test_retained_reservations_and_wait_status_timeout(self):
        server, health = self.server()
        self.samples(5.0)
        self.assertTrue(server.admit("decode"))
        self.p.release_prefill("decode")
        self.samples(5.0)
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5})
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(server.admit, "waiting")
            until = time.monotonic() + 0.1
            while "waiting" not in self.p.snapshot()["admission_waiters"] and time.monotonic() < until:
                time.sleep(0.005)
            status = self.p.snapshot()
            self.assertEqual(status["admission_waiters"]["waiting"], "memory_margin_and_retained_reservations")
            self.assertEqual(status["memory_reserved_total_gib"], 0.5)
            self.assertFalse(future.result(timeout=0.5))
        self.assertEqual(self.p.inflight, 1)
        self.assertEqual(self.p.admission_waiters, {})
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5})
        self.samples(5.25)
        self.assertTrue(server.admit("second"))
        self.assertEqual(self.p.memory_reservations, {"decode": 0.5, "second": 0.5})

    def test_unknown_native_requests_after_restart_and_failed_completion_hold(self):
        server, health = self.server()
        self.samples()
        health.return_value = {"ok": True, "queued": 0, "inflight": 1, "requests_running": 1}
        self.assertFalse(server.admit("unknown-after-restart"))
        self.assertEqual(self.p.inflight, 0)
        self.p.memory_reservations["failed"] = 0.5
        self.p.unresolved_requests.add("failed")
        self.assertFalse(server.admit("orphan-still-running"))
        self.assertEqual(self.p.memory_reservations, {"failed": 0.5})
        health.return_value = {"ok": True, "queued": 0, "inflight": 0, "requests_running": 0}
        self.assertFalse(server.admit("need-post-idle-sample"))
        self.assertEqual(self.p.unresolved_requests, set())
        self.assertEqual(self.p.memory_reservations, {})
        self.samples()
        self.assertTrue(server.admit("idle-and-fresh"))

    def test_soft_stop_then_hard_stop_escalates_and_stops_containers(self):
        with self.p.lock:
            self.p.terminal("test", time.time())
            self.assertTrue(self.p.data["terminal_stop"])
            self.assertIsNone(self.p.data["pending_action"])
            self.p.terminal("hard stop after soft stop", time.time(), hard=True)
        self.assertEqual(self.p.data["pending_action"], "stop_owned")
        self.assertTrue(self.p.data["hard_stop"])

    def test_generated_delta_probe_fragmentation_bounds_and_unknown(self):
        for delta in ({"content": "x"}, {"reasoning_content": "x"},
                      {"tool_calls": [{"function": {"arguments": "{"}}]},
                      {"function_call": {"arguments": "{"}}):
            probe = GeneratedTokenProbe()
            frame = b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\r\n\r\n"
            self.assertFalse(probe.feed(frame[:-1]))
            self.assertTrue(probe.feed(frame[-1:]))
        for frame in (b": ping\n\n", b"data: [DONE]\n\n", b"data: nope\n\n",
                      b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'):
            self.assertFalse(GeneratedTokenProbe().feed(frame))
        probe = GeneratedTokenProbe()
        self.assertFalse(probe.feed(b"data: " + b"x" * 65536))
        self.assertTrue(probe.disabled)
        self.assertFalse(probe.feed(b'\n\ndata: {"choices":[{"delta":{"content":"x"}}]}\n\n'))


class AdmissionTransportTests(TransportFixture, unittest.TestCase):
    def test_headers_role_keepalive_hold_until_generated_token(self):
        headers_sent, allow_role, role_sent = threading.Event(), threading.Event(), threading.Event()
        allow_token, second_started, finish = threading.Event(), threading.Event(), threading.Event()
        calls = []
        lock = threading.Lock()
        # Deliberately nonzero mislabeled prefilling: never use it as workspace proof.
        self.front.manager.backend.health = lambda: {
            "ok": True, "queued": 0, "inflight": 0, "requests_running": 0, "streams": {"prefilling": 99},
        }
        role = b': keepalive\n\ndata: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        token = b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
        tail = b"data: [DONE]\n\n"

        def post(handler):
            body = handler.rfile.read(int(handler.headers["Content-Length"]))
            with lock:
                calls.append((handler.headers.get("Authorization"), body))
                first = len(calls) == 1
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(role + token + tail)))
            handler.end_headers()
            handler.wfile.flush()
            if first:
                headers_sent.set()
                allow_role.wait(1)
            else:
                second_started.set()
            handler.wfile.write(role)
            handler.wfile.flush()
            if first:
                role_sent.set()
                allow_token.wait(1)
            handler.wfile.write(token)
            handler.wfile.flush()
            finish.wait(1)
            handler.wfile.write(tail)

        for event in (finish, allow_token, allow_role):
            self.addCleanup(event.set)
        with patch.object(Native, "do_POST", post), concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.request)
            self.assertTrue(headers_sent.wait(0.5))
            second = pool.submit(self.request)
            time.sleep(0.05)
            self.assertEqual(len(calls), 1, "headers must not release the lane")
            allow_role.set()
            self.assertTrue(role_sent.wait(0.5))
            time.sleep(0.05)
            self.assertEqual(len(calls), 1, "role/empty content/keepalive must not release the lane")
            allow_token.set()
            self.assertTrue(second_started.wait(0.5))
            self.assertEqual(self.p.inflight, 2)
            self.assertEqual(len(self.p.snapshot()["memory_reservations_gib"]), 2)
            finish.set()
            self.assertEqual(first.result(), (200, role + token + tail))
            self.assertEqual(second.result(), (200, role + token + tail))
        self.assertEqual(calls, [("Bearer exact", b'{"image":"raw","tools":[1]}')] * 2)
        self.wait_idle()
        self.assertEqual(self.p.inflight, 0)
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.memory_reservations, {})

    def test_truncated_native_body_retains_reservation_until_idle(self):
        def truncated(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            handler.send_response(200)
            handler.send_header("Content-Length", "20")
            handler.end_headers()
            handler.wfile.write(b"partial")
            handler.wfile.flush()
            handler.close_connection = True

        with patch.object(Native, "do_POST", truncated):
            with self.assertRaises(http.client.IncompleteRead):
                self.request()
        self.wait_idle()
        self.assertEqual(len(self.p.unresolved_requests), 1)
        self.assertEqual(sum(self.p.memory_reservations.values()), 0.5)
        self.assertIsNone(self.p.prefill_request)

    def test_empty_response_and_transport_failure_release_lane(self):
        def empty(handler):
            handler.rfile.read(int(handler.headers["Content-Length"]))
            handler.send_response(204)
            handler.end_headers()

        with patch.object(Native, "do_POST", empty):
            self.assertEqual(self.request(), (204, b""))
        self.wait_idle()
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.inflight, 0)
        # Verification failure exercises cleanup before any native transport is opened.
        self.front.manager.verify_generation = Mock(side_effect=ValueError("identity changed"))
        self.assertEqual(self.request()[0], 502)
        self.wait_idle()
        self.assertIsNone(self.p.prefill_request)
        self.assertEqual(self.p.inflight, 0)
        self.assertEqual(self.p.memory_reservations, {})


if __name__ == "__main__":
    unittest.main()
