"""bench/drafts.py: the engine's acceptance lines parsed, attributed to their requests, summed and compared, a greedy
sha mismatch failing the comparison, the reference stop, the saved-log report and DRY_RUN, against a stand-in server
that streams as the engine does (``tensorfold`` and ``speculative`` on the last chunk) and writes a rank 0 log. CPU
only, 127.0.0.1."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import drafts as dr  # noqa: E402

STREAM = ("info: dspark acceptance stream: rounds=4 reached=4/3/2/1/0/0/0/0 kept=3/2/1/0/0/0/0/0 kept_per_round=1.500 "
          "conf1_rounds=0/0/0/0/0/0/0/0/1/3 conf1_kept=0/0/0/0/0/0/0/0/1/2")


def done(n: int, sha: str, rid: str = "chatcmpl-1") -> str:
    return (f"[tensorfold] done {rid} prompt=40 cached=0 thinking=False effort=none tokens={n} sha={sha} finish=stop "
            f"rounds=9 accepted=5/12 ttft=0.120 decode=1.000 elapsed=1.200 queued=0.000 tok/s=20.0")


# -- the line parsers and the arithmetic ----------------------------------------------------------------------------------
def test_a_stream_line_parses_and_gives_acceptance_by_position():
    t = dr.parse_tally("2026-10-25 " + STREAM)
    assert t["kind"] == "stream" and t["rounds"] == 4
    assert t["reached"] == [4, 3, 2, 1, 0, 0, 0, 0] and t["kept"] == [3, 2, 1, 0, 0, 0, 0, 0]
    assert t["conf1_rounds"][9] == 3 and t["kept_per_round"] == 1.5
    acc = dr.acceptance(t)
    assert acc["conditional"][:4] == [0.75, 0.6667, 0.5, 0.0] and acc["conditional"][4] is None
    assert acc["cumulative"][:3] == [0.75, 0.5, 0.25]
    assert acc["tau"] == 2.5                                       # 1 + 6 kept / 4 rounds
    assert acc["conf1"][9] == 0.6667 and acc["conf1"][0] is None


def test_malformed_and_other_lines_are_not_tallies():
    assert dr.parse_tally("info: dspark acceptance stream: rounds=4 reached=1/2 kept=1") is None
    assert dr.parse_tally("info: dspark acceptance stream: rounds=x reached=1 kept=1") is None
    assert dr.parse_tally("[tensorfold] done x tokens=3") is None
    assert dr.acceptance(dr.parse_tally(STREAM.replace("rounds=4", "rounds=0"))) is None
    total = dr.parse_tally(STREAM.replace("stream:", "total:"))
    assert total["kind"] == "total"


def test_tallies_sum_element_wise():
    t = dr.parse_tally(STREAM)
    s = dr.tally_sum([t, t, None])
    assert s["rounds"] == 8 and s["kept"][:3] == [6, 4, 2] and s["reached"][0] == 8 and s["streams"] == 2
    assert dr.acceptance(s)["tau"] == 2.5


def test_docker_timestamps_order_stdout_and_stderr_and_are_removed():
    # docker drops the fraction's trailing zeros: .15 is after .1
    lines = ["2026-10-25T07:00:02.15Z " + done(5, "abc"), "2026-10-25T07:00:02.1Z " + STREAM,
             "2026-10-25T07:00:01.5Z \r\x1b[2K[tensorfold] live 3 tok/s"]
    out = dr.clean(lines)
    assert out == ["[tensorfold] live 3 tok/s", STREAM, done(5, "abc")]


def test_attribution_takes_the_done_line_of_this_reply():
    lines = [STREAM, done(7, "aaa"), STREAM.replace("rounds=4", "rounds=2").replace("kept=3/2/1", "kept=1/1/0"),
             done(5, "bbb", "chatcmpl-2")]
    got = dr.attribute(lines, tokens=5, sha="bbb", spec_rounds=9)
    assert got["tally"]["rounds"] == 2 and got["done"]["id"] == "chatcmpl-2" and got["note"] is None
    assert dr.attribute(lines, tokens=9, sha=None, spec_rounds=9)["tally"] is None   # no done line with 9 tokens
    two = dr.attribute([STREAM, done(5, "x"), STREAM, done(5, "x")], tokens=5, sha="x", spec_rounds=9)
    assert two["tally"] is None and "not attributed" in two["note"]
    plain = dr.attribute([done(5, "x")], tokens=5, sha="x", spec_rounds=0)
    assert plain["tally"] is None and "no dspark acceptance line" in plain["note"]
    suspect = dr.attribute([STREAM, done(5, "x")], tokens=5, sha="x", spec_rounds=3)
    assert "suspect" in suspect["note"]
    assert dr.attribute([STREAM], tokens=5, sha="x", spec_rounds=9)["note"].startswith("no done line")


# -- compare --------------------------------------------------------------------------------------------------------------
def req(cls, pid, variant, rep, sha, tps, srv=None, drafts=None, tally=None, status=200):
    r = {"class": cls, "id": pid, "variant": variant, "rep": rep, "status": status, "error": None, "completion_tokens": 100,
         "decode_tps_client": tps, "decode_tps_server": srv, "ttft_s": 0.2, "token_sha256": sha, "text_sha256": f"t{sha}",
         "served_drafts": drafts, "speculative": {"rounds": 50, "drafted": 80, "accepted": 50, "tokens_per_round": 2.0}}
    if tally is not None:
        r["log"] = {"tally": tally, "note": None}
    return r


def run_json(label, reqs, path="x.json", **kw):
    normal = {"head": {"state": "normal"}, "worker": {"state": "normal"}}
    return {"kind": "drafts", "label": label, "prompts_sha256": "p", "max_tokens": 256, "model": "m", "_path": path,
            "server_log": "command 'docker logs -t c'" if label == "drafts-on" else None, "requests": reqs,
            "groups": [{"class": "code", "id": "a", "variant": "greedy", "before": normal, "after": normal}], **kw}


def test_compare_math_and_equal_greedy_pass():
    t = dr.parse_tally(STREAM)
    off = run_json("drafts-off", [req("code", "a", "greedy", i, "s1", tps, tps + 0.5, False) for i, tps in enumerate((10, 12, 11))])
    on = run_json("drafts-on", [req("code", "a", "greedy", i, "s1", tps, tps + 1, True, t) for i, tps in enumerate((20, 22, 21))])
    c = dr.compare([off], [on])
    assert c["verdict"] == "PASS", c["fails"]
    row = c["rows"][0]
    assert (row["tps_off"], row["tps_on"], row["speedup"]) == (11, 21, round(21 / 11, 4))
    assert row["server_speedup"] == round(22 / 11.5, 4)
    assert row["dspark"]["tau"] == 2.5 and row["dspark"]["conditional"][0] == 0.75
    assert row["speculative_on"]["acceptance_rate"] == 0.625 and not c["alternate"]
    assert c["identity"] == [{"id": "a", "variant": "greedy", "verdict": "EQUAL", "off": ["sha256:s1"],
                              "on": ["sha256:s1"], "reps": [3, 3]}]


def test_a_greedy_sha_mismatch_fails_and_a_sampled_one_is_a_note():
    off = run_json("drafts-off", [req("code", "a", "greedy", 0, "s1", 10), req("code", "a", "sampled", 0, "q1", 10)])
    on = run_json("drafts-on", [req("code", "a", "greedy", 0, "s2", 20), req("code", "a", "sampled", 0, "q2", 20)])
    c = dr.compare([off], [on])
    assert c["verdict"] == "FAIL"
    assert any("greedy a: DIFFERENT" in f for f in c["fails"])
    assert not any("sampled" in f for f in c["fails"]) and any("sampled a: DIFFERENT" in n for n in c["notes"])


def test_identity_needs_a_sha_or_text_and_a_repeatable_control():
    nosha = req("code", "a", "greedy", 0, "s1", 10)
    nosha["token_sha256"], nosha["text_sha256"] = None, None
    c = dr.compare([run_json("drafts-off", [nosha])], [run_json("drafts-on", [req("code", "a", "greedy", 0, "s1", 20)])])
    assert c["verdict"] == "FAIL" and c["identity"][0]["verdict"] == "NO-SHA"
    wobble = [req("code", "a", "greedy", 0, "s1", 10), req("code", "a", "greedy", 1, "s9", 10)]
    c = dr.compare([run_json("drafts-off", wobble)], [run_json("drafts-on", [req("code", "a", "greedy", 0, "s1", 20)])])
    assert c["identity"][0]["verdict"] == "OFF-NOT-REPEATABLE" and c["verdict"] == "FAIL"
    text = req("code", "a", "greedy", 0, None, 10)          # no token_sha256: the reply's text decides, and says so
    c = dr.compare([run_json("drafts-off", [text])], [run_json("drafts-on", [req("code", "a", "greedy", 0, None, 20)])])
    assert c["verdict"] == "PASS" and c["identity"][0]["off"] == ["text:tNone"]


def test_labels_drafts_flag_prompt_set_and_errors_are_checked():
    off = run_json("drafts-off", [req("code", "a", "greedy", 0, "s1", 10, drafts=True)])
    on = run_json("drafts-on", [req("code", "a", "greedy", 0, "s1", 20, drafts=True)], prompts_sha256="other")
    c = dr.compare([off], [on])
    assert any("tensorfold.drafts" in f for f in c["fails"]) and any("prompts_sha256" in f for f in c["fails"])
    swapped = dr.compare([run_json("drafts-on", [])], [run_json("drafts-off", [])])
    assert sum("is labelled" in f for f in swapped["fails"]) == 2
    bad = dr.compare([run_json("drafts-off", [req("code", "a", "greedy", 0, "s1", 10)])],
                     [run_json("drafts-on", [req("code", "a", "greedy", 0, "s1", 20, status=500)])])
    assert any("requests failed" in f for f in bad["fails"]) and any("MISSING" in f for f in bad["fails"])


def test_within_a_few_percent_says_alternate_and_a_hot_stamp_is_noted():
    off = run_json("drafts-off", [req("code", "a", "greedy", 0, "s1", 20.0)])
    on = run_json("drafts-on", [req("code", "a", "greedy", 0, "s1", 20.4)])
    on["groups"][0]["after"] = {"head": {"state": "throttled"}, "worker": {"state": "normal"}}
    c = dr.compare([off], [on])
    assert c["verdict"] == "PASS" and c["alternate"]
    assert any("alternate" in n for n in c["notes"]) and any("not comparable" in n for n in c["notes"])


# -- against a stand-in server ------------------------------------------------------------------------------------------
class Engine:
    """Streams as the engine's server does and logs as rank 0 does: the stream line, then the done line."""

    def __init__(self, log: Path, drafts: bool, differ: str | None = None, stats: bool = True) -> None:
        self.log, self.drafts, self.differ, self.stats, self.n = log, drafts, differ, stats, 0
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.log.write_text("info: dspark calibration: widths 1..8 ms 33.0/35.1/37.0\n" if drafts else "")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def reply(self, body: dict) -> tuple[str, int]:
        last = body["messages"][-1]
        words = ("reply to " + json.dumps(last)[:40]).split()
        if body.get("seed") is not None:
            words.append(f"seed{body['seed']}")
        if self.differ and self.differ in json.dumps(body["messages"]) and body.get("temperature") == 0:
            words.append("drafted-differently")
        return " ".join(words), len(words)

    def handler(self):
        eng = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send_json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send_json({"status": "ok"} if self.path == "/health" else {"data": [{"id": "MiniMax-M3.1-Flash-EXL3"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text, n = eng.reply(body)
                sha256 = hashlib.sha256(text.encode()).hexdigest()
                with eng.lock:
                    eng.n += 1
                    rid = f"chatcmpl-{eng.n}"
                    with eng.log.open("a") as f:
                        if eng.drafts and eng.stats:
                            f.write(STREAM + "\n")
                        f.write(done(n, sha256[:12], rid) + "\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                chunks = [{"choices": [{"index": 0, "delta": {"content": w + " "}}]} for w in text.split()]
                spec = {"rounds": 4 if eng.drafts else n, "drafted": 6 if eng.drafts else 0, "accepted": 3 if eng.drafts else 0,
                        "acceptance_rate": 0.5 if eng.drafts else 0, "tokens_per_round": n / 4 if eng.drafts else 1.0}
                chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                chunks.append({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": n},
                               "tensorfold": {"tokens_per_second": 20.0, "time_to_first_token": 0.1, "drafts": eng.drafts,
                                              "sampling": "greedy", "token_sha": sha256[:12], "token_sha256": sha256},
                               "speculative": spec})
                for c in chunks:
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        return H

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def engine(tmp_path):
    made = []

    def make(name, **kw):
        e = Engine(tmp_path / f"{name}.log", **kw)
        made.append(e)
        return e

    yield make
    for e in made:
        e.close()


def bench_run(run_py, e, label, out, *extra):
    return run_py("bench/drafts.py", "run", "--label", label, "--base", e.url, "--reps", "2", "--classes", "code,agent",
                  "--no-stamp", "--log-wait", "2", "--server-log", str(e.log), "--out", str(out), *extra)


def test_a_full_off_and_on_pair_passes_with_acceptance_per_class(run_py, engine, tmp_path):
    off_e, on_e = engine("off", drafts=False), engine("on", drafts=True)
    r_off = bench_run(run_py, off_e, "drafts-off", tmp_path / "off.json")
    assert r_off.returncode == 0, r_off.stdout + r_off.stderr
    r_on = bench_run(run_py, on_e, "drafts-on", tmp_path / "on.json", "--reference", str(tmp_path / "off.json"))
    assert r_on.returncode == 0, r_on.stdout + r_on.stderr
    on = json.loads((tmp_path / "on.json").read_text())
    # 2 classes x (2 greedy prompts + 1 sampled) x 2 reps, each matched to its own stream line
    assert len(on["requests"]) == 12 and all(r["log"]["tally"] for r in on["requests"])
    assert on["calibration"].startswith("dspark calibration:")
    assert on["classes_summary"]["code"]["dspark"]["tau"] == 2.5
    assert on["classes_summary"]["agent/sampled"]["attributed"] == 2
    body = json.loads((tmp_path / "off.json").read_text())
    assert body["requests"][0]["token_sha256"] and body["requests"][0]["served_drafts"] is False
    c = run_py("bench/drafts.py", "compare", "--off", str(tmp_path / "off.json"), "--on", str(tmp_path / "on.json"),
               "--out", str(tmp_path / "cmp.json"))
    assert c.returncode == 0, c.stdout + c.stderr
    assert "compare: PASS" in c.stdout and "EQUAL" in c.stdout and "2.500" in c.stdout


def test_a_drafted_greedy_reply_that_differs_stops_the_run_and_fails_compare(run_py, engine, tmp_path):
    off_e, on_e = engine("off", drafts=False), engine("on", drafts=True, differ="OrderedDict")
    assert bench_run(run_py, off_e, "drafts-off", tmp_path / "off.json").returncode == 0
    r = bench_run(run_py, on_e, "drafts-on", tmp_path / "on.json", "--reference", str(tmp_path / "off.json"))
    assert r.returncode == 1 and "differs from the reference" in r.stderr
    on = json.loads((tmp_path / "on.json").read_text())
    assert on["stopped"].startswith("FAIL: greedy code-lru rep 0")
    c = run_py("bench/drafts.py", "compare", "--off", str(tmp_path / "off.json"), "--on", str(tmp_path / "on.json"))
    assert c.returncode == 1 and "FAIL: greedy code-lru: DIFFERENT" in c.stdout


def test_without_the_stats_switch_the_requests_say_why(run_py, engine, tmp_path):
    e = engine("on", drafts=True, stats=False)
    r = run_py("bench/drafts.py", "run", "--label", "drafts-on", "--base", e.url, "--reps", "1", "--classes", "prose",
               "--no-stamp", "--log-wait", "1", "--server-log-cmd", f"cat {e.log}", "--out", str(tmp_path / "on.json"))
    assert r.returncode == 0, r.stderr
    on = json.loads((tmp_path / "on.json").read_text())
    assert all("no dspark acceptance line" in x["log"]["note"] for x in on["requests"])


def test_dry_run_prints_the_requests_and_sends_nothing(run_py, tmp_path):
    r = run_py("bench/drafts.py", "run", "--label", "drafts-on", "--base", "http://127.0.0.1:9", "--reps", "1",
               "--server-log-cmd", "false", "--out", str(tmp_path / "dry.json"), env={"DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    posts = [x for x in r.stdout.splitlines() if x.startswith("DRY POST")]
    assert len(posts) == 1 + 4 * 3                     # the warm-up, then 4 classes x (2 greedy + 1 sampled)
    assert '"temperature": 0' in posts[1] and '"ignore_eos": false' in posts[1] and '"enable_thinking": false' in posts[1]
    assert any('"seed": 4242' in p for p in posts) and any('"tool_choice": "auto"' in p for p in posts)
    js = json.loads((tmp_path / "dry.json").read_text())
    assert js["dry_run"] and js["requests"] == [] and js["max_tokens"] == 256


def test_the_log_mode_reads_a_saved_gzipped_log(run_py, tmp_path):
    log = tmp_path / "rank0.log.gz"
    with gzip.open(log, "wt") as f:
        f.write("info: dspark calibration: auto, 8 widths\n" + STREAM + "\n" + done(5, "a") + "\n" + STREAM + "\n"
                + done(6, "b") + "\n" + STREAM.replace("stream:", "total:").replace("rounds=4", "rounds=8")
                .replace("kept=3/2/1", "kept=6/4/2") + "\n"
                + "info: minimax budget: 131072 tokens a stream, 1472 pages, 8.26 GiB of KV room, fp8 K/V\n")
    r = run_py("bench/drafts.py", "log", str(log), "--out", str(tmp_path / "log.json"))
    assert r.returncode == 0, r.stderr
    js = json.loads((tmp_path / "log.json").read_text())
    assert js["streams"] == 2 and js["done_lines"] == 2 and js["total_acceptance"]["tau"] == 2.5
    assert js["streams_acceptance"]["tau"] == 2.5 and js["calibration"] == ["dspark calibration: auto, 8 widths"]
    assert any("minimax budget" in x for x in js["budget_lines"]) and "tau 2.5" in r.stdout


def test_the_prompt_set_has_the_four_classes_and_an_agent_follow_up():
    prompts, digest = dr.load_prompts()
    assert sorted(prompts["classes"]) == ["agent", "code", "prose", "structured"] and len(digest) == 64
    agent = {p["id"]: p for p in prompts["classes"]["agent"]}
    follow = agent["agent-result"]["messages"]
    assert follow[2]["tool_calls"] and [m["role"] for m in follow[-2:]] == ["tool", "tool"]
    assert agent["agent-call"]["tools"] == agent["agent-result"]["tools"]
    g = dr.plan(prompts, ["agent"], 3)
    assert [(x["id"], x["variant"]) for x in g] == [("agent-call", "greedy"), ("agent-result", "greedy"),
                                                    ("agent-call", "sampled")]
    body = dr.body_of(g[2], "m", 256)
    assert body["seed"] == 4242 and body["tool_choice"] == "auto" and body["max_tokens"] == 256
