"""bench/draft_positions.py: the acceptance arithmetic (conditional acceptance by position, survival, tau, speed
ratio), the token equality gate, the depth prior it derives, the requests it sends, and its report over synthetic runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import draft_positions as d


def rec(cls="code", id_="a", rep=0, tried=(), kept=(), rounds=0, tau=0.0, tok_s=10.0, sha="s", drafted=0, accepted=0,
        tokens=100):
    return {"class": cls, "id": id_, "rep": rep, "tried": list(tried), "kept": list(kept), "rounds": rounds, "tau": tau,
            "tok_s": tok_s, "token_sha256": sha, "text_sha256": "t", "drafted": drafted, "accepted": accepted,
            "completion_tokens": tokens}


def test_positions_pool_over_replies_of_any_depth():
    recs = [rec(tried=[10, 8, 4], kept=[8, 4, 1]), rec(tried=[5], kept=[5]), rec()]
    assert d.pooled(recs, "tried") == [15, 8, 4]
    assert d.pooled(recs, "kept") == [13, 4, 1]
    assert d.pooled([], "tried") == []


def test_conditional_acceptance_and_survival():
    a = d.conditional([20, 10, 4, 0], [10, 4, 1, 0])
    assert a == [0.5, 0.4, 0.25, None]
    assert d.survival(a) == pytest.approx([0.5, 0.2, 0.05])
    assert d.survival([None, 0.5]) == []
    assert d.survival([1.0, 1.0]) == [1.0, 1.0]


def test_tau_weights_each_reply_by_its_rounds():
    recs = [rec(rounds=10, tau=3.0), rec(rounds=30, tau=1.0), rec(rounds=0, tau=9.0)]
    assert d.tau(recs) == pytest.approx((30 + 30) / 40)
    assert d.tau([rec()]) == 0.0


def test_summary_of_a_class():
    s = d.summarize([rec(tried=[4, 2], kept=[2, 1], rounds=4, tau=2.0, drafted=8, accepted=3, tok_s=20),
                     rec(tried=[4, 2], kept=[2, 1], rounds=4, tau=1.0, drafted=8, accepted=1, tok_s=10)])
    assert (s["n"], s["tried"], s["kept"]) == (2, [8, 4], [4, 2])
    assert s["a"] == [0.5, 0.5]
    assert s["survival"] == pytest.approx([0.5, 0.25])
    assert s["tau"] == pytest.approx(1.5)
    assert s["acceptance_rate"] == pytest.approx(4 / 16)
    assert s["tok_s"] == pytest.approx(15)


def test_compare_finds_every_different_and_missing_reply_and_the_speed_ratio():
    base = [rec("code", "a", tok_s=10), rec("prose", "b", tok_s=20)]
    same = [rec("code", "a", tok_s=15), rec("prose", "b", tok_s=30)]
    c = d.compare(base, same)
    assert (c["compared"], c["different"], c["missing"]) == (2, [], [])
    assert c["speed_ratio"] == pytest.approx({"code": 1.5, "prose": 1.5, "all": 1.5})
    bad = [rec("code", "a", sha="other"), rec("agent", "z")]
    c = d.compare(base, bad)
    assert c["different"] == ["code/a/0"]
    assert c["missing"] == ["agent/z/0"]


def test_identity_falls_back_to_the_text_without_token_ids():
    assert d.identity(rec(sha="abc")) == "tokens:abc"
    assert d.identity(dict(rec(), token_sha256=None)) == "text:t"


def test_the_prior_takes_measured_survival_where_it_was_tried_enough():
    prior = d.ENGINE_DFLASH2_PRIOR
    out = d.calibrated_prior([0.9, 0.8, 0.5], [100, 100, 5], prior, min_tried=30)
    assert out[:2] == [0.9, 0.8]
    assert out[2] == round(0.8 * prior[2] / prior[1], 3), "too few tries: the prior's decay from the last measured value"
    assert len(out) == len(prior)
    assert all(x > y for x, y in zip(out, out[1:])), "strictly decreasing, as the engine's test requires"
    assert d.calibrated_prior([], [], prior) == list(prior)
    flat = d.calibrated_prior([1.0] * 7, [100] * 7, prior)
    assert all(x > y for x, y in zip(flat, flat[1:])) and flat[0] == 1.0


def test_the_requests_cover_four_classes_greedy():
    cases = d.prompts()
    assert {c["class"] for c in cases} == set(d.CLASSES)
    assert all(sum(c["class"] == k for c in cases) >= 4 for k in d.CLASSES)
    assert len({(c["class"], c["id"]) for c in cases}) == len(cases)
    agent = [c for c in cases if c["class"] == "agent"]
    assert all(c["tools"] for c in agent)
    assert any(m.get("role") == "tool" for c in agent for m in c["messages"])
    body = d.body_for(cases[0], "m", draft_off=False, thinking=False)
    assert (body["temperature"], body["stream"], "draft" in body) == (0, False, False)
    assert d.body_for(cases[0], "m", draft_off=True, thinking=False)["draft"] is False
    assert d.body_for(agent[0], "m", False, False)["tool_choice"] == "auto"


def test_a_reply_becomes_a_record():
    reply = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 12},
        "tensorfold": {"tokens_per_second": 31.5, "seconds": 0.4, "drafts": True, "token_sha256": "ab" * 32},
        "speculative": {"rounds": 4, "drafted": 14, "accepted": 8, "acceptance_rate": 8 / 14, "tokens_per_round": 3.0,
                        "positions": {"tried": [4, 3], "kept": [3, 2]}},
    }
    r = d.record({"class": "code", "id": "a"}, reply, "dflash2", 0, False, 1.0, 0.0)
    assert (r["tok_s"], r["tau"], r["rounds"], r["tried"], r["kept"]) == (31.5, 3.0, 4, [4, 3], [3, 2])
    assert (r["token_sha256"], r["completion_tokens"], r["draft"], r["drafts"]) == ("ab" * 32, 12, None, True)
    plain = d.record({"class": "code", "id": "a"}, {"choices": [{}]}, "plain", 0, True, 1.0, 0.0)
    assert (plain["rounds"], plain["tried"], plain["draft"], plain["token_sha256"]) == (0, [], False, None)


def test_node_states_wait_for_normal_and_stop_on_emergency(tmp_path):
    state = tmp_path / "state"
    state.write_text("normal\n")
    assert d.node_states([f"cat {state}", "false"]) == ["normal", "unknown"]
    assert d.wait_normal([f"cat {state}"]) < 5
    state.write_text("emergency\n")
    with pytest.raises(SystemExit, match="emergency"):
        d.wait_normal([f"cat {state}"])


def write(path: Path, recs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def test_report_passes_equal_runs_and_fails_a_different_one(run_py, tmp_path):
    plain = write(tmp_path / "plain.jsonl", [rec("code", "a"), rec("agent", "b")])
    drafted = write(tmp_path / "dflash2.jsonl", [rec("code", "a", tried=[40, 30], kept=[30, 15], rounds=40, tau=2.2,
                                                     tok_s=14), rec("agent", "b", tok_s=12)])
    out = tmp_path / "report.json"
    r = run_py("bench/draft_positions.py", "report", plain, drafted, "--json", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tokens: EQUAL (2 requests)" in r.stdout
    assert "PASS" in r.stdout
    rep = json.loads(out.read_text())
    code = rep["runs"]["dflash2"]["code"]
    assert code["a"] == [0.75, 0.5]
    assert code["survival"] == pytest.approx([0.75, 0.375])
    assert rep["compare"]["dflash2"]["speed_ratio"]["code"] == pytest.approx(1.4)
    assert "depth prior from this run's survival" in r.stdout
    other = write(tmp_path / "dspark.jsonl", [rec("code", "a", sha="different"), rec("agent", "b")])
    r = run_py("bench/draft_positions.py", "report", plain, other)
    assert r.returncode == 1
    assert "TOKENS DIFFER (1 of 2): code/a/0" in r.stdout
