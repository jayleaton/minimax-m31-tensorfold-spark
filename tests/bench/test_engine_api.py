"""The engine features of the branch mm3/api-tools (required and named tool_choice, response_format, thinking off,
thinking_budget, the effort levels, streamed tool deltas) under ENGINE_API_TOOLS, against the stand-in server told to
refuse each one as the family as imported does, to serve it, or to serve it wrongly: a refusal SKIPs with 0 and fails
with 1, a correct service passes and a wrong one fails in both modes. Harnesses: bench/structured.py, bench/toolcall.py,
scripts/canary.py; the refusal matching: bench/engine_api.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fake_minimax as fm
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import engine_api as ea

SKIP = "SKIP (engine lacks mm3/api-tools)"
MODES = ["0", "1"]


def structured(
    run_py, server, tmp_path, mode: str, *args: str
) -> tuple[int, dict, str]:
    out = tmp_path / f"st-{mode}.json"
    r = run_py(
        "bench/structured.py",
        "--base",
        server.url,
        "--out",
        out,
        *args,
        env={"ENGINE_API_TOOLS": mode},
    )
    return r.returncode, json.loads(out.read_text()), r.stdout + r.stderr


def joined(rep: dict, key: str) -> str:
    return " | ".join(
        x for s in ("schemas", "tools", "thinking") if s in rep for x in rep[s][key]
    )


# -- the shared matcher ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "text", "feature"),
    [
        (400, json.dumps({"error": {"message": fm.REFUSAL}}), "tool_choice"),
        (400, fm.BACKEND_REFUSAL, "tool_choice"),
        # as the server sends it: JSON escapes the quotes around "required"
        (400, json.dumps({"error": {"message": fm.BACKEND_REFUSAL}}), "tool_choice"),
        (400, fm.STRUCTURES, "response_format"),
        (500, "engine failed: StructuresUnsupported", "response_format"),
        (
            400,
            "this engine cannot constrain a reply's first token, and the prompt does not end in ...",
            "grammar",
        ),
        (
            400,
            "tool_choice \"required\" or a named function on backend 'lanes': the grammar cannot be enforced: x",
            "grammar",
        ),
        (400, fm.BUDGET_REFUSAL, "thinking_budget"),
        (400, fm.EFFORT_REFUSAL, "reasoning_effort"),
    ],
)
def test_refusal_texts(status, text, feature):
    got = ea.refusal(status, text)
    assert got is not None and got.feature == feature
    assert "{" not in got.message  # the error's message, not its JSON


def test_other_errors_are_not_refusals():
    assert ea.refusal(400, '{"error": {"message": "bad JSON"}}') is None
    assert ea.refusal(401, fm.REFUSAL) is None  # a refused key is not a refused feature
    assert ea.refusal(200, fm.REFUSAL) is None


def test_switch_values(monkeypatch):
    monkeypatch.delenv("ENGINE_API_TOOLS", raising=False)
    assert ea.enabled() is False
    monkeypatch.setenv("ENGINE_API_TOOLS", "1")
    assert ea.enabled() is True and ea.enabled("0") is False
    monkeypatch.setenv("ENGINE_API_TOOLS", "yes")
    with pytest.raises(SystemExit, match="0 or 1"):
        ea.enabled()


def test_validator_subset():
    s = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
            "x": {"type": ["string", "null"]},
        },
        "required": ["n"],
        "additionalProperties": False,
    }
    assert ea.validate({"n": 2, "tags": ["a"], "x": None}, s) is None
    assert "missing n" in ea.validate({}, s)
    assert "not integer" in ea.validate({"n": "2"}, s)
    assert "not integer" in ea.validate({"n": True}, s)
    assert "minimum" in ea.validate({"n": 0}, s)
    assert "not in" in ea.validate({"n": 1, "tags": ["c"]}, s)
    assert "3 items" in ea.validate({"n": 1, "tags": ["a", "a", "a"]}, s)
    assert "extra key" in ea.validate({"n": 1, "y": 1}, s)


# -- structured.py ---------------------------------------------------------------------------------------------------


REFUSED = [
    ({"tool_choice": "refuse"}, "tools", "required/chat"),
    ({"tool_choice": "refuse-backend"}, "tools", "named-single/think"),
    ({"tool_choice": "refuse"}, "tools", "stream-required/chat"),
    ({"response_format": "refuse"}, "schemas", "json_object/chat"),
    ({"response_format": "refuse"}, "schemas", "response_format json_schema (think)"),
    ({"thinking_budget": "refuse"}, "thinking", "budget/64"),
    ({"reasoning_effort": "refuse"}, "thinking", "effort/max"),
]


@pytest.mark.parametrize(("api", "suite", "tag"), REFUSED)
@pytest.mark.parametrize("mode", MODES)
def test_structured_refusal(fake, run_py, tmp_path, mode, api, suite, tag):
    server = fake(api=dict(fm.NEW_ENGINE, **api))
    rc, rep, out = structured(run_py, server, tmp_path, mode, "--suites", suite)
    if mode == "0":
        assert rc == 0, out
        assert f"{tag}: {SKIP}" in joined(rep, "skipped") and not rep[suite]["fails"]
    else:
        assert rc == 1
        assert f"{tag}: " in joined(rep, "fails") and "(ENGINE_API_TOOLS=1)" in joined(
            rep, "fails"
        )


@pytest.mark.parametrize("mode", MODES)
def test_structured_correct_service_passes(fake, run_py, tmp_path, mode):
    server = fake(api=fm.NEW_ENGINE)
    rc, rep, out = structured(run_py, server, tmp_path, mode)
    assert rc == 0, out
    assert all(rep[s]["status"] == "PASS" for s in ("schemas", "tools", "thinking")), (
        out
    )
    assert not joined(rep, "skipped") and rep["engine_api_tools"] is (mode == "1")
    assert rep["tools"]["cases"]["stream-required/chat"]["bytes_equal"] is True
    assert rep["thinking"]["cases"]["budget/64"]["reasoning_tokens"] == 64
    assert rep["thinking"]["cases"]["effort/minimal"]["ran"] == "low"
    bodies = server.bodies()
    choices = [b.get("tool_choice") for b in bodies]
    assert (
        "required" in choices
        and {"type": "function", "function": {"name": "search_flights"}} in choices
    )
    assert any(
        b.get("parallel_tool_calls") is False and isinstance(b.get("tool_choice"), dict)
        for b in bodies
    )
    assert any(b.get("stream") and b.get("tool_choice") == "required" for b in bodies)
    assert {(b.get("response_format") or {}).get("type") for b in bodies} >= {
        "json_object",
        "json_schema",
    }
    assert {b.get("thinking_budget") for b in bodies} >= {64, 256}
    assert {b.get("reasoning_effort") for b in bodies} >= {*ea.EFFORTS, "minimal"}


WRONG = [
    (
        {"tool_choice": "wrong"},
        "tools",
        "named/chat: call 'get_weather' (want search_flights)",
    ),
    ({"tool_choice": "wrong"}, "tools", "required/think: call 'not_offered'"),
    ({"tool_choice": "badjson"}, "tools", "required/chat: arguments not JSON"),
    (
        {"tool_choice": "parallel"},
        "tools",
        "named-single/chat: 2 calls with parallel_tool_calls false",
    ),
    ({"response_format": "badjson"}, "schemas", "json_object/chat: not JSON"),
    ({"response_format": "schema"}, "schemas", "person/think: $: [] not object"),
    (
        {"response_format": "leak"},
        "schemas",
        "city/chat: markup in content (</mm:think>)",
    ),
    (
        {"thinking_off": "leak"},
        "thinking",
        "off/plain: markup in content (</mm:think>)",
    ),
    ({"thinking_off": "leak"}, "tools", "single/chat: markup in content (</mm:think>)"),
    (
        {"thinking_budget": "over"},
        "thinking",
        "budget/64: 600 reasoning tokens over the budget's 80",
    ),
    (
        {"thinking_budget": "noanswer"},
        "thinking",
        "budget/256: no answer after the close",
    ),
    ({"stream_tools": "wrong"}, "tools", "stream-required/chat: streamed calls"),
]


@pytest.mark.parametrize(("api", "suite", "want"), WRONG)
@pytest.mark.parametrize("mode", MODES)
def test_structured_wrong_service_fails(fake, run_py, tmp_path, mode, api, suite, want):
    server = fake(api=dict(fm.NEW_ENGINE, **api))
    rc, rep, out = structured(run_py, server, tmp_path, mode, "--suites", suite)
    assert rc == 1 and rep[suite]["status"] == "FAIL"
    assert want in joined(rep, "fails"), out


OLD_WAYS = [
    (
        {"thinking_off": "thinks"},
        "thinking",
        "off/plain",
        "thinking off, yet reasoning_content came back",
    ),
    (
        {"thinking_off": "thinks"},
        "schemas",
        "json_object/chat",
        "thinking off, yet reasoning_content",
    ),
    ({"reasoning_effort": "old"}, "thinking", "effort/high", "ran as xhigh"),
]


@pytest.mark.parametrize(("api", "suite", "tag", "why"), OLD_WAYS)
@pytest.mark.parametrize("mode", MODES)
def test_structured_old_engine_behaviour(
    fake, run_py, tmp_path, mode, api, suite, tag, why
):
    """Accepted, but as the family as imported serves it: a skip with 0, a failure with 1."""

    server = fake(api=dict(fm.NEW_ENGINE, **api))
    rc, rep, out = structured(run_py, server, tmp_path, mode, "--suites", suite)
    if mode == "0":
        assert rc == 0, out
        assert f"{tag}: {SKIP}: {why}" in joined(rep, "skipped")
    else:
        assert rc == 1 and f"{tag}: {why}" in joined(rep, "fails")


def test_structured_old_engine_whole_run(fake, run_py, tmp_path):
    server = fake(api=fm.OLD_ENGINE)
    rc, rep, out = structured(run_py, server, tmp_path, "0")
    assert rc == 0, out
    assert rep["schemas"]["status"] == "SKIP" and SKIP in out
    rc, rep, _ = structured(run_py, server, tmp_path, "0", "--engine-api-tools")
    assert rc == 1 and rep["engine_api_tools"] is True
    rc, _, out = structured(run_py, server, tmp_path, "1", "--engine-api-tools", "0")
    assert rc == 0, out


def test_bad_switch_value_is_refused(fake, run_py):
    r = run_py(
        "bench/structured.py", "--base", fake().url, env={"ENGINE_API_TOOLS": "yes"}
    )
    assert r.returncode != 0 and "ENGINE_API_TOOLS is 0 or 1" in r.stderr


# -- toolcall.py --tool-choice required --------------------------------------------------------------------------------


def toolcall(run_py, server, tmp_path, mode: str) -> tuple[int, dict, str]:
    out = tmp_path / f"tc-{mode}.json"
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--reps",
        "1",
        "--tool-choice",
        "required",
        "--only",
        "bash_4arg,nested_object",
        "--out",
        out,
        env={"ENGINE_API_TOOLS": mode},
    )
    return r.returncode, json.loads(out.read_text()), r.stdout + r.stderr


@pytest.mark.parametrize("mode", MODES)
def test_toolcall_required(fake, run_py, tmp_path, mode):
    from test_minimax_harness import TYPED_ARGS

    rc, rep, out = toolcall(
        run_py, fake(args=TYPED_ARGS, api=fm.OLD_ENGINE), tmp_path, mode
    )
    if mode == "0":
        assert rc == 0 and rep["unsupported"] == 2 and SKIP in out
    else:
        assert rc == 1 and "refused (ENGINE_API_TOOLS=1)" in out
    rc, rep, out = toolcall(
        run_py, fake(args=TYPED_ARGS, api=fm.NEW_ENGINE), tmp_path, mode
    )
    assert rc == 0 and rep["passed"] == rep["total"] == 2, out
    wrong = fake(args=TYPED_ARGS, api=dict(fm.NEW_ENGINE, tool_choice="badjson"))
    rc, rep, out = toolcall(run_py, wrong, tmp_path, mode)
    assert rc == 1 and "bad_json" in out


# -- the canary --------------------------------------------------------------------------------------------------------


def canary(run_py, server, mode: str, *args: str):
    return run_py(
        "scripts/canary.py", "--base", server.url, *args, env={"ENGINE_API_TOOLS": mode}
    )


def test_canary_probes_the_features_with_the_switch_on(fake, run_py):
    server = fake(api=fm.NEW_ENGINE)
    r = canary(run_py, server, "1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith(
        "[canary] ok: 6 probes (chat, thinking, tool, json, required, nothink)"
    )
    bodies = server.bodies()
    assert (
        bodies[4]["tool_choice"] == "required"
        and bodies[4]["chat_template_kwargs"]["enable_thinking"] is False
    )
    assert (
        bodies[5]["chat_template_kwargs"] == {"enable_thinking": False}
        and "tools" not in bodies[5]
    )


def test_canary_unchanged_with_the_switch_off(fake, run_py):
    server = fake(api=fm.NEW_ENGINE)
    r = canary(run_py, server, "0")
    assert r.returncode == 0 and r.stdout.startswith(
        "[canary] ok: 4 probes (chat, thinking, tool, json)"
    )
    assert all(b.get("tool_choice") != "required" for b in server.bodies())
    # the family as imported: JSON by prompt, as before
    r = canary(run_py, fake(api=fm.OLD_ENGINE), "0")
    assert r.returncode == 0 and "json prompted" in r.stdout


def test_canary_fails_a_refusal_with_the_switch_on(fake, run_py):
    r = canary(run_py, fake(api=fm.OLD_ENGINE), "1")
    assert r.returncode == 1
    assert "[canary] FAIL json: refused, the engine lacks mm3/api-tools" in r.stderr
    assert "[canary] FAIL required: refused, the engine lacks mm3/api-tools" in r.stderr
    assert (
        "[canary] FAIL nothink: thinking off, yet reasoning_content came back"
        in r.stderr
    )
    r = canary(run_py, fake(api=fm.OLD_ENGINE), "0", "--engine-api-tools")
    assert r.returncode == 1


@pytest.mark.parametrize(
    ("api", "want"),
    [
        ({"thinking_off": "leak"}, "nothink: raw MiniMax markup in content"),
        ({"tool_choice": "wrong"}, "required: called 'not_offered', not get_weather"),
        ({"tool_choice": "badjson"}, "required: arguments not a JSON object"),
    ],
)
def test_canary_fails_a_wrong_service(fake, run_py, api, want):
    r = canary(run_py, fake(api=dict(fm.NEW_ENGINE, **api)), "1")
    assert r.returncode == 1 and want in r.stderr
