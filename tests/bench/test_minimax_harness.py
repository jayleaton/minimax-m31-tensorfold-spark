"""The stdlib harnesses (bench/toolcall.py, tooleval/chains.py, structured.py, quality.py, stress.py, soak.py) against
the stand-in MiniMax server: clean runs pass, raw-markup leaks and typing slips are counted, refused engine features
SKIP (test_engine_api.py has the ENGINE_API_TOOLS matrix), and the key (read from a mode-600 file, sent as a header) never appears in argv, output or saved files
even when the server echoes the request headers into its replies."""

from __future__ import annotations

import json
from pathlib import Path

import fake_minimax as fm
import pytest

TYPED_ARGS = {
    "set_config": {
        "service": "api",
        "settings": {"threshold": 0.75, "retries": 3, "verbose": False, "mode": "safe"},
    },
    "tag_files": {
        "paths": ["a.py", "b.py"],
        "tags": ["core", "v2"],
        "recursive": False,
    },
    "http_request": {
        "method": "POST",
        "url": "https://api.example.com/items",
        "headers": {"Accept": "application/json"},
        "body": {"name": "widget", "count": 2},
        "timeout_s": 2.5,
    },
    "edit": {
        "filePath": "src/app.ts",
        "oldString": "var ",
        "newString": "let ",
        "replaceAll": True,
    },
    "bash": {
        "command": "ls /srv/projects",
        "description": "list projects",
        "workdir": "/tmp",
        "timeout": 10000,
    },
    "read": {"filePath": "config.yaml", "offset": 0, "limit": 20},
    "glob": {"pattern": "**/*.ts"},
    "grep": {"pattern": "42"},
}
TYPED_CASES = "bash_4arg,edit_replace_all,grep_numeric_string,nested_object,string_arrays,nested_mapping,parallel_two"


def all_text(root: Path) -> str:
    return "".join(
        p.read_text(errors="replace") for p in root.rglob("*") if p.is_file()
    )


# -- the stand-in itself: the template's rendering read back by the server's rules ----------------------------------


def test_round_trip_types_by_schema():
    tools = [
        {"type": "function", "function": {"name": n, "parameters": p}}
        for n, p in (
            (
                "set_config",
                {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "settings": {
                            "type": "object",
                            "properties": {
                                "threshold": {"type": "number"},
                                "retries": {"type": "integer"},
                                "verbose": {"type": "boolean"},
                                "mode": {"type": "string"},
                            },
                        },
                    },
                },
            ),
            (
                "tag_files",
                {
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array"},
                        "n": {"type": "string"},
                    },
                },
            ),
        )
    ]
    calls = [
        ("set_config", TYPED_ARGS["set_config"]),
        ("tag_files", {"paths": ["a.py"], "tags": [], "n": "42"}),
    ]
    raw = "<mm:think>\nplan\n</mm:think>\nok " + fm.render_calls(calls)
    content, reasoning, parsed = fm.split_reply(raw, tools)
    assert (content, reasoning) == ("ok", "plan")
    assert parsed == calls
    assert (
        fm.split_reply(raw, tools, typing="strings")[2][0][1]["settings"]["retries"]
        == "3"
    )
    broken = raw.replace(fm.NS + "</invoke>", "", 1)
    content, _, parsed = fm.split_reply(broken, tools)
    assert (
        not parsed and fm.NS in content
    )  # an invalid block stays text, as on the server


# -- toolcall.py ----------------------------------------------------------------------------------------------------


def test_toolcall_clean_run(fake, run_py, key, key_file, tmp_path):
    server = fake(key=key, args=TYPED_ARGS, echo=True)
    out = tmp_path / "r" / "toolcall.json"
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--reps",
        "1",
        "--only",
        TYPED_CASES + ",followup_answer",
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert (
        rep["passed"] == rep["total"] == 8
        and rep["corruption"] == rep["leak"] == rep["typing"] == 0
    )
    follow = next(x for x in rep["results"] if x["case"] == "followup_answer")[
        "followup"
    ]
    assert "8080" in follow["content_preview"]
    # turn 2 sent the model's own calls back, then the tool results
    turn2 = [b for b in server.bodies() if b["messages"][-1]["role"] == "tool"]
    assert (
        turn2
        and turn2[0]["messages"][-2]["tool_calls"][0]["function"]["name"] == "read"
    )
    assert all(
        x["headers"].get("Authorization") == "Bearer " + key for x in server.requests
    )
    assert key not in r.stdout + r.stderr + out.read_text()


def test_toolcall_streamed(fake, run_py, tmp_path):
    server = fake(args=TYPED_ARGS)
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--reps",
        "1",
        "--stream",
        "--mode",
        "low",
        "--only",
        TYPED_CASES,
        "--out",
        tmp_path / "s.json",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(b.get("stream") for b in server.bodies())


def test_toolcall_counts_raw_markup_as_leaks(fake, run_py, tmp_path):
    server = fake(args=TYPED_ARGS, leak=True)
    out = tmp_path / "leak.json"
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--reps",
        "1",
        "--mode",
        "high",
        "--only",
        TYPED_CASES,
        "--out",
        out,
    )
    assert r.returncode == 1
    rep = json.loads(out.read_text())
    assert rep["leak"] == rep["total"] == 7 and rep["passed"] == 0
    assert all("leaked_markup" in x["problems"] for x in rep["results"])


def test_toolcall_counts_typing_slips(fake, run_py, tmp_path):
    server = fake(args=TYPED_ARGS, typing="strings")
    out = tmp_path / "typing.json"
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--reps",
        "1",
        "--only",
        "nested_object,string_arrays,grep_numeric_string",
        "--out",
        out,
    )
    assert r.returncode == 1
    rep = json.loads(out.read_text())
    by = {x["case"]: x for x in rep["results"]}
    assert by["nested_object"]["typing"] and by["string_arrays"]["typing"]
    assert by["grep_numeric_string"][
        "ok"
    ]  # a string that looks like a number stays a string


def test_toolcall_required_is_skipped_when_refused(fake, run_py, tmp_path):
    server = fake(args=TYPED_ARGS)
    out = tmp_path / "req.json"
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
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKIP" in r.stdout and json.loads(out.read_text())["unsupported"] == 2


def agent_brain(steps: dict):
    """A model that takes a coding client's steps: ``steps[case prompt word]`` lists one reply per turn."""

    def brain(body: dict, fake) -> str:
        msgs = body["messages"]
        user = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
        turn = sum(m["role"] == "assistant" for m in msgs if m.get("tool_calls"))
        replies = next(v for k, v in steps.items() if k in user)
        reply = replies[min(turn, len(replies) - 1)]
        return reply if isinstance(reply, str) else fm.render_calls(reply)

    return brain


AGENT_STEPS = {
    "replace every": [
        [("read", {"filePath": "src/app.ts"})],
        [("edit", TYPED_ARGS["edit"])],
    ],
    "microservices": [
        [("todowrite", {"todos": [{"content": "Read README.md", "status": "pending", "priority": "high"}]})],
        [("read", {"filePath": "README.md"})],
    ],
    "What port": [
        [("glob", {"pattern": "**/config.yaml"})],
        [("read", {"filePath": "config.yaml"})],
        "It sets port 8080.",
    ],
}


def test_toolcall_credits_a_preparatory_step_and_counts_it(fake, run_py, tmp_path):
    """Read before edit, plan before read, find before read: answered from the case's canned results, then checked."""

    server = fake(brain=agent_brain(AGENT_STEPS))
    out = tmp_path / "prep.json"
    r = run_py(
        "bench/toolcall.py",
        "--base",
        server.url,
        "--reps",
        "1",
        "--only",
        "edit_replace_all,long_reasoning_tool,followup_answer",
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["passed"] == rep["total"] == 3 and rep["direct"] == 0
    assert "3/3 clean (100.0%), 0 without a preparatory step" in r.stdout
    edit = next(x for x in rep["results"] if x["case"] == "edit_replace_all")
    assert edit["prep"][0]["calls"][0]["name"] == "read" and not edit["direct"]
    # the read went back with the case's file, not a placeholder
    bodies = server.bodies()
    after_read = next(b for b in bodies[1:] if "replace every" in b["messages"][-3]["content"])
    assert after_read["messages"][-1]["content"].startswith("var a = 1;")
    follow = next(x for x in rep["results"] if x["case"] == "followup_answer")
    assert "8080" in follow["followup"]["content_preview"]
    found = next(
        b for b in bodies if b["messages"][-1]["role"] == "tool" and "What port" in b["messages"][-3]["content"]
    )
    assert found["messages"][-1]["content"] == "config.yaml\n"


def test_toolcall_prep_is_bounded_and_direct_calls_count(fake, run_py, tmp_path):
    """A model that only ever reads fails edit_replace_all after MAX_PREP answered reads; a direct edit is direct."""

    steps = {"replace every": [[("read", {"filePath": "src/app.ts"})]]}
    server = fake(brain=agent_brain(steps))
    out = tmp_path / "loop.json"
    r = run_py("bench/toolcall.py", "--base", server.url, "--reps", "1", "--only", "edit_replace_all", "--out", out)
    assert r.returncode == 1
    res = json.loads(out.read_text())["results"][0]
    assert res["problems"] == ["missing_expected:any_of(['edit'])"] and len(res["prep"]) == 2
    assert len(server.bodies()) == 3
    server = fake(args=TYPED_ARGS)
    r = run_py("bench/toolcall.py", "--base", server.url, "--reps", "1", "--only", "edit_replace_all", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["direct"] == rep["passed"] == 1 and "prep" not in rep["results"][0]


def test_toolcall_adaptive_mode_sends_no_effort(fake, run_py, tmp_path):
    server = fake(args=TYPED_ARGS)
    r = run_py(
        "bench/toolcall.py", "--base", server.url, "--reps", "1", "--mode", "adaptive", "--temperature", "-1",
        "--only", "bash_4arg", "--out", tmp_path / "a.json",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    body = server.bodies()[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in body and "temperature" not in body


# -- tooleval/chains.py ---------------------------------------------------------------------------------------------


def geo_brain(seen: list):
    def brain(body: dict, fake) -> str:
        msgs = body["messages"]
        seen.append([m for m in msgs if m["role"] == "assistant"])
        think = "<mm:think>\nnext step\n</mm:think>\n"
        tools = [m for m in msgs if m["role"] == "tool"]
        if not tools:
            return think + fm.render_calls([("geocode", {"place": "Tokyo"})])
        if len(tools) == 1:
            pos = json.loads(tools[0]["content"])
            return think + fm.render_calls(
                [("get_timezone", {"lat": pos["lat"], "lon": pos["lon"]})]
            )
        return think + "Tokyo is in Asia/Tokyo."

    return brain


def test_chains_scores_a_chain_and_sends_history_back(
    fake, run_py, key, key_file, tmp_path
):
    seen: list = []
    server = fake(key=key, brain=geo_brain(seen))
    out = tmp_path / "chains.json"
    r = run_py(
        "bench/tooleval/chains.py",
        "--base",
        server.url + "/v1",
        "--api-key-file",
        key_file,
        "--mode",
        "low",
        "--only",
        "geo-timezone",
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["scores"] == [2] and rep["leaks"] == 0
    assert rep["reps"][0][0]["calls"] == [
        ["geocode", {"place": "Tokyo"}],
        ["get_timezone", {"lat": 35.68, "lon": 139.69}],
    ]
    # the assistant turns went back with their tool_calls and reasoning_content as returned
    assert [m["reasoning_content"] for m in seen[-1]] == ["next step", "next step"]
    assert server.bodies()[0]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }


def test_chains_counts_a_leak(fake, run_py, tmp_path):
    server = fake(brain=geo_brain([]), leak=True)
    out = tmp_path / "chains.json"
    r = run_py(
        "bench/tooleval/chains.py",
        "--base",
        server.url,
        "--only",
        "geo-timezone",
        "--out",
        out,
    )
    assert r.returncode == 1
    rep = json.loads(out.read_text())
    assert rep["leaks"] == 1 and rep["scores"] == [0]
    assert "markup in content" in rep["reps"][0][0]["malformed"][0]


def edit_brain(body: dict, fake) -> str:
    """Lists, reads, writes workers 8, reads the file back, and writes again while the read still says otherwise."""

    tools = [m for m in body["messages"] if m["role"] == "tool"]
    if not tools:
        return fm.render_calls([("list_files", {"path": "/srv/app"})])
    if len(tools) == 1:
        return fm.render_calls([("read_file", {"path": "/srv/app/config.yaml"})])
    last = tools[-1]["content"]
    if '"ok": true' in last:
        return fm.render_calls([("read_file", {"path": "/srv/app/config.yaml"})])
    if "workers: 8" in last:
        return "Done: workers is now 8 and the other lines are unchanged."
    new = last.replace("workers: 4", "workers: 8")
    return fm.render_calls([("write_file", {"path": "/srv/app/config.yaml", "content": new})])


def test_chains_edit_config_reads_its_own_write_back(fake, run_py, tmp_path):
    """The simulated file keeps the write: a model that checks its edit confirms it instead of rewriting until the
    turn limit (the stateless file answered every read with workers: 4)."""

    server = fake(brain=edit_brain)
    out = tmp_path / "chains.json"
    r = run_py("bench/tooleval/chains.py", "--base", server.url, "--only", "edit-config", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())["reps"][0][0]
    assert rep["points"] == 2 and rep["turns"] == 5
    assert [c[0] for c in rep["calls"]] == ["list_files", "read_file", "write_file", "read_file"]
    assert rep["calls"][2][1]["content"] == "port: 8080\nworkers: 8\nlog_level: info\n"


def retry_brain(body: dict, fake) -> str:
    tools = [m for m in body["messages"] if m["role"] == "tool"]
    if not tools:
        return fm.render_calls([("get_weather", {"city": "Pariss"})])
    if "Did you mean Paris" in tools[-1]["content"]:
        return fm.render_calls([("get_weather", {"city": "Paris"})])
    return "Paris: " + json.loads(tools[-1]["content"])["sky"]


def test_chains_retry_after_error_asks_for_the_retry(fake, run_py, tmp_path):
    """The prompt asks for the literal city first and a corrected call when the tool rejects it (the old wording,
    "use the tool exactly with the city I wrote", left stopping after the error a valid reading); the chain scores."""

    server = fake(brain=retry_brain)
    out = tmp_path / "chains.json"
    r = run_py("bench/tooleval/chains.py", "--base", server.url, "--only", "retry-after-error", "--mode", "adaptive",
               "--temperature", "-1", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(out.read_text())["scores"] == [2]
    body = server.bodies()[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and "temperature" not in body
    prompt = body["messages"][-1]["content"]
    assert "exactly as I wrote it first" in prompt and "call it again" in prompt


# -- structured.py --------------------------------------------------------------------------------------------------


def test_structured_passes_and_skips_refused_tool_choice(fake, run_py, tmp_path):
    server = fake()
    out = tmp_path / "st.json"
    r = run_py("bench/structured.py", "--base", server.url, "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["schemas"]["status"] == "PASS" and rep["tools"]["status"] == "PASS"
    assert rep["thinking"]["status"] == "PASS"
    skipped = " ".join(rep["tools"]["skipped"])
    assert (
        "required/chat" in skipped
        and "named/think" in skipped
        and "cannot be enforced" in skipped
        and "SKIP (engine lacks mm3/api-tools)" in skipped
    )
    assert any(b.get("draft") is False for b in server.bodies())
    r = run_py(
        "bench/structured.py", "--base", server.url, "--suites", "tools", "--strict"
    )
    assert r.returncode == 1


def test_structured_response_format_refused_is_skipped(fake, run_py, tmp_path):
    server = fake(response_format="refuse")
    out = tmp_path / "st.json"
    r = run_py(
        "bench/structured.py", "--base", server.url, "--suites", "schemas", "--out", out
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    # json_schema and json_object, thinking off and on
    assert rep["schemas"]["status"] == "SKIP" and len(rep["schemas"]["skipped"]) == 4


def test_structured_fails_on_leaks(fake, run_py):
    server = fake(leak=True)
    r = run_py("bench/structured.py", "--base", server.url, "--suites", "tools")
    assert r.returncode == 1 and "markup in content" in r.stdout


# -- quality.py -----------------------------------------------------------------------------------------------------


def test_quality_mmlu_and_compare(fake, run_py, key, key_file, tmp_path):
    server = fake(key=key)
    data = Path(__file__).resolve().parents[2] / "bench/data/mmlu200.jsonl"
    rows = data.read_text().splitlines()
    assert len(rows) == 200
    small = tmp_path / "m.jsonl"
    small.write_text("\n".join(rows[:12]) + "\n")
    want = sum(
        json.loads(x)["answer"] == 1 for x in rows[:12]
    )  # the stand-in answers B
    out = tmp_path / "q.json"
    r = run_py(
        "bench/quality.py",
        "mmlu",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--data",
        small,
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads(out.read_text())
    assert (
        rec["correct"] == want
        and rec["n"] == 12
        and rec["leaks"] == 0
        and rec["errors"] == 0
    )
    assert key not in r.stdout + r.stderr + out.read_text()
    assert server.bodies()[0]["chat_template_kwargs"] == {"enable_thinking": False}
    r = run_py("bench/quality.py", "compare", "--run", out, "--ref", out)
    assert r.returncode == 0 and "delta +0.0 points" in r.stdout


def test_quality_needle_without_tokenize(fake, run_py, key, key_file, tmp_path):
    server = fake(
        key=key, echo=True
    )  # the reply carries the request headers: the saved reply must be masked
    out = tmp_path / "n.json"
    r = run_py(
        "bench/quality.py",
        "needle",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--sizes",
        "2000",
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(out.read_text())["ok"] is True
    assert (
        "<redacted>" in out.read_text()
        and key not in out.read_text() + r.stdout + r.stderr
    )


# -- stress.py, soak.py ---------------------------------------------------------------------------------------------


def test_stress_and_admit(fake, run_py, key, key_file, tmp_path):
    server = fake(key=key)
    out = tmp_path / "stress.json"
    r = run_py(
        "bench/stress.py",
        "stress",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--long",
        "300",
        "--dctx",
        "100",
        "--decode",
        "8",
        "--stagger",
        "0",
        "--streams",
        "4",
        "--require-full",
        "--out",
        out,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["model"] == fm.MODEL and len(rep["streams"]) == 4
    bodies = server.bodies("/v1/completions")
    assert all(
        b["ignore_eos"] and 1000 <= min(b["prompt"]) and max(b["prompt"]) < 190_000
        for b in bodies
    )
    r = run_py(
        "bench/stress.py",
        "admit",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--tokens",
        "64",
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("kinds", ["mix", "session"])
def test_soak_short(fake, run_py, key, key_file, tmp_path, kinds):
    server = fake(key=key, response_format="refuse")
    out = tmp_path / "soak.json"
    r = run_py(
        "bench/soak.py",
        "--base",
        server.url,
        "--api-key-file",
        key_file,
        "--minutes",
        "0.05",
        "--drain-s",
        "2",
        "--kinds",
        kinds,
        "--out",
        out,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert (
        rep["pass"]
        and rep["errors"] == 0
        and rep["leaks"] == 0
        and rep["drained"] is True
    )
    assert rep["response_format"].startswith("refused") and rep["answer_17x23"] == "391"
    assert all(b.get("tool_choice") != "required" for b in server.bodies())


def test_soak_fails_on_leaks_and_unknown_drain_passes(fake, run_py, tmp_path):
    server = fake(leak=True, health={"status": "ok"})
    out = tmp_path / "soak.json"
    r = run_py(
        "bench/soak.py",
        "--base",
        server.url,
        "--minutes",
        "0.05",
        "--drain-s",
        "1",
        "--out",
        out,
    )
    assert r.returncode == 1
    rep = json.loads(out.read_text())
    assert rep["leaks"] > 0 and rep["drained"] == "unknown"


# -- defaults from scripts/config.sh's settings ----------------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1"])
def test_defaults_from_the_environment(fake, run_py, key, key_file, tmp_path, host):
    server = fake(key=key, model="served-name-x", args=TYPED_ARGS)
    port = server.url.rsplit(":", 1)[1]
    env = {
        "HOST": host,
        "PORT": port,
        "SERVED_NAME": "served-name-x",
        "API_KEY_FILE": str(key_file),
    }
    r = run_py("bench/toolcall.py", "--reps", "1", "--only", "bash_4arg", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert server.bodies()[0]["model"] == "served-name-x"
    # API_KEY=1 with no readable file is an error; without API_KEY=1 a missing file means no key
    env.update(API_KEY_FILE=str(tmp_path / "missing"))
    assert (
        run_py(
            "bench/toolcall.py", "--reps", "1", "--only", "bash_4arg", env=env
        ).returncode
        == 1
    )
    assert (
        "No such file"
        in run_py(
            "bench/toolcall.py", "--only", "bash_4arg", env=dict(env, API_KEY="1")
        ).stderr
    )
