"""scripts/canary.py against the stand-in MiniMax server: the four probes pass on a healthy reply, each MiniMax failure
(raw markup in content, reasoning not separated, a call left as text, a degenerate reply) fails it, a refused
response_format falls back to a prompted JSON reply, and the key never shows in argv or output."""

from __future__ import annotations

import socket

import fake_minimax as fm

CANARY = "scripts/canary.py"


def test_ok_prints_one_line(fake, run_py, key, key_file):
    server = fake(key=key)
    r = run_py(
        CANARY, "--base", server.url, "--model", fm.MODEL, "--api-key-file", key_file
    )
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith(
        "[canary] ok: 4 probes (chat, thinking, tool, json)"
    )
    bodies = server.bodies()
    assert [b["chat_template_kwargs"]["enable_thinking"] for b in bodies] == [
        False,
        True,
        False,
        False,
    ]
    assert bodies[2]["tool_choice"] == "auto" and bodies[1]["reasoning_effort"] == "low"
    assert all(
        x["headers"].get("Authorization") == "Bearer " + key for x in server.requests
    )
    assert (
        len({b["messages"][0]["content"].split("]")[0] for b in bodies}) == 1
    )  # one nonce a run


def test_defaults_from_the_environment(fake, run_py, key, key_file):
    server = fake(key=key, model="served-name-x")
    env = {
        "HOST": "0.0.0.0",
        "PORT": server.url.rsplit(":", 1)[1],
        "SERVED_NAME": "served-name-x",
        "API_KEY_FILE": str(key_file),
    }
    r = run_py(CANARY, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {b["model"] for b in server.bodies()} == {"served-name-x"}


def test_required_tool_choice_is_opt_in(fake, run_py):
    server = fake(enforce_tool_choice=True)
    r = run_py(CANARY, "--base", server.url, "--tool-choice", "required")
    assert r.returncode == 0, r.stdout + r.stderr
    assert server.bodies()[2]["tool_choice"] == "required"
    r = run_py(
        CANARY, "--base", fake().url, "--tool-choice", "required"
    )  # the engine's refusal today
    assert r.returncode == 1 and "tool: request failed" in r.stderr


def test_refused_response_format_falls_back_to_a_prompt(fake, run_py):
    server = fake(response_format="refuse")
    r = run_py(CANARY, "--base", server.url)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "json prompted" in r.stdout
    assert "response_format" not in server.bodies()[-1]


def test_raw_markup_fails(fake, run_py, key, key_file):
    server = fake(key=key, leak=True, echo=True)
    r = run_py(CANARY, "--base", server.url, "--api-key-file", key_file)
    assert r.returncode == 1
    assert "[canary] FAIL thinking: raw MiniMax markup in content" in r.stderr
    assert "[canary] FAIL tool: raw MiniMax markup in content" in r.stderr
    assert key not in r.stdout + r.stderr


def test_reasoning_must_come_back_separated(fake, run_py):
    def no_think(body, f):
        return fm.default_brain(dict(body, chat_template_kwargs={}), f)

    r = run_py(CANARY, "--base", fake(brain=no_think).url)
    assert r.returncode == 1 and "thinking: no reasoning_content" in r.stderr


def test_a_call_left_as_text_fails(fake, run_py):
    def broken_call(body, f):
        raw = fm.default_brain(body, f)
        return raw.replace(
            fm.NS + "</invoke>", "", 1
        )  # the server cannot parse it: the block stays text

    r = run_py(CANARY, "--base", fake(brain=broken_call).url)
    assert r.returncode == 1
    assert (
        "tool: raw MiniMax markup in content" in r.stderr
        and "tool: no tool_calls" in r.stderr
    )


def test_degenerate_reply_fails(fake, run_py):
    def salad(body, f):
        return (
            "Paris " + "the " * 40
            if "France" in fm.visible(body["messages"][-1]["content"])
            else fm.default_brain(body, f)
        )

    r = run_py(CANARY, "--base", fake(brain=salad).url)
    assert r.returncode == 1 and "chat: repetitive reply" in r.stderr


def test_unreachable_or_refused_key_is_exit_2(fake, run_py, key, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = run_py(CANARY, "--base", f"http://127.0.0.1:{port}", "--timeout", "5")
    assert r.returncode == 2 and "cannot query" in r.stderr
    r = run_py(CANARY, "--base", fake(key=key).url)  # no key sent
    assert r.returncode == 2 and "refused the key" in r.stderr
    wrong = tmp_path / "wrong-key"
    wrong.write_text("someone-elses-key-123456\n")
    r = run_py(CANARY, "--base", fake(key=key).url, "--api-key-file", wrong)
    assert r.returncode == 2 and "someone-elses-key-123456" not in r.stdout + r.stderr
