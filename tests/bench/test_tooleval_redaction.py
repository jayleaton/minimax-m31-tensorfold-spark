"""bench/tooleval/run.sh end to end with a dummy key: stand-in tool-eval-bench and spark-bench that echo their argv,
environment and request headers into every report, log and dump, and the stand-in MiniMax server checking the key.
The key comes from API_KEY_FILE ("label: key", mode 600); it reaches the benches through their environment, never
argv, and must not appear in any saved file or on the terminal, even when TEB_ARGS / SB_ARGS carry it as
--api-key=KEY or the server echoes the request headers into its replies."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = "MiniMax-M3.1-Flash-EXL3"
RUN = ROOT / "bench/tooleval/run.sh"
pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None, reason="run.sh needs curl"
)

# the stand-ins write what they got: argv and the key from their environment, and whether argv held the key
TEB = """#!/bin/sh
json=""; prev=""
for a in "$@"; do [ "$prev" = --json-file ] && json=$a; prev=$a; done
case "$*" in *"$TOOL_EVAL_API_KEY"*) argv=dirty ;; *) argv=clean ;; esac
[ -n "$TOOL_EVAL_API_KEY" ] || argv=unset
echo "tool-eval-bench argv: $*"
echo "tool-eval-bench using key $TOOL_EVAL_API_KEY" >&2
printf '{"argv": "%s", "argv_check": "%s", "headers": {"Authorization": "Bearer %s"}}\\n' \\
    "$*" "$argv" "$TOOL_EVAL_API_KEY" > "$json"
"""
SB = """#!/bin/sh
case "$*" in *"$SPARK_BENCH_API_KEY"*) argv=dirty ;; *) argv=clean ;; esac
echo "spark-bench argv: $*"
echo "spark-bench key=$SPARK_BENCH_API_KEY argv_check=$argv"
printf '{"request_headers": {"x-api-key": "%s"}, "argv": "%s"}\\n' "$SPARK_BENCH_API_KEY" "$*" \\
    > "$SPARK_BENCH_DUMP_DIR/raw_dump.jsonl"
"""


def install_fakes(cache: Path) -> None:
    for path, text in (
        (cache / "tool-eval-bench/.venv/bin/tool-eval-bench", TEB),
        (cache / "spark-bench/.venv/bin/python", SB),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(text)
        path.chmod(0o755)


def run_sh(env: dict, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUN), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def setup(tmp_path, clean_env, key, key_file, fake):
    server = fake(key=key, echo=True)
    install_fakes(tmp_path / "cache")
    env = dict(
        clean_env,
        BASE_URL=server.url + "/v1",
        MODEL=MODEL,
        API_KEY="1",
        API_KEY_FILE=str(key_file),
        OUT=str(tmp_path / "out"),
        TOOLEVAL_CACHE=str(tmp_path / "cache"),
    )
    return server, env, tmp_path / "out"


def assert_clean(out: Path, r: subprocess.CompletedProcess, key: str) -> list[Path]:
    files = [p for p in out.rglob("*") if p.is_file()]
    assert files
    for p in files:
        assert key not in p.read_text(errors="replace"), p
    assert key not in r.stdout + r.stderr
    return files


def test_key_never_saved_or_printed(setup, key):
    server, env, out = setup
    env.update(TEB_ARGS="--api-key=" + key, SB_ARGS="--api-key=" + key)
    r = run_sh(env, "both", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    # the key still reached the server, as a header
    seen = [
        x["headers"].get("Authorization")
        for x in server.requests
        if x["path"] != "/health"
    ]
    assert seen and all(h == "Bearer " + key for h in seen)

    files = assert_clean(out, r, key)
    names = {p.name for p in files}
    assert {
        "teb.cmd",
        "teb.log",
        "teb.json",
        "sb.cmd",
        "sb.log",
        "raw_dump.jsonl",
        "models-before.json",
    } <= names
    run = next(p.parent for p in files if p.name == "teb.cmd")
    teb_cmd = (run / "teb.cmd").read_text()
    assert (
        "TOOL_EVAL_API_KEY='<redacted>'" in teb_cmd
        or "TOOL_EVAL_API_KEY=<redacted>" in teb_cmd
    )
    assert "'--api-key=<redacted>'" in teb_cmd and "--no-think" in teb_cmd
    sb_cmd = (run / "sb.cmd").read_text()
    assert "SPARK_BENCH_API_KEY" in sb_cmd and "'--api-key=<redacted>'" in sb_cmd
    assert "<redacted>" in (run / "teb.json").read_text()  # scrubbed when the run ended


def test_key_reaches_the_benches_by_environment_not_argv(setup, key):
    _server, env, out = setup
    r = run_sh(env, "both", "high")
    assert r.returncode == 0, r.stdout + r.stderr
    assert_clean(out, r, key)
    run = next(p.parent for p in out.rglob("teb.json"))
    assert '"argv_check": "clean"' in (run / "teb.json").read_text()
    assert "argv_check=clean" in (run / "sb.log").read_text()
    assert "reasoning_effort" in (run / "teb.cmd").read_text()


@pytest.mark.parametrize("bench", ["own", "toolcall"])
def test_stdlib_harnesses_read_the_key_file(setup, key, key_file, bench):
    server, env, out = setup
    r = run_sh(env, bench, "low", "t1", timeout=300)
    assert r.returncode in (0, 1), (
        r.stdout + r.stderr
    )  # the stand-in model plans nothing: the score may fail
    assert "results: " in r.stdout
    files = assert_clean(out, r, key)
    cmd = next(p for p in files if p.suffix == ".cmd")
    assert str(key_file) in cmd.read_text() and "--api-key-file" in cmd.read_text()
    assert any(
        p.suffix == ".json" and p.parent == cmd.parent and p.stem == cmd.stem
        for p in files
    )
    chats = [x for x in server.requests if x["path"] == "/v1/chat/completions"]
    assert chats and all(
        x["headers"].get("Authorization") == "Bearer " + key for x in chats
    )


def test_refuses_a_literal_key_in_api_key_without_echoing_it(setup, key):
    _server, env, _out = setup
    r = run_sh(dict(env, API_KEY=key), "teb", "off")
    assert r.returncode != 0 and "API_KEY_FILE" in r.stderr
    assert key not in r.stdout + r.stderr


def test_api_key_1_needs_a_readable_file(setup, tmp_path):
    _server, env, _out = setup
    r = run_sh(dict(env, API_KEY_FILE=str(tmp_path / "missing")), "teb", "off")
    assert r.returncode != 0 and "cannot be read" in r.stderr


def test_no_key_file_sends_no_header(setup, tmp_path, fake):
    _server, env, out = setup
    open_server = fake()
    env = dict(env, BASE_URL=open_server.url + "/v1", API_KEY="0")
    env.pop("API_KEY_FILE")
    r = run_sh(env, "teb", "off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert all("Authorization" not in x["headers"] for x in open_server.requests)
    assert "TOOL_EVAL_API_KEY" not in next(out.rglob("teb.cmd")).read_text()


def test_usage_and_bad_mode(setup):
    _server, env, out = setup
    r = run_sh(env)
    assert r.returncode == 2 and "bench/tooleval/run.sh setup" in r.stdout
    r = run_sh(env, "teb", "medium")
    assert r.returncode != 0 and "mode must be off, adaptive, low or high" in r.stderr
    assert not out.exists() or not any(out.iterdir())


def test_unlisted_model_stops_before_any_run(setup):
    _server, env, out = setup
    r = run_sh(dict(env, MODEL="some-other-model"), "teb", "off")
    assert r.returncode != 0 and "does not list some-other-model" in r.stderr
    assert os.path.exists(out) is False or not any(out.iterdir())
