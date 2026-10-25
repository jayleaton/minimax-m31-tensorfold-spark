"""Shared fixtures for the bench / canary tests: a dummy key built at runtime (no credential literal in the tree), its
mode-600 key file, the stand-in server, and a runner for the harnesses as ``python3 -I`` subprocesses with an
environment that holds none of the operator's settings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_minimax import FakeMiniMax

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = (
    "HOST",
    "PORT",
    "SERVED_NAME",
    "API_KEY",
    "API_KEY_FILE",
    "BASE_URL",
    "MODEL",
    "OUT",
    "TOOLEVAL_CACHE",
    "ENGINE_API_TOOLS",
)


@pytest.fixture
def key() -> str:
    return "DUMMY" + "-bench-" + os.urandom(6).hex()


@pytest.fixture
def key_file(tmp_path: Path, key: str) -> Path:
    path = tmp_path / "api-key"
    path.write_text(f"bench: {key}\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SETTINGS}


@pytest.fixture
def fake():
    """fake(**options): a started FakeMiniMax, shut down after the test."""

    started: list[FakeMiniMax] = []

    def make(**options) -> FakeMiniMax:
        f = FakeMiniMax(**options).__enter__()
        started.append(f)
        return f

    yield make
    for f in started:
        f.__exit__(None, None, None)


@pytest.fixture
def run_py(clean_env):
    """run_py("bench/x.py", *args, env={...}): the harness under ``python3 -I``, output captured."""

    def run(
        script: str, *args: str, env: dict | None = None, timeout: float = 120
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", str(ROOT / script), *map(str, args)],
            env=dict(clean_env, **(env or {})),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return run
