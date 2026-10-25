"""scripts/check_secrets.py fails on a planted credential, prints it masked, and passes documented placeholders;
scripts/check-public.sh fails through it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "check_secrets.py"

# Built at runtime so this file carries no credential-shaped literal.
FAKES = {
    "OpenAI / Anthropic key": "sk-ant-" + "api03-" + "Q" * 30,
    "Hugging Face token": "hf_" + "Ab3" * 12,
    "GitHub token": "ghp_" + "Zx9" * 12,
    "Slack token": "xoxb-" + "1234-5678-abcdefgh",
    "AWS access key id": "AKIA" + "ABCDEFGH23456789",
    "Bearer token": "Bearer " + "eyJ" + "x" * 30,
    "credential assignment": 'api_key = "' + "q7Rk" * 6 + '"',
    "PEM private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
}
# the shape scripts/config.sh's api_key_ensure writes
SK = "sk-" + "tf-" + "Vb7_" * 11


def scan(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(SCAN), *map(str, paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_each_planted_credential_fails_masked(tmp_path):
    for kind, fake in FAKES.items():
        f = tmp_path / f"{kind.split()[0].lower()}.txt"
        f.write_text(f"config line\n{fake}\n")
        r = scan(f)
        assert r.returncode == 1, (kind, r.stdout)
        assert f"{f}:2: {kind}:" in r.stdout
        secret = fake.split('"')[1] if kind == "credential assignment" else fake
        # only the first four characters are shown
        assert secret not in r.stdout + r.stderr


def test_placeholders_pass(tmp_path):
    f = tmp_path / "README.md"
    f.write_text(
        'api_key = "<your key>"\npassword: "$PASSWORD"\ntoken = "${HF_TOKEN}"\nsecret = "changeme"\n'
        'API_KEY="xxxxxxxx"\nAuthorization: Bearer $API_KEY\nsk-cli\n--api-key sk-env\n'
    )
    r = scan(f)
    assert r.returncode == 0, r.stdout


def test_recipe_key_handling_passes(tmp_path):
    # scripts/config.sh / start.sh / check-drafted.sh lines: a switch, file pointers, a key read at run time
    f = tmp_path / "config.sh"
    f.write_text(
        'API_KEY="${API_KEY:-0}"\n'
        'API_KEY_FILE="${API_KEY_FILE:-$STATE_DIR/api-key}"\n'
        "API_KEY_IN_CONTAINER=/run/secrets/tensorfold-api-key\n"
        'HEAD_ONLY+=(--api-key-file "$API_KEY_IN_CONTAINER")\n'
        'printf \'Authorization: Bearer %s\\n\' "$(head -n 1 "$API_KEY_FILE")"\n'
        'req.add_header("Authorization", "Bearer " + key)\n'
        'HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HF_HOME/token}"\n'
    )
    r = scan(f)
    assert r.returncode == 0, r.stdout


def test_recipe_literal_keys_fail(tmp_path):
    hf = FAKES["Hugging Face token"]
    cases = {
        "serve --api-key " + SK: "OpenAI / Anthropic key: sk-t…",
        "export HF_TOKEN=" + hf: "Hugging Face token: hf_A…",
        "curl -H 'Authorization: Bearer " + "Zq8" * 8 + "'": "Bearer token: Bear…",
    }
    for line, want in cases.items():
        f = tmp_path / "notes.md"
        f.write_text(line + "\n")
        r = scan(f)
        assert r.returncode == 1 and f"{f}:1: {want}" in r.stdout, (line, r.stdout)
        assert SK not in r.stdout and hf not in r.stdout


def test_directory_scan_and_clean_exit(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ok.py").write_text("print('hello')\n")
    assert scan(tmp_path).returncode == 0
    (tmp_path / "a" / "leak.env").write_text(
        "HF_TOKEN=" + FAKES["Hugging Face token"] + "\n"
    )
    r = scan(tmp_path)
    assert r.returncode == 1 and "leak.env:1: Hugging Face token: hf_A…" in r.stdout


def test_check_public_fails_on_a_planted_key(tmp_path):
    (tmp_path / "scripts").mkdir()
    for name in (
        "check-public.sh",
        "check_secrets.py",
        "redact.py",
        "check-secrets-allow.txt",
    ):
        shutil.copy(ROOT / "scripts" / name, tmp_path / "scripts")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    def check() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "scripts/check-public.sh"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    (tmp_path / "start.md").write_text("API_KEY=1 ./start.sh restart\n")
    clean = check()
    assert clean.returncode == 0, clean.stdout
    (tmp_path / "start.md").write_text("./start.sh restart --api-key " + SK + "\n")
    r = check()
    assert r.returncode == 1, r.stdout
    assert "== credentials (scripts/check_secrets.py)" in r.stdout
    assert "start.md:1: OpenAI / Anthropic key: sk-t…" in r.stdout
    assert SK not in r.stdout + r.stderr
    # without the scanner the check fails rather than skipping credentials
    (tmp_path / "start.md").write_text("ok\n")
    (tmp_path / "scripts" / "check_secrets.py").unlink()
    r = check()
    assert r.returncode == 1 and "check_secrets.py is missing" in r.stdout
