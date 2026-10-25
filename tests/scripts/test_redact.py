"""scripts/redact.py: what is masked in argv, env, text and JSON, and what stays readable."""

from __future__ import annotations

import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from redact import (
    MASK,
    known_secrets,
    mask_values,
    redact,
    redact_argv,
    redact_env,
    redact_text,
    scrub,
    secret_name,
)

# built at runtime: no credential literal in the tree
KEY = "DUMMY" + "-k3y-" + "0123456789abcdef"
# the shape scripts/config.sh's api_key_ensure writes to API_KEY_FILE
SK = "sk-" + "tf-" + secrets.token_urlsafe(32)
HF = "hf_" + "Q9w" * 12


def test_secret_names():
    for name in (
        "--api-key",
        "API_KEY",
        "TENSORFOLD_API_KEY",
        "--hf-token",
        "x-api-key",
        "apiKey",
        "HF_TOKEN",
        "GHCR_TOKEN",
        "GITHUB_TOKEN",
        "MM3_KEY",
        "client_secret",
        "--password",
        "Authorization",
        "--token",
    ):
        assert secret_name(name), name
    for name in (
        "--max-tokens",
        "--tokens",
        "--tokenizer",
        "TF_MINIMAX_POOL_TOKENS",
        "--model",
        "--api-key-file",
        "API_KEY_FILE",
        "HF_TOKEN_FILE",
        "TOKEN_PATH",
        "--keyword",
        "monkey",
        "--decode-tokens",
    ):
        assert not secret_name(name), name


def test_argv_both_flag_forms_env_assignments_and_headers():
    argv = [
        "bench",
        "--api-key",
        KEY,
        "--api-key=" + KEY,
        "--model",
        "m",
        "--max-tokens",
        "5",
        "API_KEY=" + KEY,
        "-H",
        "Authorization: Bearer " + KEY,
        "--header=x-api-key: " + KEY,
        "--api-key-file",
        "/run/key",
        "--url",
        "https://user:" + KEY + "@host/v1",
        "--hf-token",
        "--quiet",
    ]
    out = redact_argv(argv)
    assert KEY not in " ".join(out)
    assert out[:5] == ["bench", "--api-key", MASK, "--api-key=" + MASK, "--model"]
    assert out[5:9] == ["m", "--max-tokens", "5", "API_KEY=" + MASK]
    # a pointer at a secret stays readable; a following flag is not taken as the value
    assert "--api-key-file" in out and "/run/key" in out
    assert out[-2:] == ["--hf-token", "--quiet"]


def test_recipe_serve_command_keeps_the_key_file_and_masks_a_literal_key():
    # start.sh's rank 0 command with API_KEY=1, plus a literal key someone passed after it
    argv = [
        "docker",
        "run",
        "-v",
        "/home/user/.cache/tensorfold-minimax-m3/api-key:/run/secrets/tensorfold-api-key:ro",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_TOKEN=" + HF,
        "tensorfold-native",
        "serve",
        "--api-key-file",
        "/run/secrets/tensorfold-api-key",
        "--max-tokens",
        "32768",
        "--api-key",
        SK,
    ]
    out = redact_argv(argv)
    assert SK not in " ".join(out) and HF not in " ".join(out)
    # the mount's host side (the key file's path) stays readable; its "api-key:..." tail reads as a key assignment
    assert out[:3] == argv[:3] and out[3].startswith(argv[3].split(":")[0] + ":")
    assert out[4:6] == argv[4:6]
    assert out[7] == "HF_TOKEN=" + MASK
    assert out[8:14] == argv[8:14]
    assert out[-2:] == ["--api-key", MASK]


def test_env_and_json():
    env = redact_env(
        {
            "API_KEY": KEY,
            "API_KEY_FILE": "/home/user/.cache/tensorfold-minimax-m3/api-key",
            "HF_TOKEN": HF,
            "HF_TOKEN_FILE": "/home/user/.cache/huggingface/token",
            "TF_MINIMAX_POOL_TOKENS": "65536",
            "PATH": "/bin",
            "NOTE": "key is " + KEY,
        },
        known=[KEY],
    )
    assert env == {
        "API_KEY": MASK,
        "API_KEY_FILE": "/home/user/.cache/tensorfold-minimax-m3/api-key",
        "HF_TOKEN": MASK,
        "HF_TOKEN_FILE": "/home/user/.cache/huggingface/token",
        "TF_MINIMAX_POOL_TOKENS": "65536",
        "PATH": "/bin",
        "NOTE": "key is " + MASK,
    }
    doc = redact(
        {
            "argv": ["--api-key", KEY],
            "api_key": KEY,
            "api_key_file": "/run/secrets/tensorfold-api-key",
            "max_tokens": 5,
            "token": "Ġthe",
            "tokens": [1, 2],
            "nested": [{"headers": {"Authorization": "Bearer " + KEY}}],
        }
    )
    assert KEY not in repr(doc)
    assert doc["max_tokens"] == 5 and doc["token"] == "Ġthe" and doc["tokens"] == [1, 2]
    assert doc["api_key_file"] == "/run/secrets/tensorfold-api-key"


def test_text_masks_credentials_keeps_placeholders():
    fake_sk = "sk-" + "a1" * 12
    text = (
        f'curl -H "x-api-key: {KEY}" --api-key={KEY} "api_key": "{KEY}" {fake_sk} HF_TOKEN={KEY}\n'
        'bos_token="<s>" --api-key "$API_KEY" password: <you> max_tokens=128 token count 42\n'
    )
    out = redact_text(text)
    assert KEY not in out and fake_sk not in out
    assert (
        'bos_token="<s>"' in out
        and '"$API_KEY"' in out
        and "<you>" in out
        and "max_tokens=128" in out
    )
    assert "token count 42" in out


def test_text_recipe_lines():
    # the key-file shapes (one key, or "label: key"), the curl header file, a log line and a pasted command
    lines = {
        SK: MASK,
        "alice: " + SK: "alice: " + MASK,
        "Authorization: Bearer " + SK: "Authorization: " + MASK,
        "export HF_TOKEN=" + HF: "export HF_TOKEN=" + MASK,
        "tensorfold-native serve /model --api-key " + SK: (
            "tensorfold-native serve /model --api-key " + MASK
        ),
    }
    for line, want in lines.items():
        assert redact_text(line) == want, line
    # pointers at a key stay readable
    kept = (
        "--api-key-file /run/secrets/tensorfold-api-key API_KEY_FILE=/state/api-key"
        " HF_TOKEN_FILE=/hf/token\n"
    )
    assert redact_text(kept) == kept


def test_known_values_and_scrub(tmp_path):
    # API_KEY=1 is start.sh's on / off switch: too short to mask by literal match
    known = known_secrets(
        {"API_KEY": "1", "MM3_KEY": KEY, "SHORT_TOKEN": "1", "MODEL": "x" * 20}
    )
    assert known == [KEY]
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "report.json").write_text(
        '{"echo": "the tool said ' + KEY + '"}'
    )
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe" + KEY.encode())
    assert scrub(tmp_path, known) == 1
    assert KEY not in (tmp_path / "sub" / "report.json").read_text()


def test_cli_argv_and_text_filter():
    env = {"API_KEY": KEY, "PATH": "/usr/bin:/bin"}
    cmd = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "redact.py"),
            "argv",
            "--",
            "tool",
            "--api-key",
            KEY,
            "--api-key-file",
            "/run/key",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert cmd.strip() == f"tool --api-key '{MASK}' --api-key-file /run/key"
    pem = "-----BEGIN " + "PRIVATE KEY-----\nMIIBVQ\n-----END " + "PRIVATE KEY-----\n"
    log = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts" / "redact.py"), "text"],
        input="echo " + KEY + "\nAuthorization: Bearer " + SK + "\n" + pem,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert KEY not in log and SK not in log and "MIIBVQ" not in log


def test_cli_scrub(tmp_path):
    (tmp_path / "r.json").write_text('{"cmd": "serve --api-key ' + SK + '"}')
    r = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts" / "redact.py"), "scrub", tmp_path],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "masked credentials in 1 file(s)" in r.stderr
    assert SK not in (tmp_path / "r.json").read_text()


def test_mask_values_only_masks_given_values():
    text = '{"reply": "password: hunter22", "echo": "' + KEY + '"}'
    assert (
        mask_values(text, [KEY, ""])
        == '{"reply": "password: hunter22", "echo": "' + MASK + '"}'
    )
