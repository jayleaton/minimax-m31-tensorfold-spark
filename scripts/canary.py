#!/usr/bin/env python3
"""Post-start canary for the MiniMax-M3.1 server: a few short fixed requests that fail the start if the engine, the
chat template or the reply parser is broken. Standard library only, so the head runs it outside the image.

A load can finish, answer ``/v1/models`` and still be broken: a rank on a stale kernel cache or mismatched weights
emits word salad or one repeated token, a template mismatch leaves ``<mm:think>`` or the tool-call namespace token in
the reply, a parser miss returns a call as text. Probes (greedy, a per-run nonce so a prompt-cache hit cannot answer
them from a stored state):

- ``chat``: thinking off, "capital of France": ``Paris``, not degenerate.
- ``thinking``: thinking on (effort low), 17 + 25: ``42`` in ``content`` and the reasoning separated into
  ``reasoning_content``, with no think tag left in either.
- ``tool``: a ``get_weather`` tool and a prompt that asks for the call (``--tool-choice auto``: the engine refuses
  "required" for MiniMax's markup today): one call parsed into ``tool_calls``, arguments a JSON object with the city.
- ``json``: ``response_format`` json_schema {ok: boolean}; when the server refuses response_format (HTTP 400 / 422 /
  501), the same object asked for by the prompt alone (reported as "json prompted").

With ``ENGINE_API_TOOLS=1`` (``--engine-api-tools``: the engine includes the branch mm3/api-tools) a refusal of
``response_format`` fails the ``json`` probe instead, and two probes follow:

- ``required``: the ``tool`` probe with ``tool_choice`` "required": the call validates against the tool's schema and,
  thinking off, no reasoning comes back.
- ``nothink``: thinking off, "largest planet": ``Jupiter``, no ``reasoning_content``, no ``</mm:think>`` in content.

With 0 (the default) the canary sends neither (bench/engine_api.py has the switch and the refusal texts).

Every reply is checked for raw MiniMax markup in ``content`` (the namespace token ``]<]minimax[>[``, ``<invoke
name=``, ``<tool_call>``, ``<mm:think>``, ``</mm:think>``): any is a failure.

    scripts/canary.py --base http://127.0.0.1:8000 --model MiniMax-M3.1-Flash-EXL3 [--api-key-file F] [--json]

Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE / ENGINE_API_TOOLS (scripts/config.sh exports them). The key is read from the
file, sent as a header and never printed. Exit 0: pass, one ``[canary] ok: ...`` line on stdout. Exit 1: a probe
failed (details on stderr). Exit 2: the server could not be reached or refused the key.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# bench/engine_api.py: the ENGINE_API_TOOLS switch and the engine's refusal texts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import engine_api as ea

LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ON = {
    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    "reasoning_effort": "low",
}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
OK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class Unreachable(Exception):
    """No server, or it refused the key."""


def default_base() -> str:
    """http://HOST:PORT from scripts/config.sh's settings (a wildcard HOST is reached on 127.0.0.1)."""

    host = os.environ.get("HOST") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{host}:{os.environ.get('PORT') or 8000}"


def read_key(path: str | None) -> str:
    """The first key of an API key file (one key, or "label: key", a line); "" when there is none to read.

    ``path`` None: API_KEY_FILE from the environment, used when it exists (required when API_KEY=1)."""

    strict = path is not None or os.environ.get("API_KEY") == "1"
    path = path if path is not None else os.environ.get("API_KEY_FILE", "")
    if not path or (not strict and not os.path.exists(path)):
        return ""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise SystemExit(
            f"cannot read the API key file {path}: {exc.strerror}"
        ) from None
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(":", 1)[1].strip() if ":" in line else line
    raise SystemExit(f"[canary] {path}: no key in it")


class Api:
    """JSON over HTTP with the key as a header; ``mask`` hides the key in any text that is printed."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def post(self, body: dict, timeout: float) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", json.dumps(body).encode(), headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(
                    self.mask(r.read().decode())
                )  # a reply never holds the key
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise Unreachable(
                    f"HTTP {exc.code}: the server refused the key"
                ) from None
            if refused := ea.refusal(
                exc.code, self.mask(exc.read().decode(errors="replace"))
            ):
                raise refused from None
            raise
        except urllib.error.URLError as exc:
            raise Unreachable(str(exc.reason)) from None

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def degenerate(text: str, *, min_distinct: float = 0.3, max_run: int = 6) -> str | None:
    """Why ``text`` looks like a broken engine's output, or None. Short answers pass on content alone."""

    if not text.strip():
        return "empty reply"
    if "�" in text:
        return "replacement character (U+FFFD) in the reply"
    words = re.findall(r"\S+", text)
    if len(words) >= 12 and len(set(words)) / len(words) < min_distinct:
        return f"repetitive reply ({len(set(words)) / len(words):.2f} distinct words)"
    run = best = 1
    for a, b in itertools.pairwise(words):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best > max_run:
        return f"the same word {best} times in a row"
    if re.search(r"(.)\1{23,}", text):
        return "one character repeated 24+ times"
    return None


def message(r: dict) -> tuple[dict, str]:
    ch = (r.get("choices") or [{}])[0]
    return ch.get("message") or {}, ch.get("finish_reason") or ""


def leaks(msg: dict) -> list[str]:
    out = []
    if LEAK.search(msg.get("content") or ""):
        out.append(
            f"raw MiniMax markup in content: {(msg.get('content') or '')[:120]!r}"
        )
    if LEAK.search(msg.get("reasoning_content") or ""):
        out.append("a think tag left inside reasoning_content")
    if msg.get("reasoning") and not msg.get("reasoning_content"):
        out.append("reasoning returned in 'reasoning', not 'reasoning_content'")
    return out


def json_object(text: str) -> Any:
    """The JSON value of a reply, a ```json fence around it allowed."""

    t = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    return json.loads(m.group(1) if m else t)


def probe_chat(
    api: Api, model: str, nonce: str, timeout: float
) -> tuple[list[str], dict]:
    r = api.post(
        dict(
            OFF,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"[canary {nonce}] What is the capital of France? One word.",
                }
            ],
            temperature=0,
            max_tokens=32,
        ),
        timeout,
    )
    msg, _ = message(r)
    content = msg.get("content") or ""
    problems = leaks(msg)
    if why := degenerate(content):
        problems.append(why)
    if not re.search(r"\bParis\b", content):
        problems.append(f"expected Paris: {content[:120]!r}")
    return problems, r


def probe_thinking(
    api: Api, model: str, nonce: str, timeout: float
) -> tuple[list[str], dict]:
    ask = f"[canary {nonce}] What is 17 + 25? Think it through, then answer with the number."
    r = api.post(
        dict(
            ON,
            model=model,
            messages=[{"role": "user", "content": ask}],
            temperature=0,
            max_tokens=1024,
        ),
        timeout,
    )
    msg, finish = message(r)
    problems = leaks(msg)
    if not re.search(r"\b42\b", msg.get("content") or ""):
        problems.append(
            f"expected 42 in content (finish {finish}): {(msg.get('content') or '')[:120]!r}"
        )
    if not (msg.get("reasoning_content") or "").strip():
        problems.append(
            "no reasoning_content (thinking on: the reasoning must come back separated)"
        )
    return problems, r


def probe_tool(
    api: Api, model: str, nonce: str, timeout: float, tool_choice: str
) -> tuple[list[str], dict]:
    ask = f"[canary {nonce}] What's the weather in Paris right now? Call the get_weather tool; do not answer yourself."
    body = dict(
        OFF, model=model, messages=[{"role": "user", "content": ask}], tools=[WEATHER]
    )
    r = api.post(
        dict(body, tool_choice=tool_choice, temperature=0, max_tokens=512), timeout
    )
    msg, finish = message(r)
    problems = leaks(msg)
    calls = msg.get("tool_calls") or []
    if not calls:
        problems.append(
            f"no tool_calls (finish {finish}): {(msg.get('content') or '')[:120]!r}"
        )
        return problems, r
    fn = calls[0].get("function") or {}
    if fn.get("name") != "get_weather":
        problems.append(f"called {fn.get('name')!r}, not get_weather")
    try:
        args = json.loads(fn.get("arguments") or "")
        if not isinstance(args, dict):
            raise TypeError("not an object")
    except (ValueError, TypeError) as exc:
        problems.append(
            f"arguments not a JSON object ({exc}): {str(fn.get('arguments'))[:120]!r}"
        )
        return problems, r
    city = args.get("city")
    if not isinstance(city, str) or "paris" not in city.lower():
        problems.append(f"city {city!r}, not Paris")
    if any(LEAK.search(v) for v in args.values() if isinstance(v, str)):
        problems.append("MiniMax markup inside an argument")
    return problems, r


def probe_json(
    api: Api, model: str, nonce: str, timeout: float, api_tools: bool = False
) -> tuple[list[str], dict, bool]:
    """(problems, reply, prompted): prompted when the server refused response_format (not with ``api_tools``)."""

    ask = f"[canary {nonce}] Give a JSON object with the key ok set to true."
    body = dict(OFF, model=model, temperature=0, max_tokens=64)
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "canary", "schema": OK_SCHEMA},
    }
    prompted = False
    try:
        r = api.post(
            dict(
                body, messages=[{"role": "user", "content": ask}], response_format=fmt
            ),
            timeout,
        )
    except (urllib.error.HTTPError, ea.Refused) as exc:
        if api_tools and isinstance(exc, ea.Refused):
            raise
        if isinstance(exc, urllib.error.HTTPError) and exc.code not in (400, 422, 501):
            raise
        prompted = True
        ask += ' Reply with only the JSON object {"ok": true}, nothing else.'
        r = api.post(dict(body, messages=[{"role": "user", "content": ask}]), timeout)
    msg, _ = message(r)
    problems = leaks(msg)
    try:
        if json_object(msg.get("content") or "")["ok"] not in (True, False):
            problems.append("ok is not a boolean")
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(
            f"not a JSON object with ok ({type(exc).__name__}): {(msg.get('content') or '')[:120]!r}"
        )
    return problems, r, prompted


def probe_required(
    api: Api, model: str, nonce: str, timeout: float
) -> tuple[list[str], dict]:
    """The tool probe with tool_choice "required" (mm3/api-tools): a schema-valid call, no reasoning with thinking off."""

    problems, r = probe_tool(api, model, nonce, timeout, "required")
    msg, _ = message(r)
    for c in msg.get("tool_calls") or []:
        try:
            if e := ea.validate(
                json.loads((c.get("function") or {}).get("arguments") or ""),
                WEATHER["function"]["parameters"],
            ):
                problems.append(f"arguments break the schema: {e}")
        except ValueError:
            pass  # probe_tool reported it
    if (msg.get("reasoning_content") or "").strip():
        problems.append("thinking off, yet reasoning_content came back")
    return problems, r


def probe_nothink(
    api: Api, model: str, nonce: str, timeout: float
) -> tuple[list[str], dict]:
    """Thinking off (mm3/api-tools: the prompt ends in the think close): an answer, no reasoning, no lone close."""

    ask = f"[canary {nonce}] Which planet is the largest planet in the solar system? One word."
    r = api.post(
        dict(
            OFF,
            model=model,
            messages=[{"role": "user", "content": ask}],
            temperature=0,
            max_tokens=64,
        ),
        timeout,
    )
    msg, finish = message(r)
    content = msg.get("content") or ""
    problems = leaks(msg)
    if (msg.get("reasoning_content") or "").strip():
        problems.append("thinking off, yet reasoning_content came back")
    if why := degenerate(content):
        problems.append(why)
    if not re.search(r"\bJupiter\b", content):
        problems.append(f"expected Jupiter (finish {finish}): {content[:120]!r}")
    return problems, r


def check(
    api: Api,
    model: str,
    timeout: float = 300.0,
    tool_choice: str = "auto",
    api_tools: bool = False,
) -> dict:
    """Run every probe; the verdict and the per-probe details (JSON-able). ``api_tools``: the engine includes
    mm3/api-tools (ENGINE_API_TOOLS=1): its probes run and a refused feature fails."""

    nonce = f"{os.getpid()}-{int(time.time())}"
    probes: list[dict] = []
    prompted = False
    names = ("chat", "thinking", "tool", "json") + (
        ("required", "nothink") if api_tools else ()
    )
    for name in names:
        t0 = time.perf_counter()
        try:
            if name == "chat":
                problems, r = probe_chat(api, model, nonce, timeout)
            elif name == "thinking":
                problems, r = probe_thinking(api, model, nonce, timeout)
            elif name == "tool":
                problems, r = probe_tool(api, model, nonce, timeout, tool_choice)
            elif name == "json":
                problems, r, prompted = probe_json(
                    api, model, nonce, timeout, api_tools
                )
            elif name == "required":
                problems, r = probe_required(api, model, nonce, timeout)
            else:
                problems, r = probe_nothink(api, model, nonce, timeout)
        except Unreachable:
            raise
        except ea.Refused as exc:
            problems, r = (
                [
                    f"refused, the engine lacks mm3/api-tools (ENGINE_API_TOOLS=1): {exc.message}"
                    if api_tools
                    else f"request failed: refused: {exc.message}"
                ],
                {},
            )
        except Exception as exc:  # noqa: BLE001 - a failed request fails the probe
            problems, r = [f"request failed: {type(exc).__name__}: {exc}"[:300]], {}
        msg, _ = message(r) if r else ({}, "")
        stats = r.get("tensorfold") or {}
        probes.append(
            {
                "name": name,
                "ok": not problems,
                "problems": problems,
                "seconds": round(time.perf_counter() - t0, 2),
                "completion_tokens": int(
                    (r.get("usage") or {}).get("completion_tokens") or 0
                ),
                "decode_s": float(stats.get("decode_s") or 0.0),
                "content": (msg.get("content") or "")[:200],
            }
        )
    tokens = sum(
        max(p["completion_tokens"] - 1, 0) for p in probes if p["decode_s"] > 0
    )
    decode_s = sum(p["decode_s"] for p in probes)
    return {
        "ok": all(p["ok"] for p in probes),
        "problems": [f"{p['name']}: {x}" for p in probes for x in p["problems"]],
        "json_prompted": prompted,
        "engine_api_tools": api_tools,
        "decode_tps": tokens / decode_s if decode_s > 0 else None,
        "probes": probes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", default=default_base())
    ap.add_argument(
        "--model", default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3"
    )
    ap.add_argument(
        "--api-key-file",
        default=None,
        help="the server's key file (default: $API_KEY_FILE if present)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds a probe (the first may compile kernels)",
    )
    ap.add_argument("--tool-choice", choices=("auto", "required"), default="auto")
    ap.add_argument(
        "--json", action="store_true", help="print the full report as JSON on stdout"
    )
    ea.add_flag(ap)
    a = ap.parse_args(argv)
    api_tools = ea.enabled(a.engine_api_tools)
    api = Api(a.base, read_key(a.api_key_file))
    try:
        report = check(api, a.model, a.timeout, a.tool_choice, api_tools)
    except Unreachable as exc:
        print(api.mask(f"[canary] cannot query {api.base}: {exc}"), file=sys.stderr)
        return 2
    if a.json:
        print(api.mask(json.dumps(report, indent=1)))
    if report["ok"]:
        names = ", ".join(
            p["name"]
            + (" prompted" if p["name"] == "json" and report["json_prompted"] else "")
            for p in report["probes"]
        )
        tps = report["decode_tps"]
        print(
            f"[canary] ok: {len(report['probes'])} probes ({names})"
            + (f", decode {tps:.1f} tok/s" if tps else "")
        )
        return 0
    for p in report["problems"]:
        print(api.mask(f"[canary] FAIL {p}"), file=sys.stderr)
    for p in report["probes"]:
        print(
            api.mask(f"[canary]   {p['name']}: {p['content'][:160]!r}"), file=sys.stderr
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
