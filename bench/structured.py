#!/usr/bin/env python3
"""Structured output, tool-call shapes and thinking controls against a running MiniMax-M3.1 server, standard library only.

A reply's identity is its ``tensorfold.token_ids`` when the server returns them, else its text.

Suites (``--suites``, comma list; all by default):

- ``schemas``: 6 JSON schemas (nested objects, arrays, enums, bounds, numbers) x thinking off / on (effort low), greedy:
  each sent as is, with ``"draft": false`` (serial), and the 6 of a setting 4 at a time concurrently; plus a
  ``json_object`` reply in each setting. Gates: the three replies are identical; every reply that finished (``stop``)
  parses and validates; with thinking on the reasoning stays in ``reasoning_content`` and never leaks into the JSON; with
  thinking off no reasoning comes back; no MiniMax markup in the content.
- ``tools``: ``tool_choice`` ``required`` (at least one call, each an offered tool), a named function (only that one),
  a named function with ``parallel_tool_calls`` false (exactly one call), a ``strict`` tool under ``auto``,
  ``parallel_tool_calls`` false, two independent calls in one turn, ``tool_choice`` ``none`` (no call), a streamed
  ``auto`` call and a streamed ``required`` call (the fragments concatenate to the arguments of the same request
  answered whole): every call names a known tool, its arguments parse and validate against the schema (types
  included: the parser types MiniMax's text values by the schema), no markup leaks, no reasoning with thinking off.
- ``thinking``: thinking off (no ``reasoning_content``, an answer, no ``</mm:think>`` in it); ``thinking_budget`` 64 and
  256 on a prompt that asks for long reasoning (the reasoning tokens, ``usage.completion_tokens_details`` or a rough
  count, stay within the budget and a little slack, and the answer arrives after the close); ``reasoning_effort`` at
  low, medium, high, xhigh, max and the API's ``minimal`` (each accepted; the level the server reports running, when it
  does, is the one asked for, or the nearest: low).

Engine features (required and named ``tool_choice``, ``response_format``, thinking off, ``thinking_budget``, the effort
levels) arrive with the engine branch mm3/api-tools. With ``ENGINE_API_TOOLS`` 0 (the default; ``--engine-api-tools``
sets it) a refusal of one, or the old engine's known behaviour (a thinking-off reply that still thinks, high / max run
as xhigh), is ``SKIP (engine lacks mm3/api-tools)``; with 1 it fails. An accepted feature must behave either way. Any
other refused feature (``"draft": false``) is UNSUPPORTED, not FAIL; ``--strict`` fails both kinds.

  python3 bench/structured.py --base http://127.0.0.1:8000 --model MiniMax-M3.1-Flash-EXL3 --out R/structured.json
Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE / ENGINE_API_TOOLS. Exit status 0 when no gate fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# engine_api beside this file (python3 -I leaves the script's directory off sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_api as ea

LEAK = ea.LEAK
validate = ea.validate
MASK = "<redacted>"
UNSUPPORTED = re.compile(
    r"tool_choice|response_format|json_schema|grammar|structur|unsupported|not supported|cannot be enforced",
    re.IGNORECASE,
)
SCHEMAS: list[tuple[str, dict]] = [
    (
        "person",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 130},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        },
    ),
    (
        "order",
        {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer", "minimum": 1},
                        },
                        "required": ["sku", "qty"],
                    },
                },
                "total": {"type": "number"},
            },
            "required": ["id", "items", "total"],
        },
    ),
    (
        "status",
        {
            "type": "object",
            "properties": {
                "state": {"enum": ["ok", "degraded", "down"]},
                "uptime_s": {"type": "integer", "minimum": 0},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["state", "uptime_s", "notes"],
        },
    ),
    (
        "city",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"},
                "coords": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                },
            },
            "required": ["city", "country", "coords"],
        },
    ),
    (
        "flags",
        {
            "type": "object",
            "properties": {
                "debug": {"type": "boolean"},
                "level": {"enum": [1, 2, 3]},
                "tags": {
                    "type": "array",
                    "items": {"enum": ["a", "b", "c"]},
                    "maxItems": 3,
                },
            },
            "required": ["debug", "level", "tags"],
        },
    ),
    (
        "list",
        {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "square": {"type": "integer"}},
                "required": ["n", "square"],
            },
        },
    ),
]
PROMPTS = {
    "person": "Invent a person.",
    "order": "Invent a small web-shop order.",
    "status": "Report a service status.",
    "city": "Describe Lyon.",
    "flags": "Pick feature flags for a test build.",
    "list": "The numbers 1, 2, 3 with their squares.",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Find flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


class Unsupported(Exception):
    """The server refused a request as unsupported (its message)."""


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
    raise SystemExit(f"{path}: no key in it")


class Api:
    """JSON over HTTP with the key as a header; ``mask`` hides the key in any text that is printed or saved.
    A refusal of an engine feature raises ``engine_api.Refused``; another that names an unsupported feature,
    ``Unsupported``."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def open(self, body: dict, timeout: float):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", json.dumps(body).encode(), headers
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            text = self.mask(exc.read().decode(errors="replace"))
            if refused := ea.refusal(exc.code, text):
                raise refused from None
            if exc.code in (400, 422, 501) and UNSUPPORTED.search(text):
                raise Unsupported(ea.message(text)) from None
            raise RuntimeError(f"HTTP {exc.code}: {text[:300]}") from None

    def post(self, body: dict, timeout: float = 900.0) -> dict:
        with self.open(body, timeout) as r:
            return json.loads(
                self.mask(r.read().decode())
            )  # a reply never holds the key

    def stream(self, body: dict, timeout: float = 900.0) -> tuple[dict, str]:
        """A streamed request: (the assembled tool calls by index, the streamed content)."""

        calls: dict[int, dict] = {}
        content = ""
        with self.open(dict(body, stream=True), timeout) as r:
            for raw in r:
                line = self.mask(raw.decode()).strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                c = json.loads(line[5:])
                for ch in c.get("choices") or []:
                    delta = ch.get("delta") or {}
                    content += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        d = calls.setdefault(
                            tc.get("index", 0), {"name": "", "arguments": ""}
                        )
                        f = tc.get("function") or {}
                        d["name"] += f.get("name") or ""
                        d["arguments"] += f.get("arguments") or ""
        return calls, content

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def ident(r: dict) -> str:
    tf = r.get("tensorfold") or {}
    ids = tf.get("token_ids")
    return (
        json.dumps(ids)
        if ids
        else json.dumps(r["choices"][0]["message"].get("content"))
    )


class Suite(ea.Ledger):
    """A suite's verdicts: ``fails``, engine-feature ``skipped`` and other ``unsupported`` features."""

    def __init__(self, on: bool) -> None:
        super().__init__(on)
        self.unsupported: list[str] = []


def body_for(
    a,
    prompt: str,
    *,
    schema: tuple[str, dict] | None = None,
    fmt: dict | None = None,
    thinking: bool = False,
    effort: str = "low",
    max_tokens: int = 1024,
) -> dict:
    b = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if thinking:
        b["chat_template_kwargs"]["reasoning_effort"] = effort
        b["reasoning_effort"] = effort
    if schema:
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": schema[0], "schema": schema[1], "strict": True},
        }
    if fmt:
        b["response_format"] = fmt
    return b


def check_off(tag: str, msg: dict, led: Suite) -> None:
    """Thinking off: no reasoning comes back (the family as imported leaves it to its adaptive template)."""

    if (msg.get("reasoning_content") or "").strip():
        led.lacks(tag, "thinking off, yet reasoning_content came back")


def check_json(tag: str, r: dict, schema: dict, thinking: bool, led: Suite) -> None:
    """A response_format reply: no markup, the reasoning where it belongs, finished, JSON valid against ``schema``."""

    msg, fin = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
    content = msg.get("content") or ""
    if m := ea.leaks(content):
        led.fail(tag, f"markup in content ({m}): {content[:120]!r}")
    if ea.leaks(msg.get("reasoning_content")):
        led.fail(tag, "think tags left inside reasoning_content")
    if not thinking:
        check_off(tag, msg, led)
    if fin != "stop":
        led.fail(
            tag, f"finish {fin} (max_tokens too small or the grammar never closed)"
        )
        return
    try:
        e = validate(json.loads(content), schema)
    except ValueError as exc:
        e = f"not JSON ({exc}): {content[:120]!r}"
    if e:
        led.fail(tag, e)


def suite_schemas(a, api: Api, led: Suite) -> dict:
    rec = {}
    for thinking in (False, True):
        setting = "think" if thinking else "chat"
        max_tokens = 2048 if thinking else 768
        bodies = [
            body_for(
                a, PROMPTS[n], schema=(n, s), thinking=thinking, max_tokens=max_tokens
            )
            for n, s in SCHEMAS
        ]
        first = None
        try:
            first = [api.post(b) for b in bodies]
        except ea.Refused as exc:
            led.lacks(f"response_format json_schema ({setting})", exc.message)
        except Unsupported as exc:
            led.unsupported.append(f"response_format json_schema ({setting}): {exc}")
        if first:
            try:
                serial = [api.post(dict(b, draft=False)) for b in bodies]
            except Unsupported as exc:
                led.unsupported.append(f'"draft": false ({setting}): {exc}')
                serial = first
            with cf.ThreadPoolExecutor(4) as ex:
                batched = list(ex.map(api.post, bodies))
            for (n, s), x, y, z in zip(SCHEMAS, first, serial, batched):
                tag = f"{n}/{setting}"
                msg = x["choices"][0]["message"]
                rec[tag] = {
                    "finish": x["choices"][0]["finish_reason"],
                    "tokens": x.get("usage", {}).get("completion_tokens"),
                    "same": ident(x) == ident(y) == ident(z),
                }
                if not rec[tag]["same"]:
                    led.fail(tag, "single / serial / batched replies differ")
                if thinking and not msg.get("reasoning_content"):
                    rec[tag]["note"] = "no reasoning_content returned"
                check_json(tag, x, s, thinking, led)
        # json_object: any JSON object, no schema
        tag = f"json_object/{setting}"
        b = body_for(
            a,
            "Describe Lyon as a JSON object with the keys city and country.",
            fmt={"type": "json_object"},
            thinking=thinking,
            max_tokens=max_tokens,
        )
        try:
            r = api.post(b)
        except ea.Refused as exc:
            led.lacks(tag, exc.message)
            continue
        except Unsupported as exc:
            led.unsupported.append(f"{tag}: {exc}")
            continue
        rec[tag] = {"finish": r["choices"][0]["finish_reason"]}
        check_json(tag, r, {"type": "object"}, thinking, led)
    return rec


def check_calls(tag: str, calls: list[dict], want: str | None, led: Suite) -> None:
    if not calls:
        led.fail(tag, "no tool call")
        return
    for c in calls:
        name, raw = c["function"]["name"], c["function"]["arguments"]
        if name not in TOOL_BY_NAME or (want and name != want):
            led.fail(tag, f"call {name!r} (want {want or 'a known tool'})")
            continue
        if ea.leaks(raw):
            led.fail(tag, f"markup in arguments {raw[:120]!r}")
        try:
            e = validate(json.loads(raw), TOOL_BY_NAME[name])
        except ValueError as exc:
            e = f"arguments not JSON ({exc}): {raw[:120]!r}"
        if e:
            led.fail(tag, e)


def suite_tools(a, api: Api, led: Suite) -> dict:
    rec: dict = {}
    ask = "Check the weather in Oslo for 3 days, then find a flight from Oslo to Rome on 2026-11-02."
    email = "Email bob@example.com that the build is green. Use the send_email tool."
    both = (
        "Make both tool calls now, in this one turn: the weather in Oslo for 3 days, and flights from Oslo to Rome "
        "on 2026-11-02."
    )
    flights = {"type": "function", "function": {"name": "search_flights"}}
    cases = [
        ("required", {"tool_choice": "required"}, None, ask),
        ("named", {"tool_choice": flights}, "search_flights", ask),
        (
            "named-single",
            {"tool_choice": flights, "parallel_tool_calls": False},
            "search_flights",
            ask,
        ),
        ("strict-auto", {"tool_choice": "auto"}, "send_email", email),
        (
            "single",
            {"tool_choice": "auto", "parallel_tool_calls": False},
            None,
            ask + " Use the tools.",
        ),
        ("parallel", {"tool_choice": "auto"}, None, both),
    ]
    # the calls of a reply, for the streamed one of the same request
    whole: dict[str, list[dict]] = {}
    for thinking in (False, True):
        for name, extra, want, prompt in cases:
            tag = f"{name}/{'think' if thinking else 'chat'}"
            b = dict(
                body_for(a, prompt, thinking=thinking, max_tokens=2048),
                tools=TOOLS,
                **extra,
            )
            try:
                r = api.post(b)
            except ea.Refused as exc:
                led.lacks(tag, exc.message)
                continue
            except Unsupported as exc:
                led.unsupported.append(f"{tag}: {exc}")
                continue
            ch = r["choices"][0]
            calls = ch["message"].get("tool_calls") or []
            content = ch["message"].get("content") or ""
            whole[tag] = calls
            rec[tag] = {
                "finish": ch["finish_reason"],
                "calls": [c["function"]["name"] for c in calls],
            }
            check_calls(tag, calls, want, led)
            if m := ea.leaks(content):
                led.fail(tag, f"markup in content ({m}): {content[:120]!r}")
            if not thinking:
                check_off(tag, ch["message"], led)
            if calls and ch["finish_reason"] != "tool_calls":
                led.fail(tag, f"finish {ch['finish_reason']} with tool calls")
            if extra.get("parallel_tool_calls") is False and len(calls) > 1:
                led.fail(tag, f"{len(calls)} calls with parallel_tool_calls false")
            if name == "parallel" and len(calls) < 2:
                rec[tag]["note"] = (
                    f"{len(calls)} call(s) where two were asked for in one turn"
                )
    r = api.post(
        dict(body_for(a, ask, max_tokens=512), tools=TOOLS, tool_choice="none")
    )
    msg = r["choices"][0]["message"]
    rec["none/chat"] = {
        "finish": r["choices"][0]["finish_reason"],
        "calls": len(msg.get("tool_calls") or []),
    }
    if rec["none/chat"]["calls"]:
        led.fail("none/chat", "tool_choice none still called a tool")
    if ea.leaks(msg.get("content")):
        led.fail(
            "none/chat", f"markup in content: {(msg.get('content') or '')[:120]!r}"
        )
    calls, content = api.stream(
        dict(
            body_for(a, ask + " Use the tools.", max_tokens=1024),
            tools=TOOLS,
            tool_choice="auto",
        )
    )
    rec["stream/chat"] = {"calls": [c["name"] for c in calls.values()]}
    check_calls("stream/chat", [{"function": c} for c in calls.values()], None, led)
    if ea.leaks(content):
        led.fail("stream/chat", f"markup in streamed content: {content[:120]!r}")
    # the required case's request streamed: its argument fragments concatenate to the arguments answered whole
    tag = "stream-required/chat"
    try:
        calls, content = api.stream(
            dict(body_for(a, ask, max_tokens=2048), tools=TOOLS, tool_choice="required")
        )
    except ea.Refused as exc:
        led.lacks(tag, exc.message)
        return rec
    streamed = [calls[i] for i in sorted(calls)]
    rec[tag] = {"calls": [c["name"] for c in streamed]}
    check_calls(tag, [{"function": c} for c in streamed], None, led)
    if ea.leaks(content):
        led.fail(tag, f"markup in streamed content: {content[:120]!r}")
    if (want := whole.get("required/chat")) is not None:
        got = [(c["name"], c["arguments"]) for c in streamed]
        ref = [(c["function"]["name"], c["function"]["arguments"]) for c in want]
        rec[tag]["bytes_equal"] = got == ref
        if not same_calls(got, ref):
            led.fail(
                tag,
                f"streamed calls {got!r} differ from the whole reply's {ref!r}"[:300],
            )
    return rec


def same_calls(got: list[tuple[str, str]], ref: list[tuple[str, str]]) -> bool:
    """The same calls: names, and arguments equal as JSON (blanks aside)."""

    def parsed(raw: str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    return [(n, parsed(x)) for n, x in got] == [(n, parsed(x)) for n, x in ref]


HEAVY = (
    "A farmer has 15 sheep and three times as many goats, then sells 40% of all the animals. Reason at length, "
    "step by step, checking every step twice, then give the number of animals left."
)


def suite_thinking(a, api: Api, led: Suite) -> dict:
    rec: dict = {}
    # thinking off: no reasoning, an answer, no lone close in it
    tag = "off/plain"
    try:
        r = api.post(
            body_for(
                a,
                "What is the capital of Italy? Answer in one sentence.",
                max_tokens=256,
            )
        )
    except ea.Refused as exc:
        led.lacks(tag, exc.message)
    else:
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        rec[tag] = {"finish": r["choices"][0]["finish_reason"], "content": content[:80]}
        if not content.strip():
            led.fail(tag, "no content")
        if m := ea.leaks(content):
            led.fail(tag, f"markup in content ({m}): {content[:120]!r}")
        check_off(tag, msg, led)
    # thinking_budget: the reasoning stops at the budget and the answer follows the close
    for budget in (64, 256):
        tag = f"budget/{budget}"
        b = dict(
            body_for(a, HEAVY, thinking=True, effort="high", max_tokens=budget + 1024),
            thinking_budget=budget,
        )
        try:
            r = api.post(b)
        except ea.Refused as exc:
            led.lacks(tag, exc.message)
            continue
        msg, fin = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
        content = msg.get("content") or ""
        n, exact = ea.reasoning_tokens(r)
        limit = ea.budget_limit(budget, exact)
        rec[tag] = {
            "reasoning_tokens": n,
            "exact": exact,
            "limit": limit,
            "finish": fin,
        }
        if n > limit:
            led.fail(
                tag,
                f"{n} reasoning tokens{'' if exact else ' (estimated)'} over the budget's {limit}",
            )
        if not content.strip() or fin != "stop":
            led.fail(
                tag, f"no answer after the close (finish {fin}): {content[:120]!r}"
            )
        if m := ea.leaks(content):
            led.fail(tag, f"markup in content ({m}): {content[:120]!r}")
        if ea.leaks(msg.get("reasoning_content")):
            led.fail(tag, "think tags left inside reasoning_content")
    # reasoning_effort: the five levels, and an API name MiniMax has not (the nearest runs)
    for effort in (*ea.EFFORTS, "minimal"):
        tag = f"effort/{effort}"
        b = body_for(
            a,
            "Say hello in one short sentence.",
            thinking=True,
            effort=effort,
            max_tokens=512,
        )
        try:
            r = api.post(b)
        except ea.Refused as exc:
            led.lacks(tag, exc.message)
            continue
        msg = r["choices"][0]["message"]
        ran = (r.get("tensorfold") or {}).get("reasoning_effort")
        rec[tag] = {"ran": ran, "finish": r["choices"][0]["finish_reason"]}
        if ea.leaks(msg.get("content")):
            led.fail(tag, f"markup in content: {(msg.get('content') or '')[:120]!r}")
        if ran is None or ran == ea.run_effort(effort):
            continue
        if ran == ea.old_effort(effort):
            led.lacks(tag, f"ran as {ran}")
        else:
            led.fail(tag, f"ran as {ran}, want {ea.run_effort(effort)}")
    return rec


SUITES = {"schemas": suite_schemas, "tools": suite_tools, "thinking": suite_thinking}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default=default_base())
    ap.add_argument(
        "--model", default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3"
    )
    ap.add_argument(
        "--api-key-file",
        default=None,
        help="the server's key file (default: $API_KEY_FILE if present)",
    )
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument(
        "--strict",
        action="store_true",
        help="every refused feature fails its suite (engine features as with ENGINE_API_TOOLS=1)",
    )
    ea.add_flag(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    on = ea.enabled(a.engine_api_tools) or a.strict
    if bad := [s for s in a.suites.split(",") if s not in SUITES]:
        ap.error(f"--suites: no suite {', '.join(bad)} (have {', '.join(SUITES)})")
    api = Api(a.base, read_key(a.api_key_file))
    t0, out, bad = time.time(), {"engine_api_tools": on}, 0
    for s in a.suites.split(","):
        led = Suite(on)
        try:
            rec = SUITES[s](a, api, led)
        except Exception as exc:  # noqa: BLE001 - a crashed suite is a failed suite
            rec = {}
            led.fails.append(
                api.mask(f"suite crashed: {type(exc).__name__}: {exc}")[:300]
            )
        fails = led.fails + (
            [f"unsupported: {u}" for u in led.unsupported] if a.strict else []
        )
        if fails:
            status = "FAIL"
        elif not rec and led.skipped:
            status = "SKIP"
        elif not rec and led.unsupported:
            status = "UNSUPPORTED"
        else:
            status = "PASS"
        out[s] = {
            "status": status,
            "pass": not fails,
            "fails": fails,
            "skipped": led.skipped,
            "unsupported": led.unsupported,
            "cases": rec,
        }
        bad += bool(fails)
        lines = (
            [f"\n  - {f}" for f in fails[:20]]
            + [f"\n  - {k}" for k in led.skipped]
            + [f"\n  - unsupported: {u}" for u in led.unsupported]
        )
        print(
            api.mask(f"[structured] {s}: {status} ({len(rec)} cases)" + "".join(lines)),
            flush=True,
        )
    out["pass"], out["seconds"] = not bad, round(time.time() - t0, 1)
    if a.out:
        Path(a.out).write_text(api.mask(json.dumps(out, indent=1)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
