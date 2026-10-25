"""The engine-feature switch ``ENGINE_API_TOOLS`` and the engine's refusal texts, shared by the harnesses and the canary.

The features: ``tool_choice`` "required" or a named function, ``response_format`` (json_object, json_schema), thinking
off, ``thinking_budget`` and the five ``reasoning_effort`` levels. The family as imported (``patches/``) refuses the first
two and does not honour the rest; the engine branch mm3/api-tools serves them. ``ENGINE_API_TOOLS`` (environment, 0 by
default; ``--engine-api-tools`` on a command line) says which engine runs:

- 0: a refusal of one of these, or the old engine's known behaviour (a thinking-off reply that still thinks, high / max
  run as xhigh), is reported ``SKIP (engine lacks mm3/api-tools)`` and fails nothing;
- 1: the same is a failure.

In both modes a feature the engine accepts must behave correctly, or its case fails. Standard library only; a harness
imports it with its own directory on ``sys.path`` (``python3 -I`` leaves it off).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

SKIP = "SKIP (engine lacks mm3/api-tools)"
# raw MiniMax markup: the namespace token, an invoke, a tool-call block, a think tag
LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke\b|</?tool_call>|</?mm:think>")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The engine's refusal texts (HTTP 400 unless noted), by the feature they refuse. Sources: the family patch
# (patches/0001-minimax-m3-family.patch, MiaAI-Lab) and the mm3/api-tools branch, zig/src/server/.
REFUSALS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (feature, re.compile(pattern))
    for feature, pattern in (
        # tool_grammar.zig minimax_refusal (the family patch; removed by mm3/api-tools)
        ("tool_choice", r"cannot be enforced for MiniMax-M3\.1's tool markup"),
        # server.zig checkFeatures: an engine without grammars (Info.structures false, the family as imported)
        ("tool_choice", r"cannot enforce tool_choice \"required\" or a named function"),
        # server.zig checkFeatures, the same engine
        (
            "response_format",
            r"structured output \(response_format and the guided_\* fields\) is not supported",
        ),
        # minimax_m3/backend.zig putBits: a grammar on the first token (an engine error, any status)
        ("response_format", r"StructuresUnsupported"),
        # think_plan.zig no_first_mask (mm3/api-tools): a forced close it cannot place
        ("grammar", r"this engine cannot constrain a reply's first token"),
        # server.zig compiledStructure: xgrammar's reason (mm3/api-tools: a tool without parameters)
        ("grammar", r"the grammar cannot be enforced"),
        # think_plan.zig (chat.zig before mm3/api-tools)
        ("thinking_budget", r"thinking_budget is not supported by this native engine"),
        # fields.zig thinkingFields: a value outside the API's names
        (
            "reasoning_effort",
            r"reasoning_effort must be none, minimal, low, medium, high, xhigh or max",
        ),
    )
)


class Refused(Exception):
    """The engine refused an engine feature: ``feature`` and the server's message."""

    def __init__(self, feature: str, message: str) -> None:
        super().__init__(message)
        self.feature, self.message = feature, message


def enabled(value: str | None = None) -> bool:
    """The switch: ``value`` (a command line's) else ENGINE_API_TOOLS, "0" (the default) or "1"."""

    v = (os.environ.get("ENGINE_API_TOOLS") or "0") if value is None else value
    if v not in ("0", "1"):
        raise SystemExit(f"ENGINE_API_TOOLS is 0 or 1, not {v!r}")
    return v == "1"


def add_flag(ap: argparse.ArgumentParser) -> None:
    """``--engine-api-tools [0|1]`` (bare: 1); read the result with ``enabled(a.engine_api_tools)``."""

    ap.add_argument(
        "--engine-api-tools",
        nargs="?",
        const="1",
        default=None,
        choices=("0", "1"),
        help="1: the engine includes mm3/api-tools, so a refused engine feature fails (default: $ENGINE_API_TOOLS or 0)",
    )


def message(text: str, limit: int = 300) -> str:
    """An error reply's message (OpenAI's error.message), else its text; at most ``limit`` characters."""

    try:
        text = json.loads(text)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        pass
    return str(text)[:limit]


def refusal(status: int, text: str) -> Refused | None:
    """The engine-feature refusal an HTTP error is, or None (another error). Matched on the decoded message (JSON
    escapes the quotes around "required") and on the raw text (an engine error outside OpenAI's shape)."""

    if status < 400 or status in (401, 403, 404):
        return None
    decoded = message(text, 1 << 16)
    for feature, pattern in REFUSALS:
        if pattern.search(decoded) or pattern.search(text):
            return Refused(feature, message(text))
    return None


class Ledger:
    """One suite's verdicts: failures, engine-feature skips (``SKIP``) and other notes."""

    def __init__(self, on: bool) -> None:
        self.on = on
        self.fails: list[str] = []
        self.skipped: list[str] = []

    def fail(self, tag: str, why: str) -> None:
        self.fails.append(f"{tag}: {why}")

    def lacks(self, tag: str, why: str) -> None:
        """A refusal, or the old engine's known behaviour: a failure with the switch on, else a skip."""

        if self.on:
            self.fails.append(f"{tag}: {why} (ENGINE_API_TOOLS=1)")
        else:
            self.skipped.append(f"{tag}: {SKIP}: {why}")


def leaks(text: str | None) -> str | None:
    """The first raw MiniMax markup in ``text``, or None."""

    m = LEAK.search(text or "")
    return m.group(0) if m else None


def reasoning_tokens(reply: dict) -> tuple[int, bool]:
    """(reasoning tokens, exact): usage.completion_tokens_details.reasoning_tokens when the server reports it, else
    a rough count of reasoning_content (4 characters a token)."""

    details = (reply.get("usage") or {}).get("completion_tokens_details") or {}
    if isinstance(details.get("reasoning_tokens"), int):
        return details["reasoning_tokens"], True
    msg = ((reply.get("choices") or [{}])[0]).get("message") or {}
    return -(-len(msg.get("reasoning_content") or "") // 4), False


def budget_limit(budget: int, exact: bool) -> int:
    """The most reasoning tokens a budget allows: its close and a little slack, more for a rough count."""

    return budget + max(16, budget // 8) if exact else budget + budget // 2 + 32


def run_effort(effort: str) -> str:
    """The level mm3/api-tools runs for a requested effort (its five, the nearest for another API name)."""

    return {"minimal": "low"}.get(effort, effort)


def old_effort(effort: str) -> str:
    """The level the family as imported runs (fields.zig coerceEffort without template levels)."""

    return {"high": "xhigh", "max": "xhigh", "minimal": "low"}.get(effort, effort)


_KINDS = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is(v: Any, t: str) -> bool:
    if t == "integer":  # 3, not 3.0 (a typing slip the parser must not make)
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _KINDS.get(t, object))


def validate(v: Any, s: Any, path: str = "$") -> str | None:
    """Why ``v`` breaks the JSON schema ``s``, or None. The subset the harnesses use: type (a name or a list), enum,
    const, required, properties, additionalProperties (false or a schema), items, min/maxItems, min/maxLength,
    minimum / maximum and their exclusive forms."""

    if not isinstance(s, dict):
        return None
    if "enum" in s and v not in s["enum"]:
        return f"{path}: {v!r} not in {s['enum']}"
    if "const" in s and v != s["const"]:
        return f"{path}: {v!r} is not {s['const']!r}"
    t = s.get("type")
    types = [t] if isinstance(t, str) else t if isinstance(t, list) else []
    if types and not any(_is(v, x) for x in types):
        return f"{path}: {v!r} not {'/'.join(types)}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        for key, bad in (
            ("minimum", lambda b: v < b),
            ("maximum", lambda b: v > b),
            ("exclusiveMinimum", lambda b: v <= b),
            ("exclusiveMaximum", lambda b: v >= b),
        ):
            if isinstance(s.get(key), (int, float)) and bad(s[key]):
                return f"{path}: {v} breaks {key} {s[key]}"
    if isinstance(v, str) and not s.get("minLength", 0) <= len(v) <= s.get(
        "maxLength", 1 << 30
    ):
        return f"{path}: length {len(v)}"
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                return f"{path}: missing {k}"
        props, extra = s.get("properties", {}), s.get("additionalProperties", True)
        for k, x in v.items():
            if k in props:
                sub = props[k]
            elif extra is False:
                return f"{path}: extra key {k!r}"
            else:
                sub = extra
            if e := validate(x, sub, f"{path}.{k}"):
                return e
    if isinstance(v, list):
        if not s.get("minItems", 0) <= len(v) <= s.get("maxItems", 1 << 30):
            return f"{path}: {len(v)} items"
        for i, x in enumerate(v):
            if e := validate(x, s.get("items", {}), f"{path}[{i}]"):
                return e
    return None
