"""A stand-in for the MiniMax-M3.1 server on 127.0.0.1, for the bench and canary tests (no GPU, no model).

It answers the OpenAI routes the harnesses use (/health, /v1/models, /v1/chat/completions plain and streamed,
/v1/completions streamed) the way the engine's server does: a "brain" writes the model's raw text (``<mm:think>`` ...
``</mm:think>``, MiniMax tool markup rendered as the pack's chat template renders it), and the server half splits it,
reasoning into ``reasoning_content`` and each ``]<]minimax[>[<tool_call>`` block into ``tool_calls``, typing the
leaves by the tool's schema (a port of the engine's tool_minimax.zig rules). An invalid block stays in the content,
as on the real server. Switches make it misbehave on purpose:

- ``leak``: no split at all, the raw text is the content (a template / parser failure the harnesses must count);
- ``typing="strings"``: every leaf stays a string (a typing failure);
- ``echo``: the request's Authorization header is written at the head of every reply's content (redaction must mask the key);
- ``key``: requests without ``Authorization: Bearer <key>`` (or ``x-api-key``) get 401;
- ``response_format="refuse"``: json_schema and json_object requests get the engine's 400 (no grammar support);
- ``tool_choice`` "required" or a named function get the engine's 400 refusal unless ``enforce_tool_choice``;
- ``api={feature: mode}``: the engine features of the branch mm3/api-tools (``API``: the defaults; ``OLD_ENGINE``: the
  family as imported; ``NEW_ENGINE``: all served), each refused as the old engine does, served, or served wrongly:

  ==================  =========================================================================================
  ``tool_choice``     refuse (minimax_refusal), refuse-backend (no grammar support), ok, wrong (another function;
                      an unoffered one for "required"), badjson (arguments cut short), parallel (the named
                      function twice, parallel_tool_calls false ignored)
  ``response_format`` refuse, ok, badjson (content cut short), schema (``[]``), leak (``</mm:think>`` before it)
  ``thinking_off``    ok, thinks (reasoning anyway: the adaptive template), leak (the lone close left in content)
  ``thinking_budget`` refuse, ok (the reasoning cut at the budget), over (not cut), noanswer (cut, then no answer)
  ``reasoning_effort`` refuse, ok (the five levels as given, minimal as low), old (high / max run as xhigh)
  ``stream_tools``    ok, wrong (a streamed argument fragment loses its last character)
  ==================  =========================================================================================

  A request that asks for a long reasoning ("at length") with thinking on reasons for 600 words; a word is a token
  here (``usage.completion_tokens_details.reasoning_tokens``). ``tensorfold.reasoning_effort`` reports the level run.

- ``vision``: image input as the engine branch mm3/vision serves it (``--vision``), a dict: ``answers`` (the sha256 of an
  image's bytes -> the reply about it), ``max_images`` (4), ``max_bytes`` (10 MB an image), ``urls`` (False: an https
  part is refused, as without ``--vision-urls``), ``misbehave`` (a set: ``oversize-500`` answers an over-large image
  with HTTP 500, ``video-200`` serves a video part, ``count-200`` serves too many images). PNG, JPEG and WebP data URLs
  are taken; GIF, video parts and bad base64 get HTTP 400. An image counts its tokens as MiniMax's processor sizes it
  (28 x 28 pixels a token, 576 at most) in ``usage.prompt_tokens``. Without ``vision`` an image part gets HTTP 400.
  ``/v1/messages`` (Anthropic) takes ``image`` blocks with a base64 source and answers in Anthropic's shape.

Every request is recorded in ``requests`` (path, headers, body).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import struct
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

NS = "]<]minimax[>["
OPEN = NS + "<tool_call>"
CLOSE = NS + "</tool_call>"
MODEL = "MiniMax-M3.1-Flash-EXL3"
REFUSAL = (
    'tool_choice "required" or a named function cannot be enforced for MiniMax-M3.1\'s tool markup yet: '
    'use tool_choice "auto" and check the reply\'s tool_calls'
)
# the engine's texts: tool_grammar.zig (family patch), server.zig, think_plan.zig, fields.zig
BACKEND_REFUSAL = (
    "backend 'lanes' cannot enforce tool_choice \"required\" or a named function: it has no token grammar support; "
    'send "auto" or "none"'
)
STRUCTURES = "structured output (response_format and the guided_* fields) is not supported by this engine yet"
BUDGET_REFUSAL = "thinking_budget is not supported by this native engine"
EFFORT_REFUSAL = (
    "reasoning_effort must be none, minimal, low, medium, high, xhigh or max"
)
EFFORT_NAMES = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
API = {
    "tool_choice": "refuse",
    "response_format": "ok",
    "thinking_off": "ok",
    "thinking_budget": "ok",
    "reasoning_effort": "ok",
    "stream_tools": "ok",
}
OLD_ENGINE = dict(
    API, response_format="refuse", thinking_off="thinks", reasoning_effort="old"
)
NEW_ENGINE = dict(API, tool_choice="ok")
LONG_THINK = (
    "<mm:think>\n" + " ".join(f"step{i}" for i in range(600)) + "\n</mm:think>\n"
)


class Invalid(Exception):
    pass


# -- the template's rendering ----------------------------------------------------------------------------------------


def to_xml(val: Any) -> str:
    """chat_template.jinja's to_xml: mappings as child elements (None skipped), lists as <item>s, booleans as JSON."""

    if isinstance(val, dict):
        return "".join(
            f"{NS}<{k}>{to_xml(v)}{NS}</{k}>" for k, v in val.items() if v is not None
        )
    if isinstance(val, (list, tuple)):
        return "".join(f"{NS}<item>{to_xml(v)}{NS}</item>" for v in val)
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def render_calls(calls: list[tuple[str, dict]]) -> str:
    """One tool-call block of ``(name, arguments)`` calls, as the template writes an assistant turn's calls."""

    out = OPEN + "\n"
    for name, args in calls:
        out += f'{NS}<invoke name="{name}">'
        out += "".join(
            f"{NS}<{k}>{to_xml(v)}{NS}</{k}>" for k, v in args.items() if v is not None
        )
        out += f"{NS}</invoke>\n"
    return out + CLOSE


# -- the server's parser ---------------------------------------------------------------------------------------------


def kinds_of(schema: dict) -> set[str] | None:
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return set(t)
    return None


def sub_schema(schema: dict, key: str | None) -> dict:
    if not isinstance(schema, dict):
        return {}
    if key is None:
        s = schema.get("items")
    else:
        s = (schema.get("properties") or {}).get(key)
        if s is None and isinstance(schema.get("additionalProperties"), dict):
            s = schema["additionalProperties"]
    return s if isinstance(s, dict) else {}


def decode(text: str, kinds: set[str] | None, typing: str) -> Any:
    """A leaf's value: the type its schema names when the text spells it, else the text."""

    if typing == "strings" or not kinds:
        return text
    t = text.strip()
    if "boolean" in kinds and t in ("true", "false"):
        return t == "true"
    if "integer" in kinds and re.fullmatch(r"-?\d+", t):
        return int(t)
    if "number" in kinds and re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", t):
        return json.loads(t)
    if "null" in kinds and t == "null":
        return None
    return text


def value(node: dict, schema: dict, typing: str) -> Any:
    kinds = kinds_of(schema)
    if not node["kids"]:
        if node["text"] == "" and kinds and "string" not in kinds:
            if "array" in kinds and "object" not in kinds:
                return []
            if "object" in kinds and "array" not in kinds:
                return {}
        return decode(node["text"], kinds, typing)
    if node["text"].strip():
        raise Invalid("text beside child elements")
    as_list = all(k["name"] == "item" for k in node["kids"])
    if kinds and "array" in kinds and "object" not in kinds:
        as_list = True
    if kinds and "object" in kinds and "array" not in kinds:
        as_list = False
    if as_list:
        return [value(k, sub_schema(schema, None), typing) for k in node["kids"]]
    return {
        k["name"]: value(k, sub_schema(schema, k["name"]), typing) for k in node["kids"]
    }


def parse_block(
    payload: str, schemas: dict[str, dict], typing: str = "schema"
) -> list[tuple[str, dict]] | None:
    """The invokes of one block's payload with typed arguments; None when the block is not well formed."""

    parts = payload.split(NS)
    if parts[0].strip():
        return None
    stack: list[dict] = []
    out: list[tuple[str, dict]] = []
    try:
        for seg in parts[1:]:
            if not seg.startswith("<") or ">" not in seg:
                return None
            gt = seg.index(">")
            tag, rest = seg[1:gt], seg[gt + 1 :]
            if tag.startswith('invoke name="'):
                if (
                    stack
                    or not tag.endswith('"')
                    or len(tag) <= len('invoke name="')
                    or rest.strip()
                ):
                    return None
                stack.append(
                    {"name": tag[len('invoke name="') : -1], "text": "", "kids": []}
                )
            elif tag == "/invoke":
                if len(stack) != 1 or rest.strip():
                    return None
                node = stack.pop()
                params = schemas.get(node["name"], {})
                args = {
                    k["name"]: value(k, sub_schema(params, k["name"]), typing)
                    for k in node["kids"]
                }
                out.append((node["name"].strip(), args))
            elif tag.startswith("/"):
                if len(stack) < 2 or stack[-1]["name"] != tag[1:] or rest.strip():
                    return None
                stack.pop()
            else:
                if not stack or not tag or re.search(r'[ \t\r\n"<=/]', tag):
                    return None
                node = {"name": tag, "text": rest, "kids": []}
                stack[-1]["kids"].append(node)
                stack.append(node)
    except Invalid:
        return None
    return out if out and not stack else None


def split_reply(
    raw: str, tools: list[dict], typing: str = "schema"
) -> tuple[str, str, list[tuple[str, dict]]]:
    """(content, reasoning, calls) of a raw reply, as the server splits it."""

    schemas = {
        t["function"]["name"]: t["function"].get("parameters") or {} for t in tools
    }
    reasoning, text = "", raw
    if "</mm:think>" in text:
        head, _, text = text.partition("</mm:think>")
        reasoning = head.split("<mm:think>")[-1].strip()
    parts, calls, pos = [], [], 0
    while (s := text.find(OPEN, pos)) != -1:
        e = text.find(CLOSE, s)
        end = len(text) if e == -1 else e + len(CLOSE)
        parsed = (
            parse_block(text[s + len(OPEN) : e].strip(), schemas, typing)
            if e != -1
            else None
        )
        parts.append(text[pos:end] if parsed is None else text[pos:s])
        calls += parsed or []
        pos = end
    parts.append(text[pos:])
    return "".join(parts).strip(), reasoning, calls


# -- the model -------------------------------------------------------------------------------------------------------


def visible(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def example(schema: dict, key: str = "", text: str = "") -> Any:
    """A value that satisfies ``schema`` (deterministic); a ``city`` is read from the prompt."""

    if "enum" in schema:
        return schema["enum"][0]
    t = kinds_of(schema) or {"string"}
    if "object" in t:
        return {
            k: example(v, k, text) for k, v in (schema.get("properties") or {}).items()
        }
    if "array" in t:
        return [
            example(sub_schema(schema, None), "", text)
            for _ in range(max(schema.get("minItems", 1), 1))
        ]
    if "boolean" in t:
        return True
    if "integer" in t:
        return max(int(schema.get("minimum", 1)), 1)
    if "number" in t:
        return 1.5
    if key == "city":
        m = re.search(r"\bin ([A-Z][a-z]+)", text)
        return m.group(1) if m else "Paris"
    return "x"


def default_brain(body: dict, fake: FakeMiniMax) -> str:
    """The raw text a model would write: a think block when thinking is on, then tool calls or an answer."""

    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else {}
    text = visible(last.get("content"))
    tools = body.get("tools") or []
    choice = body.get("tool_choice", "auto")
    think = ""
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
    if thinking or fake.api["thinking_off"] == "thinks":
        think = "<mm:think>\nWork it out step by step, then answer.\n</mm:think>\n"
    if thinking and "at length" in text:
        think = LONG_THINK
    if last.get("role") == "tool":
        return think + "From the tool: " + visible(last.get("content"))[:200]
    if tools and choice != "none":
        names = [t["function"]["name"] for t in tools]
        if isinstance(choice, dict):
            chosen = [choice["function"]["name"]]
        else:
            hits = sorted(
                (m.start(), n)
                for n in names
                if (m := re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE))
            )
            chosen = [n for _, n in hits] or names[:1]
        if body.get("parallel_tool_calls") is False:
            chosen = chosen[:1]
        mode = fake.api["tool_choice"]
        if mode == "wrong" and isinstance(choice, dict):
            chosen = [n for n in names if n != chosen[0]][:1]
        elif mode == "wrong" and choice == "required":
            chosen = ["not_offered"]
        elif mode == "parallel" and isinstance(choice, dict):
            chosen = chosen[:1] * 2
        params = {
            t["function"]["name"]: t["function"].get("parameters") or {} for t in tools
        }
        return think + render_calls(
            [
                (
                    n,
                    fake.args.get(n)
                    or (example(params[n], "", text) if n in params else {}),
                )
                for n in chosen
            ]
        )
    if fake.vision is not None:
        for data in image_bytes(last):
            if (seen := fake.vision.get("answers", {}).get(hashlib.sha256(data).hexdigest())) is not None:
                return think + seen
    return think + answer(body, text)


# -- images ----------------------------------------------------------------------------------------------------------

IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"), (b"RIFF", "webp"), (b"GIF8", "gif"))


def media_parts(body: dict) -> list[dict]:
    """Every image or video part of a request's messages."""

    out = []
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            out += [p for p in content if isinstance(p, dict) and p.get("type") in ("image_url", "video_url")]
    return out


def decode_url(url: str) -> bytes:
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("only data: URLs with base64 content")
    return base64.b64decode(url.split(";base64,", 1)[1], validate=True)


def image_bytes(message: dict) -> list[bytes]:
    out = []
    content = message.get("content")
    for p in content if isinstance(content, list) else []:
        if isinstance(p, dict) and p.get("type") == "image_url":
            try:
                out.append(decode_url((p.get("image_url") or {}).get("url") or ""))
            except (ValueError, binascii.Error):
                pass
    return out


def image_tokens(data: bytes) -> int:
    """MiniMax's rows for an image: at most 672 x 672 pixels (aspect kept), 28 x 28 pixels a token; 576 for non-PNG."""

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return 576
    w, h = struct.unpack(">II", data[16:24])
    if w * h > 672 * 672:
        s = math.sqrt(672 * 672 / (w * h))
        w, h = max(28, math.floor(w * s / 28) * 28), max(28, math.floor(h * s / 28) * 28)
    else:
        w, h = max(28, round(w / 28) * 28), max(28, round(h / 28) * 28)
    return min(576, (w // 28) * (h // 28)) + 2  # the image's begin and end markers


def answer(body: dict, text: str) -> str:
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        return json.dumps(example(rf["json_schema"]["schema"]))
    if rf.get("type") == "json_object":
        return '{"city": "Lyon", "country": "France"}'
    for pattern, reply in (
        ("capital of France", "Paris"),
        ("capital of Italy", "Rome is the capital of Italy."),
        ("largest planet", "Jupiter"),
        ("17 + 25", "42"),
        ("17*23", "391"),
        ("key ok", '{"ok": true}'),
        ("multiple choice", "B"),
        ("Reply with only a JSON object", '{"title": "x", "year": 1, "tags": []}'),
    ):
        if pattern in text:
            return reply
    if m := re.search(r"passkey is (\d+)", text):
        return m.group(1)
    return "A short, plain answer: the request was understood and handled."


# -- the server ------------------------------------------------------------------------------------------------------


class FakeMiniMax:
    def __init__(
        self,
        *,
        model: str = MODEL,
        key: str = "",
        leak: bool = False,
        typing: str = "schema",
        echo: bool = False,
        response_format: str = "ok",
        enforce_tool_choice: bool = False,
        api: dict[str, str] | None = None,
        brain: Callable[[dict, FakeMiniMax], str] | None = None,
        args: dict[str, dict] | None = None,
        health: dict | None = None,
        vision: dict | None = None,
    ) -> None:
        self.model, self.key, self.leak, self.typing, self.echo = (
            model,
            key,
            leak,
            typing,
            echo,
        )
        self.response_format, self.enforce_tool_choice = (
            response_format,
            enforce_tool_choice,
        )
        self.api = dict(API, **(api or {}))
        if response_format == "refuse":
            self.api["response_format"] = "refuse"
        if enforce_tool_choice:
            self.api["tool_choice"] = "ok"
        self.brain = brain or default_brain
        self.args = args or {}
        self.health = {"status": "ok", "inflight": 0} if health is None else health
        self.vision = None if vision is None else {"answers": {}, "max_images": 4, "max_bytes": 10_000_000,
                                                   "urls": False, "misbehave": set(), **vision}
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.httpd.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> Self:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def bodies(self, path: str = "/v1/chat/completions") -> list[dict]:
        return [r["body"] for r in self.requests if r["path"] == path]

    # the reply to one chat request: (status, JSON body) or (200, the message parts for a stream)
    def refusal(self, body: dict) -> str | None:
        """The engine's refusal of a request's features, or None."""

        api, kw = self.api, body.get("chat_template_kwargs") or {}
        choice = body.get("tool_choice")
        if (choice == "required" or isinstance(choice, dict)) and body.get("tools"):
            if api["tool_choice"] == "refuse":
                return REFUSAL
            if api["tool_choice"] == "refuse-backend":
                return BACKEND_REFUSAL
        rf = (body.get("response_format") or {}).get("type")
        if rf in ("json_schema", "json_object") and api["response_format"] == "refuse":
            return STRUCTURES
        budget = body.get("thinking_budget")
        if (
            kw.get("enable_thinking")
            and isinstance(budget, int)
            and budget > 0
            and api["thinking_budget"] == "refuse"
        ):
            return BUDGET_REFUSAL
        effort = body.get("reasoning_effort", kw.get("reasoning_effort"))
        if effort is not None and (
            effort not in EFFORT_NAMES or api["reasoning_effort"] == "refuse"
        ):
            return EFFORT_REFUSAL
        return None

    def effort_run(self, body: dict) -> str | None:
        """The level ``tensorfold.reasoning_effort`` reports (fields.zig coerceEffort, old or new)."""

        kw = body.get("chat_template_kwargs") or {}
        if not kw.get("enable_thinking"):
            return "none"
        e = body.get("reasoning_effort", kw.get("reasoning_effort"))
        if e is None:
            return None
        if self.api["reasoning_effort"] == "old":
            return {"high": "xhigh", "max": "xhigh", "minimal": "low"}.get(e, e)
        return {"minimal": "low"}.get(e, e)

    def media(self, body: dict) -> tuple[int, str] | None:
        """The engine's answer to a request's image and video parts when it is not served: (status, message)."""

        parts = media_parts(body)
        if not parts:
            return None
        v = self.vision
        if v is None:
            return 400, "image input needs the server's --vision flag"
        bad = v["misbehave"]
        images = [p for p in parts if p["type"] == "image_url"]
        if len(images) != len(parts) and "video-200" not in bad:
            return 400, "video parts are not supported by the MiniMax-M3.1 engine: send images"
        if len(images) > v["max_images"] and "count-200" not in bad:
            return 400, (f"a request may hold at most {v['max_images']} images (--vision-max-images); "
                         f"it has {len(images)}")
        for p in images:
            url = (p.get("image_url") or {}).get("url") or ""
            if url.startswith("https://") and not v["urls"]:
                return 400, "image URLs need the server's --vision-urls flag; send a data: URL"
            try:
                data = decode_url(url)
            except (ValueError, binascii.Error):
                return 400, "an image_url must be a data: URL with base64 content"
            if len(data) > v["max_bytes"]:
                if "oversize-500" in bad:
                    return 500, "internal error"
                return 400, f"an image may be at most {v['max_bytes']} bytes; this one is {len(data)}"
            kind = next((k for magic, k in IMAGE_MAGIC if data.startswith(magic)), None)
            if kind not in ("png", "jpeg", "webp"):
                return 400, "unsupported image format: send JPEG, PNG or WebP"
        return None

    def prompt_tokens(self, body: dict) -> int:
        """The request's text a token per 4 characters (image data left out), plus each image's rows."""

        msgs, n = json.loads(json.dumps(body.get("messages") or [])), 0
        for m in msgs:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    try:
                        n += image_tokens(decode_url((p.get("image_url") or {}).get("url") or ""))
                    except (ValueError, binascii.Error):
                        pass
            m["content"] = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
        return len(json.dumps(msgs)) // 4 + n

    def chat(self, body: dict, headers: dict) -> tuple[int, dict]:
        if why := self.refusal(body):
            return 400, {"error": {"message": why, "type": "invalid_request_error"}}
        if refused := self.media(body):
            status, why = refused
            return status, {"error": {"message": why, "type": "invalid_request_error" if status < 500 else "server"}}
        api, kw = self.api, body.get("chat_template_kwargs") or {}
        tools = body.get("tools") or []
        raw = self.brain(body, self)
        if self.leak:
            content, reasoning, calls = raw, "", []
        else:
            content, reasoning, calls = split_reply(raw, tools, self.typing)
        finish = "tool_calls" if calls else "stop"
        budget = body.get("thinking_budget")
        if kw.get("enable_thinking") and isinstance(budget, int) and budget > 0:
            if api["thinking_budget"] in ("ok", "noanswer"):
                reasoning = " ".join(reasoning.split()[:budget])
            if api["thinking_budget"] == "noanswer":
                content, finish = "", "length"
        if not kw.get("enable_thinking") and api["thinking_off"] == "leak":
            # the old split: a lone close stays in the answer
            content = "</mm:think>\n" + content
        if (body.get("response_format") or {}).get("type") and not calls:
            content = {
                "badjson": content[:-1],
                "schema": "[]",
                "leak": "</mm:think>" + content,
            }.get(api["response_format"], content)
        if self.echo:  # what a server that reflects its request headers would show
            content = f"[echo Authorization: {headers.get('Authorization')}] " + content
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or (None if calls else ""),
        }
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            cut = api["tool_choice"] == "badjson" and body.get("tool_choice") not in (
                None,
                "auto",
                "none",
            )
            msg["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": n,
                        "arguments": json.dumps(a)[:-1] if cut else json.dumps(a),
                    },
                }
                for i, (n, a) in enumerate(calls)
            ]
        usage = {
            "prompt_tokens": self.prompt_tokens(body),
            "completion_tokens": len(raw) // 4 + 1,
            "completion_tokens_details": {"reasoning_tokens": len(reasoning.split())},
        }
        return 200, {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": self.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage,
            "tensorfold": {
                "enable_thinking": bool(kw.get("enable_thinking")),
                "reasoning_effort": self.effort_run(body),
            },
        }

    def messages(self, body: dict, headers: dict) -> tuple[int, dict]:
        """Anthropic /v1/messages: image blocks (base64 source) become image_url parts; the reply in Anthropic's shape."""

        msgs = []
        for m in body.get("messages") or []:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if b.get("type") == "image":
                        src = b.get("source") or {}
                        url = f"data:{src.get('media_type')};base64,{src.get('data')}"
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        parts.append({"type": "text", "text": b.get("text", "")})
                content = parts
            msgs.append({"role": m.get("role"), "content": content})
        thinking = (body.get("thinking") or {}).get("type") == "enabled"
        chat = {"model": body.get("model"), "messages": msgs, "chat_template_kwargs": {"enable_thinking": thinking}}
        status, reply = self.chat(chat, headers)
        if status != 200:
            return status, {"type": "error", "error": {"type": "invalid_request_error",
                                                       "message": reply["error"]["message"]}}
        msg = reply["choices"][0]["message"]
        blocks = [{"type": "thinking", "thinking": msg["reasoning_content"]}] if msg.get("reasoning_content") else []
        blocks.append({"type": "text", "text": msg.get("content") or ""})
        return 200, {"id": "msg_fake", "type": "message", "role": "assistant", "model": self.model,
                     "content": blocks, "stop_reason": "end_turn",
                     "usage": {"input_tokens": reply["usage"]["prompt_tokens"],
                               "output_tokens": reply["usage"]["completion_tokens"]}}

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def send_json(self, status: int, obj: Any) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self) -> bool:
                if not fake.key:
                    return True
                return (
                    self.headers.get("Authorization") == "Bearer " + fake.key
                    or self.headers.get("x-api-key") == fake.key
                )

            def record(self, body: Any) -> None:
                with fake.lock:
                    fake.requests.append(
                        {"path": self.path, "headers": dict(self.headers), "body": body}
                    )

            def do_GET(self):
                self.record(None)
                if self.path == "/health":  # no key needed, as on the real server
                    return self.send_json(200, fake.health)
                if not self.authorized():
                    return self.send_json(
                        401, {"error": {"message": "invalid api key", "type": "auth"}}
                    )
                if self.path == "/v1/models":
                    return self.send_json(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": fake.model, "object": "model"}],
                        },
                    )
                self.send_json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self.send_json(400, {"error": {"message": "bad JSON"}})
                self.record(body)
                if not self.authorized():
                    return self.send_json(
                        401, {"error": {"message": "invalid api key", "type": "auth"}}
                    )
                if self.path == "/v1/chat/completions":
                    status, reply = fake.chat(body, dict(self.headers))
                    if status != 200 or not body.get("stream"):
                        return self.send_json(status, reply)
                    return self.stream_chat(reply, body)
                if self.path == "/v1/messages":
                    return self.send_json(*fake.messages(body, dict(self.headers)))
                if self.path == "/v1/completions" and body.get("stream"):
                    return self.stream_completion(body)
                self.send_json(404, {"error": {"message": "not found"}})

            def sse(self, chunks: list[dict]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    for c in chunks:
                        self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                ):  # a client that cancels mid-stream
                    pass

            def stream_chat(self, reply: dict, body: dict) -> None:
                choice = reply["choices"][0]
                msg = choice["message"]
                deltas: list[dict] = [{"role": "assistant"}]
                if msg.get("reasoning_content"):
                    deltas.append({"reasoning_content": msg["reasoning_content"]})
                content = msg.get("content") or ""
                half = len(content) // 2
                deltas += [
                    {"content": part}
                    for part in (content[:half], content[half:])
                    if part
                ]
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    args = tc["function"]["arguments"]
                    first = {"index": i, "id": tc["id"], "type": "function"}
                    first["function"] = {
                        "name": tc["function"]["name"],
                        "arguments": args[: len(args) // 2],
                    }
                    tail = args[len(args) // 2 :]
                    if fake.api["stream_tools"] == "wrong":
                        tail = tail[:-1]
                    rest = {"index": i, "function": {"arguments": tail}}
                    deltas += [{"tool_calls": [first]}, {"tool_calls": [rest]}]
                chunks = [
                    {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": d}],
                    }
                    for d in deltas
                ]
                chunks.append(
                    {
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": choice["finish_reason"],
                            }
                        ],
                    }
                )
                if (body.get("stream_options") or {}).get("include_usage"):
                    chunks.append(
                        {
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "usage": reply["usage"],
                        }
                    )
                self.sse(chunks)

            def stream_completion(self, body: dict) -> None:
                n = int(body.get("max_tokens") or 16)
                chunks = [
                    {
                        "object": "text_completion",
                        "choices": [{"index": 0, "text": "a"}],
                    }
                    for _ in range(n)
                ]
                usage = {
                    "prompt_tokens": len(body.get("prompt") or []),
                    "completion_tokens": n,
                }
                chunks.append(
                    {"object": "text_completion", "choices": [], "usage": usage}
                )
                self.sse(chunks)

        return Handler
