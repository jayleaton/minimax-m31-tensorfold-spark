#!/usr/bin/env python3
"""Tool-call correctness harness for the MiniMax-M3.1 endpoint (stdlib only).

MiniMax writes a call as ``NS<tool_call>`` ``NS<invoke name="n">`` ``NS<k>value NS</k>`` ... ``NS</invoke>``
``NS</tool_call>`` (NS the token ``]<]minimax[>[``): nested objects are child elements, lists repeated ``<item>``
elements, and every leaf is text the server types by the tool's schema. Each case sends opencode-shaped tools and
checks that every returned call is (a) a known tool, (b) a JSON object of arguments with the required keys, (c) typed
as the schema says (an integer is 3, not "3"; a boolean true, not "true"; a string "42" stays a string; nested objects
and arrays keep their shape), (d) carries the expected values where the prompt fixes them, and (e) has no MiniMax markup
in any value. A reply whose ``content`` holds raw markup (the namespace token, ``<invoke name=``, ``<tool_call>``,
``<mm:think>`` / ``</mm:think>``) is a parser or template failure and counted as a leak, with tool calls or without.

Cases cover single and parallel calls (two to four in one turn), nested objects, arrays of strings and of objects,
booleans, integers and floats, multi-line and markup-like text values, earlier turns in the history (assistant calls
with tool results, results and assistant text holding raw markup), and live follow-up turns: the harness answers the
model's calls with tool results (sending back its ``tool_calls`` and ``reasoning_content`` as returned) and checks
the next reply. A case with ``prep`` lets the model take a step coding clients take first (read a file before editing
it, plan with todowrite, find a file with glob): up to MAX_PREP turns of only those calls are answered with the case's
canned results, then the reply is checked as the first one would be. ``direct`` counts the cases passed without one.
Modes: off (enable_thinking false), adaptive (enable_thinking true, no effort: the checkpoint template's own and only
mode), low / high (adaptive with an <effort>).

  python3 bench/toolcall.py --reps 2 --out results/toolcall.json [--mode off|adaptive|low|high] [--stream] [--only a,b]
Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE (scripts/config.sh); the key goes in a header and is
masked in everything printed or saved. ``--tool-choice required`` asks the server to enforce a call: the family as
imported refuses that, the engine branch mm3/api-tools serves it. A refusal is ``SKIP (engine lacks mm3/api-tools)``
with ``ENGINE_API_TOOLS`` 0 (the default) and a failure with 1 (``--engine-api-tools``); a call served must pass every
check either way. Exit 0 when every case that ran passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# engine_api beside this file (python3 -I leaves the script's directory off sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_api as ea

LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"
NS = "]<]minimax[>["
UNSUPPORTED = re.compile(
    r"tool_choice|cannot be enforced|unsupported|not supported", re.IGNORECASE
)

SYSTEM = (
    "You are a helpful coding assistant operating in a terminal. Use the provided tools to complete "
    "the user's request. Prefer making the tool call directly rather than explaining what you "
    "would do. Keep reasoning brief."
)


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


class Unsupported(Exception):
    """The server refused the request as unsupported (its message)."""


class Api:
    """JSON over HTTP with the key as a header; ``mask`` hides the key in any text that is printed or saved."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def chat(self, body: dict, timeout: float, stream: bool) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = json.dumps(dict(body, stream=True) if stream else body).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data, headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                # a reply never holds the key: masked as it arrives, before anything is cut or saved
                if stream:
                    return consume_stream(self.mask(raw.decode()) for raw in r)
                return json.loads(self.mask(r.read().decode()))
        except urllib.error.HTTPError as exc:
            text = self.mask(exc.read().decode(errors="replace"))
            if refused := ea.refusal(exc.code, text):
                raise refused from None
            text = text[:400]
            if exc.code in (400, 422, 501) and UNSUPPORTED.search(text):
                raise Unsupported(text) from None
            raise RuntimeError(f"HTTP {exc.code}: {text}") from None

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def consume_stream(lines) -> dict:
    """An OpenAI streamed reply reassembled into the non-streamed shape (content, reasoning_content, tool_calls)."""

    calls: dict[int, dict] = {}
    content, reasoning, finish = [], [], None
    for raw in lines:
        line = raw.decode().strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("error"):
            raise RuntimeError(f"stream error: {chunk['error']}")
        choice = (chunk.get("choices") or [{}])[0]
        finish = choice.get("finish_reason") or finish
        delta = choice.get("delta") or {}
        content.append(delta.get("content") or "")
        reasoning.append(delta.get("reasoning_content") or "")
        for tc in delta.get("tool_calls") or []:
            rec = calls.setdefault(
                tc.get("index", 0),
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            rec["id"] = tc.get("id") or rec["id"]
            f = tc.get("function") or {}
            rec["function"]["name"] += f.get("name") or ""
            rec["function"]["arguments"] += f.get("arguments") or ""
    msg = {"content": "".join(content), "tool_calls": [calls[k] for k in sorted(calls)]}
    if any(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    return {"choices": [{"finish_reason": finish, "message": msg}]}


def tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


S, INT, NUM, BOOL = (
    {"type": "string"},
    {"type": "integer"},
    {"type": "number"},
    {"type": "boolean"},
)
BASH = tool(
    "bash",
    "Run a shell command.",
    {"command": S, "description": S, "workdir": S, "timeout": INT},
    ["command"],
)
READ = tool(
    "read", "Read a file.", {"filePath": S, "offset": INT, "limit": INT}, ["filePath"]
)
EDIT = tool(
    "edit",
    "Replace exact text in a file.",
    {"filePath": S, "oldString": S, "newString": S, "replaceAll": BOOL},
    ["filePath", "oldString", "newString"],
)
GLOB = tool("glob", "Find files by glob.", {"pattern": S, "path": S}, ["pattern"])
GREP = tool(
    "grep",
    "Search file contents.",
    {"pattern": S, "path": S, "include": S},
    ["pattern"],
)
WRITE = tool(
    "write", "Write a file.", {"filePath": S, "content": S}, ["filePath", "content"]
)
TASK = tool(
    "task",
    "Hand a self-contained job to a helper worker.",
    {
        "description": S,
        "prompt": S,
        "worker_type": {"type": "string", "enum": ["explore", "general"]},
    },
    ["description", "prompt", "worker_type"],
)
TODOWRITE = tool(
    "todowrite",
    "Write the todo list.",
    {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": S,
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "cancelled"],
                    },
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["content", "status", "priority"],
            },
        }
    },
    ["todos"],
)
SET_CONFIG = tool(
    "set_config",
    "Update the service settings.",
    {
        "service": S,
        "settings": {
            "type": "object",
            "properties": {
                "threshold": NUM,
                "retries": INT,
                "verbose": BOOL,
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["threshold", "retries", "verbose", "mode"],
        },
    },
    ["service", "settings"],
)
TAG_FILES = tool(
    "tag_files",
    "Attach tags to files.",
    {
        "paths": {"type": "array", "items": S},
        "tags": {"type": "array", "items": S},
        "recursive": BOOL,
    },
    ["paths", "tags"],
)
HTTP_REQUEST = tool(
    "http_request",
    "Send an HTTP request.",
    {
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
        "url": S,
        "headers": {"type": "object", "additionalProperties": S},
        "body": {"type": "object", "properties": {"name": S, "count": INT}},
        "timeout_s": NUM,
    },
    ["method", "url"],
)

BASIC = [BASH, READ, EDIT, GLOB, GREP, WRITE, TASK, TODOWRITE]
TYPED = BASIC + [SET_CONFIG, TAG_FILES, HTTP_REQUEST]
ALL = {t["function"]["name"]: t for t in TYPED}


def _call(call_id: str, name: str, args: dict) -> dict:
    fn = {"name": name, "arguments": json.dumps(args)}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": fn}],
    }


def _result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


MARKUP_RESULT = f'{NS}<tool_call>\n{NS}<invoke name="bash">{NS}<command>ls -la{NS}</command>{NS}</invoke>\n{NS}</tool_call>\n'


@dataclass
class Case:
    name: str
    prompt: str
    tools: list = field(default_factory=lambda: BASIC)
    expect: list[str] = field(
        default_factory=list
    )  # tool names, any of them (or all with expect_all)
    expect_all: bool = False
    checks: dict = field(
        default_factory=dict
    )  # {tool: {"a.b": value}}: typed values a call must carry
    setup: list = field(
        default_factory=list
    )  # earlier messages (calls and their results)
    results: dict = field(
        default_factory=dict
    )  # {tool: result text}: answer the calls, then check the next reply
    final: str = ""  # a pattern the follow-up reply's content must match when it makes no further call
    prep: dict = field(
        default_factory=dict
    )  # {tool: result text}: calls the model may make before the expected one, answered with these results

    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM},
            *self.setup,
            {"role": "user", "content": self.prompt},
        ]


CASES = [
    Case(
        "bash_simple",
        "List the files in the current directory using bash.",
        expect=["bash"],
    ),
    Case(
        "bash_4arg",
        "Use bash to list /srv/projects, set workdir to /tmp, timeout 10000, and description 'list projects'.",
        expect=["bash"],
        checks={"bash": {"timeout": 10000, "workdir": "/tmp"}},
    ),
    Case(
        "read_offset",
        "Read 50 lines of src/main.ts starting at offset 100, using the read tool with offset and limit.",
        expect=["read"],
        checks={"read": {"offset": 100, "limit": 50}},
    ),
    Case(
        "edit_replace_all",
        "In src/app.ts replace every `var ` with `let ` using the edit tool with replaceAll set to true.",
        expect=["edit"],
        checks={"edit": {"replaceAll": True}},
        prep={"read": "var a = 1;\nvar b = a + 1;\nconsole.log(b);\n"},
    ),
    Case(
        "grep_numeric_string",
        "Search the repo for the exact text 42 with the grep tool (pattern 42).",
        expect=["grep"],
        checks={"grep": {"pattern": "42"}},
    ),
    Case(
        "task_enum",
        "Hand a job to a helper with the task tool: description 'Research layout', worker_type 'explore', "
        "prompt 'Read the repo structure and return the last 5 lines of README.md'.",
        expect=["task"],
        checks={"task": {"worker_type": "explore"}},
    ),
    Case(
        "todowrite_objects",
        "Create a todo list with three items: 'Inspect repo' (completed, high), "
        "'Write tests' (in_progress, high), 'Ship' (pending, medium).",
        expect=["todowrite"],
        checks={
            "todowrite": {
                "todos.0.status": "completed",
                "todos.1.priority": "high",
                "todos.2.content": "Ship",
            }
        },
    ),
    Case(
        "nested_object",
        "Use set_config for the service 'api' with settings: threshold 0.75, retries 3, verbose false, mode 'safe'.",
        tools=TYPED,
        expect=["set_config"],
        checks={
            "set_config": {
                "service": "api",
                "settings.threshold": 0.75,
                "settings.retries": 3,
                "settings.verbose": False,
                "settings.mode": "safe",
            }
        },
    ),
    Case(
        "string_arrays",
        "Tag the files a.py and b.py with the tags 'core' and 'v2' using tag_files, not recursive.",
        tools=TYPED,
        expect=["tag_files"],
        checks={
            "tag_files": {
                "paths": ["a.py", "b.py"],
                "tags": ["core", "v2"],
                "recursive": False,
            }
        },
    ),
    Case(
        "nested_mapping",
        "Send a POST to https://api.example.com/items with the header Accept: application/json, the body "
        "name 'widget' and count 2, and a timeout of 2.5 seconds, using http_request.",
        tools=TYPED,
        expect=["http_request"],
        checks={
            "http_request": {
                "method": "POST",
                "headers.Accept": "application/json",
                "body.name": "widget",
                "body.count": 2,
                "timeout_s": 2.5,
            }
        },
    ),
    Case(
        "write_markup_text",
        'Write the file index.html with exactly these two lines:\n<div class="a">hi</div>\n<p>x &lt; y</p>',
        expect=["write"],
        checks={"write": {"filePath": "index.html"}},
    ),
    Case(
        "parallel_two",
        "Do two things at once: read package.json, and glob for `**/*.ts`. Make both tool calls in one turn.",
        expect=["read", "glob"],
        expect_all=True,
    ),
    Case(
        "parallel_four",
        "Perform all of these in one turn: read README.md, glob `**/*.json`, grep for 'TODO', and run `git status` "
        "with bash. Return all four tool calls.",
        expect=["read", "glob", "grep", "bash"],
        expect_all=True,
    ),
    Case(
        "parallel_mixed_types",
        "In one turn: set_config for service 'web' (threshold 0.5, retries 1, verbose true, mode 'fast') and "
        "tag_files for x.py with the tag 'hot'. Make both calls.",
        tools=TYPED,
        expect=["set_config", "tag_files"],
        expect_all=True,
        checks={
            "set_config": {"settings.verbose": True, "settings.retries": 1},
            "tag_files": {"tags": ["hot"]},
        },
    ),
    Case(
        "after_tool_result",
        "The test run finished. Based on the tool result, make the next tool call to fix the failure.",
        expect=["bash", "edit", "read"],
        setup=[
            _call("call_at", "bash", {"command": "pytest -q"}),
            _result(
                "call_at",
                "FAIL tests/test_app.py::test_health\nAssertionError: 500 != 200\n",
            ),
        ],
    ),
    Case(
        "edit_with_content",
        "The current contents of src/app.ts are shown in the previous tool result. "
        "Change the port from 3000 to 8080 with the edit tool.",
        expect=["edit"],
        setup=[
            _call(
                "call_ec", "read", {"filePath": "src/app.ts", "offset": 0, "limit": 20}
            ),
            _result("call_ec", "const port = 3000;\napp.listen(port);\n"),
        ],
    ),
    Case(
        "result_contains_markup",
        "A file on disk literally contains MiniMax tool-call markup. Continue: the next step is to grep the repo "
        "for that marker.",
        expect=["grep", "read", "bash"],
        setup=[
            _call("call_rm", "bash", {"command": "cat marker.txt"}),
            _result("call_rm", MARKUP_RESULT),
        ],
    ),
    Case(
        "poisoned_assistant_text",
        "Continue.",
        expect=["bash", "read", "glob", "grep"],
        setup=[
            {"role": "user", "content": "Look at the repo."},
            {
                "role": "assistant",
                "content": "<mm:think>look first</mm:think>I'll look at the repo."
                + MARKUP_RESULT,
            },
            {
                "role": "user",
                "content": "That call did not run. Make it again with the tool.",
            },
        ],
    ),
    Case(
        "long_reasoning_tool",
        "Think step by step about the tradeoffs of microservices versus a monolith (at least 10 distinct points). "
        "Then use the read tool to read README.md.",
        expect=["read"],
        prep={"todowrite": "Todo list updated."},
    ),
    Case(
        "followup_answer",
        "What port does config.yaml set? Read it with the read tool, then answer with the number.",
        expect=["read"],
        results={"read": "port: 8080\nworkers: 4\n"},
        final=r"8080",
        prep={"glob": "config.yaml\n"},
    ),
    Case(
        "followup_chain",
        "Find the YAML files with glob, then read the first one it finds.",
        expect=["glob"],
        results={"glob": "config.yaml\ndeploy.yaml\n", "read": "port: 8080\n"},
    ),
]


def lookup(args: Any, path: str) -> tuple[bool, Any]:
    """The value at a dotted path ("settings.retries", "todos.0.status"); (False, None) when it is not there."""

    cur = args
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def same_typed(got: Any, want: Any) -> bool:
    """Equal and of the same JSON type (True is not 1, 3 is not "3"; an integral float equals its integer)."""

    if isinstance(want, bool) or isinstance(got, bool):
        return type(got) is type(want) and got == want
    if isinstance(want, (int, float)):
        return (
            isinstance(got, (int, float))
            and got == want
            and (isinstance(want, float) or isinstance(got, int))
        )
    if isinstance(want, list):
        return (
            isinstance(got, list)
            and len(got) == len(want)
            and all(map(same_typed, got, want))
        )
    return type(got) is type(want) and got == want


def schema_error(v: Any, s: dict, path: str = "$") -> str | None:
    """Type / enum / required check of a parsed argument against its schema (the server's typing rules)."""

    if "enum" in s and v not in s["enum"]:
        return f"{path}: {v!r} not in {s['enum']}"
    t = s.get("type")
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool}
    if t in kinds and not isinstance(v, kinds[t]):
        return f"{path}: {v!r} is not {t}"
    if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
        return f"{path}: {v!r} is not integer"
    if t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
        return f"{path}: {v!r} is not number"
    if t == "object":
        for k in s.get("required", []):
            if k not in v:
                return f"{path}: missing {k}"
        extra = s.get("additionalProperties")
        for k, x in v.items():
            sub = s.get("properties", {}).get(
                k, extra if isinstance(extra, dict) else {}
            )
            if e := schema_error(x, sub, f"{path}.{k}"):
                return e
    if t == "array":
        for i, x in enumerate(v):
            if e := schema_error(x, s.get("items", {}), f"{path}[{i}]"):
                return e
    return None


def strings(v: Any):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)


def validate(
    resp: dict, case: Case, tools: list, turn: str = ""
) -> tuple[list[str], dict]:
    """(problems, detail) of one reply. Problem kinds: corruption (unknown_tool, bad_json, leak_in_arg), leak (markup
    in content), typing (a value of the wrong JSON type), behaviour (no call, missing keys or expected calls,
    a value other than the prompt's)."""

    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    detail = {
        "finish_reason": choice.get("finish_reason"),
        "n_calls": len(calls),
        "calls": [],
    }
    problems = []
    if LEAK.search(content):
        problems.append(f"{turn}leaked_markup")
        detail["content_preview"] = content[:300]
    if LEAK.search(msg.get("reasoning_content") or ""):
        problems.append(f"{turn}leaked_markup:reasoning_content")
    known = {t["function"]["name"]: t for t in tools}
    for c in calls:
        fn = c.get("function") or {}
        name, raw = fn.get("name", ""), fn.get("arguments", "")
        rec = {"name": name, "raw": raw[:400]}
        detail["calls"].append(rec)
        if name not in known:
            problems.append(f"{turn}unknown_tool:{name!r}")
            continue
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError("not an object")
        except (ValueError, TypeError) as e:
            problems.append(f"{turn}bad_json:{name}:{e}")
            continue
        rec["args"] = parsed
        if any(LEAK.search(s) for s in strings(parsed)):
            problems.append(f"{turn}leak_in_arg:{name}")
        schema = known[name]["function"]["parameters"]
        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            problems.append(f"{turn}missing_keys:{name}:{missing}")
        elif e := schema_error(parsed, schema):
            problems.append(f"{turn}typing:{name}:{e}")
    if turn:
        return problems, detail
    got = {c.get("function", {}).get("name") for c in calls}
    if not calls:
        problems.append("no_tool_calls")
        detail["content_preview"] = content[:300]
    elif case.expect and case.expect_all and not set(case.expect) <= got:
        problems.append(f"missing_expected:all_of({sorted(set(case.expect) - got)})")
    elif case.expect and not case.expect_all and not got & set(case.expect):
        problems.append(f"missing_expected:any_of({case.expect})")
    for name, want in case.checks.items():
        args = [
            r.get("args")
            for r in detail["calls"]
            if r["name"] == name and r.get("args") is not None
        ]
        if not args:
            continue  # reported above as a missing call
        for path, value in want.items():
            if not any(
                same_typed(lookup(x, path)[1], value)
                for x in args
                if lookup(x, path)[0]
            ):
                seen = [lookup(x, path)[1] for x in args if lookup(x, path)[0]]
                kind = (
                    "typing"
                    if any(str(s).lower() == str(value).lower() for s in seen)
                    else "value"
                )
                problems.append(f"{kind}:{name}.{path}={seen[:1]!r}, want {value!r}")
    return problems, detail


CORRUPTION = ("unknown_tool", "bad_json", "leak_in_arg")
MAX_PREP = 2  # preparatory turns a case with ``prep`` answers before the expected call must come


def kind(problem: str) -> str:
    """A problem's kind without its turn ("followup:", "prep1:"): "typing:x" and "prep1:typing:x" are "typing"."""

    head, _, rest = problem.partition(":")
    return rest.split(":")[0] if head == "followup" or re.fullmatch(r"prep\d+", head) else head


def mode_fields(mode: str) -> dict:
    if mode == "off":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if mode == "adaptive":
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": mode},
        "reasoning_effort": mode,
    }


def history(msg: dict) -> dict:
    """The assistant turn as a coding client sends it back: content, its tool_calls and reasoning_content."""

    hist = {
        "role": "assistant",
        "content": msg.get("content") or "",
        "tool_calls": msg["tool_calls"],
    }
    if msg.get("reasoning_content"):
        hist["reasoning_content"] = msg["reasoning_content"]
    return hist


def is_prep(resp: dict, case: Case) -> bool:
    """A reply of only the case's preparatory calls (none of them an expected one)."""

    calls = (resp["choices"][0].get("message") or {}).get("tool_calls") or []
    names = {(c.get("function") or {}).get("name") for c in calls}
    return bool(calls) and names <= set(case.prep) and not names & set(case.expect)


def run_case(a, api: Api, case: Case) -> dict:
    tools = TYPED if a.toolset == "typed" else case.tools
    payload = {
        "model": a.model,
        "max_tokens": a.max_tokens,
        "tools": tools,
        "messages": case.messages(),
    }
    payload.update(mode_fields(a.mode))
    if a.temperature >= 0:
        payload["temperature"] = a.temperature
    if a.tool_choice != "auto":
        payload["tool_choice"] = a.tool_choice
    t0 = time.time()
    prep_problems, prep_turns = [], []
    try:
        resp = api.chat(payload, a.timeout, a.stream)
        while case.prep and len(prep_turns) < MAX_PREP and is_prep(resp, case):
            p, d = validate(resp, case, tools, turn=f"prep{len(prep_turns) + 1}:")
            prep_problems += p
            prep_turns.append(d)
            if any(kind(x) in CORRUPTION for x in p):
                break
            msg = resp["choices"][0]["message"]
            answers = [
                _result(c.get("id", ""), case.prep[c["function"]["name"]])
                for c in msg["tool_calls"]
            ]
            payload = dict(
                payload, messages=payload["messages"] + [history(msg)] + answers
            )
            resp = api.chat(payload, a.timeout, a.stream)
    except ea.Refused as exc:
        on = ea.enabled(a.engine_api_tools)
        return {
            "case": case.name,
            "ok": False if on else None,
            "problems": [
                f"refused (ENGINE_API_TOOLS=1): {exc.message}"
                if on
                else f"{ea.SKIP}: {exc.message}"
            ],
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Unsupported as exc:
        return {
            "case": case.name,
            "ok": None,
            "problems": [f"unsupported:{exc}"],
            "elapsed_s": 0.0,
        }
    except Exception as exc:  # noqa: BLE001 - a failed request fails the case
        return {
            "case": case.name,
            "ok": False,
            "problems": [api.mask(f"request_error:{type(exc).__name__}: {exc}")[:300]],
            "elapsed_s": round(time.time() - t0, 1),
        }
    problems, detail = validate(resp, case, tools)
    problems = prep_problems + problems
    if prep_turns:
        detail["prep"] = prep_turns
    msg = resp["choices"][0].get("message") or {}
    if (
        case.results
        and msg.get("tool_calls")
        and not any(kind(p) in CORRUPTION for p in problems)
    ):
        follow = [history(msg)] + [
            _result(c.get("id", ""), case.results.get(c["function"]["name"], "ok"))
            for c in msg["tool_calls"]
        ]
        body = dict(payload, messages=payload["messages"] + follow)
        try:
            resp2 = api.chat(body, a.timeout, a.stream)
            p2, d2 = validate(resp2, case, tools, turn="followup:")
            content2 = resp2["choices"][0]["message"].get("content") or ""
            if not d2["n_calls"] and not content2.strip():
                p2.append("followup:empty_reply")
            elif (
                case.final and not d2["n_calls"] and not re.search(case.final, content2)
            ):
                p2.append(f"followup:answer_missing:/{case.final}/")
            problems += p2
            detail["followup"] = {**d2, "content_preview": content2[:300]}
        except Exception as exc:  # noqa: BLE001
            problems.append(
                api.mask(f"followup:request_error:{type(exc).__name__}: {exc}")[:300]
            )
    corruption = any(kind(p) in CORRUPTION for p in problems)
    return {
        "case": case.name,
        "ok": not problems,
        "direct": not problems and not prep_turns,
        "problems": problems,
        "elapsed_s": round(time.time() - t0, 1),
        "corruption": corruption,
        "leak": any("leaked_markup" in p for p in problems),
        "typing": any(kind(p) == "typing" for p in problems),
        **detail,
    }


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
    ap.add_argument(
        "--mode",
        default="off",
        choices=("off", "adaptive", "low", "high"),
        help="thinking off, adaptive (the template's own), or adaptive at an effort",
    )
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; <0 omits it (the server default)",
    )
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument(
        "--stream", action="store_true", help="use SSE streaming (as coding clients do)"
    )
    ap.add_argument(
        "--toolset",
        choices=["case", "typed"],
        default="case",
        help="typed: offer every tool each time",
    )
    ap.add_argument("--tool-choice", choices=["auto", "required"], default="auto")
    ea.add_flag(ap)
    a = ap.parse_args(argv)
    # a bad ENGINE_API_TOOLS stops here, before any request
    ea.enabled(a.engine_api_tools)
    api = Api(a.base, read_key(a.api_key_file))
    cases = [c for c in CASES if not a.only or c.name in a.only.split(",")]
    if not cases:
        ap.error(f"--only {a.only!r}: no such case")
    results = []
    for rep in range(a.reps):
        for case in cases:
            res = run_case(a, api, case)
            res["rep"] = rep
            results.append(res)
            mark = {True: "OK  ", False: "FAIL", None: "SKIP"}[res["ok"]]
            line = f"[{rep}] {mark} {res['case']:24s} {res.get('elapsed_s')}s"
            print(
                api.mask(
                    line
                    + (" | " + "; ".join(res["problems"]) if res["problems"] else "")
                ),
                flush=True,
            )
    ran = [r for r in results if r["ok"] is not None]
    n = len(ran)
    passed = sum(1 for r in ran if r["ok"])
    direct = sum(1 for r in ran if r.get("direct"))
    counts = {
        k: sum(1 for r in ran if r.get(k)) for k in ("corruption", "leak", "typing")
    }
    unsupported = len(results) - n
    print(
        f"\nmode {a.mode}, temperature {a.temperature}, {a.reps} reps: {passed}/{n} clean "
        f"({100 * passed / max(n, 1):.1f}%), {direct} without a preparatory step, "
        f"corruption {counts['corruption']}, markup leaks {counts['leak']}, "
        f"typing {counts['typing']}"
        + (f", skipped {unsupported}" if unsupported else ""),
        flush=True,
    )
    report = {
        "config": vars(a),
        "passed": passed,
        "direct": direct,
        "total": n,
        "unsupported": unsupported,
        **counts,
        "results": results,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(api.mask(json.dumps(report, indent=2)))
        print(f"wrote {a.out}")
    return 0 if passed == n else 1  # skipped (refused) cases are reported, not failed


if __name__ == "__main__":
    sys.exit(main())
