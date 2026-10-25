#!/usr/bin/env python3
"""Soak test: mixed traffic against the MiniMax-M3.1 server for --minutes (30), standard library only.

--workers threads (4); a controller moves the number of active ones between 1 and that every 45-120 s (requests
beyond the server's PARALLEL wait in its queue). Each request draws a
kind: code / prose (T = 0 or 0.7), structured (response_format json_schema; a JSON-only prompt instead when the server
refuses response_format, found by one probe at the start), tool (tools + tool_choice auto and a prompt that asks for
the call: the engine refuses "required" for MiniMax), thinking (enable_thinking, effort low), long (an 8K-48K-token
document + a question), nonstream. Streamed requests are cancelled at a random chunk ~15% of the time (the client
closes the socket); ~5% are "disconnects" (non-streamed, a 2-5 s client timeout mid-reply). Both are intentional and
not errors.

Checked: errors (HTTP >= 400, exceptions, empty replies of normal requests) = 0; MiniMax markup in a reply's content
(the server's parser or template failed) = 0; content checks (structured parses, tool calls present) counted
separately; latency (first token, total) per kind; /health every 10 s; at the end inflight back to 0 within --drain-s
(no slot leak; reported "unknown" when /health has no inflight field) and 17*23 = 391.

  python3 bench/soak.py --minutes 30 --out soak.json [--base URL] [--model M] [--api-key-file F]
Exit 0 on PASS (0 errors, 0 leaks, drained or unknown, 391, no fatal).

``--kinds session``: tool-using sessions on top of the mix above. ``session`` is a coding assistant's session that
grows a turn at a time (40 tools, a tool call and its result appended each turn, 1-4K tokens a turn) up to
``--session-max-k`` thousand
tokens (120), then starts over; each worker keeps its own session, so up to four grow side by side. The controller
also idles every worker for 30-90 s with probability ``--idle-p`` (0.15), as a client between turns does.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"
KINDS = [
    ("code", 22),
    ("prose", 18),
    ("structured", 14),
    ("tool", 14),
    ("thinking", 10),
    ("long", 10),
    ("nonstream", 7),
    ("disconnect", 5),
]
KINDS_SESSION = [
    ("session", 34),
    ("code", 13),
    ("prose", 11),
    ("structured", 9),
    ("tool", 10),
    ("thinking", 7),
    ("long", 8),
    ("nonstream", 5),
    ("disconnect", 3),
]
CODE = [
    "Write a Python function that merges two sorted lists, with a docstring and two doctests.",
    "Implement an LRU cache class in Python with get / put in O(1). Code only.",
    "Write a bash script that rotates log files older than 7 days into a tar.gz archive.",
]
PROSE = [
    "Write a short essay about the history of lighthouses.",
    "Describe a rainy morning in a harbor town.",
    "Explain to a ten-year-old why the sky is blue, in two paragraphs.",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["title", "year", "tags"],
}
TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
SESSION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Workspace operation {i}: reads, edits or searches files. "
            * 3,
            "parameters": {
                "type": "object",
                "properties": {
                    f"arg{j}": {
                        "type": "string",
                        "description": "path, pattern or text",
                    }
                    for j in range(4)
                },
                "required": ["arg0"],
            },
        },
    }
    for i in range(40)
]
WORDS = [
    "amber",
    "basin",
    "cobalt",
    "drift",
    "ember",
    "fable",
    "grove",
    "harbor",
    "island",
    "jetty",
    "kelp",
    "lumen",
    "marsh",
    "north",
    "orbit",
    "pine",
    "quay",
    "ridge",
]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


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
    """HTTP with the key as a header; ``mask`` hides the key in any text that is printed or saved."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def open(self, path: str, body: dict | None, timeout: float):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(
            urllib.request.Request(self.base + path, data, headers), timeout=timeout
        )

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def _text(rng: random.Random, tokens: int) -> str:
    return " ".join(
        f"{rng.choice(WORDS)}_{rng.randrange(10000)}"
        for _ in range(max(1, tokens // 3))
    )


def session_body(sess: dict, rng: random.Random, max_k: int) -> dict:
    """The next turn of a worker's session (``sess``: its messages, grown here)."""

    msgs = sess.get("messages")
    if not msgs or sess.get("chars", 0) > max_k * 1000 * 4:
        msgs = sess["messages"] = [
            {
                "role": "system",
                "content": "You are a coding assistant working in a repository. "
                + _text(rng, 2000),
            },
            {
                "role": "user",
                "content": "Fix the failing tests. "
                + _text(rng, rng.randrange(200, 2000)),
            },
        ]
        sess["chars"], sess["turn"] = sum(len(m["content"]) for m in msgs), 0
    else:
        sess["turn"] += 1
        cid = f"call_{sess['turn']}_{rng.randrange(1 << 30)}"
        result = _text(rng, rng.randrange(1000, 4000))
        args = json.dumps({"arg0": f"src/{rng.choice(WORDS)}.py"})
        msgs += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": f"tool_{rng.randrange(40)}",
                            "arguments": args,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": cid, "content": result},
        ]
        sess["chars"] += len(result) + 120
    return {
        "messages": list(msgs),
        "tools": SESSION_TOOLS,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
        "reasoning_effort": "low",
        "max_tokens": 400,
        "temperature": 0.6,
    }


def body_for(
    kind: str,
    model: str,
    rng: random.Random,
    sess: dict | None = None,
    session_max_k: int = 120,
    response_format: bool = True,
) -> tuple[dict, bool]:
    """(request body, streamed)."""

    t = rng.choice([0.0, 0.7])
    if kind == "session":
        b = session_body(sess if sess is not None else {}, rng, session_max_k)
    elif kind == "code":
        b = dict(
            OFF,
            messages=[{"role": "user", "content": rng.choice(CODE)}],
            max_tokens=384,
            temperature=t,
        )
    elif kind in ("prose", "nonstream", "disconnect"):
        b = dict(
            OFF,
            messages=[{"role": "user", "content": rng.choice(PROSE)}],
            max_tokens=384,
            temperature=t,
        )
    elif kind == "structured":
        ask = "Describe a classic film as JSON."
        b = dict(OFF, max_tokens=256, temperature=0)
        if response_format:
            b["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "film", "schema": SCHEMA},
            }
        else:
            ask += ' Reply with only a JSON object {"title": string, "year": integer, "tags": [strings]}, no fences.'
        b["messages"] = [{"role": "user", "content": ask}]
    elif kind == "tool":
        city = rng.choice(["Oslo", "Lima", "Kyiv"])
        ask = f"What is the weather in {city}? Call the get_weather tool."
        b = dict(
            OFF,
            messages=[{"role": "user", "content": ask}],
            tools=TOOL,
            tool_choice="auto",
            max_tokens=256,
        )
        b["temperature"] = 0
    elif kind == "thinking":
        b = {
            "messages": [
                {
                    "role": "user",
                    "content": "How many weekdays are there in March 2027? Think it through.",
                }
            ],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
            "reasoning_effort": "low",
            "max_tokens": 2048,
            "temperature": t,
        }
    else:  # long: an 8K-48K-token document, one question
        n = rng.choice([8, 16, 32, 48]) * 1024
        doc = " ".join(
            f"Entry {i}: the {rng.choice(WORDS)} {rng.choice(WORDS)} met the {rng.choice(WORDS)}."
            for i in range(n // 11)
        )
        ask = doc + "\n\nHow many entries are there? One number."
        b = dict(
            OFF,
            messages=[{"role": "user", "content": ask}],
            max_tokens=32,
            temperature=0,
        )
    b["model"] = model
    return b, kind not in ("nonstream", "disconnect")


class Soak:
    def __init__(self, a, api: Api) -> None:
        self.a, self.api, self.lock, self.stop = (
            a,
            api,
            threading.Lock(),
            threading.Event(),
        )
        self.active = 1
        self.rec: list[dict] = []
        self.health: list[dict] = []
        self.kinds = KINDS_SESSION if a.kinds == "session" else KINDS
        self.sessions = [{} for _ in range(a.workers)]
        self.response_format = True
        self.response_format_note = ""

    def probe_response_format(self) -> None:
        """One structured request: a refusal switches the structured kind to a JSON-only prompt."""

        body, _ = body_for("structured", self.a.model, random.Random(0))
        try:
            with self.api.open("/v1/chat/completions", body, self.a.timeout) as r:
                r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 422, 501):
                self.response_format = False
                self.response_format_note = self.api.mask(
                    exc.read().decode(errors="replace")
                )[:200]

    def one(self, kind: str, rng: random.Random, worker: int = 0) -> dict:
        body, streamed = body_for(
            kind,
            self.a.model,
            rng,
            self.sessions[worker],
            self.a.session_max_k,
            self.response_format,
        )
        r = {
            "kind": kind,
            "t": round(time.time() - self.t0, 1),
            "error": None,
            "check": None,
            "cancel": False,
        }
        r["leak"] = False
        cancel_at = (
            rng.randint(1, 20) if streamed and rng.random() < self.a.cancel_p else None
        )
        timeout = rng.uniform(2, 5) if kind == "disconnect" else self.a.timeout
        if streamed:
            body["stream"] = True
        t0, text, content, calls, chunks = time.time(), "", "", 0, 0
        try:
            with self.api.open("/v1/chat/completions", body, timeout) as resp:
                if not streamed:
                    m = json.loads(self.api.mask(resp.read().decode()))["choices"][0][
                        "message"
                    ]
                    content = m.get("content") or ""
                    text = content + (m.get("reasoning_content") or "")
                    calls = len(m.get("tool_calls") or [])
                else:
                    for raw in resp:
                        line = self.api.mask(raw.decode()).strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        c = json.loads(line[5:])
                        if c.get(
                            "error"
                        ):  # an in-stream error event after the 200 headers
                            err = c["error"]
                            msg = err.get("message") if isinstance(err, dict) else err
                            r["error"] = ("stream error: " + str(msg))[:200]
                            break
                        for ch in c.get("choices") or []:
                            d = ch.get("delta") or {}
                            if r.get("ttft_s") is None and (
                                d.get("content")
                                or d.get("reasoning_content")
                                or d.get("tool_calls")
                            ):
                                r["ttft_s"] = round(time.time() - t0, 3)
                            content += d.get("content") or ""
                            text += (d.get("content") or "") + (
                                d.get("reasoning_content") or ""
                            )
                            calls += len(d.get("tool_calls") or [])
                        chunks += 1
                        if cancel_at is not None and chunks >= cancel_at:
                            r["cancel"] = True
                            break  # leaving the with-block closes the socket mid-stream
        except urllib.error.HTTPError as exc:
            r["error"] = (
                f"HTTP {exc.code}: "
                + self.api.mask(exc.read().decode(errors="replace"))[:200]
            )
        except TimeoutError as exc:
            if kind == "disconnect":
                r["cancel"] = True
            else:
                r["error"] = f"timeout: {exc}"
        except (OSError, http.client.HTTPException, ValueError) as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"[:200]
        if r["error"]:
            r["error"] = self.api.mask(r["error"])
        r["total_s"] = round(time.time() - t0, 3)
        if LEAK.search(
            content
        ):  # a cancelled stream's partial content counts too: markup never reaches a client
            r["leak"] = True
            r["check"] = self.api.mask(f"markup in content: {content[:80]!r}")
        if r["error"] is None and not r["cancel"] and not r["leak"]:
            if kind == "structured":
                try:
                    json.loads(
                        content.strip().removeprefix("```json").removesuffix("```")
                    )
                except ValueError:
                    r["check"] = self.api.mask(
                        f"structured reply not JSON: {content[:80]!r}"
                    )
            elif kind == "tool" and not calls:
                r["check"] = "no tool call"
            elif (
                kind != "tool"
                and not text.strip()
                and not (kind == "session" and calls)
            ):
                r["error"] = "empty reply"
        return r

    def worker(self, i: int) -> None:
        rng = random.Random(self.a.seed * 100 + i)
        names, weights = zip(*self.kinds)
        while not self.stop.is_set():
            if i >= self.active:
                time.sleep(1)
                continue
            r = self.one(rng.choices(names, weights)[0], rng, i)
            with self.lock:
                self.rec.append(r)
            if r["error"] or r["leak"]:
                print(
                    f"[soak] {'ERROR' if r['error'] else 'LEAK'} {r['kind']} at {r['t']} s: "
                    f"{r['error'] or r['check']}",
                    flush=True,
                )

    def control(self) -> None:
        rng = random.Random(self.a.seed)
        while not self.stop.wait(rng.uniform(45, 120)):
            idle = rng.random() < self.a.idle_p
            self.active = 0 if idle else rng.randint(1, self.a.workers)
            if idle:  # every worker idles for 30-90 s, then 1..workers again
                if self.stop.wait(rng.uniform(30, 90)):
                    break
                self.active = rng.randint(1, self.a.workers)
            print(
                f"[soak] {round(time.time() - self.t0)} s: {self.active} concurrent; {len(self.rec)} done",
                flush=True,
            )

    def poll(self) -> None:
        while not self.stop.wait(10):
            self.health.append(dict(health(self.api), t=round(time.time() - self.t0)))

    def run(self) -> dict:
        self.t0 = time.time()
        self.probe_response_format()
        th = [
            threading.Thread(target=self.worker, args=(i,), daemon=True)
            for i in range(self.a.workers)
        ]
        th += [
            threading.Thread(target=self.control, daemon=True),
            threading.Thread(target=self.poll, daemon=True),
        ]
        for t in th:
            t.start()
        time.sleep(self.a.minutes * 60)
        self.stop.set()
        for t in th[: self.a.workers]:
            t.join(self.a.timeout + 30)
        drained, t1 = False, time.time()
        if "inflight" not in health(self.api):
            drained = None  # this server's /health does not say
        while drained is False and time.time() - t1 < self.a.drain_s:
            if health(self.api).get("inflight") == 0:
                drained = True
                break
            time.sleep(2)
        return self.report(drained, round(time.time() - t1, 1))

    def report(self, drained: bool | None, drain_s: float) -> dict:
        rec = self.rec
        by: dict[str, dict] = {}

        def pct(v: list, p: float):
            return v[min(len(v) - 1, int(p * len(v)))] if v else None

        for k, _ in self.kinds:
            xs = [r for r in rec if r["kind"] == k]
            ok = [r for r in xs if not r["error"] and not r["cancel"]]
            tt = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)
            tot = sorted(r["total_s"] for r in ok)
            by[k] = {
                "n": len(xs),
                "ok": len(ok),
                "cancelled": sum(r["cancel"] for r in xs),
                "errors": sum(bool(r["error"]) for r in xs),
                "leaks": sum(r["leak"] for r in xs),
                "check_fails": sum(bool(r["check"]) for r in xs),
                "ttft_p50": pct(tt, 0.5),
                "ttft_p95": pct(tt, 0.95),
                "total_p50": statistics.median(tot) if tot else None,
                "total_p95": pct(tot, 0.95),
            }
        ans = ""
        try:
            body = dict(
                OFF,
                model=self.a.model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is 17*23? Answer with the number only.",
                    }
                ],
                max_tokens=64,
                temperature=0,
            )
            with self.api.open("/v1/chat/completions", body, 300) as r:
                ans = (
                    json.loads(self.api.mask(r.read().decode()))["choices"][0][
                        "message"
                    ].get("content")
                    or ""
                )
        except Exception as exc:  # noqa: BLE001
            ans = self.api.mask(f"ERROR {exc}")
        errors = sum(bool(r["error"]) for r in rec)
        leaks = sum(r["leak"] for r in rec)
        fatal = any(h.get("fatal") for h in self.health)
        out = {
            "minutes": self.a.minutes,
            "requests": len(rec),
            "errors": errors,
            "leaks": leaks,
            "check_fails": sum(bool(r["check"]) for r in rec),
            "cancelled": sum(r["cancel"] for r in rec),
            "response_format": "used"
            if self.response_format
            else "refused: " + self.response_format_note,
            "drained": "unknown" if drained is None else drained,
            "drain_s": drain_s,
            "answer_17x23": ans[:40],
            "fatal_seen": fatal,
            "stalled_seen": any(h.get("stalled") for h in self.health),
            "inflight_max": max(
                (h.get("inflight") or 0 for h in self.health), default=None
            ),
            "by_kind": by,
            "health_end": health(self.api),
            "error_samples": [r for r in rec if r["error"]][:20],
            "check_samples": [r for r in rec if r["check"]][:10],
        }
        out["pass"] = (
            errors == 0
            and leaks == 0
            and drained is not False
            and "391" in ans
            and not fatal
        )
        return out


def health(api: Api) -> dict:
    try:
        with api.open("/health", None, 10) as r:
            h = json.loads(r.read())
        out = {
            "fatal": h.get("fatal"),
            "stalled": h.get("stalled"),
            "running": h.get("requests_running"),
        }
        if "inflight" in h:
            inf = h["inflight"]
            out["inflight"] = len(inf) if isinstance(inf, (list, dict)) else inf
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": api.mask(str(exc))[:120]}


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
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--cancel-p", type=float, default=0.15)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--drain-s", type=float, default=120)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="client threads (1..this many active at once)",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--kinds",
        choices=("mix", "session"),
        default="mix",
        help="session: + growing tool sessions, idle spells",
    )
    ap.add_argument(
        "--session-max-k",
        type=int,
        default=120,
        help="sessions start over past this many K tokens",
    )
    ap.add_argument(
        "--idle-p",
        type=float,
        default=None,
        help="chance a control step idles all workers (session: 0.15)",
    )
    a = ap.parse_args(argv)
    if a.idle_p is None:
        a.idle_p = 0.15 if a.kinds == "session" else 0.0
    api = Api(a.base, read_key(a.api_key_file))
    out = Soak(a, api).run()
    brief = {
        k: v
        for k, v in out.items()
        if k not in ("by_kind", "error_samples", "check_samples")
    }
    print(api.mask(json.dumps(brief)), flush=True)
    for k, v in out["by_kind"].items():
        print(f"[soak] {k}: {v}", flush=True)
    if a.out:
        Path(a.out).write_text(api.mask(json.dumps(out, indent=1)))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
