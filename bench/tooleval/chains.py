#!/usr/bin/env python3
"""Multi-step tool-call chains against the MiniMax-M3.1 endpoint, simulated tools, standard library only.

The shape of tool-eval-bench's category C (multi-step chains) and spark-bench's agentic domain, without an install:
each scenario needs several dependent calls (a result feeds the next call's arguments), the history is sent back the
way coding clients do (assistant ``tool_calls`` + ``reasoning_content`` as returned, ``tool`` messages), and the final
answer is checked. A scenario scores 2 points: 1 for the required calls with the right arguments in a valid order, 1
for the final answer. Malformed calls (unknown tool, arguments that are not a JSON object of the declared keys, MiniMax
markup in a value) and raw markup in a reply's content (the server's parser or template missed a call or a think block)
are counted and fail the scenario's call point.

  python3 bench/tooleval/chains.py --base http://127.0.0.1:8000 --model MiniMax-M3.1-Flash-EXL3 --mode off \\
      --out R/chains-off.json [--reps 1] [--temperature 0] [--api-key-file F] [--only geo-timezone,...]
Modes: off (enable_thinking false: the server starts the reply after </mm:think>), adaptive (enable_thinking true and
no effort: the checkpoint template's own and only mode), low / high (adaptive with an <effort>). --temperature below 0
leaves it out, so the server samples with generation_config.json's settings (MiniMax's 1.0 / top_p 0.95).
The simulated files keep what write_file writes, so a model that reads its edit back sees the edit.
Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE (scripts/config.sh). The key is read from the file and sent
as a header; it is masked in everything printed or saved. Exit 0 when the score is >= --min-score (default 10 of 12).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# MiniMax's namespace token, its invoke / call tags and its think tags: never in a value or in content
LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"


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
    """JSON over HTTP with the key as a header; ``mask`` hides the key in any text that is printed or saved."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def post(self, path: str, body: dict, timeout: float) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), headers
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(
                self.mask(r.read().decode())
            )  # a reply never holds the key

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


S = {"type": "string"}
N = {"type": "number"}
TOOLS = {
    "get_weather": fn(
        "get_weather", "Current weather for a city", {"city": S}, ["city"]
    ),
    "search_flights": fn(
        "search_flights",
        "Flights between two cities on a date (YYYY-MM-DD)",
        {"origin": S, "destination": S, "date": S},
        ["origin", "destination", "date"],
    ),
    "find_user": fn(
        "find_user",
        "Look up a customer by email; returns the user id",
        {"email": S},
        ["email"],
    ),
    "get_orders": fn(
        "get_orders", "A user's orders with their amounts", {"user_id": S}, ["user_id"]
    ),
    "list_files": fn(
        "list_files", "List the files of a directory", {"path": S}, ["path"]
    ),
    "read_file": fn("read_file", "Read a file", {"path": S}, ["path"]),
    "write_file": fn(
        "write_file",
        "Overwrite a file with new content",
        {"path": S, "content": S},
        ["path", "content"],
    ),
    "calc": fn(
        "calc",
        "Evaluate an arithmetic expression (+ - * / ** and parentheses)",
        {"expr": S},
        ["expr"],
    ),
    "get_stock": fn(
        "get_stock", "Latest price of a stock ticker", {"symbol": S}, ["symbol"]
    ),
    "geocode": fn("geocode", "Coordinates of a place name", {"place": S}, ["place"]),
    "get_timezone": fn(
        "get_timezone",
        "IANA time zone at coordinates",
        {"lat": N, "lon": N},
        ["lat", "lon"],
    ),
}


# the simulated project (/srv/app) by file name; each scenario run starts from a copy
FILES = {
    "README.md": "(binary or empty)",
    "config.yaml": "port: 8080\nworkers: 4\nlog_level: info\n",
    "main.py": "(binary or empty)",
}


def sim(name: str, args: dict, files: dict | None = None) -> str:
    """The simulated tools (deterministic); ``files`` is the run's project, which write_file changes."""

    files = dict(FILES) if files is None else files

    a = {k: (str(v).strip() if isinstance(v, str) else v) for k, v in args.items()}
    if name == "get_weather":
        c = str(a.get("city", "")).lower()
        if c == "pariss":
            return json.dumps({"error": "city not found: Pariss. Did you mean Paris?"})
        return json.dumps(
            {
                "city": a.get("city"),
                "sky": {"oslo": "sunny", "paris": "rain"}.get(c, "cloudy"),
                "temp_c": {"oslo": 12, "paris": 15}.get(c, 18),
            }
        )
    if name == "search_flights":
        if (
            str(a.get("origin", "")).lower() == "oslo"
            and str(a.get("destination", "")).lower() == "rome"
        ):
            return json.dumps(
                {
                    "flights": [
                        {"number": "SK4711", "dep": "08:15"},
                        {"number": "AZ609", "dep": "17:40"},
                    ]
                }
            )
        return json.dumps({"flights": []})
    if name == "find_user":
        found = str(a.get("email", "")).lower() == "ana@example.com"
        return json.dumps({"user_id": "u-8842"} if found else {"error": "no such user"})
    if name == "get_orders":
        if a.get("user_id") == "u-8842":
            return json.dumps(
                {
                    "orders": [
                        {"id": "o1", "amount": 120.50},
                        {"id": "o2", "amount": 79.25},
                        {"id": "o3", "amount": 300.00},
                    ]
                }
            )
        return json.dumps({"error": "unknown user id"})
    if name == "list_files":
        return json.dumps({"files": sorted(files)})
    if name == "read_file":
        return files.get(os.path.basename(str(a.get("path", ""))), "(binary or empty)")
    if name == "write_file":
        content = str(args.get("content", ""))
        files[os.path.basename(str(a.get("path", "")))] = content
        return json.dumps({"ok": True, "bytes": len(content)})
    if name == "calc":
        expr = str(a.get("expr", ""))
        if not re.fullmatch(r"[0-9.+\-*/() eE]+", expr):
            return json.dumps({"error": "only numbers and + - * / ** ( ) are allowed"})
        try:
            return json.dumps(
                {"result": round(eval(expr, {"__builtins__": {}}), 6)}
            )  # digits and operators only
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})
    if name == "get_stock":
        price = {"ACME": 187.4, "GLOBEX": 192.1}.get(str(a.get("symbol", "")).upper())
        return json.dumps({"symbol": a.get("symbol"), "price": price})
    if name == "geocode":
        if "tokyo" in str(a.get("place", "")).lower():
            return json.dumps({"place": a.get("place"), "lat": 35.68, "lon": 139.69})
        return json.dumps({"error": "not found"})
    if name == "get_timezone":
        try:
            lat, lon = float(a.get("lat", 0)), float(a.get("lon", 0))
        except (TypeError, ValueError):
            return json.dumps({"error": "lat and lon must be numbers"})
        near = abs(lat - 35.68) < 0.5 and abs(lon - 139.69) < 0.5
        return json.dumps({"tz": "Asia/Tokyo"} if near else {"tz": "UTC"})
    return json.dumps({"error": f"unknown tool {name}"})


def has(calls, name, **want) -> bool:
    return any(
        c[0] == name
        and all(
            str(c[1].get(k, "")).strip().lower() == str(v).lower()
            for k, v in want.items()
        )
        for c in calls
    )


def before(calls, a: str, b: str) -> bool:
    names = [c[0] for c in calls]
    return a in names and b in names and names.index(a) < names.index(b)


SCENARIOS = [
    (
        "weather-then-flight",
        ["get_weather", "search_flights"],
        (
            "If the weather in Oslo is sunny right now, find me a flight from Oslo to Rome on 2026-11-02 and tell me the "
            "earliest flight number. Use the tools."
        ),
        lambda c: (
            has(c, "get_weather", city="oslo")
            and has(
                c,
                "search_flights",
                origin="oslo",
                destination="rome",
                date="2026-11-02",
            )
            and before(c, "get_weather", "search_flights")
        ),
        lambda t: "SK4711" in t,
    ),
    (
        "user-orders-total",
        ["find_user", "get_orders", "calc"],
        "What is the total amount of all orders of the customer ana@example.com? Look it up with the tools.",
        lambda c: (
            has(c, "find_user", email="ana@example.com")
            and has(c, "get_orders", user_id="u-8842")
            and before(c, "find_user", "get_orders")
        ),
        lambda t: "499.75" in t.replace(",", ""),
    ),
    (
        "edit-config",
        ["list_files", "read_file", "write_file"],
        (
            "In the project directory /srv/app, find the YAML config, read it, and change the number of workers to 8, "
            "keeping every other line. Use the tools, then confirm."
        ),
        lambda c: (
            has(c, "read_file")
            and any(
                x[0] == "write_file"
                and "workers: 8" in str(x[1].get("content", ""))
                and "port: 8080" in str(x[1].get("content", ""))
                for x in c
            )
            and before(c, "read_file", "write_file")
        ),
        lambda t: "8" in t,
    ),
    (
        "compound-interest",
        ["calc"],
        (
            "Using the calc tool (not mental arithmetic), compute 10000 * (1 + 0.05) ** 3, then subtract 1000 from that "
            "result with a second calc call. Report the final number rounded to 2 decimals."
        ),
        lambda c: sum(x[0] == "calc" for x in c) >= 2,
        lambda t: "10576.25" in t.replace(",", ""),
    ),
    (
        "retry-after-error",
        ["get_weather"],
        (
            "What's the weather in Pariss? Call the tool with the city exactly as I wrote it first; if the tool "
            "rejects it, correct the city and call it again."
        ),
        lambda c: has(c, "get_weather", city="paris"),
        lambda t: "rain" in t.lower() or "15" in t,
    ),
    (
        "geo-timezone",
        ["geocode", "get_timezone"],
        "Which IANA time zone is Tokyo in? Find its coordinates with geocode, then ask get_timezone.",
        lambda c: (
            has(c, "geocode")
            and has(c, "get_timezone")
            and before(c, "geocode", "get_timezone")
        ),
        lambda t: "Asia/Tokyo" in t,
    ),
]


def mode_fields(mode: str) -> dict:
    if mode == "off":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if mode == "adaptive":
        return {"chat_template_kwargs": {"enable_thinking": True}}
    return {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": mode},
        "reasoning_effort": mode,
    }


def run(a, api: Api, sc) -> dict:
    name, tool_names, prompt, calls_ok, answer_ok = sc
    extra = (
        TOOLS["list_files"] if "list_files" not in tool_names else TOOLS["calc"]
    )  # one distractor
    tools = [TOOLS[t] for t in tool_names] + [extra]
    msgs = [
        {
            "role": "system",
            "content": "You are an assistant. Use the provided tools to do the task; call one tool at a "
            "time when a later call needs an earlier result.",
        },
        {"role": "user", "content": prompt},
    ]
    calls, malformed, final, t0, turns, leaks = [], [], "", time.time(), 0, 0
    files = dict(FILES)
    try:
        for turns in range(1, a.max_turns + 1):
            body = {
                "model": a.model,
                "messages": msgs,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": a.max_tokens,
            }
            if a.temperature >= 0:
                body["temperature"] = a.temperature
            r = api.post(
                "/v1/chat/completions", dict(body, **mode_fields(a.mode)), a.timeout
            )
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            if LEAK.search(content):
                leaks += 1
                malformed.append(f"markup in content: {content[:80]!r}")
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = content
                break
            hist = {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tcs,
            }
            if msg.get("reasoning_content"):
                hist["reasoning_content"] = msg["reasoning_content"]
            msgs.append(hist)
            for tc in tcs:
                f = tc.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                    assert isinstance(args, dict)
                except (ValueError, AssertionError):
                    args = {}
                    malformed.append(
                        f"{f.get('name')}: arguments {str(f.get('arguments'))[:80]!r}"
                    )
                req = (
                    TOOLS.get(f.get("name"), {})
                    .get("function", {})
                    .get("parameters", {})
                    .get("required")
                )
                if req is None:
                    malformed.append(f"unknown tool {f.get('name')!r}")
                elif set(req) - set(args):
                    malformed.append(
                        f"{f.get('name')}: missing {sorted(set(req) - set(args))}"
                    )
                if any(isinstance(v, str) and LEAK.search(v) for v in args.values()):
                    malformed.append(f"{f.get('name')}: markup in an argument")
                calls.append((f.get("name"), args))
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": sim(f.get("name"), args, files),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - a failed request fails the scenario
        malformed.append(api.mask(f"request failed: {type(exc).__name__}: {exc}")[:200])
    p_calls = bool(calls_ok(calls)) and not malformed
    p_answer = bool(final) and bool(answer_ok(final))
    return {
        "scenario": name,
        "points": int(p_calls) + int(p_answer),
        "calls_ok": p_calls,
        "answer_ok": p_answer,
        "turns": turns,
        "calls": [[c[0], c[1]] for c in calls],
        "malformed": malformed,
        "leaks": leaks,
        "final": final[:300],
        "seconds": round(time.time() - t0, 1),
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
    ap.add_argument("--mode", default="off", choices=("off", "adaptive", "low", "high"))
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; <0 omits it (the server default)",
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", default="", help="comma list of scenario names")
    ap.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="pass line (points a rep; default 10 of 12)",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    api = Api(a.base, read_key(a.api_key_file))
    scenarios = [s for s in SCENARIOS if not a.only or s[0] in a.only.split(",")]
    if not scenarios:
        ap.error(f"--only {a.only!r}: no such scenario")
    top = 2 * len(scenarios)
    min_score = (
        a.min_score
        if a.min_score is not None
        else (10 if len(scenarios) == len(SCENARIOS) else top)
    )
    reps = []
    for rep in range(a.reps):
        res = [run(a, api, sc) for sc in scenarios]
        for r in res:
            line = (
                f"[chains {a.mode} rep {rep}] {r['scenario']}: {r['points']}/2 (calls {r['calls_ok']}, answer "
                f"{r['answer_ok']}, {len(r['calls'])} calls, {r['turns']} turns, {r['seconds']} s)"
            )
            print(
                api.mask(
                    line
                    + (f" malformed: {r['malformed'][:3]}" if r["malformed"] else "")
                ),
                flush=True,
            )
        reps.append(res)
    scores = [sum(r["points"] for r in res) for res in reps]
    out = {
        "mode": a.mode,
        "model": a.model,
        "max_points": top,
        "scores": scores,
        "min_score": min_score,
        "pass": min(scores) >= min_score,
        "malformed": sum(len(r["malformed"]) for res in reps for r in res),
        "leaks": sum(r["leaks"] for res in reps for r in res),
        "reps": reps,
    }
    verdict = "PASS" if out["pass"] else "FAIL"
    print(
        f"[chains {a.mode}] score {scores} of {top} (pass line {min_score}), malformed {out['malformed']}, "
        f"markup leaks {out['leaks']}: {verdict}",
        flush=True,
    )
    if a.out:
        Path(a.out).write_text(api.mask(json.dumps(out, indent=1)))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
