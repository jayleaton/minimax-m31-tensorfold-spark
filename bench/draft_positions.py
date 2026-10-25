#!/usr/bin/env python3
"""Draft acceptance and speed at C1 (one request at a time), drafts on against drafts off (stdlib only).

Greedy requests (temperature 0, a fixed ``max_tokens``) in four classes: ``code``, ``prose``, ``structured`` (JSON,
tables, CSV, YAML) and ``agent`` (tool-call turns with earlier calls and results in the history). The code, repeat and
JSON prompts are scripts/check-drafted.sh's; the agent turns use bench/toolcall.py's system prompt. Each reply's
runtime fields are kept, one JSON line a request:

- ``tensorfold``: ``tokens_per_second`` (decode), ``seconds``, ``drafts`` (whether the request drafted),
  ``token_sha256`` (the reply's token ids hashed: drafted and plain replies must match);
- ``speculative``: ``rounds``, ``drafted``, ``accepted``, ``acceptance_rate``, ``tokens_per_round`` (tau, the bonus
  token included) and ``positions``: ``{"tried": [...], "kept": [...]}`` for the neural draft's rounds, ``tried[j]``
  the rounds that offered draft j+1 with every earlier draft kept, ``kept[j]`` those of them that kept it.

    python3 bench/draft_positions.py run --label dflash2 --out R/dflash2.jsonl [--draft-off] [--classes code,agent] [--reps 1]
    python3 bench/draft_positions.py report R/plain.jsonl R/dflash2.jsonl R/dflash2-off.jsonl [--json R/report.json]

``run --draft-off`` sends ``"draft": false`` with every request (the engine serves it plain on a drafting server) and
checks the reply did not draft. ``report`` takes the drafts-off run first (the baseline: a ``DRAFTS=0`` server's run)
and for every run and class prints the conditional acceptance a_j = kept_j / tried_j, the survival S_j = a_1 ... a_j,
tau, the acceptance rate and the mean tokens a second; then each run against the baseline: the speed ratio, and whether
every reply's token_sha256 equals the baseline's. A difference is a bug of the draft path (the target verifies every
draft): ``report`` exits 1. Speed is reported, never gated. It also prints a depth prior for the engine's DFlash2 drafter
from the measured survival (docs/DFLASH2.md, Depth).

``--state-cmd`` (repeatable; scripts/measure-drafts.sh passes each Spark's ``/run/spark-thermal/state``) is run before
every request: a node that is not ``normal`` is waited for, ``emergency`` stops the run. The limits behind the state
are the device's (``/etc/spark/thermal.env``, spark-thermal-guard); nothing here compares a temperature with a number.
Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE (scripts/config.sh); the key is read from its file, sent as
a header and never printed or saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# toolcall.py beside this file (python3 -I leaves the script's directory off sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolcall as tc

CLASSES = ("code", "prose", "structured", "agent")
# The engine's DFlash2 depth prior (mm3/dflash2, families/minimax_m3/draft_kind.zig dflash2_prior): GLM-5.3's DFlash2
# survival by depth on code, a placeholder until MiniMax's own is measured.
ENGINE_DFLASH2_PRIOR = (0.874, 0.707, 0.56, 0.437, 0.34, 0.27, 0.21)
MIN_TRIED = 30  # rounds a position needs before its measured survival replaces the prior's

FRACTION = "Write a Python module with a Fraction class (add, sub, mul, div, comparison, repr) and tests."
SNIPPET = '''def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]
'''
ARTICLE = (
    "The printing press, developed in the fifteenth century, changed how ideas travelled. Before it, books were "
    "copied by hand, slowly and at great cost, so few people owned any. Movable type let a workshop print hundreds of "
    "identical pages in a day. Prices fell, literacy rose, and scholars in distant cities could read the same text "
    "and argue about it. Within fifty years presses were at work in most of Europe's large towns."
)


def _tools() -> list[dict]:
    def fn(name: str, desc: str, props: dict, req: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req},
            },
        }

    s = {"type": "string"}
    return [
        fn("read_file", "Read a file of the project", {"path": s}, ["path"]),
        fn("run_shell", "Run a shell command in the project", {"command": s}, ["command"]),
        fn(
            "edit_file",
            "Replace text in a file",
            {"path": s, "old": s, "new": s},
            ["path", "old", "new"],
        ),
    ]


def _call(cid: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def _result(cid: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": text}


def prompts() -> list[dict]:
    """Every case: class, id, the request's messages (and tools), its max_tokens."""

    sys_msg = {"role": "system", "content": tc.SYSTEM}
    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    tools = _tools()
    return [
        {"class": "code", "id": "fraction", "messages": user(FRACTION), "max_tokens": 384},
        {"class": "code", "id": "linked-list", "messages": user("Write a Python function that reverses a linked list, with a short test."), "max_tokens": 256},
        {"class": "code", "id": "refactor", "messages": user("Add type hints and a docstring to this function, then write three pytest tests for it:\n\n" + SNIPPET), "max_tokens": 384},
        {"class": "code", "id": "bash", "messages": user("Write a bash script that finds the ten largest files under a folder given as its argument and prints their sizes in MiB."), "max_tokens": 256},
        {"class": "prose", "id": "lighthouse", "messages": user("Explain to a ten-year-old how a lighthouse works and why ships need one, in about 200 words."), "max_tokens": 320},
        {"class": "prose", "id": "summary", "messages": user("Summarize this paragraph in three sentences, then give it a title:\n\n" + ARTICLE), "max_tokens": 200},
        {"class": "prose", "id": "letter", "messages": user("Write a polite letter to a neighbour asking them to keep their dog off the shared garden."), "max_tokens": 320},
        {"class": "prose", "id": "repeat", "messages": user("Repeat this line exactly 12 times, numbered 1 to 12: 'the quick brown fox jumps over the lazy dog'."), "max_tokens": 260},
        {"class": "structured", "id": "json-languages", "messages": user("List 25 programming languages as a JSON array of objects with name, year and paradigm."), "max_tokens": 400},
        {"class": "structured", "id": "table-planets", "messages": user("Give a markdown table of the eight planets with columns: name, order from the sun, diameter in km, number of moons."), "max_tokens": 320},
        {"class": "structured", "id": "csv", "messages": user("Write CSV with a header row for 20 fictional employees: id, name, department, start_date (YYYY-MM-DD), salary."), "max_tokens": 400},
        {"class": "structured", "id": "yaml", "messages": user("Write a YAML configuration for a web service with sections server (host, port, workers), database (url, pool), logging (level, file) and three feature flags."), "max_tokens": 256},
        {
            "class": "agent",
            "id": "first-call",
            "messages": [sys_msg, *user("The tests in tests/test_merge.py fail. Find out why and fix merge.py.")],
            "tools": tools,
            "max_tokens": 256,
        },
        {
            "class": "agent",
            "id": "after-read",
            "messages": [
                sys_msg,
                *user("The tests in tests/test_merge.py fail. Find out why and fix merge.py."),
                _call("c1", "run_shell", {"command": "python3 -m pytest -q tests/test_merge.py"}),
                _result("c1", "F.\nFAILED tests/test_merge.py::test_ties - assert [1, 2, 2] == [1, 2, 2, 3]\n1 failed, 1 passed"),
                _call("c2", "read_file", {"path": "merge.py"}),
                _result("c2", SNIPPET.replace("a[i:] + b[j:]", "a[i:]")),
            ],
            "tools": tools,
            "max_tokens": 320,
        },
        {
            "class": "agent",
            "id": "after-edit",
            "messages": [
                sys_msg,
                *user("Rename the function merge to merge_sorted everywhere in the project and run the tests."),
                _call("c1", "run_shell", {"command": "grep -rn 'merge(' --include=*.py ."}),
                _result("c1", "./merge.py:1:def merge(a, b):\n./tests/test_merge.py:3:    assert merge([1], [2]) == [1, 2]\n./tests/test_merge.py:6:    assert merge([1, 2], [2, 3]) == [1, 2, 2, 3]"),
                _call("c2", "edit_file", {"path": "merge.py", "old": "def merge(a, b):", "new": "def merge_sorted(a, b):"}),
                _result("c2", "ok: 1 replacement"),
            ],
            "tools": tools,
            "max_tokens": 320,
        },
        {
            "class": "agent",
            "id": "report",
            "messages": [
                sys_msg,
                *user("How many Python files are in this project, and which is the largest?"),
                _call("c1", "run_shell", {"command": "find . -name '*.py' -printf '%s %p\\n' | sort -n"}),
                _result("c1", "\n".join(f"{120 * i + 37} ./pkg/mod_{i:02d}.py" for i in range(1, 19))),
            ],
            "tools": tools,
            "max_tokens": 200,
        },
    ]


# -- one request ---------------------------------------------------------------------------------------------------


def body_for(case: dict, model: str, draft_off: bool, thinking: bool) -> dict:
    body = {
        "model": model,
        "messages": case["messages"],
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if case.get("tools"):
        body["tools"] = case["tools"]
        body["tool_choice"] = "auto"
    if draft_off:
        body["draft"] = False
    return body


def record(case: dict, reply: dict, label: str, rep: int, draft_off: bool, wall: float, waited: float) -> dict:
    """One request's line: what it was, the runtime fields, the speculative fields."""

    rt = reply.get("tensorfold") or {}
    spec = reply.get("speculative") or rt.get("speculative") or {}
    pos = spec.get("positions") or {}
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = json.dumps([msg.get("content"), msg.get("reasoning_content"), msg.get("tool_calls")], sort_keys=True)
    return {
        "label": label,
        "class": case["class"],
        "id": case["id"],
        "rep": rep,
        "draft": False if draft_off else None,
        "completion_tokens": int((reply.get("usage") or {}).get("completion_tokens") or 0),
        "finish": choice.get("finish_reason"),
        "wall_s": round(wall, 3),
        "thermal_wait_s": round(waited, 1),
        "tok_s": float(rt.get("tokens_per_second") or 0.0),
        "decode_s": float(rt.get("seconds") or 0.0),
        "drafts": rt.get("drafts"),
        "token_sha256": rt.get("token_sha256"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "rounds": int(spec.get("rounds") or 0),
        "drafted": int(spec.get("drafted") or 0),
        "accepted": int(spec.get("accepted") or 0),
        "acceptance_rate": float(spec.get("acceptance_rate") or 0.0),
        "tau": float(spec.get("tokens_per_round") or 0.0),
        "tried": [int(x) for x in pos.get("tried") or []],
        "kept": [int(x) for x in pos.get("kept") or []],
    }


def node_states(cmds: list[str]) -> list[str]:
    """Each command's first word (a device state: normal, throttled, emergency; 'unknown' when it fails)."""

    out = []
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20, check=False)
            words = r.stdout.split()
            out.append(words[0] if r.returncode == 0 and words else "unknown")
        except subprocess.TimeoutExpired:
            out.append("unknown")
    return out


def wait_normal(cmds: list[str], poll: float = 10.0, limit: float = 1800.0) -> float:
    """Wait until every node reports ``normal``; seconds waited. ``emergency``, or ``limit`` s without ``normal``, stops."""

    t0 = time.monotonic()
    while cmds:
        states = node_states(cmds)
        if all(s == "normal" for s in states):
            break
        if "emergency" in states:
            raise SystemExit(f"a node reports emergency ({states}): the run stops; read memguard.log there")
        if time.monotonic() - t0 > limit:
            raise SystemExit(f"no node state normal for {limit:.0f} s ({states}): the run stops")
        print(f"[draft_positions] waiting for the nodes to report normal: {states}", file=sys.stderr)
        time.sleep(poll)
    return time.monotonic() - t0


def run(a: argparse.Namespace) -> int:
    api = tc.Api(a.base, tc.read_key(a.api_key_file))
    want = set(a.classes.split(","))
    bad = want - set(CLASSES)
    if bad:
        raise SystemExit(f"no such class: {', '.join(sorted(bad))} (classes: {', '.join(CLASSES)})")
    cases = [c for c in prompts() if c["class"] in want]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fails = 0
    with out.open("w") as f:
        for rep in range(a.reps):
            for case in cases:
                waited = wait_normal(a.state_cmd)
                t0 = time.perf_counter()
                try:
                    reply = api.chat(body_for(case, a.model, a.draft_off, a.thinking), a.timeout, False)
                except (RuntimeError, OSError, tc.Unsupported) as exc:
                    print(f"[draft_positions] {case['class']}/{case['id']}: {exc}", file=sys.stderr)
                    fails += 1
                    continue
                rec = record(case, reply, a.label, rep, a.draft_off, time.perf_counter() - t0, waited)
                if a.draft_off and (rec["drafted"] > 0 or rec["drafts"] is True):
                    print(f"[draft_positions] {case['id']}: \"draft\": false was not honoured (drafted {rec['drafted']}): "
                          "use a DRAFTS=0 server for the drafts-off run", file=sys.stderr)
                    fails += 1
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{rec['class']:<10} {rec['id']:<15} {rec['completion_tokens']:>4} tok {rec['tok_s']:7.2f} tok/s "
                      f"tau {rec['tau']:.2f} acc {rec['acceptance_rate']:.2f}")
    return 1 if fails else 0


# -- aggregation (pure) --------------------------------------------------------------------------------------------


def pooled(recs: list[dict], key: str) -> list[int]:
    """Element-wise sum of a position list (``tried`` or ``kept``) over records of any length."""

    n = max((len(r[key]) for r in recs), default=0)
    return [sum(r[key][j] for r in recs if j < len(r[key])) for j in range(n)]


def conditional(tried: list[int], kept: list[int]) -> list[float | None]:
    """a_j = kept_j / tried_j (None where nothing was tried)."""

    return [k / t if t else None for t, k in zip(tried, kept)]


def survival(acc: list[float | None]) -> list[float]:
    """S_j = a_1 ... a_j, up to the first position with no tries."""

    out, s = [], 1.0
    for a in acc:
        if a is None:
            break
        s *= a
        out.append(s)
    return out


def tau(recs: list[dict]) -> float:
    """Tokens a round over all rounds (each reply's tokens_per_round weighted by its rounds); 0 without rounds."""

    rounds = sum(r["rounds"] for r in recs)
    return sum(r["tau"] * r["rounds"] for r in recs) / rounds if rounds else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(recs: list[dict]) -> dict:
    """One run's (or one class's) aggregate."""

    tried, kept = pooled(recs, "tried"), pooled(recs, "kept")
    acc = conditional(tried, kept)
    drafted = sum(r["drafted"] for r in recs)
    return {
        "n": len(recs),
        "tried": tried,
        "kept": kept,
        "a": acc,
        "survival": survival(acc),
        "tau": tau(recs),
        "acceptance_rate": sum(r["accepted"] for r in recs) / drafted if drafted else 0.0,
        "tok_s": mean([r["tok_s"] for r in recs]),
        "tokens": sum(r["completion_tokens"] for r in recs),
    }


def key_of(r: dict) -> tuple:
    return (r["class"], r["id"], r["rep"])


def identity(r: dict) -> str:
    """What must match between a drafted and a plain reply: the token ids' sha256, else the reply's text."""

    return "tokens:" + r["token_sha256"] if r.get("token_sha256") else "text:" + r["text_sha256"]


def compare(base: list[dict], run_: list[dict]) -> dict:
    """A run against the baseline: requests compared, the ones that differ or are missing, speed ratio per class."""

    by = {key_of(r): r for r in base}
    diff, missing = [], []
    for r in run_:
        b = by.get(key_of(r))
        if b is None:
            missing.append("/".join(map(str, key_of(r))))
        elif identity(b) != identity(r):
            diff.append("/".join(map(str, key_of(r))))
    ratio = {}
    for cls in sorted({r["class"] for r in run_}):
        x = mean([r["tok_s"] for r in run_ if r["class"] == cls])
        y = mean([r["tok_s"] for r in base if r["class"] == cls])
        ratio[cls] = x / y if y else None
    xs, ys = mean([r["tok_s"] for r in run_]), mean([r["tok_s"] for r in base])
    ratio["all"] = xs / ys if ys else None
    return {"compared": len(run_) - len(missing), "different": diff, "missing": missing, "speed_ratio": ratio}


def calibrated_prior(surv: list[float], tried: list[int], prior=ENGINE_DFLASH2_PRIOR, min_tried: int = MIN_TRIED) -> list[float]:
    """A depth prior from measured survival: S_j where position j was tried at least ``min_tried`` times; past that the
    prior's own decay from the last measured value. Strictly decreasing (the engine's test requires it), 3 decimals."""

    out: list[float] = []
    for j, p in enumerate(prior):
        if j < len(surv) and j < len(tried) and tried[j] >= min_tried:
            v = surv[j]
        elif out:
            v = out[-1] * p / prior[j - 1]
        else:
            v = p
        if out:
            v = min(v, out[-1] - 0.001)
        out.append(round(max(v, 0.001), 3))
    return out


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fmt(xs: list[float | None]) -> str:
    return " ".join("-" if x is None else f"{x:.3f}" for x in xs) or "-"


def report(a: argparse.Namespace) -> int:
    runs = [(Path(p).stem, load(p)) for p in a.files]
    base_name, base = runs[0]
    out: dict[str, Any] = {"baseline": base_name, "runs": {}, "compare": {}}
    bad = 0
    for name, recs in runs:
        print(f"\n## {name} ({len(recs)} requests{', baseline' if name == base_name else ''})\n")
        print("| class | n | tau | acceptance | tok/s | a_j (conditional) | S_j (survival) | tried |")
        print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
        per = {}
        for cls in [*CLASSES, "all"]:
            sub = recs if cls == "all" else [r for r in recs if r["class"] == cls]
            if not sub:
                continue
            s = per[cls] = summarize(sub)
            print(f"| {cls} | {s['n']} | {s['tau']:.2f} | {s['acceptance_rate']:.3f} | {s['tok_s']:.2f} | "
                  f"{_fmt(s['a'])} | {_fmt(s['survival'])} | {' '.join(map(str, s['tried'])) or '-'} |")
        out["runs"][name] = per
        if name == base_name:
            continue
        c = out["compare"][name] = compare(base, recs)
        ratios = ", ".join(f"{k} {v:.3f}" for k, v in c["speed_ratio"].items() if v is not None)
        print(f"\nagainst {base_name}: speed ratio (tok/s) {ratios or '-'}")
        if c["different"] or c["missing"]:
            bad = 1
            print(f"TOKENS DIFFER ({len(c['different'])} of {c['compared']}): {', '.join(c['different']) or '-'}; "
                  f"missing from the baseline: {', '.join(c['missing']) or '-'}. A difference is a bug of the draft path")
        else:
            print(f"tokens: EQUAL ({c['compared']} requests)")
        al = per.get("all")
        if al and al["survival"]:
            print(f"depth prior from this run's survival (draft_kind.zig dflash2_prior, now {list(ENGINE_DFLASH2_PRIOR)}): "
                  f"{calibrated_prior(al['survival'], al['tried'], min_tried=a.min_tried)}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\n{'FAIL: a drafted reply differs from the baseline' if bad else 'PASS: every reply equals the baseline'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the requests against a running server, one at a time")
    r.add_argument("--base", default=tc.default_base())
    r.add_argument("--model", default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3")
    r.add_argument("--api-key-file", default=None)
    r.add_argument("--label", required=True, help="the run's name in its lines (plain, dflash2, dspark, ...)")
    r.add_argument("--out", required=True, help="the JSON lines file")
    r.add_argument("--classes", default=",".join(CLASSES))
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--draft-off", action="store_true", help='send "draft": false (plain decode on a drafting server)')
    r.add_argument("--thinking", action="store_true", help="enable_thinking true (default false, as the smoke test)")
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--state-cmd", action="append", default=[], help="a command printing a node's device state")
    p = sub.add_parser("report", help="aggregate runs; the first is the drafts-off baseline")
    p.add_argument("files", nargs="+")
    p.add_argument("--json", default=None)
    p.add_argument("--min-tried", type=int, default=MIN_TRIED)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
