#!/usr/bin/env python3
"""Answer quality through the serving path (a running MiniMax-M3.1 server), standard library only.

  quality.py mmlu --data bench/data/mmlu200.jsonl --out R.json [--shots 0] [--label L]
      MMLU-200 (thinking off, greedy, max_tokens 8, the reply's first A-D letter). Records each question's
      prompt_tokens, finish reason and whether the reply carried reasoning_content: the pack's chat template is
      adaptive, so a reply that thinks anyway spends the 8 tokens there (``thought`` in the report; raise
      --max-tokens if it is not 0). A reply with MiniMax markup in its content is counted as a leak.
  quality.py mmlu ... --shots 20
      the same questions after ONE fixed preamble of the first N questions with their answers (~2-3K tokens), scored
      on the other 200-N: the long-prompt path (chunked prefill, prompt cache) against the short one.
  quality.py needle --sizes 32768,100000 --out N.json
      a passkey at 1/3 depth of a filler document of about each size (sized with the server's /tokenize when it has
      one, else estimated).
  quality.py compare --run R.json --ref F.json [--kit 0.875] [--within 1.0]
      accuracy of both, the run - ref delta in points, per-question agreement, optionally vs a reference accuracy
      (--kit); exit 1 on a FAIL. For two builds, two configs, or a 0-shot vs a 20-shot run.

Every subcommand that queries takes --base (http://$HOST:$PORT), --model ($SERVED_NAME) and --api-key-file
($API_KEY_FILE when present); the key goes in a header and is masked in everything printed or saved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"
LETTERS = "ABCD"
WORDS = [
    "river",
    "mountain",
    "lantern",
    "copper",
    "harbor",
    "meadow",
    "quartz",
    "signal",
    "orchard",
    "canvas",
    "thunder",
    "velvet",
    "compass",
    "ember",
    "glacier",
    "timber",
    "falcon",
    "prairie",
    "marble",
    "beacon",
    "willow",
    "cobalt",
    "summit",
    "ferry",
    "saddle",
    "tundra",
    "pepper",
    "violet",
    "anchor",
    "bramble",
    "cinder",
    "delta",
    "fjord",
    "granite",
    "hollow",
    "ivory",
    "jasmine",
    "kettle",
    "lagoon",
    "mosaic",
    "nectar",
    "oyster",
]


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

    def post(self, path: str, body: dict, timeout: float = 1800) -> dict:
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


def chat(a, api: Api, content: str, max_tokens: int) -> dict:
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = api.post("/v1/chat/completions", body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    return {
        "content": msg.get("content") or "",
        "thought": bool(msg.get("reasoning_content")),
        "finish": ch.get("finish_reason"),
        "prompt_tokens": int((r.get("usage") or {}).get("prompt_tokens") or 0),
    }


def question(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return (
        f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n"
        f"{q['question']}\n{opts}\n\nAnswer with the letter of the correct option only."
    )


def mmlu(a, api: Api) -> int:
    rows = [
        json.loads(line)
        for line in Path(a.data).read_text().splitlines()
        if line.strip()
    ]
    shots, rest = rows[: a.shots], rows[a.shots :]
    pre = ""
    if shots:
        pre = (
            "Here are worked examples of multiple choice questions with their answers.\n\n"
            + "\n\n".join(
                question(q) + f"\nAnswer: {LETTERS[q['answer']]}" for q in shots
            )
            + "\n\nNow the real question.\n\n"
        )
    t0, correct, items = time.time(), 0, []
    for i, q in enumerate(rest):
        try:
            r = chat(a, api, pre + question(q), a.max_tokens)
            m = re.search(r"\b([ABCD])\b", r["content"])
            got = LETTERS.index(m.group(1)) if m else -1
        except Exception as exc:  # noqa: BLE001 - recorded, counted wrong
            r = {
                "content": api.mask(f"ERROR {type(exc).__name__}: {exc}")[:200],
                "prompt_tokens": 0,
            }
            got = -2
        correct += got == q["answer"]
        items.append(
            {
                "i": i + a.shots,
                "answer": q["answer"],
                "got": got,
                "prompt_tokens": r["prompt_tokens"],
                "finish": r.get("finish"),
                "thought": r.get("thought", False),
                "leak": bool(LEAK.search(r["content"])),
                "reply": r["content"][:40],
            }
        )
    acc = correct / max(len(rest), 1)
    toks = sorted(x["prompt_tokens"] for x in items)
    rec = {
        "label": a.label,
        "model": a.model,
        "shots": a.shots,
        "max_tokens": a.max_tokens,
        "accuracy": acc,
        "correct": correct,
        "n": len(rest),
        "errors": sum(x["got"] == -2 for x in items),
        "no_letter": sum(x["got"] == -1 for x in items),
        "thought": sum(bool(x["thought"]) for x in items),
        "leaks": sum(x["leak"] for x in items),
        "prompt_tokens_median": toks[len(toks) // 2] if toks else 0,
        "seconds": round(time.time() - t0, 1),
        "items": items,
    }
    print(
        f"[quality] mmlu {a.label} shots {a.shots}: {acc:.3f} ({correct}/{len(rest)}), errors {rec['errors']}, "
        f"no letter {rec['no_letter']}, thought {rec['thought']}, markup leaks {rec['leaks']}, "
        f"median prompt {rec['prompt_tokens_median']} tokens",
        flush=True,
    )
    Path(a.out).write_text(api.mask(json.dumps(rec, indent=1)))
    return 1 if rec["leaks"] else 0


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    for s in range(max(n_words // 12, 1)):
        w = [rng.choice(WORDS) for _ in range(11)]
        out.append(
            f"Record {s}: the {w[0]} near the {w[1]} kept its {w[2]} {w[3]}, and the {w[4]} {w[5]} "
            f"{w[6]} {w[7]} {w[8]} {w[9]} {w[10]}."
        )
    return " ".join(out)


def count(a, api: Api, text: str) -> int | None:
    try:
        r = api.post(
            "/tokenize",
            {"model": a.model, "messages": [{"role": "user", "content": text}]},
            600,
        )
        return int(r.get("count") or len(r["tokens"]))
    except Exception:  # noqa: BLE001 - sized by estimate then
        return None


def needle(a, api: Api) -> int:
    res, ok = [], True
    for size in [int(x) for x in a.sizes.split(",") if x]:
        key = str(random.Random(size).randrange(10_000_000, 99_999_999))
        words = int(size * 0.62)
        for _ in range(2):  # size the document with /tokenize (one correction)
            doc = filler(words, size)
            got = count(a, api, doc)
            if got is None or abs(got - size) < size * 0.03:
                break
            words = int(words * size / got)
        cut = len(doc) // 3
        doc = doc[:cut] + f" The secret passkey is {key}. Remember it. " + doc[cut:]
        ask = (
            doc
            + "\n\nWhat is the secret passkey stated in the text above? Answer with the number only."
        )
        t0 = time.time()
        try:
            r = chat(a, api, ask, 32)
        except Exception as exc:  # noqa: BLE001
            r = {
                "content": api.mask(f"ERROR {type(exc).__name__}: {exc}")[:200],
                "prompt_tokens": 0,
            }
        found = key in r["content"]
        ok &= found
        res.append(
            {
                "size": size,
                "prompt_tokens": r["prompt_tokens"],
                "found": found,
                "reply": r["content"][:80],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"[quality] needle {a.label} {size}: prompt {r['prompt_tokens']} tokens, found {found} "
            f"({res[-1]['seconds']} s)",
            flush=True,
        )
    Path(a.out).write_text(
        api.mask(json.dumps({"label": a.label, "needles": res, "ok": ok}, indent=1))
    )
    return 0 if ok else 1


def compare(a) -> int:
    r, f = json.loads(Path(a.run).read_text()), json.loads(Path(a.ref).read_text())
    by = {x["i"]: x["got"] for x in f["items"]}
    same = sum(by.get(x["i"]) == x["got"] for x in r["items"])
    delta = 100 * (r["accuracy"] - f["accuracy"])
    lines = [
        (
            f"run {100 * r['accuracy']:.1f}% vs ref {100 * f['accuracy']:.1f}% (n {r['n']}, shots {r['shots']}): "
            f"delta {delta:+.1f} points, same answer {same}/{len(r['items'])}"
        )
    ]
    ok = abs(delta) <= a.within and not r.get("errors") and not f.get("errors")
    lines.append(f"PASS run within {a.within} point of ref: {ok}")
    if a.kit:
        for tag, x in (("run", r), ("ref", f)):
            d = 100 * (x["accuracy"] - a.kit)
            good = d >= -a.within
            ok &= good
            lines.append(
                f"PASS {tag} within {a.within} point of the reference {100 * a.kit:.1f}% (or above): {good} ({d:+.1f})"
            )
    print("\n".join(lines))
    if a.out:
        rec = {
            "delta_points": delta,
            "same": same,
            "n": len(r["items"]),
            "ok": ok,
            "run": r["accuracy"],
            "ref": f["accuracy"],
            "kit": a.kit,
        }
        Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mmlu", "needle"):
        p = sub.add_parser(name)
        p.add_argument("--base", default=default_base())
        p.add_argument(
            "--model",
            default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3",
        )
        p.add_argument(
            "--api-key-file",
            default=None,
            help="the server's key file (default: $API_KEY_FILE)",
        )
        p.add_argument("--label", default="")
        p.add_argument("--out", required=True)
    sub.choices["mmlu"].add_argument(
        "--data", default=str(Path(__file__).resolve().parent / "data/mmlu200.jsonl")
    )
    sub.choices["mmlu"].add_argument("--shots", type=int, default=0)
    sub.choices["mmlu"].add_argument("--max-tokens", type=int, default=8)
    sub.choices["needle"].add_argument("--sizes", default="32768,100000")
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--kit", type=float, default=0.0)
    c.add_argument("--within", type=float, default=1.0)
    c.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        return compare(a)
    api = Api(a.base, read_key(a.api_key_file))
    return {"mmlu": mmlu, "needle": needle}[a.cmd](a, api)


if __name__ == "__main__":
    sys.exit(main())
