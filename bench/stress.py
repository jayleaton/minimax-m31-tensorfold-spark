#!/usr/bin/env python3
"""Memory stress / admission client (stdlib only) for a running MiniMax-M3.1 server (start.sh).

  stress.py stress --long 120000 --dctx 32768 --decode 2048 --out stress.json
      one stream prefills ``--long`` tokens while ``--streams`` - 1 others (default: $PARALLEL, the server's requests
      at once, else 1; with 1 only the long stream runs: a long-prefill probe)
      prefill ``--dctx`` tokens and then decode ``--decode`` tokens (``ignore_eos`` + ``min_tokens`` = ``max_tokens``:
      the decode really runs instead of stopping at EOS); the streams start ``--stagger`` s apart. Token ids are random
      text-range ids (below MiniMax's specials, which start near 200000), a different seed a stream (no prefix
      sharing). Reports each stream's first-token / total seconds, completion tokens (the server's usage when it sends
      one, else streamed chunks), its decode tok/s, its decode tok/s WHILE the long stream was still prefilling (chunk
      times before the long stream's first token), and errors. Exit 1 on an error or (``--require-full``) a decode
      stream short of ``--decode`` tokens.
  stress.py admit --tokens 512 --max-wait 10
      one request; the seconds to its first token (admission + prefill); exit 1 if over ``--max-wait``.

Both take --base (http://$HOST:$PORT), --model (the first of /v1/models) and --api-key-file ($API_KEY_FILE when
present); the key goes in a header and is masked in everything printed or saved. The defaults fit the recipe's
131072-token context (CONTEXT): --long + its decode must stay under it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

MASK = "<redacted>"
TEXT_IDS = (
    1000,
    190_000,
)  # random prompt ids: past the byte / punctuation range, below the special tokens


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


def _model(api: Api, model: str) -> str:
    if model:
        return model
    with api.open("/v1/models", None, 30) as r:
        return json.loads(r.read())["data"][0]["id"]


def _ids(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(*TEXT_IDS) for _ in range(n)]


def _stream(
    api: Api,
    model: str,
    ids: list[int],
    max_tokens: int,
    out: dict,
    timeout: float,
    t_ref: float | None = None,
) -> None:
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_ref = time.time() if t_ref is None else t_ref
    out["start_s"] = round(time.time() - t_ref, 2)
    times: list[float] = []
    t0 = time.time()
    out.update(prompt=len(ids), max_tokens=max_tokens, first_token_s=None, tokens=0)
    try:
        with api.open("/v1/completions", body, timeout) as r:
            for raw in r:
                line = api.mask(raw.decode()).strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except ValueError:
                    chunk = {}
                if chunk.get("error"):
                    err = chunk["error"]
                    out["error"] = "stream error: " + str(
                        err.get("message") if isinstance(err, dict) else err
                    )
                    break
                if chunk.get("usage"):
                    out["usage_completion"] = chunk["usage"].get("completion_tokens")
                if not chunk.get("choices"):
                    continue
                if out["first_token_s"] is None:
                    out["first_token_s"] = round(time.time() - t0, 2)
                out["tokens"] += 1
                times.append(round(time.time() - t_ref, 3))
        out["total_s"] = round(time.time() - t0, 2)
        out["chunks"] = out["tokens"]
        if out.get("usage_completion"):
            out["tokens"] = int(out["usage_completion"])
        out["chunk_times"] = times
        if not out["tokens"] and "error" not in out:
            out["error"] = (
                "no streamed tokens (does the server stream /v1/completions?)"
            )
        ft = out["first_token_s"] or out["total_s"]
        out["decode_tok_s"] = (
            round(out["tokens"] / max(out["total_s"] - ft, 1e-6), 2)
            if out["tokens"] > 1
            else None
        )
    except Exception as e:  # noqa: BLE001 - reported
        out["error"] = api.mask(f"{type(e).__name__}: {e}")[:300]
        out["total_s"] = round(time.time() - t0, 2)


def _during(r: dict, end_s: float | None) -> dict:
    """Decode tok/s of a stream while the long stream prefilled: its chunks after its own first one, before
    ``end_s`` (the long stream's first token, seconds from the shared start), scaled from chunks to tokens."""

    times = r.get("chunk_times") or []
    if end_s is None or len(times) < 2:
        return {}
    inside = [t for t in times[1:] if t <= end_s]
    span = min(end_s, times[-1]) - times[0]
    if not inside or span <= 0:
        return {"during_prefill_tokens": 0}
    scale = r["tokens"] / max(r.get("chunks") or len(times), 1)
    return {
        "during_prefill_tokens": round(len(inside) * scale),
        "during_prefill_s": round(span, 2),
        "during_prefill_tok_s": round(len(inside) * scale / span, 2),
    }


def stress(a, api: Api) -> int:
    model = _model(api, a.model)
    plan = [(_ids(a.long, 1), a.long_decode)] + [
        (_ids(a.dctx, 2 + i), a.decode) for i in range(a.streams - 1)
    ]
    res = [{"role": "long" if i == 0 else "decode"} for i in range(len(plan))]
    t0 = time.time()
    th = [
        threading.Thread(
            target=_stream,
            args=(api, model, ids, mt, res[i], a.timeout, t0),
            daemon=True,
        )
        for i, (ids, mt) in enumerate(plan)
    ]
    for t in th:
        t.start()
        time.sleep(a.stagger)
    for t in th:
        t.join()
    long = res[0]
    long_ft = (
        None
        if long.get("first_token_s") is None
        else long["start_s"] + long["first_token_s"]
    )
    for r in res[1:]:
        r.update(_during(r, long_ft))
    full = all(r.get("tokens", 0) >= a.decode for r in res[1:])
    rep = {
        "model": model,
        "long": a.long,
        "dctx": a.dctx,
        "decode": a.decode,
        "streams_n": a.streams,
        "seconds": round(time.time() - t0, 1),
        "long_first_token_s": long_ft,
        "decode_full": full,
        "streams": res,
        "ok": all("error" not in r for r in res) and (full or not a.require_full),
    }
    brief = {k: v for k, v in rep.items() if k != "streams"}
    brief["streams"] = [{k: v for k, v in r.items() if k != "chunk_times"} for r in res]
    print(api.mask(json.dumps(brief)), flush=True)
    if a.out:
        Path(a.out).write_text(api.mask(json.dumps(rep, indent=1)))
    return 0 if rep["ok"] else 1


def admit(a, api: Api) -> int:
    res: dict = {}
    _stream(api, _model(api, a.model), _ids(a.tokens, 9), 8, res, a.timeout)
    ft = res.get("first_token_s")
    res["pass"] = "error" not in res and ft is not None and ft <= a.max_wait
    print(api.mask(json.dumps(res)), flush=True)
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress")
    s.add_argument("--long", type=int, default=120_000)
    s.add_argument("--dctx", type=int, default=32_768)
    s.add_argument("--decode", type=int, default=2048)
    s.add_argument(
        "--long-decode", type=int, default=16, help="max_tokens of the long stream"
    )
    s.add_argument(
        "--streams",
        type=int,
        default=int(os.environ.get("PARALLEL") or 1),
        help="concurrent streams, the long one included (default: $PARALLEL, else 1)",
    )
    s.add_argument(
        "--stagger", type=float, default=1.0, help="seconds between the submits"
    )
    s.add_argument(
        "--require-full",
        action="store_true",
        help="exit 1 when a decode stream is short of --decode",
    )
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument("--out", default="")
    m = sub.add_parser("admit")
    m.add_argument("--tokens", type=int, default=512)
    m.add_argument("--max-wait", type=float, default=10.0)
    m.add_argument("--timeout", type=float, default=900)
    for p in (s, m):
        p.add_argument("--base", default=default_base())
        p.add_argument(
            "--model", default="", help="default: the first model /v1/models lists"
        )
        p.add_argument(
            "--api-key-file",
            default=None,
            help="the server's key file (default: $API_KEY_FILE)",
        )
    a = ap.parse_args(argv)
    api = Api(a.base, read_key(a.api_key_file))
    return {"stress": stress, "admit": admit}[a.cmd](a, api)


if __name__ == "__main__":
    sys.exit(main())
