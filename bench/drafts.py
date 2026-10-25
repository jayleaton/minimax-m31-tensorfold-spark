#!/usr/bin/env python3
"""Drafts on against drafts off for MiniMax-M3.1-Flash's own dspark draft (stdlib only): speed, acceptance, identity.

  drafts.py run --label drafts-off|drafts-on [--reps 3] [--max-tokens 256] [--classes code,prose,structured,agent]
                [--server-log PATH | --server-log-cmd CMD] [--reference OFF.json] [--worker-ssh TARGET] --out R/x.json
      One request at a time (C1), thinking off, max_tokens fixed, ignore_eos off. Every prompt of
      bench/data/drafts-prompts.json greedy (temperature 0) ``--reps`` times, and the first prompt of each class keyed-sampled
      (the file's temperature, top_p and seed) ``--reps`` times. A request is streamed; per request: completion tokens, client
      TTFT and decode tok/s (tokens after the first over the first-to-last delta span), the server's
      ``tensorfold.tokens_per_second`` and ``time_to_first_token``, ``speculative.{rounds, drafted, accepted,
      acceptance_rate, tokens_per_round}`` (these count every kind of draft, context copies too) and
      ``tensorfold.token_sha256``.
      With ``--server-log`` (a file rank 0 logs to) or ``--server-log-cmd`` (a command that prints rank 0's log, e.g.
      ``docker logs -t minimax-m3-flash-tf``), the ``dspark acceptance stream:`` line that TF_MM3_DSPARK_STATS=1 makes
      rank 0 print for each finished stream (just before the server's ``done <id> ...`` line) is matched to its request:
      the lines the log gained during the request, paired with their ``done`` line, the pair whose done line carries the
      reply's token count (and short sha). From them: acceptance by position, conditional (kept / reached) and cumulative
      (kept / rounds), and tau = 1 + kept / rounds (head rounds), per class. The last ``dspark calibration:`` line is kept.
      ``--reference OFF.json`` (on the drafts-on run): stop at the first greedy reply whose token_sha256 differs from that
      run's (a drafting bug: the target verifies every draft). Before each prompt both nodes' device state must be
      ``normal`` (waited for up to ``--thermal-wait`` s; ``emergency`` stops the run); each prompt's reps carry a stamp of
      both nodes before and after: /run/spark-thermal/state, the GPU and hottest ACPI temperature, and the device limits
      read from /etc/spark/thermal.env.
  drafts.py compare --off OFF.json [...] --on ON.json [...] [--out R/compare.json]
      Per class (greedy, then sampled): tok/s off and on (client and server, medians), the speedup, tau, conditional and
      cumulative acceptance by position, the copy-inclusive acceptance, and the token identity of the greedy replies (every
      greedy rep of a prompt, on and off, the same token_sha256). Any greedy mismatch, a greedy reply without a sha or
      text, a request error, a run whose served ``tensorfold.drafts`` contradicts its label, or prompt sets that differ is
      FAIL (exit 1). A sampled difference is reported, not failed (scripts/check-drafted.sh decides those). A throttled
      or emergency stamp is reported: that class's speed is not comparable. With every speedup within 3 % the
      comparison says to alternate off / on once more (docs/DRAFTS.md).
  drafts.py log PATH|--cmd CMD [--out R/log.json]
      A rank 0 log (also .gz, as scripts/serve.sh stop saves it): the calibration line, ``dspark acceptance total`` with
      its acceptance by position and tau, the sum of the stream lines, the ``dspark draft:`` line, the budget lines.

DRY_RUN=1 (or --dry-run) prints every request and what would be read, sends nothing, samples nothing. Defaults come from
HOST / PORT / SERVED_NAME / API_KEY_FILE (scripts/config.sh exports them) and WORKER_SSH or WORKER; the key goes in a
header and is masked in everything printed or saved. Exit 0: done / PASS; 1: a request failed / FAIL / a greedy reply
differs from the reference; 2: the server could not be reached; 3: stopped on a device state or a missing sample.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "data" / "drafts-prompts.json"
MASK = "<redacted>"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
LABELS = ("drafts-off", "drafts-on")
FEW = 0.03  # speedups within this of 1.0 on every class: alternate off / on once more
DEPTHS = 8  # the engine's tally: reached / kept for draft depths 1..8
THERMAL_RUN = os.environ.get("SPARK_THERMAL_RUN") or "/run/spark-thermal"
THERMAL_ENV = os.environ.get("SPARK_THERMAL_ENV") or "/etc/spark/thermal.env"
# one sample of a node, read-only: state|gpu reading|hottest ACPI zone (milli-degrees)|the device limits, ';'-joined
SAMPLE_SH = (
    f"s=$(head -n 1 {THERMAL_RUN}/state 2>/dev/null || echo missing); "
    f"g=$(timeout 5 nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1); "
    "a=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -n 1); "
    f"l=$(grep -E '^GUARD_[A-Z_]+=' {THERMAL_ENV} 2>/dev/null | tr '\\n' ';'); "
    'echo "${s:-missing}|${g}|${a}|${l}"'
)

# -- the engine's log lines ---------------------------------------------------------------------------------------------
TALLY = re.compile(r"dspark acceptance (stream|total): (.*)$")
DONE = re.compile(r"\[tensorfold\] done (\S+) (.*)$")
CALIB = re.compile(r"dspark calibration: .*$")
STAMP_PREFIX = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z) ")  # docker logs -t
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BUDGET = re.compile(r"minimax budget:|\[tensorfold\] CUDA |dspark draft|the dspark draft takes|memory floor:|weights: ")


def clean(lines: list[str]) -> list[str]:
    """Log lines as written: the live status line's erase sequences and carriage returns dropped, and, when every line
    carries docker's -t time, ordered by it (stdout and stderr reach ``docker logs`` as two streams) with it removed."""

    rows = []
    for raw in lines:
        m = STAMP_PREFIX.match(raw)
        line = ANSI.sub("", raw[m.end():] if m else raw).split("\r")[-1].rstrip("\n")
        if line.strip():
            rows.append((m.group(1) if m else None, line))
    if rows and all(ts for ts, _ in rows):
        # RFC 3339 with the fraction's trailing zeros dropped (docker's RFC3339Nano): compare it padded to 9 digits;
        # the sort is stable, so lines with the same time keep their order
        rows.sort(key=lambda r: (r[0][:19], r[0][20:-1].ljust(9, "0")))
    return [line for _, line in rows]


def parse_tally(line: str) -> dict | None:
    """``dspark acceptance stream|total: rounds=R reached=a1/../a8 kept=k1/../k8 kept_per_round=X conf1_rounds=..
    conf1_kept=..`` as {kind, rounds, reached, kept, kept_per_round, conf1_rounds, conf1_kept}; None when the line is
    not one or is malformed (reached and kept must be lists of the same length)."""

    m = TALLY.search(line)
    if not m:
        return None
    fields = dict(w.split("=", 1) for w in m.group(2).split() if "=" in w)
    try:
        t = {"kind": m.group(1), "rounds": int(fields["rounds"]),
             "reached": [int(x) for x in fields["reached"].split("/")],
             "kept": [int(x) for x in fields["kept"].split("/")]}
        for k in ("conf1_rounds", "conf1_kept"):
            t[k] = [int(x) for x in fields[k].split("/")] if k in fields else []
        kpr = fields.get("kept_per_round")
        t["kept_per_round"] = float(kpr) if kpr not in (None, "nan") else None
    except (KeyError, ValueError):
        return None
    return t if len(t["reached"]) == len(t["kept"]) and t["rounds"] >= 0 else None


def parse_done(line: str) -> dict | None:
    """The server's ``[tensorfold] done <id> prompt=.. tokens=N sha=S finish=.. rounds=R accepted=A/D ...`` line."""

    m = DONE.search(line)
    if not m:
        return None
    out: dict[str, Any] = {"id": m.group(1)}
    for w in m.group(2).split():
        k, eq, v = w.partition("=")
        if not eq:
            continue
        if k in ("prompt", "cached", "tokens", "rounds") and v.isdigit():
            out[k] = int(v)
        elif k == "accepted" and re.fullmatch(r"\d+/\d+", v):
            out["accepted"], out["drafted"] = (int(x) for x in v.split("/"))
        elif k in ("sha", "finish"):
            out[k] = v
    return out


def pairs(lines: list[str]) -> tuple[list[tuple[dict | None, dict]], list[dict]]:
    """The (stream tally or None, done) pairs in log order (a stream line belongs to the next done line), and the
    stream lines left without a done line after them."""

    out, pending, orphans = [], None, []
    for line in lines:
        if (t := parse_tally(line)) and t["kind"] == "stream":
            if pending is not None:
                orphans.append(pending)
            pending = t
        elif d := parse_done(line):
            out.append((pending, d))
            pending = None
    if pending is not None:
        orphans.append(pending)
    return out, orphans


def attribute(lines: list[str], tokens: int | None, sha: str | None, spec_rounds: int | None) -> dict:
    """The stream tally of one request among the lines the log gained during it: the done line whose token count (and
    short sha, when the reply has one) is the reply's. ``note`` says why there is none, or what looks wrong."""

    found, orphans = pairs(lines)
    match = [(t, d) for t, d in found if (tokens is None or d.get("tokens") == tokens)
             and (not sha or not d.get("sha") or d["sha"] == sha)]
    res: dict[str, Any] = {"tally": None, "done": None, "note": None, "lines": len(lines), "orphans": len(orphans)}
    if len(match) == 1:
        res["tally"], res["done"] = match[0]
        if res["tally"] is None:
            res["note"] = "no dspark acceptance line before its done line"
        elif spec_rounds is not None and res["tally"]["rounds"] > spec_rounds:
            res["note"] = "suspect: more head rounds than the reply's speculative rounds"
    elif not found:
        res["note"] = "no done line in the log during the request"
    else:
        res["note"] = f"{len(match)} of {len(found)} done lines match the reply: not attributed"
    return res


def tally_sum(ts: list[dict]) -> dict | None:
    """Element-wise sum of tallies (the engine's Tally.add)."""

    ts = [t for t in ts if t]
    if not ts:
        return None
    n = max(len(t["reached"]) for t in ts)
    c = max((len(t.get("conf1_rounds") or []) for t in ts), default=0)

    def add(key, width):
        return [sum((t.get(key) or [])[i] if i < len(t.get(key) or []) else 0 for t in ts) for i in range(width)]

    return {"rounds": sum(t["rounds"] for t in ts), "reached": add("reached", n), "kept": add("kept", n),
            "conf1_rounds": add("conf1_rounds", c), "conf1_kept": add("conf1_kept", c), "streams": len(ts)}


def acceptance(t: dict | None) -> dict | None:
    """Acceptance by position from a tally: conditional[i] = kept[i] / reached[i] (of the head rounds that verified a
    draft at depth i + 1, the share that kept it), cumulative[i] = kept[i] / rounds, tau = 1 + sum(kept) / rounds (the
    tokens a head round yields). Positions without a reached round are None."""

    if not t or not t["rounds"]:
        return None
    r = t["rounds"]
    return {"rounds": r, "conditional": [round(k / a, 4) if a else None for k, a in zip(t["kept"], t["reached"])],
            "cumulative": [round(k / r, 4) for k in t["kept"]], "tau": round(1 + sum(t["kept"]) / r, 4),
            "conf1": [round(k / n, 4) if n else None for k, n in zip(t.get("conf1_kept") or [], t.get("conf1_rounds") or [])]}


class LogSource:
    """Rank 0's log: a file (read from an offset) or a command that prints the whole log (marked by line count)."""

    def __init__(self, path: str | None = None, cmd: str | None = None) -> None:
        self.path, self.cmd = path, cmd

    def describe(self) -> str:
        return f"file {self.path}" if self.path else f"command {self.cmd!r}"

    def _all(self) -> list[str]:
        if self.path:
            opener = gzip.open if self.path.endswith(".gz") else open
            with opener(self.path, "rt", errors="replace") as f:
                return clean(f.readlines())
        r = subprocess.run(self.cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           errors="replace", timeout=120)
        if r.returncode != 0:
            raise OSError(f"{self.cmd!r} exited {r.returncode}: {r.stdout[-200:]}")
        return clean(r.stdout.splitlines())

    def lines(self) -> list[str]:
        return self._all()

    def mark(self) -> int:
        return len(self._all())

    def since(self, mark: int) -> list[str]:
        lines = self._all()
        return lines[mark:] if len(lines) >= mark else lines   # a shorter log: restarted or rotated, read it all


# -- the server ---------------------------------------------------------------------------------------------------------
class Unreachable(Exception):
    """No server, or it refused the key."""


def default_base() -> str:
    """http://HOST:PORT from scripts/config.sh's settings (a wildcard HOST is reached on 127.0.0.1)."""

    host = os.environ.get("HOST") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{host}:{os.environ.get('PORT') or 8000}"


def read_key(path: str | None) -> str:
    """The first key of an API key file (one key, or "label: key", a line); "" when there is none to read."""

    strict = path is not None or os.environ.get("API_KEY") == "1"
    path = path if path is not None else os.environ.get("API_KEY_FILE", "")
    if not path or (not strict and not os.path.exists(path)):
        return ""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read the API key file {path}: {exc.strerror}") from None
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(":", 1)[1].strip() if ":" in line else line
    raise SystemExit(f"{path}: no key in it")


def shrink(v: Any) -> Any:
    """A request as DRY_RUN prints it: long strings cut."""

    if isinstance(v, dict):
        return {k: shrink(x) for k, x in v.items()}
    if isinstance(v, list):
        return [shrink(x) for x in v]
    return v[:70] + f"...[{len(v)} chars]" if isinstance(v, str) and len(v) > 90 else v


class Api:
    def __init__(self, base: str, key: str = "", dry: bool = False) -> None:
        self.base, self.key, self.dry = base.rstrip("/").removesuffix("/v1"), key, dry

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text

    def _req(self, path: str, body: dict | None) -> urllib.request.Request:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.Request(self.base + path, data, h, method="GET" if body is None else "POST")

    def get(self, path: str, timeout: float = 10) -> tuple[int, Any]:
        if self.dry:
            print(f"DRY GET {self.base}{path}", flush=True)
            return 0, None
        try:
            with urllib.request.urlopen(self._req(path, None), timeout=timeout) as r:
                text = self.mask(r.read().decode(errors="replace"))
                status = r.status
        except urllib.error.HTTPError as exc:
            return exc.code, self.mask(exc.read().decode(errors="replace"))[:300]
        except (urllib.error.URLError, OSError) as exc:
            raise Unreachable(self.mask(f"{type(exc).__name__}: {exc}")) from None
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text[:300]

    def stream(self, body: dict, timeout: float) -> dict:
        """One streamed chat completion: the client's times, the text and tool calls, usage, and the ``tensorfold`` and
        ``speculative`` blocks of whichever chunk carries them (the engine puts them on the last)."""

        body = dict(body, stream=True, stream_options={"include_usage": True})
        if self.dry:
            print(f"DRY POST {self.base}/v1/chat/completions {json.dumps(shrink(body))}", flush=True)
            return {"dry": True}
        out: dict[str, Any] = {"status": None, "error": None, "content": "", "reasoning": "", "calls": {}, "usage": None,
                               "tensorfold": None, "speculative": None, "finish": None, "t_sent": time.monotonic(),
                               "t_first": None, "t_last": None}
        try:
            with urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=timeout) as r:
                out["status"] = r.status
                for raw in r:
                    line = raw.decode(errors="replace").strip()
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
                        out["error"] = self.mask(json.dumps(chunk["error"]))[:300]
                        break
                    for k in ("usage", "tensorfold", "speculative"):
                        if isinstance(chunk.get(k), dict):
                            out[k] = chunk[k]
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                            now = time.monotonic()
                            out["t_first"] = out["t_first"] or now
                            out["t_last"] = now
                        out["content"] += d.get("content") or ""
                        out["reasoning"] += d.get("reasoning_content") or ""
                        for tc in d.get("tool_calls") or []:
                            slot = out["calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                        out["finish"] = ch.get("finish_reason") or out["finish"]
        except urllib.error.HTTPError as exc:
            out["status"], out["error"] = exc.code, self.mask(exc.read().decode(errors="replace"))[:300]
        except (urllib.error.URLError, OSError) as exc:
            if out["status"] is None:
                raise Unreachable(self.mask(f"{type(exc).__name__}: {exc}")) from None
            out["error"] = self.mask(f"{type(exc).__name__}: {exc}")
        out["t_end"] = time.monotonic()
        out["content"] = self.mask(out["content"])
        return out


def num(v: Any, nd: int) -> float | None:
    return round(v, nd) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def summary(r: dict) -> dict:
    """One reply as the report keeps it."""

    usage, tf, spec = r.get("usage") or {}, r.get("tensorfold") or {}, r.get("speculative") or {}
    n = usage.get("completion_tokens")
    span = r["t_last"] - r["t_first"] if r["t_first"] and r["t_last"] else None
    calls = [r["calls"][i] for i in sorted(r["calls"])]
    text = (r["content"] or "") + "".join(f"\n<call {c['name']} {c['arguments']}>" for c in calls)
    return {
        "status": r["status"], "error": r["error"], "finish": r["finish"], "completion_tokens": n,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "ttft_s": round(r["t_first"] - r["t_sent"], 4) if r["t_first"] else None,
        "decode_tps_client": round((n - 1) / span, 3) if span and isinstance(n, int) and n > 1 else None,
        "decode_tps_server": num(tf.get("tokens_per_second"), 3), "ttft_s_server": num(tf.get("time_to_first_token"), 4),
        "served_drafts": tf.get("drafts") if isinstance(tf.get("drafts"), bool) else None,
        "sampling": tf.get("sampling"), "token_sha256": tf.get("token_sha256"), "token_sha": tf.get("token_sha"),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "speculative": {k: spec.get(k) for k in ("rounds", "drafted", "accepted", "acceptance_rate", "tokens_per_round")}
        if spec else None,
        "tool_calls": [c["name"] for c in calls], "content_head": (r["content"] or "")[:80],
    }


# -- the nodes' readings ------------------------------------------------------------------------------------------------
def sample(ssh: str | None) -> dict:
    """One read-only sample of a node (this one, or ``ssh`` target): device state, GPU and hottest ACPI temperature, the
    device limits (GUARD_*, read, never assumed)."""

    argv = (["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, SAMPLE_SH] if ssh else ["bash", "-c", SAMPLE_SH])
    try:
        text = subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "unread", "error": str(exc)[:200]}
    parts = (text[-1] if text else "").split("|")
    if len(parts) != 4:
        return {"state": "unread", "error": "no sample"}
    gpu = [x.strip() for x in parts[1].split(",")]

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    acpi = f(parts[2])
    limits = dict(kv.split("=", 1) for kv in parts[3].split(";") if "=" in kv)
    return {"t": round(time.time(), 2), "state": parts[0].strip() or "missing",
            "gpu_c": f(gpu[0]) if gpu and gpu[0] else None,
            "acpi_c": acpi / 1000 if acpi is not None and acpi > 1000 else acpi, "limits": limits}


class Nodes:
    """The head and (with a target) the worker. ``off``: no samples, no waits."""

    def __init__(self, worker: str | None, off: bool, wait_s: float) -> None:
        self.targets = {} if off else {"head": None, **({"worker": worker} if worker else {})}
        self.wait_s = wait_s

    def read(self) -> dict:
        return {name: sample(ssh) for name, ssh in self.targets.items()}

    def ready(self, what: str) -> dict:
        """Before a measurement: both nodes ``normal``. Waits (``wait_s``) through ``throttled``; stops on ``emergency``,
        on a node that cannot be read, or at the end of the wait."""

        t0 = time.time()
        while True:
            now = self.read()
            states = {k: v["state"] for k, v in now.items()}
            if all(s == "normal" for s in states.values()):
                return now
            if any(s == "emergency" for s in states.values()):
                raise Stopped(f"device state emergency before {what}: {states}")
            if any(s in ("unread", "missing") for s in states.values()):
                raise Stopped(f"no device state before {what}: {states} (spark-thermal-guard, ssh to the worker)")
            if time.time() - t0 > self.wait_s:
                raise Stopped(f"device state not normal for {self.wait_s:.0f} s before {what}: {states}")
            print(f"[drafts] waiting for normal before {what}: {states}", flush=True)
            time.sleep(10)


class Stopped(Exception):
    """The run stops: a device state, a missing sample, or a greedy reply that differs from the reference."""


# -- the run ------------------------------------------------------------------------------------------------------------
def load_prompts(path: Path = PROMPTS) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def plan(prompts: dict, classes: list[str], reps: int) -> list[dict]:
    """The requests in order: per class, each prompt greedy ``reps`` times, then its first prompt keyed-sampled."""

    s = prompts["sampled"]
    out = []
    for cls in classes:
        items = prompts["classes"][cls]
        for p in items:
            out.append({"class": cls, "id": p["id"], "variant": "greedy", "reps": reps, "prompt": p, "sampling": {"temperature": 0}})
        out.append({"class": cls, "id": items[0]["id"], "variant": "sampled", "reps": reps, "prompt": items[0],
                    "sampling": {"temperature": s["temperature"], "top_p": s["top_p"], "seed": s["seed"]}})
    return out


def body_of(g: dict, model: str, max_tokens: int) -> dict:
    p = g["prompt"]
    body = {"model": model, "messages": p["messages"], "max_tokens": max_tokens, "ignore_eos": False, **g["sampling"], **OFF}
    if p.get("tools"):
        body["tools"], body["tool_choice"] = p["tools"], "auto"
    return body


def reference_shas(path: str) -> dict:
    """The greedy token_sha256 (else text sha) of each prompt of an earlier run."""

    ref = json.loads(Path(path).read_text())
    out: dict[str, set] = {}
    for r in ref.get("requests", []):
        if r.get("variant") == "greedy" and r.get("status") == 200:
            out.setdefault(r["id"], set()).add(identity(r))
    return out


def identity(r: dict) -> str | None:
    if r.get("token_sha256"):
        return "sha256:" + r["token_sha256"]
    return "text:" + r["text_sha256"] if r.get("text_sha256") else None


def run(a: argparse.Namespace) -> int:
    prompts, digest = load_prompts()
    classes = [c for c in a.classes.split(",") if c]
    bad = [c for c in classes if c not in prompts["classes"]]
    if bad:
        raise SystemExit(f"unknown class(es) {bad}; the file has {sorted(prompts['classes'])}")
    api = Api(a.base, read_key(a.api_key_file), a.dry_run)
    src = LogSource(a.server_log, a.server_log_cmd) if (a.server_log or a.server_log_cmd) else None
    nodes = Nodes(a.worker_ssh or None, a.no_stamp or a.dry_run, a.thermal_wait)
    ref = reference_shas(a.reference) if a.reference else None
    groups = plan(prompts, classes, a.reps)
    report: dict[str, Any] = {
        "kind": "drafts", "label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": a.dry_run,
        "base": api.base, "model": a.model, "prompts_sha256": digest, "max_tokens": a.max_tokens, "reps": a.reps,
        "classes": classes, "sampled": prompts["sampled"], "server_log": src.describe() if src else None,
        "worker_sampled": "worker" in nodes.targets, "requests": [], "groups": [], "calibration": None, "stopped": None,
    }
    if a.dry_run:
        print(f"DRY server log: {src.describe() if src else 'none'}; reference: {a.reference or 'none'}", flush=True)
    rc = 0
    try:
        st, models = api.get("/v1/models")
        if not a.dry_run and st != 200:
            raise Unreachable(f"/v1/models answered {st}: {models}")
        report["server_before"] = {"models": models, "health": api.get("/health")[1]}
        # one short request first, not counted: the first reply after a load may pay one-time costs
        warm = {"model": a.model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8,
                "temperature": 0, **OFF}
        api.stream(warm, a.timeout)
        for g in groups:
            what = f"{g['class']}/{g['id']}/{g['variant']}"
            before = nodes.ready(what) if nodes.targets else {}
            t0 = time.time()
            for rep in range(g["reps"]):
                m = safe_mark(src) if src and not a.dry_run else None
                res = api.stream(body_of(g, a.model, a.max_tokens), a.timeout)
                if res.get("dry"):
                    continue
                rec = {"class": g["class"], "id": g["id"], "variant": g["variant"], "rep": rep, **summary(res)}
                if src:
                    rec["log"] = wait_attribute(src, m, rec, a.log_wait)
                report["requests"].append(rec)
                spec = rec["speculative"] or {}
                tally = (rec.get("log") or {}).get("tally")
                print(f"[drafts] {what} rep {rep}: {rec['completion_tokens']} tokens, {rec['decode_tps_client']} tok/s "
                      f"(server {rec['decode_tps_server']}), ttft {rec['ttft_s']} s, rounds {spec.get('rounds')} "
                      f"accepted {spec.get('accepted')}/{spec.get('drafted')}"
                      + (f", tau {acceptance(tally)['tau']}" if tally and tally["rounds"] else "")
                      + (f", ERROR {rec['error']}" if rec["error"] or rec["status"] != 200 else ""), flush=True)
                if rec["error"] or rec["status"] != 200:
                    rc = 1
                if ref is not None and g["variant"] == "greedy" and rec["status"] == 200:
                    want = ref.get(g["id"])
                    if want and identity(rec) not in want:
                        rc = 1
                        raise Stopped(f"FAIL: greedy {g['id']} rep {rep} differs from the reference "
                                      f"({identity(rec)} not in {sorted(want)})")
            after = nodes.read() if nodes.targets else {}
            report["groups"].append({"class": g["class"], "id": g["id"], "variant": g["variant"], "t0": round(t0, 2),
                                     "t1": round(time.time(), 2), "before": before, "after": after})
        if src and not a.dry_run:
            calib = [x for x in src.lines() if CALIB.search(x)]
            report["calibration"] = CALIB.search(calib[-1]).group(0) if calib else None
    except Unreachable as exc:
        print(api.mask(f"[drafts] cannot query {api.base}: {exc}"), file=sys.stderr)
        rc = 2
        report["stopped"] = f"unreachable: {exc}"
    except Stopped as exc:
        print(f"[drafts] stopped: {exc}", file=sys.stderr, flush=True)
        rc = rc or 3
        report["stopped"] = str(exc)
    finally:
        report["classes_summary"] = summarize(report["requests"])
        if not a.dry_run:
            try:
                report["server_after"] = {"health": api.get("/health")[1]}
            except Unreachable as exc:
                report["server_after"] = {"error": str(exc)}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(api.mask(json.dumps(report, indent=1)))
            print(f"[drafts] -> {a.out}", flush=True)
    for cls, v in report["classes_summary"].items():
        print(f"[drafts] {a.label} {cls}: {v['tps_client_median']} tok/s client, {v['tps_server_median']} server, "
              f"tau {(v['dspark'] or {}).get('tau')}, {v['requests']} requests, {v['errors']} errors", flush=True)
    return rc


def safe_mark(src: LogSource) -> int | None:
    try:
        return src.mark()
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[drafts] warning: cannot read the server log ({str(exc)[:200]})", file=sys.stderr, flush=True)
        return None


def wait_attribute(src: LogSource, mark: int | None, rec: dict, wait_s: float) -> dict:
    """Read the lines the log gained during a request until its done line is there (or ``wait_s`` passes)."""

    spec = rec.get("speculative") or {}
    if mark is None:
        return {"tally": None, "done": None, "note": "log unreadable before the request", "lines": 0, "orphans": 0}
    deadline = time.monotonic() + wait_s
    while True:
        try:
            lines = src.since(mark)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"tally": None, "done": None, "note": f"log unreadable: {str(exc)[:200]}", "lines": 0, "orphans": 0}
        res = attribute(lines, rec.get("completion_tokens"), rec.get("token_sha"), spec.get("rounds"))
        if res["done"] is not None or time.monotonic() > deadline:
            return res
        time.sleep(0.5)


def med(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 3) if xs else None


def summarize(reqs: list[dict]) -> dict:
    """Per class and variant ("code", "code/sampled"): medians, sums of the speculative counters, the dspark tally."""

    out: dict[str, dict] = {}
    keys = []
    for r in reqs:
        k = r["class"] if r["variant"] == "greedy" else f"{r['class']}/{r['variant']}"
        if k not in keys:
            keys.append(k)
    for k in keys:
        cls, _, variant = k.partition("/")
        rs = [r for r in reqs if r["class"] == cls and r["variant"] == (variant or "greedy")]
        ok = [r for r in rs if r["status"] == 200 and not r["error"]]
        specs = [r["speculative"] for r in ok if r.get("speculative")]
        rounds = sum(s.get("rounds") or 0 for s in specs)
        drafted = sum(s.get("drafted") or 0 for s in specs)
        accepted = sum(s.get("accepted") or 0 for s in specs)
        tpr = sum((s.get("tokens_per_round") or 0) * (s.get("rounds") or 0) for s in specs)
        tally = tally_sum([(r.get("log") or {}).get("tally") for r in ok])
        logged = [r for r in ok if "log" in r]
        out[k] = {
            "requests": len(rs), "errors": len(rs) - len(ok), "tokens": sum(r["completion_tokens"] or 0 for r in ok),
            "tps_client_median": med([r["decode_tps_client"] for r in ok]),
            "tps_server_median": med([r["decode_tps_server"] for r in ok]),
            "ttft_s_median": med([r["ttft_s"] for r in ok]),
            "speculative": {"rounds": rounds, "drafted": drafted, "accepted": accepted,
                            "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
                            "tokens_per_round": round(tpr / rounds, 4) if rounds else None} if specs else None,
            "tally": tally, "dspark": acceptance(tally),
            "attributed": sum(1 for r in logged if (r["log"] or {}).get("tally")) if logged else None,
            "unattributed": [f"{r['id']} rep {r['rep']}: {r['log']['note']}" for r in logged
                             if r["log"].get("note") and not r["log"].get("tally")] if logged else [],
        }
    return out


# -- compare ------------------------------------------------------------------------------------------------------------
def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for p in paths:
        js = json.loads(Path(p).read_text())
        if js.get("kind") != "drafts":
            raise SystemExit(f"{p}: not a bench/drafts.py run")
        js["_path"] = p
        runs.append(js)
    return runs


def compare(off: list[dict], on: list[dict]) -> dict:
    """The comparison of drafts-off and drafts-on runs (each side's reps pooled): rows per class and variant, the greedy
    identity per prompt, and the verdict with every reason for a FAIL or a note."""

    fails, notes = [], []
    for side, label in ((off, "drafts-off"), (on, "drafts-on")):
        for r in side:
            if r.get("label") != label:
                fails.append(f"{r['_path']} is labelled {r.get('label')!r}, given as {label}")
            if r.get("dry_run"):
                fails.append(f"{r['_path']} is a dry run")
            if not any(g.get("before") for g in r.get("groups", [])):
                notes.append(f"{r['_path']} ({label}) carries no device readings (--no-stamp or DRY_RUN)")
            if r.get("stopped"):
                notes.append(f"{r['_path']} stopped early: {r['stopped']}")
            served = {q.get("served_drafts") for q in r.get("requests", []) if q.get("served_drafts") is not None}
            if (label == "drafts-off" and True in served) or (label == "drafts-on" and False in served):
                fails.append(f"{r['_path']} ({label}): the server reported tensorfold.drafts {sorted(served)}")
            for g in r.get("groups", []):
                for when in ("before", "after"):
                    states = {n: s.get("state") for n, s in (g.get(when) or {}).items()}
                    if any(s != "normal" for s in states.values()):
                        notes.append(f"{label} {g['class']}/{g['id']}/{g['variant']} {when}: device state {states}: "
                                     "that class's speed is not comparable")
    every = off + on
    for k in ("prompts_sha256", "max_tokens", "model"):
        vals = {json.dumps(r.get(k)) for r in every}
        if len(vals) > 1:
            fails.append(f"the runs differ in {k}: {sorted(vals)}")
    req = {"off": [q for r in off for q in r.get("requests", [])], "on": [q for r in on for q in r.get("requests", [])]}
    s_off, s_on = summarize(req["off"]), summarize(req["on"])
    rows = []
    for k in list(dict.fromkeys([*s_off, *s_on])):
        a, b = s_off.get(k) or {}, s_on.get(k) or {}

        def ratio(x, y):
            return round(y / x, 4) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x else None

        rows.append({"class": k, "tps_off": a.get("tps_client_median"), "tps_on": b.get("tps_client_median"),
                     "speedup": ratio(a.get("tps_client_median"), b.get("tps_client_median")),
                     "server_off": a.get("tps_server_median"), "server_on": b.get("tps_server_median"),
                     "server_speedup": ratio(a.get("tps_server_median"), b.get("tps_server_median")),
                     "dspark": b.get("dspark"), "speculative_on": b.get("speculative"),
                     "errors": (a.get("errors") or 0) + (b.get("errors") or 0),
                     "unattributed_on": b.get("unattributed") or []})
        if (a.get("errors") or 0) + (b.get("errors") or 0):
            (fails if "/" not in k else notes).append(f"{k}: {a.get('errors') or 0} off / {b.get('errors') or 0} on requests failed")
        if any(r.get("server_log") for r in on) and b.get("requests") and not b.get("tally"):
            notes.append(f"{k}: no dspark acceptance line matched a drafts-on request (TF_MM3_DSPARK_STATS=1 on rank 0? "
                         f"{'; '.join((b.get('unattributed') or [])[:2])})")
    ident = []
    for variant in ("greedy", "sampled"):
        ids = list(dict.fromkeys(q["id"] for q in req["off"] + req["on"] if q["variant"] == variant))
        for pid in ids:
            got = {}
            for side in ("off", "on"):
                qs = [q for q in req[side] if q["id"] == pid and q["variant"] == variant and q["status"] == 200
                      and not q["error"]]
                got[side] = [identity(q) for q in qs]
            so, sn = set(got["off"]), set(got["on"])
            if not got["off"] or not got["on"]:
                verdict = "MISSING"
            elif None in so | sn:
                verdict = "NO-SHA"
            elif len(so) > 1:
                verdict = "OFF-NOT-REPEATABLE"
            elif so != sn:
                verdict = "DIFFERENT"
            else:
                verdict = "EQUAL"
            ident.append({"id": pid, "variant": variant, "verdict": verdict, "off": sorted(so, key=str),
                          "on": sorted(sn, key=str), "reps": [len(got["off"]), len(got["on"])]})
            if verdict != "EQUAL":
                msg = f"{variant} {pid}: {verdict} (off {sorted(so, key=str)}, on {sorted(sn, key=str)})"
                (fails if variant == "greedy" else notes).append(msg)
    speedups = [r["speedup"] for r in rows if "/" not in r["class"] and r["speedup"] is not None]
    alternate = bool(speedups) and all(abs(s - 1) <= FEW for s in speedups)
    if alternate:
        notes.append(f"every greedy speedup is within {FEW:.0%} of 1: alternate off / on once more (docs/DRAFTS.md)")
    return {"verdict": "FAIL" if fails else "PASS", "fails": fails, "notes": notes, "rows": rows, "identity": ident,
            "alternate": alternate, "off": [r["_path"] for r in off], "on": [r["_path"] for r in on]}


def fmt(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def print_compare(c: dict) -> None:
    print(f"off: {', '.join(c['off'])}\non:  {', '.join(c['on'])}\n")
    print(f"{'class':<20} {'tok/s off':>9} {'tok/s on':>9} {'speedup':>8} {'srv off':>8} {'srv on':>8} {'srv x':>6} "
          f"{'tau':>6} {'tok/rnd':>7} {'acc(all)':>8}  conditional by depth / cumulative by depth")
    for r in c["rows"]:
        d, s = r["dspark"] or {}, r["speculative_on"] or {}
        cond = "/".join(fmt(x) for x in d.get("conditional") or []) or "-"
        cum = "/".join(fmt(x) for x in d.get("cumulative") or []) or "-"
        print(f"{r['class']:<20} {fmt(r['tps_off']):>9} {fmt(r['tps_on']):>9} {fmt(r['speedup'], 3):>8} "
              f"{fmt(r['server_off']):>8} {fmt(r['server_on']):>8} {fmt(r['server_speedup'], 3):>6} {fmt(d.get('tau'), 3):>6} "
              f"{fmt(s.get('tokens_per_round')):>7} {fmt(s.get('acceptance_rate'), 3):>8}  {cond}  |  {cum}")
    print("\ntoken identity (token_sha256 of every rep, off against on):")
    for i in c["identity"]:
        print(f"  {i['variant']:<8} {i['id']:<16} {i['verdict']:<18} reps off/on {i['reps'][0]}/{i['reps'][1]}")
    for n in c["notes"]:
        print(f"note: {n}")
    for f in c["fails"]:
        print(f"FAIL: {f}")
    print(f"\ncompare: {c['verdict']}")


# -- a saved log --------------------------------------------------------------------------------------------------------
def log_report(src: LogSource) -> dict:
    lines = src.lines()
    totals = [t for t in map(parse_tally, lines) if t and t["kind"] == "total"]
    streams = [t for t in map(parse_tally, lines) if t and t["kind"] == "stream"]
    total = totals[-1] if totals else None
    summed = tally_sum(streams)
    return {
        "source": src.describe(), "lines": len(lines),
        "calibration": [CALIB.search(x).group(0) for x in lines if CALIB.search(x)],
        "total": total, "total_acceptance": acceptance(total), "streams": len(streams), "streams_sum": summed,
        "streams_acceptance": acceptance(summed),
        "draft_lines": [x for x in lines if "dspark draft:" in x],
        "budget_lines": [x for x in lines if BUDGET.search(x)][:20],
        "done_lines": sum(1 for x in lines if DONE.search(x)),
    }


def print_log(r: dict) -> None:
    print(f"{r['source']}: {r['lines']} lines, {r['done_lines']} done lines, {r['streams']} dspark stream lines")
    for x in r["calibration"] or ["(no dspark calibration line)"]:
        print(f"  {x}")
    for name, acc in (("total", r["total_acceptance"]), ("sum of the stream lines", r["streams_acceptance"])):
        if acc:
            print(f"  acceptance {name}: rounds {acc['rounds']}, tau {acc['tau']}, conditional "
                  f"{'/'.join(fmt(x) for x in acc['conditional'])}, cumulative {'/'.join(fmt(x) for x in acc['cumulative'])}")
        else:
            print(f"  acceptance {name}: none")
    for x in r["draft_lines"] + r["budget_lines"]:
        print(f"  {x.strip()[:220]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="one run against a live server")
    r.add_argument("--label", required=True, choices=LABELS)
    r.add_argument("--base", default=default_base())
    r.add_argument("--model", default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3")
    r.add_argument("--api-key-file", default=None, help="the server's key file (default: $API_KEY_FILE if present)")
    r.add_argument("--reps", type=int, default=3, help="requests a prompt and variant")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--classes", default="code,prose,structured,agent")
    r.add_argument("--timeout", type=float, default=900, help="seconds a request")
    r.add_argument("--server-log", default=None, help="rank 0's log file (also .gz)")
    r.add_argument("--server-log-cmd", default=None, help="a command printing rank 0's log, e.g. 'docker logs -t NAME'")
    r.add_argument("--log-wait", type=float, default=15, help="seconds to wait for a request's done line in the log")
    r.add_argument("--reference", default=None, help="a drafts-off run: stop at the first greedy reply that differs")
    r.add_argument("--worker-ssh", default=os.environ.get("WORKER_SSH") or os.environ.get("WORKER") or "",
                   help="the worker's ssh target for its readings (default $WORKER_SSH, else $WORKER; empty: head only)")
    r.add_argument("--thermal-wait", type=float, default=900, help="seconds to wait for normal before a prompt")
    r.add_argument("--no-stamp", action="store_true", help="no node readings (and no waits)")
    r.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "1")
    r.add_argument("--out", default="")
    c = sub.add_parser("compare", help="drafts-off runs against drafts-on runs")
    c.add_argument("--off", nargs="+", required=True)
    c.add_argument("--on", nargs="+", required=True)
    c.add_argument("--out", default="")
    lg = sub.add_parser("log", help="a rank 0 log: calibration, acceptance totals, budget lines")
    lg.add_argument("path", nargs="?")
    lg.add_argument("--cmd", default=None)
    lg.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.command == "run":
        if a.reps < 1 or a.max_tokens < 1:
            ap.error("--reps and --max-tokens are positive")
        if a.server_log and a.server_log_cmd:
            ap.error("--server-log or --server-log-cmd, not both")
        return run(a)
    if a.command == "compare":
        res = compare(load_runs(a.off), load_runs(a.on))
        print_compare(res)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        return 0 if res["verdict"] == "PASS" else 1
    if bool(a.path) == bool(a.cmd):
        ap.error("log: a PATH or --cmd")
    res = log_report(LogSource(a.path, a.cmd))
    print_log(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
