# Drafts: measuring the dspark draft on the pair

> **NOT MEASURED.** Whether MiniMax's own dspark draft pays on this MoE (a verify window reads more experts than a
> one-token step, [dspark-plan section 8](dspark-plan.md)) has not been run on two Sparks. This page is the plan for the
> next combined TP2 window: the profiles, the order of the runs, when to stop, and what to record in
> [RESULTS](RESULTS.md). The production profile keeps `DRAFTS=0` until it has run and passed.

Terms as in [OPERATIONS](OPERATIONS.md): **head** = rank 0 (the API, this checkout), **worker** = rank 1.

## 1. What is measured

One request at a time (`PARALLEL=1`, C1), thinking off, `max_tokens` 256, `ignore_eos` off, on four fixed workload
classes (`bench/data/drafts-prompts.json`): code, prose, structured (a JSON array, a Markdown table) and agent (a
tool-calling turn with two tools, and the same turn's follow-up with both tool results). Each prompt runs greedy
(temperature 0) three times, and each class's first prompt runs keyed-sampled (the file's temperature, top_p and seed)
three times. Per request `bench/drafts.py` keeps the completion tokens, the client's TTFT and decode tok/s, the server's
`tensorfold.tokens_per_second`, `speculative.{rounds, drafted, accepted, acceptance_rate, tokens_per_round}` and
`tensorfold.token_sha256` ([bench/README.md](../bench/README.md)).

`speculative.drafted` / `accepted` count every kind of draft, context copies too. The draft's own acceptance comes from
rank 0's log: with `TF_MM3_DSPARK_STATS=1` the engine prints, for each finished stream and just before the server's
`done <id> ...` line,

```
dspark acceptance stream: rounds=R reached=a1/a2/.../a8 kept=k1/k2/.../k8 kept_per_round=X conf1_rounds=c0/../c9 conf1_kept=d0/../d9
```

where `rounds` counts the head rounds, `reached[i]` the head rounds that verified a draft at depth i + 1 and `kept[i]` how
many of those kept it. The bench matches each line to its request (the lines the log gained during that request, paired
with the done line that carries the reply's token count and short sha) and reports per class: the conditional acceptance
by position `kept[i] / reached[i]`, the cumulative `kept[i] / rounds`, and **tau = 1 + sum(kept) / rounds**, the tokens a
head round yields. The confidence bins (`conf1_*`, the first draft's confidence in 0.1 steps) are kept beside them. At exit
the engine prints `dspark acceptance total: <the same fields>` either way.

## 2. The two profiles

| | drafts off (the control) | drafts on |
| --- | --- | --- |
| file | `config/smoke.env` (from `config/smoke.env.example`, unchanged) | `config/drafts.env` (from `config/drafts.env.example`) |
| serve flags | `--no-drafts` on both ranks | `--drafter /model/dspark` on both ranks |
| `TF_MM3_DSPARK_STATS` | not set | `1` (both ranks get it; rank 0 logs) |
| `TF_MM3_DSPARK_CHANCES`, `TF_MM3_CALIB` | not set | `1`, `auto`: each round's depth by the draft's confidences against the verify costs timed at boot (window-all-3's rule; `TF_MM3_DRAFT_DEPTH` unset, the block caps a round) |
| `MEM_GATE_GIB` | 113 | 115: the draft is spent before any KV, 2.19 GiB a rank (2.03 GiB with one stream's 21 MB K/V ring, plus 8 ring snapshots of 21 MB, `native.draftCost`) |
| everything else | the smoke profile | the same, value for value (tests/test_repo.py `DraftsProfileTests`, `DraftsScriptTests`) |

`python3 scripts/quant_budget.py --serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8 --drafter` gives the
drafts-on gate: KV room 8.26 GiB at 117.1 GiB available (188,416 tokens, 1,472 pages), 2.52 GiB more than the
131,072-token context needs. At 113 GiB available it does not fit (94,848 tokens); `start.sh`'s gate refuses such a load
before anything starts. The narrower window both halves use then is in the profile: `NARROW_RUN: CONTEXT=65536
MEM_GATE_GIB=112` (section 5). `TF_MM3_CALIB=auto` times the verify widths at boot (~28 s, or reads a matching cached
table); record its `dspark calibration: ...` line. A fixed cap (`TF_MM3_DRAFT_DEPTH=3`, window-all-2's rule) is the
second arm when the window has time.

Both files must name the same `WORKER`, `ENGINE_COMMIT`, `ENGINE_PATCHES` and `MODEL_DIR`: one image for both halves (a
different series would rebuild it between them), an engine that includes mm3/dspark. Never set `KV_SPLIT_DEDUP`,
`KV_SPLIT_EARLY` or `TF_TP_ROCE_CHUNKED_KB` for this measurement (in a profile or `EXTRA_ENV`).

## 3. Before the window (once, no server needed)

```bash
cp config/drafts.env.example config/drafts.env   # WORKER, ENGINE_COMMIT (and MODEL_DIR) exactly as in config/smoke.env
diff <(grep -vE '^[[:space:]]*(#|$)' config/smoke.env) <(grep -vE '^[[:space:]]*(#|$)' config/drafts.env)
#   only DRAFTS, DRAFTER, TF_MM3_DSPARK_STATS, TF_MM3_DSPARK_CHANCES, TF_MM3_CALIB and MEM_GATE_GIB may show
CONFIG=config/drafts.env scripts/serve.sh preflight static     # the pack's dspark/ folder on both nodes, by size
DRY_RUN=1 python3 bench/drafts.py run --label drafts-on        # the requests it will send; nothing is sent
tests/run.sh                                                   # on the head's CPU
```

## 4. The window, in order

Run on the head from this checkout. The watchdog stands down for the whole window (`systemctl --user stop
tf-watchdog.timer` where it is installed, or touch `WATCH_LEASE`), so no probe request lands between the bench's.

```bash
R=~/drafts-$(date +%Y%m%d-%H%M); mkdir -p "$R"
W=$(sed -n 's/^WORKER=//p' config/drafts.env | tr -d '"')
LOG='docker logs -t minimax-m3-flash-tf'          # rank 0's log, with docker's times (stdout and stderr in order)
```

1. **Drafts off: start the control.**

   ```bash
   CONFIG=config/smoke.env scripts/serve.sh restart 2>&1 | tee "$R/off-start.txt"
   python3 bench/drafts.py log --cmd "$LOG" --out "$R/off-r0-boot.json"
   python3 bench/drafts.py log --cmd "ssh $W $LOG" --out "$R/off-r1-boot.json"
   ```

2. **Drafts off: capture and bench.**

   ```bash
   CONFIG=config/smoke.env scripts/check-drafted.sh capture "$R/plain"
   python3 bench/drafts.py run --label drafts-off --worker-ssh "$W" --server-log-cmd "$LOG" --out "$R/off-1.json"
   ```

3. **Drafts on: restart with the drafts profile.**

   ```bash
   CONFIG=config/drafts.env scripts/serve.sh restart 2>&1 | tee "$R/on-start.txt"
   python3 bench/drafts.py log --cmd "$LOG" --out "$R/on-r0-boot.json"          # the calibration and budget lines
   python3 bench/drafts.py log --cmd "ssh $W $LOG" --out "$R/on-r1-boot.json"
   ```

4. **Drafts on: capture, compare, bench.** `check-drafted.sh compare` must print EQUAL for all seven captures before
   the bench runs; the bench stops at the first greedy reply that differs from the drafts-off run (`--reference`).

   ```bash
   CONFIG=config/drafts.env scripts/check-drafted.sh capture "$R/drafted"
   scripts/check-drafted.sh compare "$R/plain" "$R/drafted"
   python3 bench/drafts.py run --label drafts-on --worker-ssh "$W" --server-log-cmd "$LOG" \
     --reference "$R/off-1.json" --out "$R/on-1.json"
   python3 bench/drafts.py compare --off "$R/off-1.json" --on "$R/on-1.json" --out "$R/compare-1.json"
   ```

5. **Alternate once more** when `compare` says so (every greedy speedup within 3 % of 1): the same two halves again,
   then all four runs together.

   ```bash
   CONFIG=config/smoke.env scripts/serve.sh restart 2>&1 | tee "$R/off-start-2.txt"
   python3 bench/drafts.py run --label drafts-off --worker-ssh "$W" --server-log-cmd "$LOG" --out "$R/off-2.json"
   CONFIG=config/drafts.env scripts/serve.sh restart 2>&1 | tee "$R/on-start-2.txt"
   python3 bench/drafts.py run --label drafts-on --worker-ssh "$W" --server-log-cmd "$LOG" \
     --reference "$R/off-1.json" --out "$R/on-2.json"
   python3 bench/drafts.py compare --off "$R/off-1.json" "$R/off-2.json" --on "$R/on-1.json" "$R/on-2.json" \
     --out "$R/compare-2.json"
   ```

6. **Stop the drafts-on server and read its totals.** `scripts/serve.sh stop` saves each rank's whole log, gzipped,
   under `~/.cache/tensorfold-minimax-m3/logs/` on its own node; the newest rank 0 file holds `dspark acceptance total`
   and the `dspark draft: N passes, ...` line (when the engine ended cleanly; otherwise the sum of the stream lines
   stands in).

   ```bash
   CONFIG=config/drafts.env scripts/serve.sh stop
   python3 bench/drafts.py log "$(ls -t ~/.cache/tensorfold-minimax-m3/logs/*-rank0.log.gz | head -n 1)" --out "$R/on-final.json"
   ```

   Then the pair goes back to what the window's schedule says (the production profile keeps `DRAFTS=0`), and the
   watchdog timer is started again where it was stopped.

## 5. Stop conditions

Stop the drafts measurement at the first of these, keep `$R` and both ranks' logs, and add the finding to
[KNOWN-ISSUES](KNOWN-ISSUES.md):

- **Device state.** `start.sh` refuses unless both nodes read `normal`; the bench waits (up to 900 s,
  `--thermal-wait`) before each prompt until both read `normal` and stops (exit 3) on `emergency` or a node it cannot
  read; memguard stops a rank on `emergency` and the pair watch the other. On any of these: `scripts/serve.sh stop`,
  read `memguard.log` on the node, do not clear a thermal-stop record before reading it, never raise a limit. A
  `throttled` stamp in a run (compare notes it) makes that class's speed not comparable: rerun that pair once both
  nodes are `normal`.
- **Any greedy mismatch.** `check-drafted.sh compare` prints DIFFERENT, the drafts-on bench exits 1 with "differs from
  the reference", or `bench/drafts.py compare` prints `FAIL` for a greedy prompt. That is a bug in the draft path, never
  an acceptable loss: no further drafted run, `CONFIG=config/smoke.env scripts/serve.sh restart` (drafts off) if the pair
  is to keep serving. A sampled difference in `bench/drafts.py compare` is a note; `check-drafted.sh` decides those.
- **The memory gate.** The drafts-on start stops at `MemFree ... still under MEM_GATE_GIB 115`, or the engine refuses the
  load (`the dspark draft takes N GiB of that budget`). Do not lower the gate, the reserve or a floor. Free the memory
  (other containers, a desktop session), or run **both** halves again from step 1 with the profile's narrower window from
  the environment, the same for both:
  `CONTEXT=65536 MEM_GATE_GIB=112 CONFIG=config/smoke.env scripts/serve.sh restart` and
  `CONTEXT=65536 MEM_GATE_GIB=112 CONFIG=config/drafts.env scripts/serve.sh restart`.
  memguard killing a rank, or MemAvailable under 4 GiB on either node while the draft runs, stops the measurement.
- **A failed request or a rank that exits** (a bench exit 1 with ERROR lines, `fail-fast` in `scripts/serve.sh status`).

## 6. What to record

In [RESULTS](RESULTS.md), with the window's date, `ENGINE_COMMIT` and the pack revision:

| Item | From |
| --- | --- |
| the per-class table: tok/s off and on (client and server), speedup, tau, conditional and cumulative acceptance by position, copy-inclusive acceptance | `bench/drafts.py compare` (`$R/compare-*.json`) |
| token identity: check-drafted's seven EQUAL lines, compare's greedy and sampled verdicts | the two compare commands |
| `dspark calibration: ...` (or that there was none) | `$R/on-r0-boot.json`, `$R/on-*.json` (`calibration`) |
| `dspark acceptance total` and `dspark draft: N passes, ...` | `$R/on-final.json` |
| rank memory, both halves: `minimax budget: ...` and the `[tensorfold] CUDA ...` pool line of each rank, and start's `Memory after the load` line | `$R/*-r0-boot.json`, `$R/*-r1-boot.json`, `$R/*-start.txt` |
| both nodes' device state, temperatures before and after each prompt, and the device limits | the runs' `groups` (`before` / `after`) |

## 7. Reading it

The draft pays when the measured decode speedup is above 1 on the classes that matter for this pair's use (agent first),
not when tau looks high: plan section 8's byte model puts the break-even of a full 7-draft block at about 3.4 tokens a
round and of a 4-row window at about 2.3, and copy drafts cost the same windows with no memory. Production keeps
`DRAFTS=0` / `DRAFTER=` until a window shows a speedup on every class with every greedy reply identical, twice
(alternated); then the profile change goes through [OPERATIONS section 7](OPERATIONS.md#7-qualifying-the-profile) like any
other.

## 8. What the bench assumes of the engine

- The `speculative` and `tensorfold` blocks ride on the stream's last chunk (top level, beside `choices`), as the server's
  `response_extras` puts them; `tensorfold.token_sha256` is the reply's token hash and `tensorfold.drafts` whether the
  server drafted. A run whose `drafts` contradicts its label fails the comparison.
- The stream line comes before its `done` line (the engine prints it as the stream is released). A stream line the
  request's lines hold after its done line is counted as an orphan, not given to it; one written only after the bench
  moved on would pair with the next request's done line, and a matched line with more head rounds than that reply's
  `speculative.rounds` is marked suspect. The done line is stdout and the stream line stderr: with `docker logs -t` the
  bench orders them by docker's times; a file without them keeps the file's order.
- Conditional acceptance is the engine's own definition, `kept[i] / reached[i]`; tau counts head rounds only.
