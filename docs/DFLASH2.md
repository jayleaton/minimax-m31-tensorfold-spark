# The DFlash2 draft: the `DRAFTS=dflash2` profile and its TP2 measurement

> **Not run at TP2.** The engine side is TensorFold's `mm3/dflash2` branch (the engine's, which this recipe describes
> and vendors nothing). Every script here is checked on the CPU only (`tests/run.sh`). Acceptance and speed on this pack
> are NOT MEASURED until the window in section 4 has run; [RESULTS](RESULTS.md) gets its numbers.

## 1. The draft

`--drafter <folder>` takes either MiniMax's dspark draft or a DFlash2 draft; the engine reads the kind from the folder's
`config.json` `architectures`: `DSparkMiniMaxDraftModel` is dspark, `DFlash2DraftModel` or `MiniMaxDFlash2DraftModel` is
DFlash2. There is no new flag.

| Checkpoint | Revision | License | Here |
| --- | --- | --- | --- |
| [`incoai/MiniMax-M3.1-Flash-DFlash2`](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2) | `ac07a7c1ecaae26a7ace595756124dca974b432d` | Apache-2.0 | **supported**: `DRAFTS=dflash2` |
| [`RadixArk/Minimax-M3.1-Flash-Dflash2`](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2) | `923bf1d4b567d5ccb71b0488c6e61b5e4794e92a` | none declared on its card | **not supported yet**: the engine refuses it (its block attention is causal by the reference's default rule, which is not built); no profile |

The incoai draft is public (no token). Its files, pinned in `scripts/config.sh` (`DFLASH2_PINS`) and checked on both
Sparks by `scripts/prepare.sh`:

| File | sha256 | Bytes |
| --- | --- | ---: |
| `config.json` | `64c32db3c3bc5de64a048d949c5acd91352ae9b0f02ef636e3382c838f1d0815` | |
| `README.md` | `8ec2de932fdd0e2d72a36ba17a0c593cfcba053609661a22fcb841047ae93282` | |
| `model.safetensors` | `ad861f4ddd8f7204cda1f407837403e9095e1fb1f36d4c9f4188185261cc4fd1` | 4,247,776,488 |

What it is (its `config.json`): 5 bf16 sliding-window layers (window 2048) of 64 query and 8 kv heads, MLP 12288, `fc`
over the hidden states of 6 tapped target layers (5, 16, 26, 37, 47, 58), dynamic convolutions in groups of 16, a
selector of rank 256 offering 32 candidates a row, blocks of 8 rows: an anchor and 7 drafts a pass, bidirectional inside
the block (`is_causal: false`). The target verifies every draft, so a drafted reply is the reply plain decode gives.

**The one-GPU exactness knob is not a serving setting.** `TF_MM3_DRAFT_TAPS=<ids>` with `TF_MM3_LAYERS=<n>` (a thin model)
remaps the draft's taps so it can be checked on one GPU. Every exported `TF_MM3_*` variable reaches both ranks
(`rank_env`), so never put either in `config/*.env` or the environment of a start.

## 2. The profile

| Setting | Value | What it does |
| --- | --- | --- |
| `DRAFTS` | `0` / `1` / `dflash2` | `0`: `--no-drafts`; `1`: copy drafts (and `DRAFTER` when set), as before; `dflash2`: copies and the DFlash2 draft. `DRAFTER` then defaults to `dflash2`; any other `DRAFTER` is refused |
| `DRAFTER` | empty / `dspark` / `dflash2` / a folder | `dflash2` is the folder below (also with `DRAFTS=1`) |
| `DFLASH2_DIR` | default: beside the pack | this Spark's folder of the draft (absolute or `~/`). `WEIGHTS=dir`: `MODEL_DIR`'s parent folder; otherwise `HF_CACHE/dflash2/` |
| `WORKER_DFLASH2_DIR` | default: `DFLASH2_DIR` when set, else beside the worker's pack | the worker's folder (`WORKER_MODEL_DIR`'s parent, or the worker's own HF cache) |

Each Spark's folder is mounted read-only at `/dflash2` in its rank and both ranks get the same `--drafter /dflash2`
(a follower must agree with rank 0 on the drafts). The serve profile (`MM31_SERVE_PROFILE`) reads
`drafts=dflash2,drafter=dflash2,dflash2=<revision>`; the banner says `drafts copies + the DFlash2 draft (...)`.

- `scripts/serve.sh prepare --dflash2` (or `scripts/prepare.sh --dflash2`): only the draft. It downloads the pinned
  revision into `DFLASH2_DIR` (`hf download --local-dir`, resumable; run with `HF_TOKEN` unset and
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, so no token of the head is sent), checks every file against its sha256 pin, copies the
  three files to the worker (`rsync --partial` over ssh, resumable) and checks them there. Each folder is checked in full
  once per revision; later runs check `config.json`'s sha256 and the weights' size (`SHA=1` checks in full again). A full
  `prepare` does the same when `DRAFTS=dflash2`. It refuses while the server runs: 4.2 GB through the page cache is GPU
  memory on GB10.
- `preflight` (with `DRAFTS=dflash2`): a DFlash2 class in each node's `config.json` and the pinned `config.json` and
  weights' size there.
- `start.sh`: refuses a folder whose `config.json` names no DFlash2 class (`not a DFlash2 draft`), and a missing folder on
  either Spark after setup; a `MiniMaxDFlash2DraftModel` folder is a warning (the engine refuses that layout).

## 3. Memory a rank

The engine's own cost model (`dflash2_config.cost` with a verify window of 16 rows, 2,048-row absorbs and 8 ring
snapshots), reproduced by `scripts/quant_budget.py --serve --drafter dflash2` and pinned in `tests/test_repo.py`:

| Term (one rank of two) | Bytes | GiB |
| --- | ---: | ---: |
| weights: the rank's heads and MLP rows (1,698,693,120), convs (189,235,200), `fc` (452,984,832), selector and codebooks (208,011,264), norms (150,016) | 2,549,074,432 | 2.374 |
| runtime estimate (CUDA module, graphs, NCCL; a planning figure, UNVERIFIED) | 536,870,912 | 0.500 |
| scratch (2,048 absorb rows, one block) | 59,568,384 | 0.055 |
| 8 ring snapshots (the prompt cache's) | 169,082,880 | 0.157 |
| **a ring a stream**: (2048 + 16) positions x 5 layers x K,V x 4 kv heads x 128 x bf16 | **21,135,360** | 0.020 |
| **one stream / four streams** | | **3.11 / 3.17** |

The dspark draft, for comparison: 1.34 GiB of weights and 21.05 MB a stream (2.03 GiB with one stream, its runtime
estimate and scratch included, docs/dspark-plan.md section 6; 2.19 GiB with its 8 ring snapshots, the calculator's
term, docs/DRAFTS.md). The whole DFlash2 file is 4.25 GB; each rank holds 2.55 GB
of it. The draft is spent before any KV, so the gate rises:

| Profile | `DRAFTS=0` | `DRAFTER=dspark` | `DRAFTS=dflash2` | Calculator |
| --- | ---: | ---: | ---: | --- |
| smoke, 131,072 x 1, all resident | 113 | 115 | **116** | `--serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8 [--drafter [dflash2]]` |
| production, 1,048,576 x 4, KV tier | 112 | 114 | **115** | `--serve --context 1048576 --parallel 4 --kv fp8 --reserve-gib 8 --kv-tier [--drafter [dflash2]]` |

With the KV tier the DFlash2 draft leaves a slice cache of 4.47 GiB (1.82 GiB needed): it fits. Give the gate on the
command line of a drafting start (`MEM_GATE_GIB=116 ...`); do not edit the floors.

## 4. The TP2 measurement window

The question: at C1 (one request at a time), how many of the DFlash2 draft's tokens does the target keep on this pack, by
position, and is drafted decode faster than plain? Token equality with plain decode is the hard gate; speed is reported,
not gated.

**Before the window.** The engine image is built from `mm3/dflash2` (`ENGINE_PATCHES`, `ENGINE_COMMIT`; `scripts/serve.sh
build`). With the pair stopped, `DRAFTS=dflash2 scripts/serve.sh prepare --dflash2` puts the draft on both Sparks. Pick the
profile under test (`CONFIG=config/smoke.env` for the first TP2 load, else `config/prod.env`) and keep it for every start
of the window; only `DRAFTS`, `DRAFTER` and `MEM_GATE_GIB` change. Touch `WATCH_LEASE` (or stop the watchdog's timer) for
the window, and keep other workloads off both nodes.

**The order** (on the head; `R` a fresh folder for this window, `<gate>` from the table in section 3):

```bash
export CONFIG=config/smoke.env                                   # the profile under test, the same for every start
R=~/drafts-$(date +%Y%m%d-%H%M)

# 1. the baseline: plain decode
DRAFTS=0 scripts/serve.sh restart
scripts/check-drafted.sh capture "$R/check-plain"
scripts/measure-drafts.sh run plain "$R"

# 2. copies + the DFlash2 draft (drafts on, then the same requests with "draft": false on this server)
DRAFTS=dflash2 MEM_GATE_GIB=<gate> scripts/serve.sh restart
scripts/check-drafted.sh capture "$R/check-dflash2"
scripts/check-drafted.sh compare "$R/check-plain" "$R/check-dflash2"
scripts/measure-drafts.sh run dflash2 "$R"

# 3. optional, the same window: the dspark draft, for the comparison with the dspark work
DRAFTS=1 DRAFTER=dspark MEM_GATE_GIB=<gate> scripts/serve.sh restart
scripts/check-drafted.sh capture "$R/check-dspark"
scripts/check-drafted.sh compare "$R/check-plain" "$R/check-dspark"
scripts/measure-drafts.sh run dspark "$R"

# 4. the report, then back to the profile's own drafts setting
scripts/measure-drafts.sh report "$R"                            # exit 1: a drafted reply differs from plain
scripts/serve.sh restart
```

`scripts/measure-drafts.sh run` checks that the running server's profile matches the label (`drafts=0`, `drafts=dflash2`,
`drafter=dspark`), then runs `bench/draft_positions.py`: 16 greedy requests (temperature 0, fixed `max_tokens`, thinking
off as the smoke test asks) in four classes, one at a time:

| Class | Requests (ids) |
| --- | --- |
| code | `fraction` (check-drafted's), `linked-list` (AGENTS.md's), `refactor`, `bash` |
| prose | `lighthouse`, `summary`, `letter`, `repeat` (check-drafted's; copy-friendly) |
| structured | `json-languages` (check-drafted's), `table-planets` (markdown), `csv`, `yaml` |
| agent | `first-call`, `after-read`, `after-edit`, `report`: tool-call turns with bench/toolcall.py's system prompt, three tools, earlier calls and results in the history |

`BENCH_ARGS="--reps 3"` repeats them (more rounds for the deep positions).

**What is recorded**, one JSON line a request (`$R/<label>.jsonl`, `$R/<label>-off.jsonl`): class, id, completion tokens,
finish reason, wall time; the runtime's `tokens_per_second`, `seconds`, `drafts` and `token_sha256`; the speculative
fields `rounds`, `drafted`, `accepted`, `acceptance_rate`, `tokens_per_round` (tau, the bonus token included) and
`positions.tried` / `positions.kept` (neural rounds only: `tried[j]` the rounds that offered draft j+1 with every earlier
draft kept, `kept[j]` those that kept it); the seconds waited for the nodes' state. `$R/<label>.profile` is the server's
`MM31_SERVE_PROFILE`. Also record, from rank 0's log after each stop, the drafter's line (`... draft: N passes, M absorbs
of R rows; the streams drafted D tokens and the target kept K`), and both ranks' MemAvailable after each load (`start`'s
memory line).

**The report** (`$R/report.json` and a table a run): per class and over all, the conditional acceptance a_j = kept_j /
tried_j, the survival S_j = a_1 x ... x a_j, tau, the acceptance rate and the mean tokens a second; then each run against
`plain`: the speed ratio (mean tok/s, per class and over all) and token equality, request by request.

**Pass / fail.**

- **Hard gate: token equality.** Every reply of `dflash2`, `dflash2-off` (and `dspark`, `dspark-off`) has the
  `token_sha256` of the same request in `plain`, and `scripts/check-drafted.sh compare` prints EQUAL for its seven
  requests (greedy, keyed sampled, the 6k-token prompts across the prompt chunks, the prompt-cache resume of the ring
  snapshot). Any difference is a bug of the draft path, never an acceptable loss: stop, keep `$R` and both ranks' logs,
  and add it to [KNOWN-ISSUES](KNOWN-ISSUES.md).
- **`"draft": false` is honoured.** The `-off` passes must not draft (`drafted` 0, runtime `drafts` false); a pass that
  drafts anyway fails its run, and `plain` (the `DRAFTS=0` server) stays the only baseline.
- **Speed is reported, not gated**: the ratio per class into RESULTS, with tau and a_j. Whether to serve the draft is
  decided from these numbers afterwards (docs/dspark-plan.md section 8 gives the MoE window cost: a full block's verify
  window costs about 3.4 one-token steps, so tau must clear that plus the draft pass).

**Heat.** The limits are each machine's (`/etc/spark/thermal.env`, docs/OPERATIONS.md section 4); this plan defines none.
`start` refuses unless both nodes report `normal` in `/run/spark-thermal/state` (and memguard stops a rank on
`emergency`). `measure-drafts.sh` reads each node's state before every request: it waits while a node is not `normal`
and stops on `emergency`; the wait is recorded with the request, and a request that waited is compared with care (its
GPU limit was lower). Never change the limits or the GPU setting for a measurement.

## 5. Depth: replacing the prior

The engine's DFlash2 drafter offers up to 7 drafts a pass (the block's 8 rows less the anchor). How deep a round goes is
the engine's depth rule, which starts each stream from a prior of acceptance by depth until the stream has its own
measurements. For dflash2 that prior is GLM-5.3's DFlash2 survival on code, a placeholder (`draft_kind.zig`
`dflash2_prior`: 0.874, 0.707, 0.56, 0.437, 0.34, 0.27, 0.21); dspark's is its own planning guess.

From the window: the report prints a replacement from the measured survival over all classes, S_j where position j was
tried at least 30 times (`--min-tried`), and past that the old prior's decay from the last measured value, strictly
decreasing as the engine's test requires. Use the pooled value unless the production traffic is one class (then that
class's row); keep the per-class rows in RESULTS. Positions the depth rule rarely offered are measured on few rounds:
`tried` says how many, and `--reps` adds more.

**The knobs.** This recipe exposes no depth setting: there is no flag or environment variable for the prior or the
depth, and the depth limit is the draft's block size in its `config.json` (which the pin fixes). Replacing the prior is an
engine change: a commit on the engine branch that edits `dflash2_prior`, the patch series cut from it (`ENGINE_PATCHES`,
`ENGINE_COMMIT`), `scripts/serve.sh build`, and the window again from step 1 (the new prior must give the same tokens,
and the speed is measured again).

## 6. Open

- Nothing here has run at TP2: the draft's load at TP2, its memory after the load (the runtime term is an estimate) and its
  speed.
- RadixArk's draft waits for the engine's causal block attention and a license on its card.
- The prompts are C1 and greedy only; concurrency (`PARALLEL` 4 with the draft) and sampled requests beyond
  check-drafted's two are not in this plan.
