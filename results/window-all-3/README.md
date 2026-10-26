# window-all-3: the final combined measurement (2026-10-26, 02:44-05:47 UTC)

One combined TP2 window on the pair of Sparks: every change behind its own switch, each job a fresh start. The files in
this folder are copied from that run with machine names and local paths replaced.

**Engine**: TensorFold v1.1.2 + the `minimax_m3_vl` family, branch `mm3/window-all-3` @ `0547afe80` (window-all-2 +
decode-GEMV prefill fix, 4 slots + drafts, API markup fixes, S5 draft depth, plan-inline, KV-tier I/O fallback, prefix
cache fix, draft-pass speed, KV formats). Side image `mm3/window-all-3s` @ `37da4d05b` (+ serve-latency, prefill prompt
kernels), job 461 only. **Profile**: CONTEXT 131,072 x 1, FP8 KV, CUDA graphs on, thinking off, vision off.

## The baseline recipe

`DRAFTS=1 DRAFTER=dspark TF_MM3_DSPARK_CHANCES=1 TF_MM3_CALIB=auto` (the draft sizes each round's depth by its own
confidences against boot-measured verify costs; `TF_MM3_DRAFT_DEPTH` unset) + `TF_MM3_ROCE_SUM=1 TF_MM3_SHARE_INPLACE=1
TF_ROCE_FAST=1`, every other switch at its default. Job `4315-release-chances`.

## Decode, one stream (tok/s, server)

Probe: four fixed prompts, 256 tokens each, greedy (T0) and seeded sampled (T1.0, top_p 0.95). All off: every switch off
(S11 sampling, S8 expert loads, decode-GEMV fuse and batch, multi-slot prefill, the prefix-cache fix, drafts).

| Workload | All off | Plain (all on) | Drafts, depth 3 | **Drafts, depth by confidence** | vs all off |
| --- | ---: | ---: | ---: | ---: | ---: |
| code T0 | 19.7 | 20.1 | 26.5 | **30.8** | +56 % |
| tools T0 | 19.2 | 19.5 | 26.0 | **33.5** | +74 % |
| math T0 | 19.8 | 20.1 | 26.6 | **32.7** | +65 % |
| prose T0 | 19.8 | 20.8 | 20.7 | **20.1** | +2 % |
| code T1 | 19.2 | 19.9 | 25.3 | **30.7** | +60 % |
| tools T1 | 18.6 | 19.3 | 25.6 | **32.2** | +73 % |
| math T1 | 19.0 | 20.0 | 22.3 | **30.3** | +59 % |
| prose T1 | 19.2 | 20.0 | 20.5 | **20.2** | +5 % |
| **median T0 / T1** | **19.75 / 19.1** | **20.1 / 19.95** | **26.25 / 23.8** | **31.75 / 30.5** | **+61 % / +60 %** |
| long story, 512 tokens (`bench.py decode`, 2 reps) | 18.95-19.1 | 20.0 | 19.4-19.7 | 20.2-21.9 | |
| decode after a 32K prompt (256 tokens) | | | 15.9 | **20.8** | |

- Every reply's hash is equal across every job (greedy and sampled): drafts, depth rules and every switch change no token.
- Against the 18.4 tok/s all-off figure of window 1: median +73 %; code +67 %, tools +82 %, math +78 %.
- Prose gains nothing and loses nothing (the depth rule backs off when the draft is unsure).
- Plain decode is about 5 % below window-all-2's (21.2): the decode-GEMV prefill fix keeps batched router loads to 4-16
  row calls, giving back the 1-row batching.
- Draft-pass speed (`TF_MM3_DRAFT_BATCH/TOPC/ATTN_SPLIT=1`) at one stream: greedy +1.6 %, sampled -11 %: off.
- Depth 5 (fixed) is no better than depth 3 (median 26.15).

## Prefill and first token (one cold prompt each)

| | All off | Baseline recipe |
| --- | ---: | ---: |
| Prefill 2K (tok/s) | 668 | 687 |
| Prefill 32K (tok/s) | 513 | 531 |
| Time to first token, 32K prompt (s) | | 66.7 |

The window-all-2 prefill regression (612 / 484) is gone: prefill is now at or above the all-off figure.

## Prefix cache (70,006-token prompt, drafts on)

| Request | Fixed store (default) | Old half-pool cap (`TF_MM3_CACHE_HALF=1`) |
| --- | ---: | ---: |
| fresh: first token (s) | 167.2 | 169.4 |
| exact repeat: first token (s), cached tokens | **0.76**, 69,888 | 175.6, 0 |
| follow-up on the same prefix: first token (s), cached tokens | **0.79**, 69,888 | 175.7, 0 |

The repeat's reply equals the fresh reply, and the cached follow-up's reply equals the uncached one (`revisit.json`).

## Four slots (`PARALLEL=4`, CONTEXT 40,960, drafts at depth 3, `tools/bench_concurrent.py --alone`)

| Aggregate tok/s | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| code T0 | 35.3 | 31.8 | 52.0 |
| chat T0 | 20.1 | 30.9 | 41.9 |
| code T1 | 35.8 | 42.0 | 48.1 |
| chat T1 | 24.2 | 29.5 | 36.1 |

Every concurrent reply equals its alone reply (48/48). Draft batching adds +5 % (code T0) to +9 % (chat T1) at 4 streams.
Plain decode without drafts reached 65.5 / 62.4 (code / chat T0) at 4 streams in window-all-2: drafts pay at one or two
streams, plain decode at four.

## Tool calls and quality (baseline recipe unless noted)

| Check | Result |
| --- | --- |
| Canary (strict) | pass |
| Smoke (12 checks incl. json_schema strict, named tool) | 12/12 |
| Tool chains, own set (`bench/tooleval own`), greedy, thinking off / adaptive | 12/12 / 12/12 |
| Tool calls (`bench/toolcall.py`, 21 cases x 2), greedy, off / adaptive | 42/42 / 42/42 |
| Depth-3 drafts, MiniMax's sampling (T1.0, top_p 0.95), 3 reps, off / adaptive | own 36/36 / 36/36; toolcall 58/63 / 58/63 |
| MMLU-200, 0-shot | 0.780 (156/200), no markup leaks |
| `bench/structured.py` | schemas 14/14, thinking 9/9, tools 14/15 (`tool_choice: none`: tool-call markup reaches content) |
| MemAvailable after the load, head / worker | 9.5 / 10.6 GiB |
| Load time, warm | 54 s (boot-measured verify widths add ~28 s) |

## What is on and off

| Switch | State | Why |
| --- | --- | --- |
| DSpark draft, depth by confidence (`TF_MM3_DSPARK_CHANCES=1 TF_MM3_CALIB=auto`) | **on** | +56-74 % code / tools / math, prose even, tool calls pass |
| RoCE fused sum, in-place shares, lean doorbell | on | bit-identical; within noise with drafts on |
| S11, S8, decode-GEMV fuse / batch, multi-slot prefill, prefix-cache fix | on | bit-identical; each within noise off |
| Draft-pass speed (batch, top-C, split attention) | off | sampled -11 % at one stream |
| `TF_MM3_PLAN_INLINE` | off | crashes at start (integer overflow) |
| `TF_MM3_INDEX_KEYS=int8` | off | equal on short prompts (sparse path not reached); not measured long |
| KV NVMe tier (262K exact, 4 x 1M approximate) | off | starts now (I/O threads), panics at the first prompt (`tier_io.zig:153`) |
| Serve-latency, prompt kernels (side image) | not measured | job 461 equal hashes, 693 tok/s 2K; the A/B was cut by heat |

## Heat

Three hours of continuous load ended in thermal stops: the worker reached the device emergency (94 C held 30 s) at
05:34:57 and 05:37:59 UTC, the head held 94 C for 33 s at 05:43:32 UTC. The guards stopped the containers each time; the
window cleared the record once both nodes read 86 C or less, then stopped early at 05:47 UTC. Jobs 4591 / 4592 (S11 and
S8 off) were cut by these stops and are not reported.

## Files

`<job>/probe.json` (per prompt: reply hash, tokens, server tok/s), `<job>/*.txt` (bench, smoke, tool-eval, structured,
decode-after-32K summary lines), `402-release/tool-*.log` (the tool matrix), `43*/revisit.json`,
`41*/concurrent.json`, `window.txt` (the run log).
