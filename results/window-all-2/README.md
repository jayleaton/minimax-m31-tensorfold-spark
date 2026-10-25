# window-all-2: the baseline recipe's measurement (2026-10-25, 17:26-20:26 UTC)

One combined TP2 window on the pair of Sparks: every change behind its own switch, each job a fresh start. The files in
this folder are copied from that run with machine names and local paths replaced.

**Engine**: TensorFold v1.1.2 + the `minimax_m3_vl` family, branch `mm3/window-all-2` @ `4e6390030` (image
`v1.1.2-all2`). **Recipe**: this branch's tools plus `recipe/toolfix` 39724b7 (tool-eval harness).
**Profile**: CONTEXT 131,072 x 1, FP8 KV, CUDA graphs on (`TF_MM3_GRAPHS_TP=1`), thinking off, vision off unless noted,
measured on both Sparks with the same device limits.

## The baseline recipe

`DRAFTS=1 DRAFTER=dspark TF_MM3_DRAFT_DEPTH=3` + `TF_MM3_ROCE_SUM=1 TF_MM3_SHARE_INPLACE=1 TF_ROCE_FAST=1`, every
other switch at its default (on). Job `331-release-dspark-roce`.

## Decode, one stream (tok/s, server)

Probe: four fixed prompts, 256 tokens each, greedy (T0) and seeded sampled (T1.0, top_p 0.95). The all-off column turns
off every switch (S11 sampling, S8 expert loads, decode-GEMV fuse and batch, multi-slot prefill, drafts).

| Workload | All off | Plain (all on) | Drafts on (DSpark) | Drafts vs all off |
| --- | ---: | ---: | ---: | ---: |
| code T0 | 19.7 | 21.2 | 29.4 | +49 % |
| tools T0 | 19.1 | 20.5 | 27.8 | +46 % |
| math T0 | 19.7 | 21.2 | 27.1 | +38 % |
| prose T0 | 19.8 | 21.2 | 19.7 | -1 % |
| code T1 | 19.2 | 20.9 | 27.6 | +44 % |
| tools T1 | 18.7 | 20.2 | 26.8 | +43 % |
| math T1 | 19.2 | 21.0 | 26.3 | +37 % |
| prose T1 | 19.2 | 21.0 | 20.0 | +4 % |
| **median, T0 / T1** | **19.7 / 19.2** | **21.2 / 20.9** | **27.5 / 26.5** | **+39 % / +38 %** |
| long story, 512 tokens (`bench.py decode`, 2 reps) | 19.0-19.1 | 20.2-20.4 | 19.6-21.7 | |

- Every reply's hash is equal across every job (greedy and sampled): drafts, RoCE, and each switch change no token.
- DFlash2 (`DRAFTS=dflash2`, depth 5) matches DSpark within noise (code 27.0, tools 28.2, math 28.2, prose 19.8 at T0).
- Prose drafts poorly at depth 3; the gain is in code, tool calls and math.
- Against the 18.4 tok/s all-off figure of the first window: code +60 %, tools +51 %, math +47 %, median +49 %.

## Prefill and first token (one cold prompt each)

| | All off | Baseline recipe |
| --- | ---: | ---: |
| Prefill 2K (tok/s) | 667 | 612 |
| Prefill 32K (tok/s) | 524 | 484 |
| Time to first token, 32K prompt (s) | 62.5 | 67.7 |
| Time to first token, short prompt (s) | 0.58 (window 1) | |

Prefill is 8 % slower with the switches on. No one switch turned off brings it back (each OFF run: 602-615 / 466-480);
it returns only with all of them off. Not resolved in this window.

## Concurrency (PARALLEL, `tools/bench_concurrent.py --alone`, drafts off)

| PARALLEL=4, MAX_TOKENS 32768 | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| code T0, aggregate tok/s | 22.2 | 40.7 | 65.5 |
| chat T0, aggregate tok/s | 21.1 | 32.0 | 62.4 |
| code T1, aggregate tok/s | 22.1 | 34.0 | 47.1 |

Every concurrent reply equals its alone reply (48/48 cells). PARALLEL=2: 40.0 / 31.0 tok/s at 2 streams (code / chat
T0). Four streams at CONTEXT 53,248 hit the worker's thermal stop (93 C held 33 s) and were not measured.

## Tool calls and quality

| Check | Drafts on (baseline recipe) | Plain |
| --- | --- | --- |
| Canary (strict: chat, reasoning, tool call, `json_object`, `tool_choice` required) | pass | pass |
| Smoke (12 checks incl. json_schema strict, named tool) | 12/12 | 12/12 |
| Tool chains, own set (`bench/tooleval own`), greedy, thinking off / adaptive | 12/12 / 12/12 | 12/12 / 12/12 |
| Tool calls (`bench/toolcall.py`, 21 cases x 2), greedy, off / adaptive | 42/42 / 42/42 | 21/21 / 21/21 (1 rep) |
| Same, MiniMax's sampling (T1.0, top_p 0.95), 3 reps, off / adaptive | | 58/63 / 57/63 (1 markup leak) |
| Own set, MiniMax's sampling, 3 reps | | 36/36 / 36/36 |
| Passkey needle, 120K-token prompt, depth 0.4 | | pass |
| Image input (`VISION=1`, CONTEXT 23,680): smoke / 10 visual questions | | 9/9 / 10/10; 672x672 first token 1.48 s |

Not measured in this window: decode after a 32K prompt, MMLU-200, tool-eval-bench, `bench/structured.py`'s full set,
MemAvailable after the load. Load time, warm: 25.8 s (drafts on), 24.0 s (plain).

## What is on and off

| Switch | State | Why |
| --- | --- | --- |
| S11 exact sampling (`TF_MM3_FAST_SAMPLE`) | on | bit-identical |
| S8 expert loads a k step ahead (`TF_EXL3_EXPERT_LOADS`) | on | bit-identical, neutral at TP2 |
| Decode GEMV fuse / batch (`TF_MM3_GEMV_FUSE`, `TF_MM3_GEMV_BATCH`) | on | +5 % / +1.4 % decode |
| Multi-slot prefill (`TF_MM3_MANY_ROWS`) | on | concurrent = alone |
| RoCE fused sum, in-place shares, lean doorbell | on | +4 % decode, bit-identical |
| DSpark draft, depth 3 | **on** | +38-49 % code / tools / math, tool calls pass |
| DFlash2 draft | off (works) | same speed as DSpark; one draft is enough |
| KV NVMe tier | off | admission fixed (262,144 admitted), start fails in the tier's I/O setup (`IoSetup`) |
| Vision | off by default | works (`VISION=1`); costs KV room |

## Files

`<job>/probe.json` (per prompt: reply hash, tokens, server tok/s), `<job>/*.txt` (bench, smoke and tool-eval summary
lines), `302-allon/tool-*.log` (the tool matrix), `*slots*/concurrent.json`, `309-vision/`, `window.txt` (the run log).
