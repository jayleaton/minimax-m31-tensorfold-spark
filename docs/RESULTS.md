# Results

**Serving measurements: [results/window-all-2](../results/window-all-2/README.md)** (one combined run on our pair,
2026-10-25: speed, prefill, concurrency, tool calls, structured output, needle, images, load time; the README's Results
section summarises it). This page keeps the calculator and the pack's offline figures, and lists what that run did not
measure. Mark any figure not taken through the server NOT MEASURED rather than rounding up.

## R0. Calculator (arithmetic, not measurement)

`python3 scripts/quant_budget.py --serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8`, with the family's budget
terms (patches/0001, budget.zig) and the pack's tensor classes (the release's index at revision 54e4e69d):

| Term | GiB a rank |
| --- | ---: |
| weights (half the split + replicated; no draft) | 92.66 |
| reserve (`TENSORFOLD_MEMORY_RESERVE_GIB`) | 8.00 |
| CUDA context, NCCL, graphs | 4.50 |
| prompt workspaces | 1.50 |
| KV for 131,072 tokens at 47,040 B | 5.74 |
| **MEM_GATE_GIB** | **113** |

At 117.1 GiB available before the load (an idle Spark, measured earlier on our pair): KV room 10.44 GiB, 238,418 fp8
tokens (238,336 in whole 128-token pages, 1,862 pages). With images (`VISION=1`, `--vision-image-tokens 2304`): rank 0's
vision tower and scratch (2.27 GiB) and the image rows (0.03 GiB) come out of the pool on both ranks: 8.15 GiB, 185,984
tokens, 1,453 pages (52,352 tokens and 409 pages fewer, +2.41 GiB over the 131,072-token context); the gate stays 113.
The variants below are text only. The profile runs text only (`VISION=0`): at 110 GiB available the pool with images
holds 23,808 tokens, so an image run uses `CONTEXT=23680` (docs/OPERATIONS.md section 3a). Variants: the engine's
default reserve (12.2 GiB) leaves 6.28 GiB (143,261 tokens; gate 117); `--drafter` costs 2.19 GiB, its 8 ring snapshots
included (188,486 tokens; gate 115); `--drafter dflash2` 3.11 GiB (167,506 tokens; gate 116; docs/DFLASH2.md section 3);
`--kv bf16` gives 146,031 tokens (gate 117).

## R1. The pack (its own card, offline)

jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31 at `5bff3d73` (the card; the shards are those of `f76ad09a`):
teacher-forced against the source, assistant tokens, rows up to 2,048, 95 % bootstrap intervals. All 152 conversations: KL 0.0369 [0.0254, 0.0506], top-1 94.9 %. Code: KL 0.0082,
top-1 97.2 %. Math 0.0078 / 97.1 %. Reasoning 0.0078 / 97.3 %. Science 0.0177 / 94.6 %. Tool-agentic: KL 0.0416
[0.0078, 0.0796], top-1 95.2 % on 18 conversations, **failing** its bar (0.0127 / 97.1 %); a re-test on 66 tool conversations is in progress. These say
nothing about serving.

## R2. CPU checks (this branch)

`tests/run.sh` on a Spark's CPU: bash -n and shellcheck (errors) on every script, pytest over `tests/` (the recipe's
config, rank contract, pair watch and calculator; the guard; the credential tools; the bench harnesses against a stand-in
server), `scripts/check-public.sh`. The results of the commit that last ran them are in that commit's pull request.

## To fill on the first two-node run (docs/OPERATIONS.md section 7)

| Item | Where it comes from |
| --- | --- |
| load time, first and warm | `scripts/serve.sh start` ("Server answered after N s") |
| MemAvailable on both ranks after the load, and the minimum under a long prompt | `start`'s memory line; a sampler during R-mem |
| the engine's boot memory and pool lines | rank 0's log; they set `boot_floor_log_regex` |
| canary | `scripts/canary.py` |
| greedy repeatability, drafted == plain | two identical runs; `scripts/check-drafted.sh` |
| the dspark draft: decode tok/s off and on per class, tau and acceptance by position, token identity, the calibration line, acceptance totals, rank memory with the draft | [DRAFTS](DRAFTS.md) section 6 (`bench/drafts.py`) |
| DFlash2 (and dspark) at C1: a_j, survival, tau, tok/s against plain, token equality | `scripts/measure-drafts.sh` (docs/DFLASH2.md section 4, `bench/draft_positions.py`) |
| decode and prefill tok/s, TTFT | `bench/stress.py`, `bench/soak.py` |
| tool calling | `bench/tooleval/run.sh`, `bench/toolcall.py` |
| image input: smoke, time to the first token, image tokens, VQA accuracy; rank 0's memory with images on and off | `bench/vision.py smoke`, `bench/vision.py bench`; `start`'s memory line |
| structured output, quality | `bench/structured.py`, `bench/quality.py` |
| pair watch and watchdog heal | `docker kill` of one rank |
