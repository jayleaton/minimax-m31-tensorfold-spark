Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# MiniMax-M3.1-Flash on TensorFold, 2x NVIDIA DGX Spark

A recipe for serving MiniMax-M3.1-Flash on two NVIDIA DGX Sparks behind an OpenAI-compatible API: our EXL3 pack, the
TensorFold Zig engine with the `minimax_m3_vl` family, one rank on each Spark, and the exact serving profile we measured.
It is built on [Mia / MiaAI-Lab's MiniMax-M3.1-Flash TensorFold recipe and family patch](#credits).

| | |
| --- | --- |
| Model | [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (revision `54e4e69d`) |
| Pack | **EXL3 3.55 bpw, 199.9 GB**: [`jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31`](https://huggingface.co/jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31) (revision `5bff3d73`; routed experts 3.55 bpw, attention 6 / 8, head 8, the base model's `dspark/` draft copied unchanged) |
| Hardware | 2x DGX Spark (GB10, 128 GB each), tensor parallel 2 over the 200 Gb/s CX7 link |
| Engine | [TensorFold](https://github.com/ashhart/TensorFold) v1.1.2 Zig engine plus the `minimax_m3_vl` family, vendored as [`engine/patches`](engine/SOURCE-EXPORT.md) (MiaAI-Lab's family patch and ours; exported from engine commit `4e6390030139`) |
| Serving | 131,072 tokens x 1 slot, FP8 KV, KV tier off, tool calling, structured output (xgrammar), drafts on, MiniMax's dspark draft, image input off by default (`VISION=1` measured) |

The served model ID is `MiniMax-M3.1-Flash-EXL3`.

> **Work in progress.** Measured on one pair of Sparks. Read [what is not solved](docs/KNOWN-ISSUES.md) and the
> [limits](#limits) before relying on it.

## Results

Measured on our pair in one combined run (window-all-2, 2026-10-25), at the profile in
[`config/prod.env.example`](config/prod.env.example): 131,072 x 1, FP8 KV, CUDA graphs on, the dspark draft at depth 3
and the RoCE switches on, thinking off. Raw files, method and every switch: [results/window-all-2](results/window-all-2/README.md).

**Speed**, one stream, decode tok/s at the server, 256 tokens a reply; T0 greedy, T1 sampled (1.0, top_p 0.95):

| Workload | All switches off | Plain decode | Drafts on | Drafts vs all off |
| --- | ---: | ---: | ---: | ---: |
| code T0 | 19.7 | 21.2 | **29.4** | +49 % |
| tool calls T0 | 19.1 | 20.5 | **27.8** | +46 % |
| math T0 | 19.7 | 21.2 | **27.1** | +38 % |
| prose T0 | 19.8 | 21.2 | 19.7 | -1 % |
| code T1 | 19.2 | 20.9 | 27.6 | +44 % |
| tool calls T1 | 18.7 | 20.2 | 26.8 | +43 % |
| **median T0 / T1** | **19.7 / 19.2** | **21.2 / 20.9** | **27.5 / 26.5** | **+39 % / +38 %** |
| Prefill 2K / 32K (tok/s) | 667 / 524 | | 612 / 484 | -8 % |
| Time to first token, 32K prompt (s) | 62.5 | | 67.7 | |

Against the 18.4 tok/s of our first all-off run: code **+60 %**, tool calls **+51 %**, math **+47 %**, prose +7 %.
Drafted replies are verified: every reply's hash is equal with drafts on and off, greedy and sampled. incoai's DFlash2
draft (`DRAFTS=dflash2`, depth 5) matches the dspark draft within noise (code 27.0, tool calls 28.2, math 28.2) and is
documented as the alternative ([DFLASH2](docs/DFLASH2.md)).

**Four requests at once** (`PARALLEL=4`, drafts off, an option, not the default profile): 65.5 tok/s aggregate on code
(22.2 alone, 40.7 at two), 62.4 on chat; each of the four replies equals its alone reply (48/48).

**Serving checks** (drafts on unless noted):

| Check | Result |
| --- | ---: |
| Canary, strict (chat, reasoning, tool call, `json_object`, `tool_choice` required) | pass |
| Smoke (12 checks incl. `json_schema` strict and a named tool) | 12/12 |
| Tool chains (`bench/tooleval own`), thinking off / adaptive | 12/12 / 12/12 |
| Tool calls (`bench/toolcall.py`, 21 cases x 2), thinking off / adaptive | 42/42 / 42/42 |
| Plain decode, MiniMax's sampling, 3 reps: tool chains / tool calls | 36/36 / 58/63 |
| Passkey needle in a 120K-token prompt | pass |
| Image input (`VISION=1`, CONTEXT 23,680): smoke / 10 visual questions | 9/9 / 10/10 |
| Load time, warm | 25.8 s |

**The pack against the release** (offline, teacher-forced, 152 conversations, assistant tokens): KL 0.0369, top-1
94.9 %; code 0.0082 / 97.2 %, math 0.0078 / 97.1 %, tool-agentic 0.0416 / 95.2 % (above its own bar of 0.0127; the tool checks above pass through the
server). Method: [PACKS](docs/PACKS.md),
[RESULTS](docs/RESULTS.md).

### Limits

- **Prose gains about nothing from the draft** at depth 3 (19.7 vs 21.2 plain at T0); the gain is in code, tool calls
  and math.
- **Prefill is ~8 % slower** than with every switch off (612 / 484 against 667 / 524 tok/s at 2K / 32K). No single switch
  explains it; a fix is in progress.
- **The KV tier and 4 x 1M are not in the profile**: the tier's start fails in its I/O setup, so the context is 131,072
  tokens, all resident.
- Not measured yet: decode after a long prompt, MMLU, tool-eval-bench, memory after the load.
- One pair of Sparks, one run per job.

## Quick start

Bring-up, step by step (node checks, the link, scoped sudo, the device's heat limits): [AGENTS.md](AGENTS.md). On the
head Spark (rank 0, serves the API):

```bash
git clone https://github.com/jayleaton/minimax-m31-tensorfold-spark.git
cd minimax-m31-tensorfold-spark
cp config/prod.env.example config/prod.env && $EDITOR config/prod.env   # set WORKER=user@<worker link address>
hf auth login                      # head only; a read token for the pack
scripts/serve.sh build             # the engine image on both Sparks
scripts/serve.sh prepare           # the pack here and on the worker, checked against SHA256SUMS
scripts/serve.sh preflight         # both nodes, the link, memory
scripts/serve.sh start             # -> MiniMax-M3.1-Flash-EXL3 on http://127.0.0.1:8000/v1
```

- `scripts/serve.sh build` builds the engine from upstream TensorFold v1.1.2 plus [`engine/patches`](engine/SOURCE-EXPORT.md);
  keep `ENGINE_PATCHES` and `ENGINE_COMMIT` as the example sets them.
- Install each Spark's heat limits first (`scripts/install-device-limits.sh`, [OPERATIONS section 4](docs/OPERATIONS.md#4-temperature-the-devices-limits));
  the recipe never starts without them.
- We suggest a 2100-2200 MHz GPU clock cap to help with temperature control.
- Check the canary passes (`scripts/serve.sh canary`) before serving traffic.

Then `scripts/serve.sh status | health | logs 0 | logs 1 | restart | stop`. The API listens on 127.0.0.1 without auth;
`API_KEY=1` requires a key from a mode-600 file ([API key](docs/OPERATIONS.md#2-configuration)).

## Configuration

[`config/prod.env.example`](config/prod.env.example) documents every setting; the ones that matter most:

| Setting | Profile | What it does |
| --- | --- | --- |
| `MODEL_ID`, `MODEL_REVISION` | the pack above, pinned | the checkpoint both ranks load |
| `ENGINE_PATCHES`, `ENGINE_COMMIT` | `engine/patches`, the exported commit | the engine `scripts/serve.sh build` makes: TensorFold v1.1.2 plus the series |
| `CONTEXT`, `PARALLEL` | `CONTEXT=131072`, `PARALLEL=1` | tokens a request holds, and requests at once |
| `KV` | `fp8` | e4m3 K/V with BF16 index keys: 47,040 B a token a rank |
| `KV_TIER_DIR`, `TF_MM3_KV_TIER_GIB` | `KV_TIER_DIR` empty: off, every page resident | the engine's KV tier (pages on each Spark's NVMe); empty: every page resident |
| `DRAFTS`, `DRAFTER` | `DRAFTS=1`, `DRAFTER=dspark` | `0` off; `1` + `DRAFTER=dspark` the pack's own draft; `dflash2` incoai's DFlash2 draft |
| `VISION` | `VISION=0` | `1`: image input (`image_url` / Anthropic `image`, JPEG, PNG, WebP) |
| `MEM_GATE_GIB`, `TENSORFOLD_MEMORY_RESERVE_GIB` | from `scripts/quant_budget.py --serve` | the MemFree gate before a load, and the engine's reserve; they move together |

Memory, floors, the pair watch and the watchdog: [OPERATIONS](docs/OPERATIONS.md). Pack sizes and the per-rank budget
for any profile: `python3 scripts/quant_budget.py --serve --help`.

## Rollback

Each lever turns off on its own; stop the pair (`scripts/serve.sh stop`), change `config/prod.env`, start again:

- **Drafts**: `DRAFTS=0`, `DRAFTER=` (plain decode; the same tokens, without the draft's memory).
- **RoCE switches**: `TF_MM3_ROCE_SUM=0`, `TF_MM3_SHARE_INPLACE=0`, `TF_ROCE_FAST=0` (same tokens, ~4 % slower decode).
- **Image input**: `VISION=0` (no tower loaded; an image part gets HTTP 400).
- **KV tier** (when a profile turns it on): empty `KV_TIER_DIR`, with `CONTEXT=131072`, `PARALLEL=1` and
  `MEM_GATE_GIB=113`, all resident.
- **Engine**: check out the previous release's `engine/patches` and `config/prod.env.example` engine lines,
  `scripts/serve.sh build`, then `start`; `ENGINE_PATCHES=patches` builds MiaAI-Lab's family alone.
- **Recipe**: check out the previous release tag of this repository; the pack revision is pinned in its config.

## Troubleshooting

The full table of symptoms and fixes is [AGENTS.md section 6](AGENTS.md#6-common-failures-and-fixes); the common ones:

| Symptom | Fix |
| --- | --- |
| `MemFree ... still under MEM_GATE_GIB` | stop other workloads; add the scoped sudoers line for the cache drop ([AGENTS.md](AGENTS.md#1-check-both-nodes)) |
| `rank 1 exited` early, NCCL timeout | `scripts/serve.sh preflight`; check the link ([AGENTS.md section 2](AGENTS.md#2-the-link)) |
| `a file does not match SHA256SUMS` | delete that file from the cache; `scripts/serve.sh prepare` |
| `the canary failed (CANARY=strict)` | read `canary-fail-r{0,1}.log`; open an issue with it |
| a rank's container gone, `memguard.log` says `killed container` | lower `CONTEXT` or turn drafts off; never lower the floors |
| `no device thermal limits on this Spark` | `scripts/install-device-limits.sh`, then `--apply` |

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | bring-up step by step: node checks, the link, config, start, verify, failures |
| [OPERATIONS](docs/OPERATIONS.md) | commands, settings, memory, images, heat limits, guard, watchdog, qualifying a profile |
| [PACKS](docs/PACKS.md) / [RESULTS](docs/RESULTS.md) | the pack and its per-rank budget; every measurement and how it was taken |
| [DRAFTS](docs/DRAFTS.md) / [DFLASH2](docs/DFLASH2.md) | the dspark and DFlash2 drafts: profiles, memory, how they are measured |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [PORT](docs/PORT.md) / [CHANGELOG](docs/CHANGELOG.md) | what came from MiaAI-Lab's recipe and our GLM / DeepSeek recipes; every change |
| [bench/](bench/README.md) | tool calling, structured output, quality, stress, soak, images, drafts |
| engine notes | MiaAI-Lab's: [rank contract](docs/rank-contract.md), [FP8 KV](docs/fp8-kv.md), [dspark plan](docs/dspark-plan.md), [loader speed](docs/loader-speed.md), [engine port plan](docs/engine-port-plan.md), [build notes](docs/tensorfold-build-notes.md) |

## Checks

```bash
tests/run.sh                         # bash -n, shellcheck, pytest over tests/, check-public (CPU only)
scripts/check-public.sh --history    # before publishing anything
```

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (ashhart) and its contributors, and on
[MiniMax](https://huggingface.co/MiniMaxAI)'s model, dspark draft and papers. The base of this recipe, its
`minimax_m3_vl` engine family patch and its engine notes are Mia / MiaAI-Lab's
([MiaAI-Lab on GitHub](https://github.com/MiaAI-Lab), [Mia-AiLab on Hugging Face](https://huggingface.co/Mia-AiLab):
[MiniMax-M3.1-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/MiniMax-M3.1-Flash-EXL3-2x-DGX-Sparks-TensorFold),
with her [EAQ v1 3.06 bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw) and
[3.5 bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw) packs). The DFlash2 draft is
[incoai](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2)'s, and both drafts follow
[z-lab's DFlash](https://github.com/z-lab/dflash); EXL3 is by
[turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3); structured output is
[xgrammar](https://github.com/mlc-ai/xgrammar) (mlc-ai); image input decodes with
[libjpeg-turbo](https://libjpeg-turbo.org) and [libwebp](https://chromium.googlesource.com/webm/libwebp). The operations come from our
[GLM-5.3](https://github.com/jayleaton/glm53-tensorfold-spark) and
[DeepSeek-V4.1](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) recipes. Every contributor, project and
benchmark we used is credited in [CREDITS](docs/CREDITS.md).

## Licensing

This project's code, scripts, benchmarks and docs: **Apache License 2.0** ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)), including the parts from MiaAI-Lab's recipe (Apache-2.0, her copyright and NOTICE kept). No
weights are included: the pack and the dspark draft are under the MiniMax Community License (non-commercial terms; read it), the DFlash2 draft under
Apache-2.0. TensorFold, ExLlamaV3, xgrammar and the base image keep their own terms:
[`docs/LICENSING.md`](docs/LICENSING.md).

"MiniMax" belongs to MiniMax; "DGX Spark" and "GB10" belong to NVIDIA. This recipe is not affiliated with or endorsed by
MiniMax, NVIDIA or the TensorFold authors.
