Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# MiniMax-M3.1-Flash on TensorFold, 2x NVIDIA DGX Spark

A recipe for serving MiniMax-M3.1-Flash on two NVIDIA DGX Sparks behind an OpenAI-compatible API: our EXL3 pack, the
TensorFold Zig engine with the `minimax_m3_vl` family, one rank on each Spark, and the exact serving profile we measured.
It is built on [Mia / MiaAI-Lab's MiniMax-M3.1-Flash TensorFold recipe and `minimax_m3_vl` family patch](#credits)
(private repo).

| | |
| --- | --- |
| Model | [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (revision `54e4e69d`) |
| Pack | **EXL3 3.55 bpw, 199.9 GB**: [`jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31`](https://huggingface.co/jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31) (revision `5bff3d73`; routed experts 3.55 bpw, attention 6 / 8, head 8, the base model's `dspark/` draft copied unchanged) |
| Hardware | 2x DGX Spark (GB10, 128 GB each), tensor parallel 2 over the 200 Gb/s CX7 link |
| Engine | [TensorFold](https://github.com/ashhart/TensorFold) v1.1.2 Zig engine plus the `minimax_m3_vl` family, vendored as [`engine/patches`](engine/SOURCE-EXPORT.md) (MiaAI-Lab's family patch and ours; exported from engine commit `0547afe802fa`) |
| Serving | 131,072 tokens x 1 slot, FP8 KV, KV tier off, tool calling, structured output (xgrammar), drafts on, MiniMax's dspark draft, image input off by default (`VISION=1` measured) |

The served model ID is `MiniMax-M3.1-Flash-EXL3`.

> **Work in progress.** Measured on one pair of Sparks. Read [what is not solved](docs/KNOWN-ISSUES.md) and the
> [limits](#limits) before relying on it.

## Results

Measured on our pair in one combined run (window-all-3, 2026-10-26), at the profile in
[`config/prod.env.example`](config/prod.env.example): 131,072 x 1, FP8 KV, CUDA graphs on, the dspark draft with each
round's depth sized by its own confidences (`TF_MM3_DSPARK_CHANCES=1 TF_MM3_CALIB=auto`, `TF_MM3_DRAFT_DEPTH` unset) and
the RoCE switches on, thinking off. Raw files, method and every switch: [results/window-all-3](results/window-all-3/README.md)
(the earlier run: [results/window-all-2](results/window-all-2/README.md)). Both Sparks were running at their thermal
limits during this run ([heat](#limits)).

**Speed**, one stream, decode tok/s at the server, 256 tokens a reply; T0 greedy, T1 sampled (1.0, top_p 0.95):

| Workload | All switches off | Plain decode | Drafts, depth 3 | **Drafts, depth by confidence** | vs all off |
| --- | ---: | ---: | ---: | ---: | ---: |
| code T0 | 19.7 | 20.1 | 26.5 | **30.8** | +56 % |
| tool calls T0 | 19.2 | 19.5 | 26.0 | **33.5** | +74 % |
| math T0 | 19.8 | 20.1 | 26.6 | **32.7** | +65 % |
| prose T0 | 19.8 | 20.8 | 20.7 | 20.1 | +2 % |
| code T1 | 19.2 | 19.9 | 25.3 | **30.7** | +60 % |
| tool calls T1 | 18.6 | 19.3 | 25.6 | **32.2** | +73 % |
| math T1 | 19.0 | 20.0 | 22.3 | **30.3** | +59 % |
| prose T1 | 19.2 | 20.0 | 20.5 | 20.2 | +5 % |
| **median T0 / T1** | **19.75 / 19.1** | **20.1 / 19.95** | **26.25 / 23.8** | **31.75 / 30.5** | **+61 % / +60 %** |
| Decode after a 32K prompt (tok/s) | | | 15.9 | 20.8 | |
| Prefill 2K / 32K (tok/s) | 668 / 513 | | | 687 / 531 | +3 % |
| Time to first token, 32K prompt (s) | | | | 66.7 | |

Drafted replies are verified: every reply's hash is equal across every job, drafts on and off, greedy and sampled, every
depth rule and every switch. Prose neither gains nor loses (the depth rule backs off when the draft is unsure); the gain
is in code, tool calls and math. Against the 18.4 tok/s of our first all-off run: median **+73 %**; code +67 %, tool
calls +82 %, math +78 %. incoai's DFlash2 draft (`DRAFTS=dflash2`) is documented as the alternative
([DFLASH2](docs/DFLASH2.md); measured in window-all-2 only).

**Prefix cache** (a 70,006-token prompt, drafts at depth 3): fresh, first token in 167.2 s; the exact repeat in **0.76 s**
and a follow-up on the same prefix in 0.79 s (69,888 tokens cached); the repeat's reply equals the fresh one.

**Four requests at once** ([`config/slots4.env.example`](config/slots4.env.example): `PARALLEL=4`, CONTEXT 40,960, drafts
at depth 3, an option, not the default profile): **52.0 tok/s** aggregate on code T0 (35.3 alone, 31.8 at two), 41.9 on
chat T0; every concurrent reply equals its alone reply (48/48). Without drafts the same four streams decoded 65.5 / 62.4
(code / chat T0, window-all-2): drafts pay at one or two streams, plain decode at four.

**Serving checks** (the profile above unless noted):

| Check | Result |
| --- | ---: |
| Canary, strict (chat, reasoning, tool call, `json_object`, `tool_choice` required) | pass |
| Smoke (12 checks incl. `json_schema` strict and a named tool) | 12/12 |
| Tool chains (`bench/tooleval own`), thinking off / adaptive | 12/12 / 12/12 |
| Tool calls (`bench/toolcall.py`, 21 cases x 2), thinking off / adaptive | 42/42 / 42/42 |
| Drafts at depth 3, MiniMax's sampling, 3 reps: tool chains / tool calls | 36/36 / 58/63 |
| MMLU-200, 0-shot | 0.780 (156/200), no markup leaks |
| `bench/structured.py`: schemas / thinking / tools | 14/14 / 9/9 / 14/15 (`tool_choice: "none"`, [limits](#limits)) |
| Passkey needle in a 120K-token prompt (window-all-2) | pass |
| Image input (`VISION=1`, CONTEXT 23,680; window-all-2): smoke / 10 visual questions | 9/9 / 10/10 |
| MemAvailable after the load, head / worker | 9.5 / 10.6 GiB |
| Load time, warm | 54 s (timing the verify costs at boot adds ~28 s) |

**The pack against the release** (offline, teacher-forced, 152 conversations, assistant tokens): KL 0.0369, top-1
94.9 %; code 0.0082 / 97.2 %, math 0.0078 / 97.1 %, tool-agentic 0.0416 / 95.2 % (above its own bar of 0.0127; the tool checks above pass through the
server). Method: [PACKS](docs/PACKS.md),
[RESULTS](docs/RESULTS.md).

### Limits

- **Heat**: the numbers come from a run in which both Sparks were thermally limited. After three hours of load the worker
  reached the device emergency (94 °C held 30 s) at 05:34:57 and 05:37:59 UTC and the head held 94 °C at 05:43:32 UTC;
  the guards stopped the containers each time and the run stopped early at 05:47 UTC. The S11-off and S8-off checks
  (jobs 4591 / 4592) were cut by those stops and are missing; several measurements waited for a throttled node to
  cool before they ran.
- **`tool_choice: "none"` leaks tool-call markup into the reply's content**: the model may still write a tool call, and
  its markup reaches `content` (`bench/structured.py` tools 14/15). Clients that send `"none"` must not rely on clean content; leave tools out of
  the request instead.
- **Prose gains nothing from the draft** (20.1 against 20.8 plain at T0); the gain is in code, tool calls and math.
- **Plain decode is ~5 % below window-all-2's** (20.1 against 21.2 at T0): the prefill fix that removed window-all-2's
  prefill regression gives back the one-row router batching.
- **Off, they fail**: the KV tier (it starts, then panics at the first prompt), so the context is 131,072 tokens, all
  resident, and 4 x 1M is not in the profile; `TF_MM3_PLAN_INLINE=1` (crashes at start).
- **Depth by confidence is measured at one stream only**; four streams were measured at depth 3.
- Not measured: tool-eval-bench, `TF_MM3_INDEX_KEYS=int8` on long prompts, the side image's serve-latency switches.
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
scripts/serve.sh status | health | ready | logs 0 | logs 1 | restart | stop
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
- **RoCE switches**: `TF_MM3_ROCE_SUM=0`, `TF_MM3_SHARE_INPLACE=0`, `TF_ROCE_FAST=0` (same tokens; ~4 %
  slower plain decode in window-all-2, within noise with drafts on in window-all-3).
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
| [CLIENTS](docs/CLIENTS.md) | connecting a client: endpoints, reply fields, request options, efficient use, examples |
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
`minimax_m3_vl` engine family patch and its engine notes are Mia / MiaAI-Lab's: her MiniMax-M3.1-Flash TensorFold
recipe and `minimax_m3_vl` family patch (private repo; imported here at `fd1d379`, the patch shipped unchanged)
([MiaAI-Lab on GitHub](https://github.com/MiaAI-Lab), [Mia-AiLab on Hugging Face](https://huggingface.co/Mia-AiLab),
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
