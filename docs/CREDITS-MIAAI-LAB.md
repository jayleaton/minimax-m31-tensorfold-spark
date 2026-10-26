# Credits of the base recipe (MiaAI-Lab)

This is MiaAI-Lab's `CREDITS.md` from the base recipe, kept as imported with the lines added since (a link to a
repository of hers that is private is written as its name). This repository's own credits,
which include everyone named here, are [CREDITS.md](CREDITS.md).

This repository is a recipe: scripts, a Dockerfile and a list of pins that put other people's work together on two DGX
Sparks. Almost everything that makes it work was built by others. Its own work is licensed under the Apache License 2.0
([`LICENSE`](../LICENSE)); [`NOTICE`](../NOTICE) carries the third-party notices that go with it. It does not yet serve the
model on its own: TensorFold v1.1.2 has no MiniMax family, so the engine work in `patches/` (listed below) adds it.

## Model

- **[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)** by
  [MiniMax](https://huggingface.co/MiniMaxAI): the model's design, training and evaluations. Its license, on the model card,
  governs any use of the weights, including the quantized checkpoint below. The weights are not part of this repository;
  `scripts/prepare.sh` downloads them from Hugging Face.
- **Mia's AI Lab**: the checkpoint this recipe serves,
  [`Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw`](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
  (private): the EAQ v1 EXL3 quantization at 3.06 bits a weight on average (the head at 8 bits), made with
  [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp ([turboderp-org](https://github.com/turboderp-org)) (MIT):
  the EXL3 format and its quantizer.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache 2.0): the native Zig engine at v1.1.2, its OpenAI-compatible server, lane core, CUDA runtime,
  EXL3 module and tensor-parallel fabric (`zig/src/fabric/tp`, whose RoCE notes `scripts/nodes.sh` follows). The contributors
  of each part are named in TensorFold's own [credits](https://github.com/ashhart/TensorFold/blob/v1.1.2/docs/credits.md) and
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v1.1.2/THIRD_PARTY_NOTICES.md), which apply here too.
  Everything in this recipe that is TensorFold's own code, behavior or format (the server and its flags, the API key file's
  format, the capabilities report the scripts check against, the `TF_TP_*` settings) is credited to them; this recipe only runs it.
- **[vcruz305](https://github.com/vcruz305)** authored TensorFold's EXL3 module for CUDA (TensorFold
  [#42](https://github.com/ashhart/TensorFold/issues/42), shipped in 0.3.6), the grouped-expert design that the native EXL3 kernels
  in TensorFold v1.1.2 follow, as TensorFold's `docs/recipes/exl3-universal-experts.md` records. Only work merged into
  TensorFold's history is used here, and nothing from outside it.
- **The patches below** are Mia's AI Lab's own work on top of TensorFold, listed in full under "Patches (this repository's own work)".
- **The dspark draft** (the `dspark/` folder of the checkpoint, class `DSparkMiniMaxDraftModel`) was trained and published by
  MiniMax ([MiniMaxAI](https://huggingface.co/MiniMaxAI)); its architecture family is z-lab's DFlash (MIT) and the DSpark method is
  DeepSeek's (MIT). `patches/tasks/0017-minimax-m3-dspark-device.patch` wires it into the engine on TensorFold's lane drafter
  contract (`core/lanes` `Drafter` and `Drafted`, `native/cuda_drafted.zig`) and re-implements, for this model, the semantics of
  TensorFold's DSpark and DFlash2 drafters (`families/deepseek_v41/draft/dspark.zig`, `families/qwen3_5/cuda_draft*.zig`,
  `families/glm/dflash*.zig` and `cuda_taps.zig`; the GLM DFlash2 CUDA drafter is credited in TensorFold's CHANGELOG to
  BobClawblaw, #979), with the mxfp8 lane matmul of `cuda/mxfp8.zig` and `cuda/qmmf.zig`. Written by Mia's AI Lab with Claude Code;
  no file of those was copied.
- **The DFlash2 draft** that `DRAFTS=dflash2` loads,
  [`incoai/MiniMax-M3.1-Flash-DFlash2`](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2) (revision `ac07a7c1`, class
  `DFlash2DraftModel`), was trained and published by [incoai](https://huggingface.co/incoai) under the Apache License 2.0; its
  architecture family is z-lab's [DFlash](https://github.com/z-lab/dflash) (MIT). The weights are not part of this repository:
  `scripts/prepare.sh` downloads them from Hugging Face and checks them against sha256 pins. The engine support (the draft's
  kind read from its `config.json`, its device half and its cost model) is TensorFold's `mm3/dflash2` branch, another work
  stream that this recipe runs and describes (docs/DFLASH2.md). RadixArk's DFlash2 draft for this model is named there only as
  not supported.
- `patches/0001-minimax-m3-family.patch` (with the per-task history in `patches/tasks/`) holds the MiniMax-M3.1-Flash engine work, which adds the `minimax_m3_vl` family to TensorFold
  v1.1.2: written for this recipe by Mia's AI Lab with Claude Code, on top of TensorFold's code (ashhart and contributors, Apache
  2.0) and reusing its EXL3, router, rotary and attention reference code. Their semantics are NOT yet verified against MiniMax's own
  model code. Each further patch added there is credited here, with its author, before the recipe uses it.

## Patches (this repository's own work)

Every patch in `patches/` was written for this recipe by Mia's AI Lab with [Claude Code](https://claude.com/claude-code), on top of
TensorFold v1.1.2 (Ash Hart and the TensorFold contributors, Apache 2.0) and reusing its EXL3, router, rotary, attention, lane and
fabric code. `patches/0001-minimax-m3-family.patch` is the applied series, the diff of all of the following against v1.1.2;
`patches/tasks/` keeps each one. None of them copies code from outside TensorFold, and nothing credited to r0b0tlab or vcruz305
outside TensorFold's history is used. What each one does is recorded in `patches/README.md`, with what was run against it.

| Patch | What it adds |
| --- | --- |
| `0001-minimax-m3-cpu-reference` | float64/CPU reference of the model: the semantics every kernel is checked against |
| `0002-minimax-m3-config-loader` | `minimax_m3_vl` config parsing (`bits` or `avg_bits`), tensor names, shard plan and loader |
| `0003-minimax-m3-template-tools` | chat template and tool-call handling for the family |
| `0004-minimax-m3-swigluoai-ffn` | swigluoai experts, shared expert and the sigmoid router with correction bias |
| `0005-minimax-m3-sparse-attention` | block-sparse attention with the indexer: score, top-k select, attend |
| `0006-minimax-m3-kv-pool` | paged KV pool, page maps and the memory budget |
| `0007-minimax-m3-tp` | tensor-parallel plan over two ranks with rank-order fp32 reductions |
| `0008-minimax-m3-weights-native` | device weights for the family (EXL3 dense, experts, head, norms) |
| `0009-minimax-m3-rows-kernels` | `tf_mm3_rows`: qk-norm and partial RoPE writing paged K/V/IK, fp32 router GEMV |
| `0010-minimax-m3-forward` | the device forward planner and executor, and the prompt driver |
| `0011-minimax-m3-engine`, `0011b-minimax-m3-engine-glue` | the engine loop: slots, chunked prompts, windows, follower control link, backend |
| `0012-minimax-m3-proj-pick-kernels` | fp16/bf16 index projection and the per-rank logit pick kernels |
| `0013-minimax-m3-rows-fix` | contract-test fixes for the rows kernel |
| `0014-minimax-m3-dspark-host` | host half of the dspark speculative decoder |
| `0015-minimax-m3-loader-speed` | coalesced multi-threaded weight streamer (loading went from 1874 s to about 90 s) |
| `0016-minimax-m3-kv-fp8`, `0017-minimax-m3-kv-fp8-contract-fix` | `--kv-cache fp8`: e4m3 K/V with per-token scales (47,040 B a token) and its contract fix |
| `0017-minimax-m3-dspark-device` | device half of dspark: taps, draft kernels, rings, engine wiring, `--drafter` |
| head-stride and head-width commits (in the series) | per-row head decode on the narrower rank; 7/8-bit lm_head; `avg_bits` for the 3.06 bpw pack |

The speed patches (`0020-speed-*`: profiler, prefill attention, EXL3 kernels, TP2 exchange) are credited here when they land.
Their numbers and anything not run are recorded in `patches/README.md`.

## Scripts

- The scripts' structure, the layered settings (`scripts/config.sh`), the node discovery and NFS volumes (`scripts/nodes.sh`),
  `stop.sh`, the start banner, the loading progress line and the image's build pattern come from Mia's AI Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) and
  from Mia's AI Lab's native-Zig
  DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-TensorFold (private repo),
  with code those contributors authored in the GLM recipe:
  - **webzone**: both PCIe links of a cabled QSFP port as rails (the DGX Spark "twins") in `scripts/nodes.sh`
    ([`d5afeb0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/d5afeb0));
  - **[Alexbob0](https://github.com/Alexbob0)**: workers given by host name and `FABRIC_PEER` resolution in
    `scripts/nodes.sh` ([`5e1bf27`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/5e1bf27));
  - **john**: each rank given its link's address of the head as its rendezvous (`LINK_HEAD_ADDR`), kept in
    `start.sh` ([`58eaed0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/58eaed0)).
- `start.sh`'s `FOREGROUND=1` watch loop (`foreground()`) is adapted from the GLM recipe's `start.sh` (Mia's AI Lab), cut down to
  the one worker.
- `scripts/memguard.sh` (the memory watchdog), the memory preflight (`MIN_AVAIL_GIB`), the `API_KEY=1` option and the serve-flags
  check (`check_serve_flags` in `scripts/nodes.sh`, after `scripts/zig_flags.py`) were written by Mia's AI Lab for the DeepSeek
  recipe above and are adapted here. The API key file's format and permission rule are the engine's (TensorFold, Apache 2.0).
- **BobClawblaw**: the setting `NCCL_CUMEM_ENABLE=0` / `NCCL_CUMEM_HOST_ENABLE=0` on the ranks (`scripts/nodes.sh`), as the DeepSeek
  recipe's credits record it; `--init` on the containers follows the same recipe.
- `scripts/publish-image.sh` and the registry pull in `scripts/prepare.sh` are from the GLM recipe (Mia's AI Lab).
- The family gate, the checkpoint check in `scripts/prepare.sh`, `rank_env` and the private-token check were written for this
  recipe by Mia's AI Lab, developed with [Claude Code](https://claude.com/claude-code).

## Tools and platform

- [Zig](https://ziglang.org) 0.17.0 (MIT), the compiler of the engine.
- NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, the image's base: its CUDA toolchain (nvcc), NCCL, cuBLAS and
  libibverbs (NVIDIA's container license and the components' own).
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0), the `hf` download.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server on the GPU
  in a container.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell, 128 GB unified
  memory), two of them linked by their ConnectX-7 ports. The numbers in this repository come from spark3 and spark4 (GB10) and are stated where they appear; the weights and quality have not been evaluated against a reference.

"MiniMax" belongs to MiniMax; "DGX Spark" and "GB10" belong to NVIDIA. This recipe is not affiliated with or endorsed by MiniMax,
NVIDIA or the TensorFold authors.

If you believe something here is missing or credited wrongly, please open an issue.
