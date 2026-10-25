# Credits

This recipe puts other people's work together on two DGX Sparks. Everyone whose code, kernels, models, benchmarks or
ideas it uses is named here; [CREDITS-MIAAI-LAB](CREDITS-MIAAI-LAB.md) keeps the base recipe's own credits as imported.
If something is missing or credited wrongly, please open an issue.

## Model and drafts

- [MiniMax](https://huggingface.co/MiniMaxAI): [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
  (its design, training, chat template and tool-call format), the **dspark draft** shipped in the model's `dspark/`
  folder (copied unchanged into our pack), and the papers behind them, including MiniMax's sparse attention
  ([arXiv 2606.13392](https://arxiv.org/abs/2606.13392)). The weights are under the MiniMax Community License
  ([LICENSING](LICENSING.md)).
- [incoai](https://huggingface.co/incoai): the **DFlash2 draft**
  [`incoai/MiniMax-M3.1-Flash-DFlash2`](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2) (revision `ac07a7c1`,
  Apache-2.0) that `DRAFTS=dflash2` loads.
- [z-lab / DFlash](https://github.com/z-lab/dflash) (MIT): the draft architecture family of both drafts; the engine's
  DFlash2 reference port follows its `model.py`. The DSpark method is DeepSeek's
  ([DSpark](https://arxiv.org/abs/2607.05147), MIT).
- The serving engines whose dspark semantics the engine's reference follows (ideas, no code): the
  [vLLM project](https://github.com/vllm-project/vllm) (#63232), [SGLang](https://github.com/sgl-project/sglang) and
  TokenSpeed.

## Mia / MiaAI-Lab

This recipe is built on Mia's work ([MiaAI-Lab](https://github.com/MiaAI-Lab) on GitHub,
[Mia-AiLab](https://huggingface.co/Mia-AiLab) on Hugging Face):

- The **MiniMax-M3.1-Flash TensorFold recipe**,
  [MiaAI-Lab/MiniMax-M3.1-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/MiniMax-M3.1-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (Apache-2.0), imported at `fd1d379` as this repository's base: `start.sh`, `stop.sh`, node and link discovery, the memory
  guard, the image build, and the engine notes in `docs/` (rank contract, FP8 KV, dspark plan, loader speed, engine port
  plan, GLM engine reuse, speed surveys, build notes). What changed since: [PORT](PORT.md), [CHANGELOG](CHANGELOG.md).
- The **`minimax_m3_vl` TensorFold family patch** (`patches/0001-minimax-m3-family.patch`, per task in `patches/tasks/`;
  commit author `Mia's AI Lab`): the CPU reference, config loader, template and tool calls, swigluoai experts and router,
  block-sparse attention with the indexer, the paged and FP8 KV pool, the two-rank plan, the device weights and kernels,
  the forward, the engine loop, the loader speed-up and the dspark draft's host and device halves. The engine this
  recipe serves builds on it; [`engine/patches`](../engine/SOURCE-EXPORT.md) ships it unchanged as `0001`.
- Her packs [MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
  and [MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw):
  our pack's 4-bit layers start from the layers her H35 pack widened ([PACKS](PACKS.md)), and her packs are our baseline.
- Her [GLM-5.3 TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) and her
  DeepSeek-v4.1-Flash-1017 TensorFold recipe, which the scripts' structure, the API key option, the serve-flags check and
  `scripts/memguard.sh` come from, with code by their contributors:
  - **webzone**: both PCIe links of a cabled QSFP port as rails in `scripts/nodes.sh`;
  - **[Alexbob0](https://github.com/Alexbob0)**: workers given by host name and `FABRIC_PEER` resolution;
  - **john**: each rank given its link's address of the head as its rendezvous (`LINK_HEAD_ADDR`).

## Inference engine

- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold) and the TensorFold contributors (Apache-2.0):
  the native Zig engine at v1.1.2, its OpenAI-compatible server, lanes and drafting, CUDA runtime, EXL3 kernels and
  tensor-parallel fabric. Its own [credits](https://github.com/ashhart/TensorFold/blob/v1.1.2/docs/credits.md) and
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v1.1.2/THIRD_PARTY_NOTICES.md) apply here too. The
  TensorFold work the `minimax_m3_vl` family is shaped on, by its authors:
  - **[vcruz305](https://github.com/vcruz305)**: TensorFold's EXL3 module (#42) and its grouped-expert design, with
    **justincasey** and **BHCC2025**;
  - **Jürgen Schmied (jschmied)**: the native-server registry and the CUDA serving path (#443), and pooled KV pages with
    **cwschroeder** (#935);
  - **Bizuayeu**: the GLM family's two-rank engine (TP over hosts #804), idle sleep (#1053), structured output (#1017) and
    tool calls (#1018), whose interfaces and forward shape the family follows;
  - **[BobClawblaw](https://github.com/BobClawblaw)**: the rank watch (#855, #976), the GLM DFlash2 CUDA drafter (#979)
    the dspark device half re-implements for this model, and the `NCCL_CUMEM_ENABLE=0` setting on the ranks;
  - **BHCC2025**: the 128-unit vocabulary cut (#593);
  - **BlivionIaG** and **ThinkOffApp**: the backend-owned radix prompt cache (#625, #626);
  - **Luke Alonso and the [b12x](https://github.com/local-inference-lab/b12x) contributors** (Apache-2.0): the one-shot
    RoCE all-gather in TensorFold that the family's small gathers use.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format, the quantizer our pack
  was made with (its `minimax-m3` architecture branch) and the reference the engine's parity tools compare against.
- [mlc-ai / xgrammar](https://github.com/mlc-ai/xgrammar) (Apache-2.0): the grammar engine behind structured output and
  forced tool calls, built into the image at the engine's pinned tag.
- [libjpeg-turbo](https://libjpeg-turbo.org) and [libwebp](https://chromium.googlesource.com/webm/libwebp) (BSD-style):
  the image decoders the image builds for image input.

## Recipes and operations

- Our own [GLM-5.3](https://github.com/jayleaton/glm53-tensorfold-spark) and
  [DeepSeek-V4.1](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) TensorFold recipes: `scripts/serve.sh`,
  the documented production config, the MemFree gate and page-cache drops, floors on both ranks, the pair fail-fast, the
  watchdog and guard, link checks, the credential scanner and redaction, the bench harnesses and the canary
  ([PORT](PORT.md)). Their contributors' work carried here is named above (BobClawblaw); the rest of their credits are in
  those repositories.

## Benchmarks and data

- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (MIT) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks `bench/tooleval/run.sh`
  installs at pinned commits; `bench/tooleval/chains.py` follows the shape of their multi-step categories.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al., MIT): the 200 questions in `bench/data/mmlu200.jsonl`.

## Tools and platform

- [Zig](https://ziglang.org) 0.17.0 (MIT), the engine's compiler.
- NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`), the image's base: CUDA, NCCL, cuBLAS, libibverbs.
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0), [Docker](https://www.docker.com/) and the
  [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) (Apache-2.0).
- NVIDIA: the [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).
