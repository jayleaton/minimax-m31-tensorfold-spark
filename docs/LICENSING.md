# Licensing

| Part | License |
| --- | --- |
| This project's code, scripts, configs, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)), Copyright 2026 Jay Leaton. Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| Files from Mia / MiaAI-Lab's MiniMax-M3.1-Flash TensorFold recipe (private repo) (`start.sh`, `stop.sh`, most of `scripts/`, `zig/Dockerfile`, `patches/`, her engine notes in `docs/`, [`CREDITS-MIAAI-LAB`](CREDITS-MIAAI-LAB.md), [`CHANGELOG`](CHANGELOG.md)) | Apache License 2.0, Copyright 2026 MiaAI-Lab; imported at `fd1d379`, her NOTICE reproduced in ours as section 4(d) requires. The full list and our changes: [`NOTICE`](../NOTICE), [PORT](PORT.md). |
| The engine series (`engine/patches/`, [SOURCE-EXPORT](../engine/SOURCE-EXPORT.md)) | `0001` is MiaAI-Lab's family patch (a link to `patches/0001`, Mia's AI Lab's commit, Apache-2.0). `0002` is this project's engine work on top of it, Apache-2.0, Copyright 2026 Jay Leaton; each file keeps the licence and attribution headers it carries, and the TensorFold code it modifies stays under TensorFold's license. Inside `0002`: the DFlash2 reference port of z-lab/dflash's `model.py` (MIT, `tools/dflash2_reference.py`); MiniMax's chat template as a test fixture (`chat_template.m31-54e4e69d.jinja`, from `MiniMaxAI/MiniMax-M3.1-Flash`, MiniMax Community License); the published `config.json` and tensor shapes of incoai's DFlash2 draft (Apache-2.0). Every other fixture is this project's own: a synthetic MiniMax-class DFlash2 config (`dflash2_minimax_layout.json`) and synthetic images made by `tools/zig/mm3_vision/gen_preprocess_golden.py`. |
| The engine family patch (`patches/`) | Mia's AI Lab's changes to TensorFold, Apache-2.0. The TensorFold code they modify, and quote as diff context, stays under TensorFold's license; keep TensorFold's notices with any patched tree. |
| TensorFold (not included; `zig/Dockerfile` fetches upstream v1.1.2 and applies `engine/patches`) | Apache License 2.0, Copyright 2026 TensorFold contributors. The image carries its `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` under `/opt/tensorfold/share/doc/tensorfold`. |
| Code TensorFold carries from others | under the terms its notices record, e.g. b12x's RoCE all-gather (Apache-2.0, Luke Alonso and the b12x contributors). |
| xgrammar (structured output) | Apache-2.0 ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)); built into the image at the engine's pinned tag, not vendored. |
| libjpeg-turbo, libwebp (image input) | libjpeg-turbo's IJG / BSD-3-Clause / zlib terms and libwebp's BSD-3-Clause; built into the image from their release tarballs, not vendored. |
| ExLlamaV3 | MIT ([turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)); the EXL3 format and quantizer the pack was made with; not included. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) and its components' own licenses. |
| Benchmarks fetched at run time | [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (MIT) and [spark-bench](https://github.com/Weschera/spark-bench) (its README says MIT), cloned by `bench/tooleval/run.sh setup`; not included. |
| Bench data | `bench/data/mmlu200.jsonl`: 200 questions of [MMLU](https://github.com/hendrycks/test) (MIT, Hendrycks et al.). `bench/data/drafts-prompts.json`: this project's, Apache-2.0. |

## Model weights (not included)

No weights are in this repository; `scripts/prepare.sh` downloads them, and each is used under its own card's terms.

| Weights | License |
| --- | --- |
| [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) and its `dspark/` draft | **MiniMax Community License** (its `LICENSE`): free for non-commercial use; commercial use has its own conditions. Read it before serving or sharing anything. |
| [`jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31`](https://huggingface.co/jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31) (the pack) | a quantized derivative of the model above: the **MiniMax Community License** applies to it unchanged and ships with it as `LICENSE`. |
| [`incoai/MiniMax-M3.1-Flash-DFlash2`](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2) (`DRAFTS=dflash2`) | Apache-2.0, per its card. It drafts for a MiniMax model; the model's own license still governs serving it. |
| Mia-AiLab's MiniMax-M3.1-Flash EXL3 packs (baselines) | derivatives of the model above; their cards say they do not change the base model's terms. |

The Apache-2.0 license of this recipe grants nothing for any model weights.
