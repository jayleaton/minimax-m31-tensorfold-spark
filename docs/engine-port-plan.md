# MiniMax-M3.1-Flash on TensorFold (native CUDA, Zig): engine-port design

Status: design only. Nothing here has been built or run. No GPU was used.
Target: `Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` (private; EXL3 1.6.1, codebook `mul1`, 3.55 bpw, 179.4 GiB),
`model_type` `minimax_m3_vl`, architecture `MiniMaxM3SparseForConditionalGeneration`, on two DGX Spark (GB10) ranks.
Engine: TensorFold **v1.1.2** (read from a shallow clone at `~/TensorFold/design-clone`, commit
`4ff784410f70b95767684d337e59dfd6d94fec8b`, read only). All TensorFold paths below are relative to that clone and were
opened and read unless marked otherwise.

Conventions: **VERIFIED** = read in the checkpoint files or TensorFold source. **REFERENCE** = taken from a public
third-party implementation page (see Sources); not yet checked against the upstream MiniMax code. **UNVERIFIED** =
not checked, an assumption, or an estimate. Every UNVERIFIED item must be settled by the task named in section 6.

Provenance rule: every kernel and loader reused below is inside TensorFold. That includes the EXL3 module that vcruz305
contributed to TensorFold (`docs/recipes/exl3.md` credits #42); the owner's exception (2026-10-07) allows it. Nothing is
taken from r0b0tlab or from vcruz305 outside TensorFold. The third-party references (SGLang, vLLM docs, NVIDIA
NeMo-Automodel) were read for semantics only; no code is copied from them, and the CPU reference in task T0 must be
written from the formulas in section 2, not pasted.

Input limits: the upstream `MiniMaxAI/MiniMax-M3.1-Flash` repo returned HTTP 401 without a token (not public, or gated, or a
different name), so its modeling code was NOT read. The token was used only against the private checkpoint repo for
`configuration_minimax_m3_vl.py`, `chat_template.jinja` and `tokenizer_config.json`. `tokenizer.json` and
`processing_minimax.py` were not fetched.

---

## 1. Checkpoint: exact tensor-name map (VERIFIED from `model.safetensors.index.json` and `quantization_config.json`)

89,832 tensors, 25 shards, `total_size` 192,592,265,620 B (179.37 GiB by the per-tensor sizes). Only the language model is
present: no vision tower, no `dspark` draft module, no MTP head. Text weights are `language_model.model.*`; note
`lm_head` is `language_model.lm_head.*` (not under `.model.`).

`config.json`: top-level `text_config` (hidden 6144, 60 layers, 64 q heads, 4 kv heads, head_dim 128, vocab 200064,
rms_eps 1e-6, rope_theta 1e7, `rotary_dim` 64, `partial_rotary_factor` 0.5, `hidden_act` `swigluoai`,
`swiglu_alpha` 1.702, `swiglu_limit` 7.0, `use_gemma_norm` true, `use_qk_norm` true with `qk_norm_type` `per_head`,
`attention_output_gate` false, `tie_word_embeddings` false, `dense_intermediate_size` 12288, `intermediate_size` 3072,
`shared_intermediate_size` 3072, `n_shared_experts` 1, 128 experts, top-4, `scoring_func` sigmoid,
`use_routing_bias` true, `routed_scaling_factor` 2.0, `moe_layer_freq` = [0,0,0,1 x57]). `quantization_config` is in the
top level (not in `text_config`): `{quant_method exl3, version 1.6.1, bits 3.55, head_bits 6, out_scales always, codebook mul1}`.
`sparse_attention_config`: `use_sparse_attention` true, index dim 128, 4 index heads, topk 16 blocks, block 128,
score `max`, `sparse_init_block` 0, `sparse_local_block` 1, and `sparse_attention_freq` and `sparse_disable_index_value` are
1 on all 60 layers (every layer is sparse; no layer has an index value or index output projection).
Generation: bos 200019 `]~b]`, eos 200020 `[e~[`, temperature 1.0, top_p 0.95.

Let `P = language_model.model.layers.{L}`. `suh/svh/mul1/trellis` = the 4-tensor EXL3 group of a matrix `M`
(`M.suh` fp16 [K], `M.svh` fp16 [N], `M.mul1` int32 scalar, `M.trellis` int16 [K/16, N/16, 16*bits]).
Every `mul1` marker is a scalar int32 and its recorded `mul1_multiplier` is 2212286765 = 0x83DCD12D for all tensors (one
distinct value in the file). `exl3_decode.cuh` hard-codes 0x83DCD12D (`decode2<CB_MUL1>`), so the loader must **assert** the
marker's multiplier equals it.

| Tensor(s) | dtype / shape | bits | Count |
| --- | --- | --- | --- |
| `language_model.model.embed_tokens.weight` | bf16 [200064, 6144] (2.29 GiB) | none | 1 |
| `P.input_layernorm.weight`, `P.post_attention_layernorm.weight` | bf16 [6144] | none | 60 each |
| `P.self_attn.q_proj.{suh,svh,mul1,trellis}` | suh [6144], svh [8192], trellis [384, 512, 96] | 6 | 60 |
| `P.self_attn.k_proj.*`, `P.self_attn.v_proj.*` | svh [512], trellis [384, 32, 96] | 6 | 60 each |
| `P.self_attn.o_proj.*` | suh [8192], svh [6144], trellis [512, 384, 96] | 6 | 60 |
| `P.self_attn.q_norm.weight`, `P.self_attn.k_norm.weight` | bf16 [128] (per head, shared across heads) | none | 60 each |
| `P.self_attn.index_q_proj.weight` | **fp16** [512, 6144] (4 heads x 128) | none | 60 |
| `P.self_attn.index_k_proj.weight` | **fp16** [128, 6144] (one shared key head) | none | 60 |
| `P.self_attn.index_q_norm.weight`, `P.self_attn.index_k_norm.weight` | bf16 [128] | none | 60 each |
| Dense MLP, layers 0-2: `P.mlp.{gate_proj,up_proj,down_proj}.*` | gate/up svh [12288], trellis [384, 768, 80]; down trellis [768, 384, 80] | 5 | 3 layers |
| MoE layers 3-59: `P.block_sparse_moe.gate.weight` | **fp32** [128, 6144] | none | 57 |
| `P.block_sparse_moe.e_score_correction_bias` | fp32 [128] (shape per VERIFIED count 57; dtype fp32 per README, UNVERIFIED in the header) | none | 57 |
| `P.block_sparse_moe.experts.{E}.w1.*` (gate, suh [6144], svh [3072]), `.w3.*` (up, same shape), `.w2.*` (down, suh [3072], svh [6144]) | 4-bit trellis [384, 192, 64], 3-bit [384, 192, 48] | 3 or 4 | 128 x 3 x 57 |
| `P.block_sparse_moe.shared_experts.{gate_proj,up_proj,down_proj}.*` | width 3072 | 5 | 57 |
| `language_model.model.norm.weight` | bf16 [6144] | none | 1 |
| `language_model.lm_head.{suh,svh,mul1,trellis}` | svh [200064], trellis [384, 12504, 96] | 6 | 1 |

Absent (VERIFIED by the pattern count): `index_v_proj`, `index_o_proj` (the index branch only selects), a shared-expert
gate, any MTP/`eh_proj` tensors, vision tensors, biases on EXL3 groups (no `.bias`), any `rotary` buffers.

Gate/up/down naming: `w1`=gate, `w3`=up, `w2`=down follows the Mixtral convention; vLLM's page says w13 is stored "all
gates then all ups" (REFERENCE), i.e. gate and up are not interleaved. Which of `w1`/`w3` is the SiLU side is
UNVERIFIED (settle in T0 by dequantizing one expert and checking against upstream outputs, or from the upstream loader).

Per-matrix bit widths (VERIFIED by counting `bits_per_weight` per layer; uniform across the 128 experts of a layer):

| Layers | w1 (gate) | w3 (up) | w2 (down) |
| --- | --- | --- | --- |
| 3-16 | 4 | 4 | 4 |
| 17 | 4 | 4 | **3** (the only layer mixed within a layer) |
| 18-45 | 3 | 3 | 3 |
| 46-59 | 4 | 4 | 4 |

Attention q/k/v/o: 6-bit in all 60 layers. Dense MLP and shared experts: 5-bit. lm_head: 6-bit. So the kernels need
`halves` = 2*bits in {6, 8, 10, 12} only (K2 = 6 for 3-bit, 8 for 4-bit, 10 for 5-bit, 12 for 6-bit). No half-bit widths
occur in this pack (the "3.55" average is only the mean).

Sizes by class (VERIFIED, from `n_bytes`): 4-bit routed experts 96.94 GiB, 3-bit routed experts 71.91 GiB, embedding
2.29, q_proj 2.11, o_proj 2.11, lm_head 0.86, shared experts 1.88, index_q 0.35, router gate 0.17, k_proj 0.13, v_proj 0.13,
dense MLP 0.40, index_k 0.09. One 4-bit expert matrix is 9.46 MB, one 3-bit matrix 7.10 MB.

Quant details that matter: `out_scales` "always" means `svh` carries the output scales; the layer is
`y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh` with H the 128x128 Hadamard / sqrt(128) (`docs/recipes/exl3.md`, VERIFIED).
All K and N are multiples of 128 (6144, 8192, 512, 3072, 12288, 200064 = 1563 x 128).

---

## 2. Forward pass, op by op

Decode of R rows (R = 1, or a verify window) at a position `pos`; prefill is the same with R = chunk rows.
Notation: bf16 activations unless stated; `gnorm(x, w)` = Gemma RMSNorm: `x * rsqrt(mean(x^2, fp32) + 1e-6) * (1 + w)`
with the multiply in fp32 (REFERENCE: NeMo `_rms_norm_core` "weight = weight + 1.0", fp32 variance; vLLM
`gemma_rmsnorm`). Rounding points of the upstream model (where it casts to bf16) are UNVERIFIED and decide whether
exactness vs upstream is achievable; the plan targets "exact against our own CPU reference", as the existing families do.

Embedding: `h = embed_tokens[token]` (bf16 gather). No scaling, no hyper-connections, no residual streams other than `h`.

Each layer L (0..59):

1. `a = gnorm(h, input_layernorm)`.
2. EXL3 projections (mul1, 6-bit, shared input rotation of `a` by each matrix's own `suh`, so q,k,v each rotate separately
   unless their `suh` are bit-identical; `exl3_experts` has a `shared_input` check for this on experts; UNVERIFIED for
   q/k/v): `q = Wq a` [R, 64 x 128], `k = Wk a` [R, 4 x 128], `v = Wv a` [R, 4 x 128].
3. Per-head QK norm (before RoPE, REFERENCE): `q[h] = gnorm(q[h], q_norm)` for each of 64 heads, `k[g] = gnorm(k[g], k_norm)`
   for each of 4 kv heads; weights are [128] shared by all heads (`qk_norm_type` per_head, shape VERIFIED).
4. Partial RoPE on the first 64 of 128 dims of q and k, theta 1e7, `inv_freq[i] = 1e7^(-2i/64)` for i in 0..31. Layout is
   NeoX rotate-half (REFERENCE: vLLM "partial NeoX RoPE", SGLang reads `is_neox_style` from `get_rope`'s default);
   UNVERIFIED against upstream. No rope scaling key exists in `config.json` (VERIFIED), so plain RoPE at all lengths.
   v is not rotated or normed.
5. Index branch (sparse layers, all 60): `iq = index_q_proj a` as **fp16 weights, fp16-or-fp32 accumulate** [R, 4 x 128],
   `ik = index_k_proj a` [R, 128] (one shared head, REFERENCE: NeMo `view(..., 1, index_head_dim)`);
   `iq[j] = gnorm(iq[j], index_q_norm)` per index head, `ik = gnorm(ik, index_k_norm)`; the same partial RoPE (rotary_dim 64,
   shared with the main rotary in vLLM: `index_rotary_emb = rotary_emb`, REFERENCE). Precision of the fp16 matmul (input
   cast of bf16 `a` to fp16, accumulate type) is UNVERIFIED; `a` in bf16 can exceed fp16 range only if |a| > 65504, which
   the pre-norm makes unlikely but must be tested (T0).
6. KV write: `K[pos] = k`, `V[pos] = v` (bf16, 4 heads), `IK[pos] = ik` (bf16, 1 head, post norm and RoPE).
7. Block selection per (row, kv head g = index head g; one index head per kv head, REFERENCE: `repeat_interleave(rep, dim=1)`
   expands per-kv-head selections to q heads, rep = 16):
   - `s[j] = (iq[g] . IK[j]) * 128^-0.5` for keys `j <= pos` (fp32).
   - Blocks of 128 keys, `nb = floor(pos/128) + 1`; padded keys of the last block get -inf, so the block score is
     `bs[b] = max over valid j in block b of s[j]` (score type `max`; `lse` is not used here).
   - Forced blocks get +inf: the query's **own** block (`sparse_local_block` 1 is a switch, not a count: any value > 0 forces
     the current block) and block indices `< sparse_init_block`. `sparse_init_block` = 0 is therefore a count of zero: **no
     initial block is forced** (REFERENCE: NeMo `forced = (blk == cur_block & local_blocks > 0) | (blk < init_blocks)`).
     The task brief's "init block 0" must not be read as "force block 0"; UNVERIFIED against the upstream CUDA kernel.
   - `sel = topk(bs, k = min(16, nb))` blocks (16 total including the forced current block). Ties: `torch.topk` order is
     unspecified (REFERENCE); we define ties by lower block index and process selected blocks in ascending order so
     sums are order-fixed. If `nb <= 16` every block is selected (dense attention).
8. Attention for q head h with kv head g = h / 16: keys = all positions of the blocks in `sel[g]`, causal inside the own
   block (`j <= pos`); `softmax(q . K^T * 128^-0.5) V` (scale `head_dim^-0.5`, REFERENCE: SGLang `scaling = head_dim ** -0.5`);
   no attention sinks, no sliding window, no output gate (config VERIFIED). Max 16 x 128 = 2048 keys a row.
9. `o = Wo attn` (EXL3 6-bit, K = 8192); `h = h + o`.
10. `m_in = gnorm(h, post_attention_layernorm)`.
11. FFN:
    - Layers 0-2 (dense): `g = Wgate m_in`, `u = Wup m_in` (5-bit, N = 12288), `act = swigluoai(g, u)`, `f = Wdown act`.
    - Layers 3-59 (MoE):
      a. `logits = gate_fp32 @ m_in.float()` [R, 128] (fp32 matmul, REFERENCE: `self.gate(hidden_states.to(torch.float32))`).
      b. `sel = top4(sigmoid(logits) + e_score_correction_bias)`; weights `w = sigmoid(logits)[sel]` (the bias is for
         selection only, REFERENCE vLLM "used for selection"), `w = w / sum(w)` (renormalize=True, REFERENCE),
         `w = w * 2.0` (routed_scaling_factor, applied on the routed output; REFERENCE SGLang
         `apply_routed_scaling_factor_on_output=True`). Equivalent to `tf_moe_topk_sigmoid(normalize=1, scale=2.0)`.
         Tie rule: lower expert id (`moe_glue.cu`).
      c. Each selected expert e: `act = swigluoai(W1_e x, W3_e x)` (4 or 3 bit), `y_e = W2_e act`; `routed = sum_k w_k y_ek`.
      d. Shared expert (5-bit, width 3072, ungated, no `shared_expert_gate` tensor, VERIFIED): `s = Wd swigluoai(Wg x, Wu x)`.
      e. `f = routed + s` (REFERENCE: `final = final + shared_output`).
    - `swigluoai(g, u)` (REFERENCE, NeMo `swiglu_oai`): `g = min(g, 7.0)`, `u = clamp(u, -7.0, 7.0)`,
      `out = g * sigmoid(1.702 * g) * (u + 1.0)`, computed in fp32. **This differs from every existing TensorFold epilogue**
      (`silu(g) * u` with clamp; section 3).
12. `h = h + f`.

Head: `n = gnorm(h, model.norm)`; `logits = lm_head n` (EXL3 6-bit, N = 200064), fp32 logits, then the shared sampler.

### KV cache layout per layer (per rank, TP2; section 5)

| Cache | Per token, per layer, full model | Per rank |
| --- | --- | --- |
| K | 4 heads x 128 x bf16 = 1024 B | kv heads {2r, 2r+1}: 512 B |
| V | 1024 B | 512 B |
| IK (index key) | 1 head x 128 x bf16 = 256 B | 256 B (replicated: every index head scores the same shared key) |

Per rank, 60 layers: (512+512+256) x 60 = **76,800 B/token** (75 KiB). No recurrent or conv state (every layer is
attention), so a stream's fixed bytes are about zero and a kept prompt snapshot is just page references. Use
`core/kv_pool.zig` with `page_tokens = 128` (one page = one selection block, so a block is one contiguous page and block
tables are page tables) and `token_bytes = 76,800` for 2 ranks. Layout inside a page, per layer, head-major:
`K[kvh][128 tok][128]`, `V[kvh][128][128]`, `IK[128][128]`; this keeps one block's keys contiguous for the selected-block
attention kernel and one index-key block contiguous for the scoring kernel. An optional FP8 K/V mode can reuse
`kernels/cuda/kv_fp8.cuh`, but `docs/recipes/adding-a-zig-family.md` says "No int8 or fp8 activations, and no lossy cache" for
the exactness checks, so bf16 is the default and FP8 stays an opt-in like the GLM `kv8` option.

---

## 3. Reuse map: what exists in TensorFold v1.1.2 and what is new

### 3.1 Closest families

| Family | Why it is close | Path (read) |
| --- | --- | --- |
| `deepseek_v41` | The only native-Zig family whose weights are **EXL3 mul1** with mixed widths: dense `exl3_block_dense.zig` ("mul1: linear.cu, x3seg segment tables, dense3"), grouped routed experts `x3gm` with `gm_k2s = {3,4,5,6,7,8,10}` (so K2 = 6 and 8, our 3/4-bit, plus 10 = 5-bit), prefill kernels `pfdense_k{8,10,12,16}` (4,5,6,8 bits), a router, KV pages, 2-rank paths, its own serve engine. It has CSA/indexer top-k attention but with a different mechanism. | `zig/src/families/deepseek_v41/`, `zig/src/cuda/exl3_block*.zig`, `zig/kernels/cuda/exl3_block/` |
| `glm` (CUDA, model_type `glm5_next`) | The template for **a TP2 EXL3 family on GB10**: `cuda_shard.zig` (`Share.of`: heads one by one, EXL3 widths cut in 128s, vocab by `head_unit` 128), `cuda_exl3.zig` (per-rank cut of expert tiles, `shx` shared-rotation check), `cuda_ranks.zig` (rank fingerprint agreement), `cuda_native.zig` (the `Loaded` contract and the `snap_*` hooks), `cuda_sparse.zig` (a sparse-attention indexer). Its EXL3 is mcg 4-bit experts with bf16 everything else, and `admit` refuses anything else (checked: it requires `bits == 4` and codebook `mcg`). | `zig/src/families/glm/` |
| `glm_dsa` | Metal/CPU-style MLA + DSA indexer reference with a `Config`, `moe.zig`, `model.zig`; useful for how a CPU-checkable family is laid out. Its indexer is token-pool based (pools of 4, top-512), which is **not** the MiniMax block-max selection. | `zig/src/families/glm_dsa/` |
| `flashnext` | Sparse "qsa" indexer scores on tensor cores, shared lane rounds, TP, `lane_host`. A model for the *scoring-kernel* style only. | `zig/kernels/cuda/flashnext_attn.cu`, `flashnext_qsa_tile.cuh`, `zig/src/families/flashnext/` |

None of them is a fit by itself. The port is "GLM's shell and TP, deepseek_v41's mul1 EXL3 blocks, new attention".

### 3.2 Reuse as is (VERIFIED present)

- EXL3 format, loader, oracle: `zig/src/core/exl3_format.zig` (scalar decoder, all three codebooks, half-bit widths),
  `exl3_load.zig` (scan of trellis groups + markers from safetensors headers), `exl3_layout.zig`, `exl3_prepare.zig`.
- Decode linear, any codebook and width incl. `mul1`, 1-128 rows, row-invariant: `zig/src/cuda/exl3.zig`,
  `zig/kernels/cuda/exl3_linear.cu`, `exl3_decode.cuh` (variants for halves 2..16 and mul1 odd halves 3..13, 4 and 8 warps).
  Covers q/k/v/o (K2 12), dense MLP and shared experts (K2 10), lm_head (K2 12). `qlinear.zig` has an `.exl3` arm.
- Grouped routed experts for decode, **per-expert, per-matrix `k2`, one codebook per layer**: `zig/src/cuda/exl3_experts.zig`
  (`Layer` has `gate_k2/up_k2/down_k2` int32 arrays, `codebook`, `limit`, `act_mode`) and `zig/kernels/cuda/exl3_experts.cu`
  (`tf_exl3_grouped_{0,1,2}` per codebook; `TF_EXL3X_CASE` covers k2 2..16). So layer 17's 4/4/3 mix and 3-bit/4-bit layers
  need no new GEMV, only descriptors. `exl3_upload.zig` (`layer(...)`) builds the pointer tables.
- Prompt-path experts for mul1 at 3/4/5 bits: `zig/kernels/cuda/exl3_block/x3gm.cu` + `src/cuda/exl3_block_x3gm.zig`
  (tile tables, `gm_k2s` incl. 6 and 8). Prompt dense at 4/5/6/8 bits: `pfdense_k8/k10/k12/k16.cu`. The common
  `exl3_prompt.cuh` path is **mcg 4-bit only** (`eligible()` requires `codebook == .mcg and uniform_k2 == 8`) so it does not serve us.
- Sigmoid top-k router with correction bias, normalize and scale, tie to lower id: `zig/kernels/cuda/moe_glue.cu`
  (`tf_moe_topk_sigmoid`, `_warp`; `tf_moe_combine`), `zig/src/cuda/moe_glue.zig` (`Route{experts, topk, shared, normalize, scale}`).
  Needs an fp32 logits GEMV: `deepseek_v41/router_gemv.cu`, `nemotron_route.cu` (fp32 logits in K slices) are the models;
  the exact kernel to take is UNVERIFIED (K = 6144, N = 128, input fp32).
- Per-head QK norm + rotate-half RoPE on `rot` dims + cache write in one launch: `zig/kernels/cuda/qk_rows.cu`
  (`tf_qk_rows`, `rot` argument). Needs the Gemma `1 + w` weight (fold at load into an fp32 weight, or add a flag) and an
  index-branch variant (4 + 1 heads, writes IK).
- RMS norm rows incl. fused residual add: `zig/kernels/cuda/norms.cu` (`tf_rms_rows`, `tf_add_rms_rows*`); Gemma variant is a weight transform.
- Prefill attention with GQA tensor-core tiles: `prefill_attention.cu` and decode `window_attention.cu` are the structural
  models for the selected-block kernels (not directly usable: they take dense or windowed key ranges).
- Sampler, lanes, server: `zig/src/cuda/sample.zig`, `zig/src/core/lanes/*` (lane round, copy drafts, depth rule), `zig/src/core/lane_host.zig`,
  `engine_api.zig`, `zig/src/server/*` (OpenAI/Anthropic, `tool_stream.zig`), `zig/src/core/prompt_cache.zig`, `kv_pool.zig`,
  `zig/src/native/cuda_kv_pool.zig`, `cuda_memory.zig`.
- TP: `zig/src/cuda/tp_group.zig` (`Group.open`, `allGather`, `send/recv`, RoCE small gathers `openRoce`), `nccl.zig`.
  Fingerprint agreement across ranks: `families/glm/cuda_ranks.zig` (`Fingerprint`, `mismatch`).
  `tf-cuda-test exl3-tp2 RANK HOST:PORT DIR` already checks output-column cuts, input-row cuts with fp32 partials summed rank 0 first, and the experts' intermediate cut (`docs/recipes/exl3.md`).
- Tokenizer and Jinja engine: `zig/src/core/tokenizer/{tokenizer,tiktoken,regex,steps,eos}.zig`, `zig/src/core/template/*`
  (`parity.zig` suggests a parity suite against Python Jinja; UNVERIFIED which filters the MiniMax template needs).

### 3.3 Must be new

| New item | Why | Closest code to start from |
| --- | --- | --- |
| **swigluoai epilogue** (`g=min(g,7)`, `u=clamp(u,-7,7)`, `g*sigmoid(1.702 g)*(u+1)`) in every place an expert or MLP activation is fused | Existing epilogues are `act = g/(1+exp(-g)) * u` with `upTo/within` clamps and `act_mode` 0 (GLM bf16 roundings) or 1 (fp32), e.g. `exl3_experts.cu` `tf_exl3_gateup_epilogue` lines ~290-312 and `exl3_block/x3gm.cu` lines ~240-243, `exl3_block/exl3_experts.cu`. Add `act_mode = 2` (alpha as a launch arg) to: decode `tf_exl3_gateup_epilogue`, the `tgateup` path (line ~427), `x3gm.cu` prompt epilogue, `x3ld_epi.cu`. For dense L0-2 and the shared expert (plain dense linears) add one pointwise kernel. |
| **Sparse-block index kernels**: score + block-max, forced-block top-16, selected-block GQA attention (decode and prefill) | No TensorFold kernel does block-max selection. GLM does pooled-token top-512 (`cuda_sparse.zig`, `glm_glue_index.cuh`); DeepSeek CSA is different. | Style of `flashnext_attn.cu` (`qsa` scores) and `prefill_attention.cu` GQA tiles |
| **fp16 index projections** (`index_q_proj` [512,6144], `index_k_proj` [128,6144]) | No fp16-weight GEMV family; dense bf16 rows exist (`dense_rows.cu`, bf16 weights). | `dense_rows.cu`; fuse with Gemma norm + RoPE to avoid two launches |
| **Gemma norm weights** (`1 + w`) | Load-time transform is enough; must not double-add. | `norms.cu` |
| **fp32 router GEMV for a 128-row fp32 weight** | Existing routers take bf16 or have fixed K. | `router_gemv.cu`, `nemotron_route.cu` |
| **Family shell** `zig/src/families/minimax_m3/` (config, names, loader, engine, `native`) and registration | See section 4 T1, T10. | `families/glm/*` |
| **MiniMax tool-call parser** (`]<]minimax[>[<tool_call>` ... `<invoke name="f">` `]<]minimax[>[<k>v]<]minimax[>[</k>`, nested objects as `<item>`) | `server/tool_parse.zig` handles DSML, `<tool_call>` JSON, `<function=...>`, GLM `<arg_key>` and Gemma `<|tool_call>`, but a grep for "minimax" in `zig/src` finds nothing outside H3, so this markup is not handled (the `<tool_call>` branch at line 57 would see the *token* `<tool_call>` preceded by the namespace token, and its payload is XML, not JSON; behavior UNVERIFIED). | `server/tool_parse.zig`, `tool_stream.zig`, `tool_closers.zig` |
| **Thinking handling** for `<mm:think>`/`</mm:think>` (ids 200059/200060) and modes enabled/disabled/adaptive | `think.zig` (Flash Next budget) and server think handling assume other tokens. | `families/flashnext/think.zig`, `core/engine_api.zig` |

### 3.4 mul1 codebook and mixed 3/4-bit expert support in the CUDA EXL3 kernels: finding

VERIFIED present, no new kernel work needed for the GEMV/linear arithmetic:

- `mul1` in `exl3_decode.cuh`: `x = s * 0x83DCD12D; h = 1024 + bytesum(x)` via `__dp4a`, then `hfma2(h, fp16(0x1EEE), fp16(0xC931))`, identical to
  the reference in `exl3_format.zig` and `docs/recipes/exl3.md`.
- Whole-bit widths 3..6 are K2 6, 8, 10, 12; the K2 switch in `exl3_experts.cu` has cases 2..16, and `exl3.zig`'s `variants` has all
  whole widths for mul1.
- Mixed widths inside one layer are supported by the `Layer` descriptor (per-expert, per-projection `k2` arrays), which
  covers layer 17 (w1/w3 4-bit, w2 3-bit, uniform across experts).

Gaps that remain (these are the real work):

1. The common EXL3 prompt kernels (`exl3_prompt.cuh/.zig`) are mcg-4-bit-only. Use deepseek_v41's `x3gm` (mul1, K2 6/8/10) and `pfdense` (K2 8/10/12/16)
   for prompts. Those are "Copy of src/tensorfold/families/deepseek_v41/cuda/..." files built for DeepSeek's shapes (K = 5120, etc.); their shape
   constraints for our N = 512 (k/v), 8192, 3072, 12288 and for **TP row/column cuts** are UNVERIFIED. The 3-bit prompt GEMM instantiation (K2 = 6) exists in `gm_k2s`, but whether
   every tile config compiles and runs for D = 6144, I = 1536 (TP-cut) is UNVERIFIED. Fallback that already exists: `exl3.Dense` runs prompt rows as 128-row blocks of the decode kernel
   (row-invariant, slower).
2. swigluoai (above).
3. `docs/recipes/exl3.md` says "No family serves EXL3 natively yet" for Metal/CUDA in its native section, and the GLM recipe states the native binary "has no GLM CUDA backend yet"
   while the tree contains `glm/cuda_exl3.zig`. So the end-to-end status of any native-Zig EXL3 serving path in v1.1.2 is UNVERIFIED; treat unit-level kernel coverage (`tf-cuda-test exl3`) as the only proven part.
4. `exl3_experts` limits: `grouped.max_experts` (checked in `exl3_prompt.zig`) and `slots <= 32`; with 128 experts and top-4 the values are expected to be fine; exact limit UNVERIFIED.
5. TP cuts need 128-aligned intermediate: 3072/2 = 1536 = 12 x 128 (ok); dense 12288/2 = 6144 (ok); `head_unit` 128 for the vocab (1563 units: ranks get 782/781 units).

---

## 4. Work breakdown (dependency order)

Assume no usable GPU locally. Every task has a CPU path (host test run with `zig build test-...` or plain `zig test`) as its acceptance gate; GPU gates (marked GPU) run later on the two Sparks.
New family directory: `zig/src/families/minimax_m3/`. New kernels: `zig/kernels/cuda/minimax_m3_*.cu` (+ `.cuh`), registered in `zig/build/cuda_kernels.zig`
(entry list with flags, see the `moe_glue`/`exl3_*` lines) and `zig/src/cuda/kernels.zig`. Module wiring: `zig/build/cuda.zig` (see `glmCuda(...)` and `runner.addImport("glm_cuda", ...)`).

**T0. CPU reference model and fixtures (first; unblocks T4-T9).** Files: `zig/src/families/minimax_m3/ref.zig`, `zig/src/core/minimax_ref.zig` (float64 helpers beside `attention_ref.zig`, `route_ref.zig`, `qk_ref.zig`), `zig/tests/minimax_ref_tests.zig`.
Write in float64 from section 2: gnorm, partial NeoX RoPE, per-head QK norm, index scoring, block max, forced blocks, topk with defined ties, selected-block attention, sigmoid router with bias, swigluoai, shared expert, dense MLP, full layer, 2-layer model, head.
Synthetic weights: random int16 trellis + random suh/svh with `exl3_format.dequantize` (any bit pattern is a valid trellis, so no quantizer is needed) at widths 6/8/10/12 halves with mul1, plus fp16/fp32/bf16 plain tensors. Tiny config (hidden 256, 2 q heads group...) must keep 128-aligned dims and block size 128 so selection is exercised with > 16 blocks (use 4096+ context, topk 4, or shrink `topk` in config for tests).
Acceptance: (a) swigluoai and selection match hand-computed vectors including partial last block, own-block forcing, `nb <= topk` dense case, ties; (b) resolve the UNVERIFIED items in section 6 by reading the upstream modeling code once access exists and encode the answers as test vectors; (c) a Python generator is not required in-tree.

**T1. Config, tensor-name map, admission.** Files: `families/minimax_m3/config.zig`, `names.zig`, `shard.zig` (model on `glm/config.zig`, `glm/names.zig`, `glm/cuda_shard.zig`).
Parse `text_config` (not top level) and `quantization_config` from the top level (`glm/cuda_exl3.zig` `block()` handles both places). Admit only `quant_method exl3`, `codebook mul1`, `version 1.6.1` or newer after review; refuse vision requests (no tower in the pack).
Name map from section 1 (`language_model.model.` / `language_model.lm_head.`); per-matrix bits from each trellis' last dim (`16*bits`), never from `bits`. Assert `mul1` multiplier == 0x83DCD12D and all K, N % 128 == 0.
Acceptance (host): a test that parses the real `config.json` and `model.safetensors.index.json` (copy to `zig/src/families/minimax_m3/fixtures/`, index trimmed to layers 0, 3, 17, 18, 46, 59) and asserts the counts in section 1, the bit table per layer, the 89,832 tensor total on the full index (kept out of tree if too large).

**T2. Safetensors loader and uploader for EXL3 groups.** Files: `families/minimax_m3/load.zig`, reuse `core/exl3_load.zig`, `cuda/exl3_upload.zig` (`Uploader.dense`, `layer`), `core/direct_io.zig`.
Per-rank TP cuts (T7 defines them), expert pointer tables with per-expert `k2`, embeddings and plain tensors, Gemma `1+w` transform, fp16/fp32 plain tensors as-is. Stream shards from disk with direct I/O into device memory; do not mmap the 180 GiB set (unified memory counts page cache against the host budget).
Acceptance (host): synthetic mini-checkpoint written with safetensors headers loads, the pointer tables and `k2` arrays equal the expected, layer 17's 4/4/3 is represented, truncated/duplicate/misshaped tensors fail closed (follow `core/safetensors_security_test.zig` style). GPU: `tf-cuda-test` style upload of one real layer.

**T3. Tokenizer, chat template, tool-call parser, think handling.** Files: `core/tokenizer/*` only if a gap is found; `families/minimax_m3/template.zig` (fixture tests); `server/tool_parse.zig` (new MiniMax branch), `server/tool_stream.zig`, `tool_closers.zig`; tests beside `tool_stream_test.zig`.
Needs: `tokenizer.json` (not yet fetched; UNVERIFIED that it uses the supported ByteLevel BPE path), special ids from `tokenizer_config.json` (VERIFIED: 200019 `]~b]` bos, 200020 `[e~[` eos, 200034 `]~!b[`, 200058 `]<]minimax[>[`, 200059/200060 `<mm:think>`/`</mm:think>`, 200025/200026 image/video). The template (`chat_template.jinja`, 13,151 B, read) uses `tojson(ensure_ascii=False)`, `split`, `strip`, `namespace`, macros with recursion, `is mapping`, `is iterable`, `loop.index0`, `raise_exception`; run it through `core/template` and compare to Python Jinja output on a corpus (system/root/developer/tool/assistant-with-reasoning/tool-call/multi-tool/images).
Behaviors to encode: the template's thinking modes (`thinking_mode` enabled/disabled/adaptive, default adaptive: generation prompt has no think prefix; disabled appends `</mm:think>`; enabled appends `<mm:think>`), history assistant turns re-render reasoning as `<mm:think>...</mm:think>` or a bare `</mm:think>` prefix, tool results in `\n<response>...</response>` after `]~b]tool`.
Parser: extract `<invoke name="n">` blocks between the `]<]minimax[>[<tool_call>` ... `]<]minimax[>[</tool_call>` markers; each parameter is `NS<k>value NS</k>`; nested mapping -> child elements, list -> repeated `<item>`; booleans via `tojson` (`true`/`false`); `None` omitted. Value typing against the tool schema (numbers, bools, JSON) is UNVERIFIED (the template does not say how scalars are typed back); settle with upstream's parser or a fixture.
Acceptance: golden template renderings match Python Jinja byte for byte; parser round-trip: render the template's tool-call example, parse it, get the same arguments; streaming deltas split at every byte boundary produce one call.

**T4. Dense/MoE FFN block (decode first).** Files: `families/minimax_m3/ffn.zig`, `zig/kernels/cuda/minimax_m3_act.cu` (swigluoai pointwise for dense and shared), edits to `kernels/cuda/exl3_experts.cu`, `kernels/cuda/exl3_block/x3gm.cu`, `exl3_block/exl3_experts.cu`, `x3ld_epi.cu` (new `act_mode == 2`), `cuda/exl3_experts.zig` (`Layer.act_mode`, `alpha`), `cuda/moe_glue.zig` for the router.
Wire: fp32 router GEMV -> `tf_moe_topk_sigmoid(normalize=1, scale=2.0)` -> grouped gate/up -> swigluoai -> grouped down -> `tf_moe_combine` -> add shared expert (dense EXL3 5-bit) -> TP reduction (T7).
Acceptance (host, CPU-checkable): the kernel *sources* are checked by `kernel-contract` records and by host tests of the plan/descriptor code; arithmetic is checked in GPU runs against T0 (GPU gate: layer 3 (4-bit), 17 (mixed), 18 (3-bit) of the real pack vs the float64 reference, row invariance for windows 1,2,3,16,17,64,128 per the contract in `docs/recipes/cuda.md`). Register new kernels in `core/contract/registry.zig` with float64 references and the planted controls (`docs/kernel-contract.md`: gates 1-5; swigluoai must propagate NaN like the reference, so use `propagate`, and note `fminf` drops NaN, use the `upTo/within` pattern in `exl3_experts.cu`).

**T5. Sparse attention kernels and attention block.** Files: `zig/kernels/cuda/minimax_m3_index.cu` (score + block max), `minimax_m3_select.cu` (forced-block top-16), `minimax_m3_attn.cu` (selected-block GQA decode and prefill), `families/minimax_m3/attn.zig`, `cuda/qk_rows.cu` additions (Gemma weight, index heads), `core/minimax_ref.zig` (from T0).
Design: kernel A computes `s = iq . IK` per (row, index head) tile-wise on mma and writes only per-128-key block maxima (fp32, `[rows, 4, nblocks]`; at 128k context a 8192-row chunk needs 8192 x 4 x 1024 x 4 B = 134 MB). Kernel B forces the own block, takes top-16 by (score desc, block asc), emits ascending block lists `[rows, 4, 16]` (int32, -1 padded). Kernel C: a CTA per (row tile, kv head), the 16 q heads of the group share one K/V stream over the selected pages; fixed key order and fixed split so a row's bits do not depend on the window width (`docs/recipes/adding-a-cuda-family.md` Exactness).
Decode cost note: scoring reads the whole `IK` cache (256 B/token) every token every layer: at 128k that is 32 MiB x 60 = 1.9 GiB per generated token per rank, about 9 ms at ~220 GB/s (UNVERIFIED estimate). Exact block maxima cannot be cached without dropping exactness; an approximate summary is a later, explicitly separate option.
Acceptance (host): selection/attention reference tests from T0 at block boundaries (pos = 127, 128, 129, 2047, 2048, 2049, 128k), and `registry.zig` gate-5 launch records computed without a GPU. GPU: contract gates 1-4 and a layer test vs T0.

**T6. KV pool, snapshots, memory budget.** Files: `families/minimax_m3/pool.zig`, `state.zig`, hooks `snap_*` as in `glm/cuda_native.zig`, uses `core/kv_pool.zig` (`Layout{fixed_bytes ~ 0, token_bytes 76,800, page_tokens 128, snap_bytes}`), `native/cuda_kv_pool.zig`.
Paged K/V/IK per section 2; block tables = page tables (the attention kernel takes page ids, not addresses); prompt cache keeps page refs (no recurrent state to copy).
Acceptance (host): `kv_pool` plan arithmetic for budgets (section 5), pages needed for prompt+reply+window, eviction, snapshot save/restore tests modeled on `core/kv_pool_store_test.zig` and `kv_pool_host_test.zig`.

**T7. Two-rank TP: split plan and collectives.** Files: `families/minimax_m3/shard.zig`, `ranks.zig`, `tp.zig`, `cuda/tp_group.zig` use (`allGather`, `send/recv`, `openRoce` small gathers), a rank-ordered fp32 partial sum (see `kernels/cuda/rank_sum.cu`, `cuda/rank_scatter.zig`; their exact contract is UNVERIFIED).
Split (section 5). Reductions are "gather both fp32 partials, add in rank order 0 then 1" so both ranks hold identical bits (`adding-a-cuda-family.md`: "gather and add in rank order").
Acceptance (host): `Share.of`-style tests for every axis (heads, kv heads, index heads, 128-aligned cuts, vocab 782/781 units, refuses empty ranks); fingerprint disagreement (`cuda_ranks.zig`) refuses mismatched config, quantization and weights hash. GPU: `tf-cuda-test exl3-tp2` extended with mul1 6-bit o_proj (row cut) and 3/4-bit expert cuts; two-rank output equals the single-rank reference where memory allows (a 4-layer slice fits one GPU).

**T8. Engine: prefill, decode, verify windows, lanes.** Files: `families/minimax_m3/engine.zig`, `forward.zig`, `prefill.zig`, `backend.zig` (the `lanes.backend.Backend` vtable: `prefill`, `first`, `queue`, `read`, `verify`, `keep`, `draft`), model on `families/glm/cuda_engine.zig`, `cuda_forward.zig`, `plain_rounds.zig`.
v1 scope: plain decode plus verify windows with the shared copy-draft proposer (`core/lanes`); no MTP/dspark (absent from the pack); no vision. Graph capture after eager exactness is proven (GLM/Flash Next practice).
Acceptance (host): backend vtable contract with the fake backend in `core/lanes/fake.zig`; chunk/resume equivalence logic. GPU: eager vs graph equal; window width invariance; resumed prompt equals fresh prompt (`snap_grid`).

**T9. Server/native integration.** Files: `zig/src/families/minimax_m3/native.zig` (`pub const model_type = "minimax_m3_vl"; formats = &.{"exl3"}` with `Options`, `Loaded`, `open`, `explain`, `capacity`, `weightShare`), `zig/src/native/cuda.zig` (add to `registry`), `zig/src/cli/hub.zig` (`titles`: `.{ .model_type = "minimax_m3_vl", .title = "MiniMax-M3.1-Flash" }`), `zig/src/native/cuda_refusal.zig`, `zig/src/native/cuda_family_memory.zig`, `zig/build/cuda.zig`.
Behaviors: model_type `minimax_m3_vl` -> text-only engine, vision/image requests refused with a clear message, `--parallel` serialized at first like GLM, TP rank/world from environment (GLM uses `TF_GLM_RANK`, `NCCL_COMM_ID`; define `TF_MM3_RANK/WORLD`), `generation_config.json` defaults (T=1.0, top_p 0.95), eos 200020, think tokens.
Acceptance (host): `cuda_startup_test.zig`-style startup/refusal tests; `families` table includes the new entry; `cuda_refusal.zig` tests for tp mismatch. GPU: served health, one request, tool call round trip.

**T10. Test and validation harness.** Files: `zig/tests/cuda/minimax_*_tests.zig` (model on `zig/tests/cuda/glm_exl3_tests.zig`), `zig/build/minimax_host.zig` (a `test-minimax-host` step with no GPU, like `test-exl3-cuda-host` in `zig/build/cuda.zig`), kernel-contract entries in `zig/tests/cuda/contract_*.zig` and `core/contract/registry.zig`.
Acceptance: `zig build test-minimax-host` runs T0-T3, T6, T7 host tests; GPU suites are separate steps run on the Sparks. Quality gate after the engine runs: KL/top-1 agreement against the README's reported numbers on the same 2,048-token set is not reproducible without the original model; instead record greedy tokens on fixed prompts and compare TP2 to a single-rank slice, and check the tool-calling and agentic strata first (the README says that is where the 3.55 bpw pack is strongest).

Dependency order: T0 -> (T1, T3 in parallel) -> T2 -> (T4, T5, T6, T7 in parallel; T4 and T5 need T0) -> T8 -> T9 -> T10 (T10 host tests grow with each task, final GPU suite last).
Independent engineers: loader (T1+T2), tokenizer/template/parser (T3), FFN/kernels (T4), attention kernels (T5), KV/pool (T6), TP (T7). Engine/integration (T8, T9) after at least T1, T4, T5, T6, T7 interfaces are fixed; define those interfaces (function signatures in `families/minimax_m3/iface.zig`) in a 1-day contract step before parallel work.

---

## 5. Two-rank TP split and memory budget (GB10 x 2)

Split (heads whole, EXL3 cuts on 128-aligned tiles; mirrors `glm/cuda_shard.zig`):

| Object | Rank r of 2 | Reduction |
| --- | --- | --- |
| q_proj (8192 out) | q heads [32r, 32r+32): output columns 4096 each | none |
| k_proj, v_proj (512 out) | kv heads {2r, 2r+1}: 256 columns each; GQA group g uses q heads [16g, 16g+16), all on one rank | none |
| Q/K norms | replicated (128 floats) | none |
| index_q_proj (512 out, fp16) | index heads {2r, 2r+1} (index head g selects for kv head g): 256 rows each | none |
| index_k_proj (128 out, fp16) | **replicated** (one shared key head); each rank scores with its own two index heads over the full IK | none |
| o_proj (8192 in) | input rows [4096r, 4096r+4096) (whole tiles and Hadamard blocks), `svh` whole | fp32 partials, sum rank 0 then 1 |
| Dense MLP gate/up (12288 out), down (12288 in) | 6144 each | with FFN reduction |
| Router gate fp32 [128, 6144] | replicated (identical logits and picks on both ranks, same kernel) | none |
| Routed experts | every expert on both ranks, intermediate cut 1536 each (gate/up columns, down rows) | fp32 partial, one reduction per layer |
| Shared expert (3072) | 1536 each | added into the same partial as routed, one reduction |
| Embedding (bf16 2.29 GiB) | replicated (simple; vocab-parallel gather is a later saving of 1.15 GiB) | none |
| lm_head (200064 out) | 128-wide units 782/781 per rank | all-gather of logits, or per-rank argmax/top-k then merge (use the existing sampler's rule; UNVERIFIED) |

Two reductions a layer (attention, FFN), 120 a token; each moves R x 6144 fp32 (24 KB at R = 1). `tp_group.zig` has RoCE small gathers for two ranks; per-exchange latency is UNVERIFIED (estimate 20-40 us, 2.4-4.8 ms a token).
Expert-parallel (64 experts a rank) is rejected: top-4 routing would leave the ranks unequal on single rows. Row-cut TP keeps equal bytes per rank for every pick.

Per-rank weights (computed from `quantization_config.json` sizes, VERIFIED inputs, my arithmetic):
routed experts 84.42 + shared 0.94 + q/o 2.11 + k/v 0.13 + index_q 0.18 + index_k 0.09 + router 0.17 + dense MLP 0.20 + lm_head 0.43 + embedding 2.29 = **about 91.0 GiB per rank** (the 179.4 GiB total halves to 89.7 plus about 1.3 GiB replicated).

Per-rank KV: 76,800 B/token = 75 KiB (section 2). 32k = 2.3 GiB, 128k = 9.4 GiB, 256k = 18.8 GiB, 1M = 75 GiB (does not fit).

Budget (UNVERIFIED hardware numbers: GB10 is 128 GB LPDDR5x unified; I did not read `/proc/meminfo` on a Spark, so total/available must be measured): assume about 119 GiB physical.
TensorFold's admission leaves a tenth of RAM (`docs/recipes/cuda.md`, `TENSORFOLD_MEMORY_RESERVE_GIB`) and the GLM recipe measured 3.5-4.8 GiB of CUDA context, NCCL and graph memory outside the estimate. So: 119 - 11.9 reserve = about 107 GiB; minus 91.0 weights = about 16 GiB; minus 4-5 GiB context/NCCL/graphs; minus 1-2 GiB prompt workspaces (prompt W_q decode scratch, rotation buffers, the 134 MB block-score chunk, expert plan scratch) = **about 9-10 GiB for KV, i.e. roughly 130k tokens at bf16** on one stream. That is far below the declared 1,048,576 context, so v1 must advertise a smaller context (e.g. 65,536-131,072) and refuse larger ones with the fitting capacity, as the existing CUDA hosts do.
Load-time trap: a 90 GiB read through the page cache plus 90 GiB of device allocations on unified memory can double count; use direct I/O (`core/direct_io.zig`) and the CUDA copy-source owner (`cuda/copy_source.zig`) and verify with the memory receipts (`docs/cuda-memory-receipts.md`, unread beyond its existence).

Rough decode throughput bound (estimate): bytes read a token about 14.5 GB total (attention 4.8 GB, routed experts 5.7 GB, shared 2.0 GB, dense 0.4, index projections 0.5, router 0.2, head 0.9), about 7.3 GB a rank, at the 221 GB/s that `docs/recipes/exl3.md` measured for 6-bit EXL3 on GB10: about 33 ms, so a ceiling near 30 tok/s per stream before collectives and IK scoring. UNVERIFIED; do not quote it as a result.

---

## 6. Risks and unknowns (stated as such)

Semantics not yet confirmed against the upstream MiniMax model code (blocked: upstream repo returned 401):
1. `sparse_init_block` 0 and `sparse_local_block` 1: reference reading is "count of forced initial blocks = 0, own block forced". If upstream instead forces block 0 or uses a window of local blocks, selection changes. Highest-impact unknown for long context. Also tie order, and whether the block score for the current (partial) block includes only causal keys (NeMo does via -inf padding and causal masking).
2. Whether index RoPE shares the main rotary exactly (vLLM) and uses rotate-half (NeoX) over the first 64 dims; whether q/k RoPE is NeoX or GPT-J; the bf16/fp32 rounding points around norm, RoPE and the fp16 index matmul.
3. Router details: whether sigmoid score is renormalized over the 4 picks before the 2.0 scale (REFERENCE says yes), and that shared output is not scaled.
4. `w1`/`w3` assignment (gate vs up) and whether `swigluoai` uses `(up + 1)` for routed experts exactly as for dense/shared (vLLM calls the routed activation `swigluoai_uninterleave`; REFERENCE).
5. Whether the `+1` Gemma offset applies to `q_norm/k_norm/index_*_norm` as well as the layer norms (REFERENCE says Gemma norms on index norms and qk norms; per-layer norms assumed too).
6. `e_score_correction_bias` dtype in the header (README says fp32; not read in the index).

Engine risks:
7. Native EXL3 serving on CUDA in the Zig engine has not been shown end to end for any family in v1.1.2 (docs say so); only kernel-level coverage is documented. Expect integration bugs in `exl3_block` / `x3gm` paths when fed our shapes and TP cuts.
8. Exactness against the original model is not a goal that can be verified (the README measured KL 0.0566 mean, top-1 94.3% at 2,048 tokens only). Layers 25+ route to the same experts as the original only 50-70% of tokens (README), so quality at long context and in chat/multilingual may be poor regardless of port correctness; long context, long agentic prompts, vision and speculative decoding are all untested by the pack's author.
9. Decode cost of exact block scoring grows with context (about 9 ms a token at 128k estimated). Prefill scoring is O(T^2) per layer (about 8.4 TFLOP a layer at 128k for 4 heads, 2 heads a rank); acceptable but unmeasured.
10. Memory: about 130k tokens of bf16 KV is the realistic ceiling; the 1M context cannot be offered. Hardware RAM numbers are assumed.
11. The 128-aligned cuts rely on `exl3-tp2` checks that exist for GLM shapes only; mul1 6-bit o_proj row cut at K = 8192 is new.
12. Template/Jinja: `core/template` may lack a filter or test the MiniMax template uses (`is boolean`, `tojson(ensure_ascii=False)`, `split(...)[-1]`, `strip('\n')`, `namespace`); to be discovered in T3.
13. Licensing: base model license (MiniMax) and the private checkpoint terms were not read; the pack's README says follow the base license. EXL3 credit: format and kernels derive from exllamav3 (turboderp, MIT), as the headers in `exl3_decode.cuh` state; keep that credit in `docs/credits.md` when the family lands.
14. Vision: `minimax_m3_vl` implies a tower (1280-wide, 32 layers, 3D RoPE, patch merge 2x2, temporal 2) but the pack ships none. Vision is out of scope; the template still emits image tokens, so requests with images must be refused explicitly.

## Sources

- Checkpoint files (private repo, small files only): `config.json`, `model.safetensors.index.json`, `quantization_config.json`, `generation_config.json`, `README.md`, `configuration_minimax_m3_vl.py`, `chat_template.jinja`, `tokenizer_config.json`.
- TensorFold v1.1.2: `docs/recipes/adding-a-zig-family.md`, `exl3.md`, `exl3-universal-experts.md`, `cuda.md`, `docs/kernel-contract.md`, and the source files cited inline.
- REFERENCE pages (semantics only): SGLang `python/sglang/srt/models/minimax_m3.py` (raw GitHub), NVIDIA NeMo-Automodel `components/models/minimax_m3_vl/layers.py` (raw GitHub, Apache 2.0) and its docs page, vLLM `vllm.models.minimax_m3.nvidia.model` docs page, InferenceX summary of the M3 config. Both were read through a summarizing fetch tool, not line by line, so quotes above are only as exact as that tool.

## Addendum: weights reach rank 1 over NFS (owner requirement, 2026-10-22)

Rank 1 reads the checkpoint from the head's Hugging Face cache over a read-only NFS mount (`/hf` in its container); the recipe's
`prepare.sh` makes that mount, and no copy lives on the worker. Consequences for the engine (T2/T8/T9): the loader must use
the per-rank byte runs with direct I/O and no mmap (T2 does); a refused O_DIRECT read falls back to cached reads
(`core/direct_io.zig`), and that page cache competes with the KV budget on a unified-memory Spark, so T9 should drop the cache
(posix_fadvise DONTNEED) after each shard run and the budget in T6 should keep a reserve for it. The marker check in
`load.Streamer` does about 22k small random reads; over NFS that should be measured, and batched per shard if slow. The weights
digest in the rank fingerprint must be computed from headers and sizes, not by reading 91 GiB over NFS.
