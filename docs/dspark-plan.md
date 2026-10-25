# dspark on MiniMax-M3.1-Flash: design, tensor map, memory, work breakdown

Status: the host half is patch `patches/tasks/0014-minimax-m3-dspark-host.patch` (design, tensor map, float64 reference, twins,
budget). The device half is patch `patches/tasks/0017-minimax-m3-dspark-device.patch` (branch `mm3-dspark-dev`): the tap ops
in the MiniMax forward, the draft's CUDA kernels with contract entries, the draft's device pass (mxfp8 lane matmuls, K/V rings),
the engine, follower and lane plumbing, `--drafter`, the recipe and section 13 below. Written without the model: the kernels
passed the contract gates on the Sparks' GPUs and the plumbing equals plain decode on the host's fakes, but **no draft token has
been produced from the real weights and no acceptance or speed is measured** (section 14 is the first check to run).

Conventions as in `engine-port-plan.md`. **VERIFIED** = read in the pack's files or in TensorFold v1.1.2 source. **UNVERIFIED** =
an assumption. The upstream modeling code for `DSparkMiniMaxDraftModel` was not available (the pack ships `config.json`, the
weights and the model card only; the model card says the draft "was never run here"), so **every semantic that a tensor shape
does not force is a named switch** in `zig/src/families/minimax_m3/dspark_ref.zig` (`Switches`), listed in section 9 with how to
settle it. A wrong switch can only cost acceptance, never correctness: the target verifies every draft.

Provenance: nothing here comes from github.com/r0b0tlab or from vcruz305 outside TensorFold. The code of this patch is new; it
follows the semantics of TensorFold files named in section 4 (credits in section 12).

## 1. What the pack's `dspark/` holds (VERIFIED)

Revision `0fbb6795c5b2fb99d0fb7578cfde2ffae51d9975`, folder `dspark/`: `config.json` (class `DSparkMiniMaxDraftModel`,
`model_type` `minimax_m3_vl`), `model-dspark-00001.safetensors` (2,301,852,674 data bytes, header 14,744 bytes, 97 tensors),
`model.safetensors.index.json`, tokenizer files. The model card: copied "byte-for-byte from the base model, not requantized"; the
base model has no MTP head.

**Correction to the brief: the draft is not an unquantized bf16 model.** "Not requantized" means the pack's author did not touch
it; MiniMax shipped it quantized. Its `quantization_config` is `mxfp8` (`weight_block_size` [1, 32], `activation_scheme`
`dynamic`): the 35 linear matrices are float8 e4m3 codes `weight` [n, k] plus one ue8m0 byte per 32 inputs `weight_scale_inv`
[n, k/32]. Only `fc`, both Markov tables, the confidence projection and the norms are bf16 (the config's `ignored_layers` lists
exactly `fc`, `markov_w1`, `markov_w2` and `confidence_head.proj`). The config also carries leftovers of the base model
(`intermediate_size` 3072, 128 experts, `moe_layer_freq` all 0, NVFP4 keys in `quantization_config`): the shapes below, not those
keys, are the truth (`use_dense_ffn` true, `dspark_ffn_hidden_size` 12288).

Draft shape: 5 layers, hidden 6144, 64 q heads / 4 kv heads, head_dim 128, rotary_dim 64 (theta 1e7, plain RoPE), per-head q/k
norm, `swigluoai` dense MLP 12288 (alpha 1.702, limit 7), every layer `sliding_attention` with window 4096, block size 7, mask
(noise) token 200058, Markov rank 256 head type `vanilla`, confidence head with Markov, taps `[3, 17, 31, 45, 59]` of the 60
target layers, vocabulary 200064, bf16 aux GEMMs, no hybrid context fusion. `dflash_config`, `dspark_config` and the top-level
`dspark_*` keys repeat these values; `dspark_config.parse` requires every copy to agree.

Small reads (data only, parsed with `python3 -I` at fixed offsets, nothing executed): the norm weights, the confidence head, the
256 B q/k norms, and the 98 KB scale tensor of layer 0's `k_proj` plus its first 4 KB of codes. Evidence, not proof:

- ue8m0 scale bytes of `layers.0 k_proj` are 114 to 118 (5 / 35,404 / 62,485 / 408 / 2 of 98,304 each): with bias 127 that is
  2^-13 to 2^-9, so e4m3 codes up to 448 give weights of at most about 0.2. TensorFold's own mxfp8 code (`kernels/cuda/qmmf.cu`
  `e8m0f`: `byte << 23`; `core/exlr8_dense.zig`: `ldexp(1, byte - 127)`) uses the same bias, and `dspark_test.zig` checks the
  reference's decode against that codec for all 256 bytes. No NaN code in the first 4096 code bytes.
- `input_layernorm` / `post_attention_layernorm` weights have means -0.04 to +0.10 and -0.19 to -0.15: they only make sense as the
  Gemma `1 + w` form, as the target's do. `hidden_norm` mean 0.052 (std 0.019) also fits only `1 + w`. `final_norm` mean 0.41 and
  q/k norm means 0.29 to 0.80 fit either form: they stay switches.
- Confidence head: bias +0.0162; hidden part std 0.0029; Markov part std 0.097 (the Markov embedding carries the signal).

## 2. Exact tensor map (VERIFIED from the header)

Prefix `P = language_model.model.dspark.`; layer tensors under `P layers.L.decoder_layer.` for L in 0..4 (names in
`dspark_names.zig`; 97 tensors = 27 bf16 + 35 `.weight` e4m3 + 35 `.weight_scale_inv` u8). No tensor of any other namespace is in
the file; there is **no embedding, no lm_head and no per-layer router**: the draft must reuse the target's `embed_tokens` and
`lm_head` (UNVERIFIED, but nothing else is possible).

| Tensor | dtype | shape | bytes |
| --- | --- | --- | ---: |
| `P fc.weight` | bf16 | [6144, 30720] (5 taps x 6144) | 377,487,360 |
| `P hidden_norm.weight`, `P final_norm.weight` | bf16 | [6144] | 12,288 each |
| `P markov_head.markov_w1.weight`, `.markov_w2.weight` | bf16 | [200064, 256] | 102,432,768 each |
| `P confidence_head.proj.weight` / `.bias` | bf16 | [1, 6400] (6144 + 256) / [1] | 12,800 / 2 |
| `P layers.L.decoder_layer.input_layernorm.weight`, `post_attention_layernorm.weight` | bf16 | [6144] | 12,288 each |
| `... self_attn.q_norm.weight`, `k_norm.weight` | bf16 | [128] (shared by all heads) | 256 each |
| `... self_attn.q_proj.weight` / `_scale_inv` | e4m3 / u8 | [8192, 6144] / [8192, 192] | 50,331,648 / 1,572,864 |
| `... self_attn.k_proj`, `v_proj` | e4m3 / u8 | [512, 6144] / [512, 192] | 3,145,728 / 98,304 each |
| `... self_attn.o_proj` | e4m3 / u8 | [6144, 8192] / [6144, 256] | 50,331,648 / 1,572,864 |
| `... mlp.gate_proj`, `up_proj` | e4m3 / u8 | [12288, 6144] / [12288, 192] | 75,497,472 / 2,359,296 each |
| `... mlp.down_proj` | e4m3 / u8 | [6144, 12288] / [6144, 384] | 75,497,472 / 2,359,296 |

Totals: codes 1,667,235,840 + scales 52,101,120 + fc 377,487,360 + Markov 204,865,536 + 162,818 small = 2,301,852,674
(`dspark_config.zig` reproduces the sum from the config alone and the test pins it to the index's `total_size`).
Not in the file (VERIFIED by absence): `index_*` tensors (the draft has plain sliding-window attention, no block selection),
biases on linears, router or expert tensors, rotary buffers, a `norm` for the context beyond `hidden_norm`.

## 3. The draft in ops (what `dspark_ref.zig` computes)

Notation: `gn(x, w)` = RMS norm, eps 1e-6, times `1 + w` (switch per norm: `gemma_hidden_norm`, `gemma_final_norm`, and the target's
`gemma_layer_norms`, `gemma_qk_norms`). Pending token `t0` at position P (the token the target just chose, not yet in any cache);
context = positions `< P`.

**A. Absorb (once per committed token, after the target's verify keeps rows; or per prompt chunk)**

1. Taps `x_t` for t = 0..4: the target's residual stream after layer `id_t` (`tap_site`, `last_tap_post_norm`, `tap_bf16`).
2. `f = concat(x_0 .. x_4)` [30720] (`concat_order`), bf16.
3. `c = fc f` (bf16 GEMM, replicated), then `ctx = gn(c, hidden_norm)`.
4. For each draft layer l: `k_l = rope(knorm(Wk_l ctx'), p)`, `v_l = Wv_l ctx'` where `ctx' = ctx` (`ctx_input_norm` false) or
   `gn(ctx, input_layernorm_l)`; stored in a per-layer ring slot `p mod R` (K/V of the rank's kv heads).
   The context states never pass through the draft layers (they only supply K/V), which is why a K/V ring can replace a feature
   cache (UNVERIFIED for this model; it is how DFlash2 works in TensorFold: `qwen3_5/cuda_draft.zig` `absorb`, `ring_put`).

**B. Pass (once per round and stream)**

1. Block rows i = 0..6: ids `[t0, 200058 x 6]`, `h_i = embed[id_i]` (target's table), positions `P + i` (`block_positions`).
2. Per layer l (pre-norm): `a = gn(h, input_layernorm)`; `q = Wq a` [64 x 128], `k, v = Wk a, Wv a` [4 x 128]; per-head `gn` with
   `q_norm` / `k_norm` and partial RoPE (first 64 dims, rotate-half, theta 1e7) at the row's position; GQA attention (16 q heads a
   kv head, scale 128^-0.5) of each row over the ring's context keys it can see (`window_rule`) plus the block's own keys
   (`block_attention`: bidirectional); `h += Wo attn`; `m = gn(h, post_attention_layernorm)`; `g, u = Wgate m, Wup m`;
   `act = swigluoai(g, u)`; `h += Wdown act`.
3. `post_i = gn(h_i, final_norm)`.
4. `base_i = lm_head(post_i)` over the whole vocabulary (the target's head).
5. Chain, i = 0..6 (`draft_rows`, `markov_domain`, `first_prev`, `markov_swap`): `prev` = the anchor `t0` for i = 0, else draft i-1;
   `z[id] = base_i[id] + markov_w2[id] . markov_w1[prev]`; draft i = argmax z (ties to the lower id; sampled requests use the
   keyed `lanes.sampling.choose` over the live candidates, as `deepseek_v41/draft/dspark.zig` does).
6. `conf_i = sigmoid(w . [post_i ; markov_w1[prev]] + b)` (`conf_input`, `conf_sigmoid`).
7. The lanes round builds the verify window `[t0, d_1 .. d_k]` (k <= 7, so at most 8 rows, within `window_rows` 16) and cuts k where
   the running product of confidences falls under the depth rule's threshold (`gateLen`, `conf_gate`).

## 4. What follows TensorFold's DSpark and what is new

| Piece | Verdict | Where in TensorFold (all read) | Differences to handle |
| --- | --- | --- | --- |
| External drafter contract: `taps`, `facts`, `open`, `absorb`, `hold`, `held`, `release`; `Features` (bf16 rows, `row_bytes` = taps x hidden) | **reuse verbatim** | `core/lanes/drafter.zig`, `backend.zig`, `drafted.zig`, `native/cuda_drafted.zig` (`F.openDrafter`, `--drafter`, `--drafter-bits`) | the family sets `takes_drafter = true` once wired; `lanes.Model` needs `drafts`, `draft_streams` |
| Copy-draft proposer, verify windows, depth/cost rule, lane policy | **reuse verbatim** | `core/lanes/proposer.zig`, `depth.zig`, `cost_rule.zig`, `windows.zig`; `iface.laneFacts` already exact up to 16 rows | the cost table must include the MoE expert-union growth (section 8) |
| Markov bias, confidence, chain order, candidate lists, block ids `[anchor, noise x n-1]`, row positions `P..P+n-1` | **adapt (re-implement; semantics identical by design)** | `deepseek_v41/draft/dspark.zig` (`Markov.bias`, `confidence`, `markovZ`, `chain`, `candidates`, `blockIds`, `Params`) | MiniMax's confidence has a bias and bf16 weights; tables are named `markov_w1/w2`; candidates gather over 2 ranks of 782/781 head units |
| taps -> `fc` -> `hidden_norm` -> per-layer K/V ring -> block pass | **adapt** | `qwen3_5/cuda_draft*.zig` (absorb, ring), `glm/dflash*.zig`, `glm/cuda_taps.zig` (taps after layers, bf16 slots), GLM's block pass split across ranks | no dynamic-conv kernels; 64/4 GQA; sliding window 4096; swigluoai; mxfp8 not affine-4 |
| mxfp8 linears | **reuse verbatim** | `cuda/mxfp8.zig` (`packCodes`, `packScales`, `weight`), `cuda/qmmf.zig` + `kernels/cuda/qmmf.cu` mode `.mxfp8` (lane rows, 16/32/64-row tiles), `qlinear.zig` `.mxfp8` arm; Flash Next's `cuda_projection.mxfp8` is the loader pattern | every K here is a multiple of 64 (qmmf stage) and every N of a rank a multiple of 64 |
| `fc` bf16 GEMM, 2048-row ingest | reuse | `qmmf` mode `.bf16` or `dense_rows.cu` | a prompt ingests only its last window (section 5) |
| swigluoai, partial-RoPE qk norm | reuse the target's kernels | `minimax_m3_act.cu`, `minimax_m3_rows.cu` (`tf_mm3_*`) | the ring variant of the K/V writer is new |
| Draft block attention over a 4096 K/V ring plus bidirectional block rows | **new** | style of `window_attention.cu`, `dflash block attention` (#746) | GQA 16, window rule, ring slots |
| Tap emission in the MiniMax forward (bf16 copy of `res` after the tapped layers) | **new** | `forward_plan.zig` `norm_add`; `glm/cuda_taps.zig` is the model | section 5 |
| Rank layout of the draft (heads, kv groups, FFN rows cut as the target's; Markov/fc replicated) | **new**, uses `shard.Share` | section 7 | |
| Config, names, catalog/plan, refusals, budget | **new, in this patch** | `config.zig`, `names.zig`, `load.zig`, `budget.zig` are reused as the base | |
| Ring snapshot for radix-cache resume | **new** | | section 10, risk 4 |

## 5. How the five feature streams are produced on both TP ranks: per rank, no collective

After every layer the target's residual `res` is fp32 and **bit-identical on both ranks**: each block's partial sums are exchanged
and added in rank order, and `norm_add` adds the same sum on both (`tp.zig` "rank-order sums are identical on both ranks", plan T7).
The tapped layers 3, 17, 31, 45, 59 therefore need **no all-gather**: each rank copies its own `res` rows to bf16 at the `norm_add`
that opens the next layer (and the final `norm_add` for layer 59), into a `taps_out[row][tap][6144]` bf16 buffer. New op
`tap { slot, rows }` in `forward_plan.zig`; cost: 5 bf16 row writes (61,440 B) per forward row on each rank, nothing on the link.
With `last_tap_post_norm` the tap of layer 59 is the model's final-normed state instead (UNVERIFIED which one the draft was trained
on; HF's last `hidden_states` entry is post-norm).

`taps_out` is transient: a verify window needs 16 rows x 61,440 B = 0.98 MB, a prompt chunk 2048 rows = 126 MB. The draft needs the
**last 4112 positions only** (window 4096 plus the verify slack), so a prompt of N tokens emits and absorbs taps for rows
`N - 4112 ..` only and skips the others (the prefill driver knows N at the first chunk). The ingest then runs per rank
**replicated**: `fc` (377 MB bf16) and `hidden_norm` on all rows, then K/V only for the rank's own two kv heads (k/v rows of 256).
Alternative (not v1): `fc` cut by output columns and the 3072-wide halves all-gathered before `hidden_norm`: saves 189 MB of reads per
ingest (about 0.85 ms at 221 GB/s) for one more exchange.

## 6. Memory budget per rank (the 2-rank TP plan below)

All figures are computed by `dspark_config.zig` and pinned in `dspark_test.zig` (GiB = 2^30).

| Item | Bytes | GiB | Note |
| --- | ---: | ---: | --- |
| Layers: codes 833,617,920 + scales 26,050,560 | 859,668,480 | 0.801 | half of the 35 matrices (heads, kv groups, FFN rows cut) |
| `fc` (replicated) | 377,487,360 | 0.352 | |
| Markov tables (replicated; cut by head units: -102 MB) | 204,865,536 | 0.191 | replicated so both ranks run the chain on the gathered candidates identically |
| norms, confidence | 162,818 | 0.000 | |
| **Weights** | **1,442,184,194** | **1.343** | |
| CUDA module, graphs, NCCL buffers | 536,870,912 | 0.500 | planning figure, **UNVERIFIED**, replace by a measurement |
| Absorb chunk scratch (2048 rows x (61,440 + 24,576)) | 176,160,768 | 0.164 | shrinks with a smaller chunk; only the last window of a prompt is absorbed |
| Per stream: K/V ring (4112 rows x 5,120 B) | 21,053,440 | 0.020 | bounded by the window, does not grow with the context |
| Per stream: pass scratch (7 rows) | 4,135,936 | 0.004 | |
| **One stream, everything** | **2,180,405,250** | **2.031** | `budget.draftBytes`; each extra stream adds 25,189,376 B |

The feature cache the brief names (5 x 6144 bf16 = 61,440 B a token, on top of the target's 76,800 B a token of KV) is **never
kept**: held as a K/V ring it is 10,240 B a token for the whole model, 5,120 B a rank, and only for the last 4112 positions, 21 MB a
stream instead of growing 61 KB a token (a 128k context would otherwise need 7.5 GiB a rank).

Against the ~12 GiB free after weights today: the draft takes 2.03 GiB (17%) of it, leaving about 10 GiB, i.e. **28,160 fewer
context tokens** in the KV pool (the budget test, assuming 119 GiB total and 91 GiB of weights: pool 140,160 -> 112,000 tokens, one
stream). Nothing of this can be tried on the live server: it holds 92 GiB per GPU, so the draft needs a restart with a smaller
`CONTEXT`. `budget.zig` gained `Terms.draft` (`withDraft`, `draftBytes`): weights and runtime are spent before any KV and every lane
adds its ring; a refusal then says "the dspark draft takes N GiB of that budget: serve without --drafter to use it for KV".

## 7. Tensor-parallel plan of the draft (follows `shard.Share`)

| Object | Rank r of 2 | Exchange |
| --- | --- | --- |
| `q_proj` (8192 rows) | q heads [32r, 32r + 32): rows 4096 each | none |
| `k_proj`, `v_proj` (512 rows) | kv heads {2r, 2r+1}: 256 rows; q groups of those heads are on the rank | none |
| K/V ring | the rank's two kv heads only | none |
| `o_proj` (K 8192) | input rows [4096r, +4096) | fp32 partial, rank-order sum (10 a pass: 2 per layer) |
| `gate_proj`, `up_proj` (12288 rows) | 6144 rows each | none |
| `down_proj` (K 12288) | input rows [6144r, +6144) | fp32 partial, rank-order sum |
| `fc`, `hidden_norm`, norms, q/k norms, Markov tables, confidence | replicated | none |
| head logits | the target's vocabulary units (782 / 781) | top-c per rank, gathered (existing logits/pick machinery) |

The twin `blockForwardTp` in `dspark_twin.zig` is this layout executed in float64; the tests require it to equal the unsplit
reference and catch four planted cuts (dropped head, swapped kv group, overlapping FFN rows, a missing rank-1 partial).
The chain, the Markov bias and the confidence run on both ranks from the same gathered candidates and base logits, so their results
are identical without a broadcast. Per pass: 10 exchanges of 7 x 6144 fp32 (172 KB), plus the candidate gather. Latency
estimates (UNVERIFIED): 20-40 us each, so 0.2-0.4 ms; weight reads per rank about 0.86 GB layers + 0.46 GB head slice, about 6 ms
at the 221 GB/s measured for EXL3 (plan section 5), against a one-token target step estimated at 33 ms.

## 8. Does it pay? The MoE window cost (expectation, not measurement)

A verify window of W rows reads the union of the rows' experts. With independent top-4 of 128, expected distinct experts are
`128 (1 - (124/128)^W)`: 4.0, 7.9, 15.3, 28.7, 51.0 for W = 1, 2, 4, 8, 16. With the plan's bytes (8.8 GB non-expert, 5.7 GB expert
a token, both ranks) a W-row window costs 1.00, 1.38, 2.11, 3.43, 5.62 one-token steps (`dspark_config.windowCostRatio`, tested).
A full 7-draft block (W = 8) therefore needs about **3.4 accepted tokens per round plus the draft pass (6 of 33 ms = 0.18)** just to
break even against plain decode; W = 4 breaks even at about 2.3. Real routing of adjacent tokens is correlated (fewer distinct
experts), so this is pessimistic, but the draft was trained against the original model's hidden states and the model card measures
17-23% relative L2 error at three of the five tapped layers in this pack: acceptance may be well under what the break-even needs.
Copy drafts have the same window cost and need no memory; the draft must beat them. **Decision gate before D8**: measure accepted
drafts per round on real agent transcripts (D6) and compare with these ratios.

## 9. UNVERIFIED register (every switch of `dspark_ref.Switches`)

All defaults are the closest reading to TensorFold's existing DSpark/DFlash2. `dspark_model_test.zig` proves each switch is live
(its other setting changes the float64 result), so none is dead code. Impact of a wrong default: acceptance only.

| Switch (default) | Meaning | How to settle |
| --- | --- | --- |
| `tap_site` (layer_output) | tap = residual after layer id, or before it | upstream code; else sweep with recorded taps, pick the higher acceptance |
| `last_tap_post_norm` (false) | layer 59's tap after the final norm | same |
| `tap_bf16` (true), `concat_order` (listed) | taps rounded to bf16; fc input order | same |
| `gemma_hidden_norm`, `gemma_final_norm` (true) | `1 + w` on hidden_norm / final_norm | weights' means support true for hidden_norm only (section 1) |
| `base.gemma_layer_norms` (true), `gemma_qk_norms` (true), `base.rope_layout` (neox_half), swigluoai `gate_clamp` (upper_only) and `up_plus_one` (true), `bf16_*` rounding points | inherited from the target's switches (`minimax_ref.Switches`) | same as the target's open items; settle once for both |
| `ctx_input_norm` (false) | context state passes each layer's input norm before k/v | upstream code / sweep |
| `window_rule` (per_row) | sliding window per row vs the last 4096 context rows for every row | upstream code; changes which ring rows are read |
| `block_attention` (bidirectional), `block_positions` (consecutive) | block rows see each other; positions P+i | upstream code / DFlash paper |
| `draft_rows` (next_token_all_rows) | 7 drafts from rows 0..6 (TensorFold DSpark) or 6 from rows 1..6 (DFlash) | upstream; also fixes the window width (8 vs 7 rows) |
| `markov_domain` (full_vocab), `markov_swap` (false), `first_prev` (anchor) | where the bias applies; which table is indexed by prev; first prev | upstream; `candidates` is TensorFold's approximation for speed |
| `conf_input` (post_final_norm), `conf_sigmoid` (true), `conf_gate` (product) | confidence input, link, gating rule | upstream / calibrate on acceptance |
| `aux_bf16` (true) | head, Markov and confidence GEMMs take bf16 inputs | follows `dspark_aux_gemm_precision` |
| `scale_bias` (127) | ue8m0 exponent bias | same as TensorFold's mxfp8 codec (VERIFIED there); plausible for this pack (section 1) |
| draft uses the target's embedding and `lm_head` | no such tensors in the draft | unavoidable |
| dynamic fp8 *activations* (`activation_scheme: dynamic`) | the base runs W8A8; we run bf16 activations | acceptable, but unmeasured |

## 10. Ordered work breakdown

| # | Task | Gate | Size |
| --- | --- | --- | --- |
| D0 | **This patch**: config, names, plan, float64 reference, twins with mutants, budget, refusals, this document | `test-minimax-dspark` and all host steps | done |
| D1 | Get the upstream `DSparkMiniMaxDraftModel` code (private base repo, token) and settle section 9 | switches become test vectors | S |
| D2 | `tap` op in `forward_plan.zig`/`forward_exec.zig` (bf16 taps buffer), host plan tests | forward step counts unchanged with taps off | S, done in 0017 (every prompt row is tapped and the first 4112 rows are not skipped: see 13) |
| D3 (done in 0017, `dspark_weights.zig`) | Draft weight upload: `cuda.mxfp8.pack*` for the 35 matrices, bf16 `fc`/tables, per-rank slots from `dspark_load.Plan`, direct-I/O stream | plan bytes = device bytes; truncated/misshaped fail closed | S |
| D4 (done, `minimax_m3_dspark.cu`) | Ingest kernels: fc (qmmf bf16), hidden_norm, k/v mxfp8 rows, k-norm + RoPE + ring write | contract gates 1-4 vs `dspark_ref` | M |
| D5 (done) | Block pass kernels: ring-window attention (GQA 16, bidirectional block), qk norm/RoPE, swigluoai reuse, mxfp8 lane rows 7, rank-order exchange | gates vs reference; TP2 equals TP1 | L |
| D6 (done) | Head slice -> top-c candidates -> gather -> Markov chain + confidence kernel (replicated), keyed sampling | chain equals `dspark_ref.chain` (full-vocab ties) | M |
| D7 (done, `dspark_drafter.zig`, `dspark_engine_test.zig`) | `lanes.drafter` adapter (`openDrafter`, `Absorb`, `Hold`, `facts`), follower `Kind.draft`, `Model.drafts` | `core/lanes/fake` tests; drafted replies equal undrafted | M |
| D8 (done) | Memory admission (`budget.withDraft` in `native.admission`), `takes_drafter = true`, fingerprint with draft config + digest, ring snapshots at prompt marks | refusal text tests; resume test | M |
| D9 | GPU measurement: accepted drafts per round on agent transcripts, switch sweep, break-even check (section 8) | keep only if above break-even | M |
| D10 (done) | Recipe: `DRAFTER`/`DRAFTS` in `start.sh`, `config.sh`, README text (section 11), image rebuild | | S |

D1 is the cheapest risk reducer. D2-D8 can be built before the target forward runs on a GPU (every kernel has a float64 reference),
but D9 needs the target's forward on both Sparks (still `ForwardNotBuilt`), and nothing here may run while the live server holds the GPUs.

## 11. Recipe-facing changes (made in 0017: `README.md`, `scripts/config.sh`, `start.sh`, `CREDITS.md`)

- `README.md` line 50 and `scripts/config.sh` lines 106-108 say the pack "has no draft module (`dspark` is not in it)": false for
  revision `0fbb679...`, whose `dspark/` folder holds the draft. Reword: "the pack's `dspark/` draft is not wired into the engine yet".
- Until D8, `--drafter` now refuses with `dspark_config.drafter_refusal` (`native.zig`: `takes_drafter = false`,
  `drafter_refusal`) instead of TensorFold's generic "this CUDA family drafts with its own MTP head", which is wrong for MiniMax.
- Later: pass `--drafter <snapshot>/dspark` to both ranks (rank 1 reads it through the same NFS mount as the weights; the draft is
  2.3 GB read once with direct I/O, 1.34 GiB resident a rank). `DRAFTS=1` alone must keep meaning "copy drafts".
- `dspark_config.admit` refuses: a different hidden width, vocabulary, head counts, rope or eps than the target; a tap past the last
  layer; a block that does not fit one verify window (block + 1 > 16); a world other than 1 or 2; a mask token outside the vocabulary.
  `parse` refuses a different class, hybrid context fusion, another Markov head type, an aux precision other than bf16, a
  non-dense FFN, MTP, a sparse attention config, a non-mxfp8 pack, or settings that disagree between the three copies.

## 12. What this patch contains, how it was checked, credits

Files (`zig/src/families/minimax_m3/`): `dspark_config.zig` (parse, admit, byte arithmetic, cost model), `dspark_names.zig`,
`dspark_load.zig` (rank plan over `load.Catalog`/`Item`/`Slot`, strided runs for row cuts, no Gemma fold), `dspark_ref.zig`,
`dspark_twin.zig` (K/V ring, two-rank block, per-rank candidate gather, each with planted mutants), `dspark_test.zig`,
`dspark_model_test.zig`, `fixtures/dspark_config.json`, `fixtures/dspark_header.json` (the pack's header, data only);
`budget.zig` (`Terms.draft`, `withDraft`, `draftBytes`, `Refusal.draft`), `native.zig` (two decls), `zig/build/minimax_dspark.zig`
and one line in `zig/build/minimax_family.zig` (step `test-minimax-dspark`, also under `test-cuda-host`).

What the tests prove: the real header yields the 97 tensors and bytes of section 2; the rank plans (1 and 2 ranks) read exactly the
slot bytes and two ranks tile every cut tensor exactly once; refusals fire; the float64 forward is deterministic; each of 24 switches
changes it; the ring twin equals recomputing the context from features bit for bit (with `window + 4` and exactly `window` rows)
and four planted ring faults are caught; the two-rank twin equals the reference to 1e-10 and four planted cuts are caught; the
Markov chain follows a forced `prev -> sigma(prev)` table; confidence and gate arithmetic by hand; the e4m3/ue8m0 decode equals
TensorFold's codec for all 256 bytes. What they do not prove: that the semantics match the trained draft.

Credits: the draft's architecture family is z-lab's DFlash (MIT) and the DSpark method is DeepSeek's (MIT); MiniMax trained and
published this draft. TensorFold (Ash Hart and contributors, https://github.com/ashhart/TensorFold) supplies the semantics this
code follows and the code it will reuse: `core/lanes` drafter contract and `Drafted` wrapper, `families/deepseek_v41/draft/dspark.zig`
(Markov head, confidence, chain), `families/qwen3_5/cuda_draft*.zig` and `families/glm/dflash*.zig`/`cuda_taps.zig` (taps, fc, K/V
rings, a block pass split across ranks; the GLM DFlash2 CUDA drafter is credited in TensorFold's CHANGELOG to @BobClawblaw, #979),
`cuda/mxfp8.zig` and `cuda/qmmf.zig` (mxfp8 lane matmul; Flash Next's loader pattern), `core/exlr8_codec.zig` (the e4m3 oracle the
tests compare against). No file of those was copied into this patch; no code from r0b0tlab or vcruz305 was read or used.

## 13. The device half as built (patch 0017)

**Taps.** `forward_plan.zig` gains `Op.tap { slot, rows }`, emitted after the closing `norm_add` of layers 3, 17, 31, 45 and 59
(the residual `res` there is fp32 and bit-identical on both ranks, so no exchange): `minimax_m3_dspark.cu tf_mm3_ds_tap` writes
bf16 rows `[row][tap][6144]` into a buffer of `prefill_rows + slots x window_rows` rows (126 MB + 1 MB a stream). A prompt chunk
lands at rows 0..2047 and is absorbed before the next forward; a verify window lands at its slot's own 16 rows, so another
stream's forward cannot overwrite it before the draft reads it; the prompt's last row is copied there when the pass ends (the first
draft needs it). Every prompt row is tapped and absorbed (the plan's tail rule, skipping all but the last 4112 rows, is not built:
the ring keeps the last 4112 anyway; absorbing the rest costs one fc matmul a chunk). `tap_site` stays `layer_output` and
`last_tap_post_norm` false (section 9).

**Draft device (`dspark_device.zig`, `dspark_weights.zig`).** Per rank: 35 mxfp8 matrices repacked on load for `qmmf` (`mxfp8.packCodes`,
`packScales`; the raw slots are freed), `fc` in `qmmf`'s bf16 order, the rest as stored. A stream's ring (`layers x 2 x 4112 x
(2 kv heads x 128) bf16`, 21 MB) is made for every slot at `attach`. `absorb` = fc, hidden_norm, then per layer the k and v matmuls and
`ringPut` (k norm and rope, slot = position mod 4112); `pass` = embed of `[anchor, mask x 6]` with the target's table, per layer
norm, q/k/v, `blockHeads`, `blockAttn` (the ring window `[max(lo, p + i + 1 - 4096), p)` plus the 7 block rows, bidirectional), o
(fp32 partial, rank-order sum), post norm, gate/up, `swigluoai` (the target's kernel), down (partial, sum); final norm; the target's
head on this rank's vocabulary rows, the best 64 of each row, the candidates gathered, the Markov chain and the confidences
(`chain`, replicated on both ranks, so no broadcast). The ring knows its first valid position: a prompt resumed past row 0 or a
gap makes it start there; a pass over a ring that does not end at the pending position reads it as empty. Draft tokens the chain
cannot find (-1) are filled with the pending token. The confidences are computed and kept but do not shorten a block yet
(`gateLen` is not used: the depth rule and its prior decide, section 9, `conf_gate`).

**Engine and lanes.** `Kind.absorb`, `Kind.draft` and `Kind.snap` cross to the follower (a header of 64 bytes, no ids). The lanes
backend has `prepare_features`, `features` and `prefill_features`: the core's `Drafted` wrapper absorbs each prompt chunk's rows
as they are made, absorbs a verify's kept rows, asks for `depth` drafts and reads them on the host (`held`), and the target
verifies them in its existing windows (copy drafts and head drafts alternate; no tree is built, so `context_copies` is off with a
draft). Rollback is free: only committed rows are ever absorbed. **Prompt cache**: a kept prompt mark saves the stream's ring (a
device copy into one of 8 snapshot buffers a rank, 168 MB) and a resume restores it into the new stream's slot, so the draft sees
the cached prefix; with all 8 in use a snapshot keeps none and the resumed stream drafts from its resume point. `ranks.Fingerprint`
carries a digest of the draft's config and shard headers: the ranks must load the same draft.

**Costs the round loop is told (UNVERIFIED estimates, `dspark_drafter.zig`).** One block pass 6 ms, a one-token step 33 ms, the window
costs of section 8, acceptance by depth 0.70, 0.55, 0.45, 0.38, 0.32, 0.27, 0.23. Replace them with `TF_TIMING` measurements.

**Budget.** `native.draftCost` = `dspark_config.cost` plus the 8 ring snapshots, spent before any KV (`budget.withDraft`).

**`--drafter`.** `takes_drafter` is true. The value `dspark` (also empty, `auto`, `checkpoint`) is the checkpoint's own `dspark/`
folder; a folder with a `config.json`, or a name inside the checkpoint, also works. Hugging Face ids are not resolved. The folder
must be a `DSparkMiniMaxDraftModel` that `dspark_config.admit` accepts beside the model (width, vocabulary, heads, rope, eps, taps,
block + 1 <= 16 rows, one or two ranks); `--no-drafts` leaves it unloaded; `--drafter-bits` is read and ignored (the draft is mxfp8
as shipped). Refusals are worded in `dspark_config.Refusal` and `dspark_folder.words`.

**Found on the way, not fixed (existing forward code).** `exl3.Dense.decode` writes row r of the head's logits at `r x w.n`, the
rank's own width; the forward's `hlog` buffer, `logits_gather` and `local_pick` read row r at `r x slot`, the largest rank's width.
Rank 1's head has 781 of 782 units, so with more than one head row rank 1's logits are mis-strided (rank 0's are right). The
draft's own head pass uses `w.n` as the stride; the forward's verify windows (copy drafts too) need the same attention before
a GPU run: check `tf_mm3_pick` and the gathered logits of rank 1 at 2 and more rows.

## 14. First GPU check (for the session that starts the servers)

1. `MODEL_REVISION=0fbb6795c5b2fb99d0fb7578cfde2ffae51d9975` (the revision that carries `dspark/`), a small `CONTEXT` (for example
   `CONTEXT=24576`): the draft takes about 2 GiB a rank from the KV pool.
2. Plain: `DRAFTS=0 ./start.sh restart`, then `scripts/check-drafted.sh capture /tmp/mm3-plain`.
3. Drafted: `DRAFTS=1 DRAFTER=dspark ./start.sh restart`, then `scripts/check-drafted.sh capture /tmp/mm3-drafted`.
4. `scripts/check-drafted.sh compare /tmp/mm3-plain /tmp/mm3-drafted` must print EQUAL for the seven requests (greedy and keyed
   sampled, a 6k-token prompt, the same prefix again for the ring snapshot's resume). A DIFFERENT is a bug.
5. Speed (section 8): the same request, plain against drafted, tokens a second. When the server ends, rank 0's log prints
   `dspark draft: N passes, M absorbs of R rows; the streams drafted D tokens and the target kept K` (D and K count every kind of
   draft, copies included, so run it once with `DRAFTER` empty and once with `dspark` to read the draft's share).
Run on spark1 (GB10) with synthetic weights: `tf-cuda-test mm3-dspark-pass` (2 layers of the real shapes, one rank, ring wrap, two absorbs) gave ring K/V within 2.4e-3 and the block rows within 8.1e-3 relative L2 of the float64 reference, 5.1 ms for the cold 2-layer pass. Two ranks, the head and the real weights are not covered.
The kernels alone: `tf-cuda-test mm3-dspark-contract` (gates 1-4) and `tf-cuda-test mm3-dspark-pass` (the pass on synthetic weights
against float64, one rank) need no model and no server.
