# Speed techniques to borrow from the GLM-5.3-Flash recipe (decode and prefill) for MiniMax-M3.1-Flash

Status: PLAN ONLY (read-only research, 2026-10-22). No GPU is usable here (driver down), so **nothing below was measured by us**.
Every number is a claim copied from the GLM repo and carries its location; "UNVERIFIED" means I could not confirm it from a
primary measurement record. Anything marked "derived" is my arithmetic from a claim plus the shapes in `engine-port-plan.md`,
not a result.

Sources read:
- GLM repo: `~/newMod/GLM-5.3-Flash-EXL3-TensorFold`, branch `zig-engine` (read with `git show zig-engine:<path>`; nothing checked out, nothing modified).
  Files: `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE`, `docs/zig-engine.md`, `patches/0001..0109` (106 files, three numbers unused; the 96 the
  task mentions are v1.10's), `zig/patches/0001..0007` (the Zig port against TensorFold v1.0.7, `zig/PORT_REF` = `r107-int`).
  I read the Zig port by extracting added lines per file from `zig/patches/0003-zig-glm-kernels.patch` and `0004-zig-glm-engine.patch`
  (one-liner at the end of this file). I read file headers and doc comments, not every body; where I say "not read" I mean it.
- Our tree: `~/TensorFold/tf-wt-int` (branch `mm3-int`, TensorFold v1.1.2 + MiniMax T0-T7 merges): `zig/src/families/minimax_m3/`,
  `zig/kernels/cuda/minimax_m3_{act,index,select,attn}.*`, `zig/tests/minimax_sparse_twin.zig`, `zig/build/minimax_host.zig`, plus TensorFold's own `glm`, `deepseek_v41`, `core/lanes`.
- `docs/engine-port-plan.md` sections 2-6.

## 0. Result in brief

Top 5 picks (ranked by expected value for MiniMax x feasibility without a GPU):

1. **Tensor-core tiles for the prompt-side sparse attention and block scoring** (design of GLM `msa.cu`, patch 0101, and the "one row, many blocks, q loaded once" scoring of 0086). Prefill.
2. **Hide the per-layer collective and weight-load latency in decode**: RoCE one-shot small gathers with GLM's launch policy (0006/0052/0100) plus **L2 prefetch of the next kernels' weights during each gather** (0046). Decode.
3. **Copy-draft policy refinements** (0032 `reply_match`, `miss_most`, padding to captured window widths; 0013 windows up to 16 rows). Decode. This is the only speculation MiniMax has (no drafter in the pack).
4. **Prompt chunk sizing for the grouped experts and overlapped (optionally row-split) exchanges** (0104's measurement, 0033/0010 overlap and split). Prefill.
5. **Make the sparse path cost track the row's real context**: skip index scoring/selection while `nb <= 16`, and never let scoring or selection touch capacity-sized scratch (GLM 0009/0029 `sparse_from`, 0085). Decode and prefill. Bit-safe and fully CPU-testable.

Hard provenance outcome: **no code authored by r0b0tlab was found anywhere in the GLM repo, its branches, or its history; vcruz305 appears
only as a credit for code merged into TensorFold (allowed by the owner's exception)**. Details and the UNCLEAR items are in section 4.

What the GLM repo is, so the transfer question is clear: the `zig-engine` branch is Mia's AI Lab's Zig port of TensorFold's *Python* GLM engine
(`glm5_next`) plus the recipe's patches. Much of its model math is **replayed Triton cubins** (`zig/aot/glm-aot-sm121.tar.xz`, 183 cubins; generated
`cuda_triton_generated.zig`), so the Triton sources exist only in the Python `patches/`, and the Zig side has CUDA C for the EXL3 experts, the 4-bit
matmul, `msa`, `kda`, `hc`, ops. The measured results in README/CHANGELOG are mostly from the **Python engine** on 2-4 Sparks; the Zig engine's
own status (`docs/zig-engine.md`): "the server has not been qualified through these scripts yet", greedy equal to Python on two prompts at TP=2,
"Decode at 8 streams is still under the Python engine". So every GLM gain below is a claim about a different engine and model, and most are
about TP=2/3/4 of a model with hyper-connections, KDA linear attention and DSA pooled-token top-512 sparse attention (see section 5 for what does not transfer).

Important overlap with what we already have: TensorFold v1.1.2 already contains a native `families/glm` (with `Pieces` overlap, `kv8`, RoCE small gathers via
`tp_group.openRoce`, `copy_index.zig`, lanes with `proposer.zig`, `depth.zig`, `recall*`) and `deepseek_v41/{l2pf.zig,block_prefetch.zig}` +
`kernels/cuda/deepseek_v41/{l2pf.cu,l2pace.cu}` (header: "Copy of src/tensorfold/families/glm5_next/spark/l2pf.cu"). For those, port from **our own tree**, and use
the GLM repo for the *policy, sites and measurements*.

## 1. How MiniMax differs (the filter used for "applies?")

From `engine-port-plan.md` and the merged kernels:
- 60 attention layers, no recurrent state, no hyper-connections. Block-sparse attention: 128-key blocks, per (row, kv head) block-max of an fp16-weight
  index score over the whole IK cache, forced own block, top-16 blocks, then GQA over at most 2,048 keys (16 q heads per kv head).
- MoE: 128 experts, top-4, sigmoid router, `swigluoai`, EXL3 `mul1` 3-6 bit; shared expert 5-bit; 3.55 bpw.
- TP2: two fp32-partial reductions per layer (120 a token); each moves R x 6144 fp32 (24 KB at R = 1; about 50 MB at R = 2048 by simple multiplication).
- Plan ceiling (UNVERIFIED estimate in the plan): about 33 ms/token weight streaming per rank at 221 GB/s, so near 30 tok/s before collectives and IK scoring;
  IK scan about 9 ms/token at 128k (also an estimate).
- No MTP or DFlash/dspark drafter in the pack: only prompt-lookup ("copy") proposers apply.
- Merged attention kernels are the *reference-order* versions: `minimax_m3_index.cu` is one CTA per (block, index head, row) with a scalar fp32 dot;
  `minimax_m3_attn.cu` is a sequential fp32 dot over d, 32 keys a step, one CTA per (row, kv head). `minimax_m3_select.cu` already early-returns a dense list when `nb <= 16`.
  None of these has run on a GPU ("Never run on a GPU in this tree's history", `src/cuda/minimax_m3_sparse.zig`).

## 2. Ranked candidates

Legend for "Provenance verdict": OK = no r0b0tlab/vcruz305 involvement found and origin known; BANNED = none found; UNCLEAR = see section 4.
Task IDs are the T-tasks of `engine-port-plan.md` section 4.

### C1. Tensor-core prompt attention and scoring (rank 1, prefill)

(a) What / claim. GLM `msa.cu` (patch 0101, Zig `zig/kernels/cuda/glm_msa.cu`, 310 added lines): one CTA per prompt row, the rank's heads are the MMA M dimension in m16
tiles, so each key tile is loaded and converted to bf16 **once for all heads**; online softmax per m tile; deterministic and chunking-independent.
Claimed (CHANGELOG "Unreleased", README "Prompt and decode switches", measured on one GB10 at a rank's shapes, 2,048-row chunk-layer): 32 heads 9.5 -> 7.1 ms at 32k and
10.1 -> 9.1 at 128k; 22 heads 9.9 -> 6.8 and 9.9 -> 8.4; 16 heads 4.8 -> 3.1 and 5.7 -> 4.6. Also "6.6 -> 4.8 ms" for the default one-pass kernel moved to 4 warps at <= 16 heads.
Not end-to-end: the changelog says whole-prefill effect "+2.6% at four" and `msa.cu` is **off by default because it changes bits** (one of four replies changed).
Scoring side (0086, by BadAd84): the prompt `_scores` took 4 rows x one 64-pool block a program and reloaded each row's 8 KB of queries per block; now one row x 32 blocks a program,
queries and head weights loaded once. Claimed: scoring 36% faster at long context; `_scores` 16.3 -> 103.6 us per prompt token of kernel time from 35k to 226k tokens (nsys, rank 0), cold prefill at 226k 138.3 -> 126.6 s, 35k 17.87 -> 17.47 s, replies byte-identical
(CHANGELOG "Long-context cold prefill"). UNVERIFIED by us; the measured live run is the author's.

(b) Port from. The *design* only; there is no reusable MiniMax shape. Zig CUDA: `git show zig-engine:zig/patches/0003-zig-glm-kernels.patch` -> file `zig/kernels/cuda/glm_msa.cu` (header comment lines 1-40
describe the tiling) and `glm_qmm_frag.cuh` (cp.async/ldmatrix/mma helpers, 84 lines; Ash Hart / TensorFold lineage via TensorFold `qmm.cu`). Scoring: Python Triton `patches/0086-glm-prompt-scores-loop.patch` and
`patches/0009-glm-prefill-kernels.patch` (`_scores`), and `0031-glm-decode-index-regs.patch` (`_scores_rows`: query regs loaded once, blocks g, g+G, ...). Triton bodies are not CUDA; write CUDA from the description.
In our tree start from `kernels/cuda/prefill_attention.cu` (GQA tensor-core tiles) and `flashnext_attn.cu` / `flashnext_qsa_tile.cuh` as `engine-port-plan.md` 3.1/3.3 already say.

(c) Applies to MiniMax? Yes, and better than for GLM: a MiniMax kv group is exactly **16 q heads = one m16 tile**, and the 16 heads of a group share one selected-block list, so a CTA per (row, kv head)
with M = heads maps 1:1. The block-max scoring is the O(T^2) part of prefill (plan section 6 item 9: about 8.4 TFLOP/layer at 128k for 4 heads) and our merged kernel does it as scalar fp32 dots, so a
tile of rows x 128 keys with `mma` bf16 (rows as M, keys as N, 128-dim K) is the obvious fix; the GLM "reload q per block" fix matters little for MiniMax decode (q is 256 B per (row, head) against a 32 KB IK block)
but the mma tiling matters a lot for prefill. Cost: bits differ from the sequential-fp32 reference (GLM kept `msa.cu` off for that reason). Our contract demands row-invariance and window-independence (kept by per-row CTAs and fixed key order), not equality with a scalar chain,
so the right shape is: keep the scalar kernels as the reference twin (they are, in `minimax_sparse_twin.zig`), add the mma kernels behind a flag, compare with a tolerance against the float64 reference, and fold the flag into the arithmetic identity (C9).

(d) Effort / risk. High effort (two new kernels plus tuning for GB10, sm_121), medium risk (numerics vs the reference, selection flips when scores tie within rounding: scores feed top-16, so mma-rounded block maxima can select a different block than the scalar chain on near-ties; this is a model-quality decision, not just a speed one).

(e) Dependency. T0 (reference), T5 (kernels A, B, C), T6 (page layout: IK block contiguous), T10 (contract records). It replaces, not precedes, the T5 reference-order kernels.

(f) Test without a GPU. Extend `zig/tests/minimax_sparse_twin.zig` with a *tile-order twin* (same rounding as mma: bf16 products into fp32 in the m16n8k16 accumulation order) and check it against `core/minimax_ref.zig` with a stated tolerance and against the scalar twin for selection agreement on random data
with planted near-ties (count flips; accept a stated flip rate or require none above a gap). Register launch records in `core/contract/registry.zig` (gate 5 computed without a GPU). `sass_check`-style compile-only check: nvcc `-arch=sm_121` build in `zig build` is possible without a working driver (compiling needs only the toolkit) and catches register/shared-memory budget errors.

Provenance verdict: **OK**. `0101` credits drowzeys (commit `e91312a`, branch `glm53-tp4-2026-10-05`) and BertholomusAI/Albert Lee (commit `8f649f1`) for the design points and says "no code was copied from either"; `0086` is BadAd84's. `glm_qmm_frag.cuh` lineage is TensorFold (Ash Hart + contributors). No banned handle in these files (grepped).

### C2. Hide decode communication latency: RoCE small gathers + L2 prefetch (rank 2, decode)

(a) What / claim.
- RoCE one-shot all-gather (patches 0006, 0052, 0100; Zig `cuda_comm.zig` in `zig/patches/0004`, `roce.zig` in 0001): gathers up to 512 KiB (`TF_ROCE_MAX_KB`) as one-shot RDMA writes (b12x's RoCEnante), larger ones and broadcasts stay on NCCL; the per-gather choice depends only on byte count so both ranks agree; a buffer RoCE cannot take is an error, never a silent switch; ACK timeout 20 (~4 s per try) instead of b12x's 14 because a stall once failed an RDMA write and stopped the server (0100).
  Claims (README "Link" row): "11 us a 16 KiB gather against NCCL's 45, decode +6%"; "`COMM=roce` ... code at 4 streams +1.4%"; four-Spark ring numbers (109 us vs 184 us decode-sized gather) are not relevant to TP2.
- L2 prefetch (patch 0046, Zig `cuda_l2pf.zig` 325 lines, `cuda_l2pf_plan.zig` 135 lines in `zig/patches/0004`; kernel `glm_l2pf` in `glm_ops.cu`, `zig/patches/0003`): a small read-only kernel on a side stream, forked at three sites (a: at the attention block's gather -> FFN glue weights, router, shared expert; f: at the FFN's gather -> next layer's glue weights and input projections; o: after the input projections -> the output projection) pulls the weights the main stream reads next into L2 while it waits. Reads only, so bits are unchanged. Modes: `cp.async.bulk.prefetch.L2` (sm_90+), `prefetch.global.L2::evict_last` per 128-byte line, or `ld.global.cg` touch.
  Claim (README "Decode, memory" row, which attributes the figure to 0046+0047 together, with `TF_ROCE_MAX_KB=512`): one request prose 48.36 -> 49.68 tok/s, code 59.54 -> 61.49 (+2.7% / +3.3%); four at once prose 74.8 -> 76.6, code 100.0 -> 102.7; "two starts each". Small, noisy, and shared with 0047, so the L2PF share is UNVERIFIED.

(b) Port from. RoCE: **our tree already has it** (`zig/src/cuda/roce.zig`, header "after b12x's RoCEnante (local-inference-lab/b12x, Apache-2.0)", `tp_group.openRoce`, `rank_scatter.zig`). From GLM take the policy and robustness bits: 512 KiB threshold, byte-count-only choice, "both ranks must have opened it or neither uses it" handshake, ACK timeout 20 with an env override, and the eager-gather barrier before the engine is built (0052). Files: `git show zig-engine:patches/0100-cuda-roce-ack-timeout.patch`, `patches/0052-cuda-roce-startup.patch`, `zig/patches/0004` -> `cuda_comm.zig` (496 lines).
L2PF: **port from TensorFold's own** `zig/src/families/deepseek_v41/l2pf.zig`, `block_prefetch.zig`, `kernels/cuda/deepseek_v41/l2pf.cu` (`prefetch_segments`, and `prefetch_experts` = the router's picked experts, graph-capturable) and `l2pace.cu`. Use GLM's `cuda_l2pf_plan.zig` as the template for *which tensors at which site* (tables of (address, bytes) pieces cut at 32 KiB, one wave = 192 pieces, range end rounded down to 16 B so nothing reads past an allocation).

(c) Applies? Yes, strongly in structure: MiniMax has 120 gathers a token and 60 layers of small kernels between them, the same pattern L2PF was built for. Sites for MiniMax: after the attention reduction gather -> router weight, shared-expert and (if known) next-layer norms/projections; after the FFN gather -> next layer's q/k/v/index projections and o_proj;
`prefetch_experts` for the four routed experts' trellis words as soon as the router has picked them (they are read by the very next kernels, so the overlap window is only the router-to-GEMV gap; probably small, UNVERIFIED). Derived, not measured: if GLM's 11 vs 45 us per 16 KiB gather carried over to our 24 KB gathers, 120 gathers a token would differ by about 4 ms a token; our TP2 plan assumes `tp_group` RoCE already, so the real work is verifying our fallback and the 512 KiB rule, not porting. L2 capacity matters: GB10's L2 vs the working set of a layer's weights (EXL3 3-4 bit experts for 1536-wide intermediate, about 3.5 MB each per matrix; four experts x three matrices) is UNVERIFIED; prefetching more than L2 holds evicts itself.

(d) Effort / risk. RoCE verification: low. L2PF: medium (site tables for our weight set, side stream capture inside CUDA graphs: GLM and `deepseek_v41` both do it so the pattern exists), low numerical risk (read-only), medium performance risk (can slow if it competes for bandwidth with the main stream on a bandwidth-bound decode: GB10 has one LPDDR5x pool, so prefetch bandwidth is not free; GLM kept it default-on only after measuring +2-3%).

(e) Dependency. T7 (collectives), T8 (graphs, streams), T4/T5 (the weight set per layer). Not before the eager engine is exact.

(f) Test without a GPU. Pure host tests of the site tables: for each layer and site, assert every (address, bytes) piece is 16-byte aligned, lies inside a recorded allocation, pieces <= 32 KiB, sum equals the intended tensors, and that the plan never names a tensor the next kernel does not read (GLM has `plan_test.zig` in `zig/patches/0004` as the pattern; `deepseek_v41/launch_bounds_test.zig` exists in our tree). Mock-RoCE handshake tests with `exl3_mock.zig`-style fakes for the "both open or neither" rule and the byte-threshold rule. Speed: GPU only.

Provenance verdict: RoCE (`b12x`, Apache-2.0, local-inference-lab): **OK**. L2PF: our `deepseek_v41/l2pf.*` copy of TensorFold `glm5_next/spark/l2pf.cu`: **OK** (TensorFold). GLM's own 0046 is "adapted from jayleaton/glm53-tensorfold-spark patch 0460" (Apache-2.0, Jay Leaton): handle is neither banned account, but I could not inspect that repo's contributors: **UNCLEAR for the GLM 0046 text, OK for the TensorFold copy** -> use the TensorFold copy.

### C3. Copy-draft policy refinements (rank 3, decode)

(a) What / claim. GLM's copy ("prompt-lookup") proposer is a TensorFold feature (`glm5_next/cuda/copy_drafts.py`, Ash Hart; wider windows after TensorFold PR #115). Mia's refinements, all of which only change what is *proposed* (verification keeps the serial sample):
- `reply_match` (`TF_GLM_COPY_REPLY_MATCH`): an occurrence that starts inside the reply must match the last `reply_match` tokens, not `match`. Claim (0032 header): over 39 live code replies (2026-09-30) reply-sourced copies kept 23% of their drafts, prompt-sourced 74%.
- `miss_most` (`TF_GLM_COPY_MISS_MAX`): after a copied round whose drafts were not all kept, the next copied round proposes at most N drafts until a copied round keeps all again.
- `pad` (`TF_GLM_WIDE_GRAPHS` widths): a window of 8..W rows is padded to a captured graph width by repeating the last draft so it replays a graph instead of running eagerly.
- Windows of up to 16 rows (`COPY_MAX` 15, 0013).
Claims (README "Decode" and "Drafts" rows): `COPY_MAX` 15 "edit replies 84 -> 119 tok/s"; `COPY`: quote/edit replies 80.3 -> 84.4 tok/s; `COPY_CODE`: code 55.4 -> 56.2, edit 120 -> 125. These are for GLM with DFlash2 behind the copy drafts; the gains come from replies that quote the prompt (edits, repeated tool output, code review). UNVERIFIED by us; the 39-reply sample is the author's.

(b) Port from. Zig: `zig/patches/0004-zig-glm-engine.patch` -> `zig/src/families/glm_cuda/cuda_copy_drafts.zig` (103 lines; `Settings{match 8, most 15, reply_match 16, pad {16}, miss_most 0}` with `validate`) and `cuda_draft_policy.zig` (only for candidate merging, not needed). Python: `patches/0007-glm-copy-drafts.patch`, `0013-glm-decode-rounds.patch`, `0032-glm-code-copy-drafts.patch`. Our integration points: `zig/src/core/copy_index.zig`, `core/lanes/proposer.zig` (`Proposer.vtable.propose/last_match/observe/round`), `core/lanes/depth.zig` (cost-based depth), `core/lanes/context_copy.zig`, `recall*.zig`.

(c) Applies? Yes, and it is the *only* drafting path for MiniMax (plan T8 scope). Two MiniMax-specific facts decide the policy: (1) a verify window of W rows with top-4 of 128 experts touches up to min(128, 4W) distinct experts per layer, so the window's cost grows steeply with W (at W = 16 up to 64 distinct experts, half of all routed weights): a *missed* wide window is expensive, which is exactly what `miss_most` and `reply_match` are for; (2) MiniMax is a thinking/agentic model whose tool-call JSON and repeated file text copy well. Our `depth.zig` is cost-driven from measured round times; MiniMax needs its cost table from the GPU (T8 `tune`), so the thresholds must stay settings with conservative defaults.
`pad`-to-captured-width is a graph-capture detail of GLM's engine (widths captured: 8, then wider); check how our `graph_cache.zig` keys windows before adopting it.

(d) Effort / risk. Low effort (a few settings in the proposer, tests), low risk (proposals only; output exactness untouched). The risk is only wasted verify cost if defaults are wrong.

(e) Dependency. T8 (lanes backend `verify`, `draft`), T3 (tokenizer for corpora). Window width invariance (T8 GPU gate) is what makes drafting exact.

(f) Test without a GPU. Entirely CPU: (1) property tests on the proposer with `core/lanes/fake.zig` (an `accept` rule with a scripted target stream): proposals equal across settings in what is finally emitted; (2) offline "draft simulator" over recorded token traces (GLM patch `0011-glm-draft-sim` is the idea: dump proposals and replay stop rules offline) to count kept-draft rates for `match`/`reply_match` on any transcripts we tokenize with the MiniMax tokenizer (T3). That measures *acceptance* only, never speed: window cost needs the GPU.

Provenance verdict: **OK**. Base proposer: TensorFold (Ash Hart + contributors). The refinements: MiaAI-Lab. No banned handle in 0007/0013/0032 (grepped).

### C4. Prefill chunk sizing for grouped experts, overlapped and optionally row-split exchanges (rank 4, prefill)

(a) What / claim.
- 0104's measurement (CHANGELOG, "Patch 0104"): "on one GB10 the routed experts cost 23-73% more a row in 1,024-row calls than in 2,048-row ones (2,048-row ones 4-31% more than 4,096-row ones), while the 4-bit prompt matmuls cost the same a row from 1,024 to 4,096 rows". This is GLM's expert shapes; a general property of grouped GEMM tile occupancy.
- Overlap (0017, 0022, 0033, `TF_GLM_PREFILL_OVERLAP`, `TF_GLM_OVERLAP_PIECES`): the exchange of a prompt chunk's partials runs on a side stream in row pieces under the next rows' work.
- Row split (0010 `TF_GLM_HC_SPLIT`; Zig `cuda_prefill_split.zig` 221 lines): rank r owns rows [rH, rH+H); at each reduction site it sends the peer its fp32 partial of the peer's rows, receives the peer's partial of its own, sums them rank 0 first (the same bits as the unsplit order), runs the next norm/glue on its own rows only and swaps normed rows (bf16). Claim (README "Prompt, two Sparks"): "50k prefill ~1,270 -> ~1,730 tok/s with 0009 and 0020; decode rounds pay ~1.5%" (the figure is for 0010+0033+0017+0022+0064 *and* the kernels 0009/0020 together; not separable).
- Micro-batches (0104, `TF_GLM_PROMPT_MICROBATCH`): chunk in two halves, each half's exchange under the other's compute; "same bits".
- bf16 partials (0106 `TF_GLM_PROMPT_PARTIALS=bf16`): half the bytes; claim "four Sparks: prefill +9%, all four reference replies changed". Changes bits; for us an opt-in at most.

(b) Port from. Chunk sizing: no code, a setting plus the tile-occupancy planning: `zig/src/families/glm_cuda/cuda_prefill_grid.zig` (187 lines: `chunk_rows 2048`, `busy_rows 1024` while others decode) and `cuda_prompt_micro.zig` (381 lines, pure functions `cut`, `applies`, step orders, CPU-tested with fakes). Overlap: **our tree's** `families/glm/cuda_forward.zig` `Pieces` (second stream and events, `TF_GLM_OVERLAP_PIECES` 3) is the in-tree version; use GLM's Zig `cuda_prefill_split.zig` only for the row-split arithmetic/ordering. Python: `patches/0010-glm-hc-split.patch`, `0033-glm-prefill-overlap2.patch`, `0102`, `0104`, `0106`.

(c) Applies? Partly.
- Chunk size: yes. With 128 experts and top-4, a 2,048-row chunk yields 8,192 (row, expert) pairs, about 64 per expert if balanced; 4,096 rows about 128. `x3gm` tiles and the `grouped.max_experts`/plan limits decide how much a 64-pair expert wastes (plan 3.4 item 4: UNVERIFIED). The prompt scratch grows with rows (plan section 5: the 134 MB block-score chunk is for 8,192 rows at 128k), and the KV budget is only about 9-10 GiB, so the choice trades memory for expert efficiency: make `PREFILL_ROWS` a setting (GLM: `TF_GLM_PREFILL_ROWS`, 4,096 only with micro-batches).
- Exchange overlap: yes, and it is more important for MiniMax than for GLM per row, because MiniMax has no HC mixing to hide behind and does two fp32 reductions a layer: derived by multiplication, one 2,048-row chunk moves 2 x 60 x (2,048 x 6,144 x 4 B) about 6 GB per direction between the Sparks (plan section 5 gives the per-reduction size). Whether that is a few percent or a third of prefill time depends on the RoCE/NCCL bandwidth of the Sparks' link (UNVERIFIED), so measure before building the split.
- Row split (HC-specific as written): the part that transfers is "reduce-scatter instead of all-gather of fp32 partials, norm/residual on own rows, then all-gather the normed bf16 rows"; derived bytes per rank per site: split 2 + 1 = 3 bytes x R x D (fp32 half-rows out + bf16 half-rows out) against the plain 4 x R x D, so about 25% fewer bytes and half the glue work, bits unchanged because the sum order stays rank 0 then 1. It needs a MiniMax-specific rewrite of which kernels run on own rows (the QK-norm/RoPE/index projections are row-independent and could run on own rows too; k/v/IK must then be exchanged: IK, K, V per row = 1,280 B per layer which is small next to 24 KB of hidden).

(d) Effort / risk. Sizing knob: low. Overlap on pieces: medium (in-tree pattern). Row split: high, medium risk (deadlock-prone collective ordering; GLM needed 0040 "parallel deadlocks" and 0064 "split connect early" fixes). bf16 partials: low code, but breaks the bits promise.

(e) Dependency. T7 (comm), T8 (prefill loop), T4 (expert plan), T6 (page layout for exchanged K/V/IK rows).

(f) Test without a GPU. Host: chunk planners as pure functions with the GLM test style (`cuda_prompt_micro.zig` "pure functions and no CUDA ... CPU tests check with fakes"); pairs-per-expert histograms from a synthetic router to compare tile fill at 1,024/2,048/4,096 rows (an *occupancy* number, not a speed); collective schedule simulation (the ring-exchange file `cuda_ring_exchange.zig` in GLM has a byte-level simulation test as a model for proving every byte crosses exactly once); comm deadlock tests with fake ranks using `gather_test.zig`-style fixtures.

Provenance verdict: **OK**. 0010/0033/0017/0022/0104/0106: MiaAI-Lab, designs credited to drowzeys (`compute_prompt_sp`, `SP_SELECT`, `PROMPT_OVERLAP`, branch `glm53-tp4-2026-10-05`, "the code is ours"). No banned handle.

### C5. Cost of the sparse path should track the row's context (rank 5, decode and prefill)

(a) What / claim.
- Dense below the sparse threshold: GLM prompt chunks run *dense* attention where selection would pick everything (`sparse_from = 2051` tokens, `cuda_prefill_grid.zig`; patch 0009 "dense attention only where the sparse pass needs it") and segmented decode windows skip rows that are dense (`SPR` table in `_scores_rows`, patch 0029/0031).
- Bound all index/selection work by the row's visible context, not the scratch's capacity: patch 0043 (decode token scoring bounded to the pools a row can see; TensorFold PR #140 by mikolaj92 did it for the one-program selection) and **0085** (by BadAd84): the split selection's last pass summed a table sized for the whole pool (512 chunks a row at PARALLEL=4) masked to the earlier chunks. Claim (CHANGELOG): selection call at 16 rows 0.96 -> 0.10 ms at 35k, 5.79 -> 0.28 at 226k, 15.68 -> 0.64 at 590k, 27.87 -> 1.08 at 1.04M; "Live on three Sparks, one request: decode at 35k / 226k / 590k 182.6 / 111.2 / 66.2 -> 203.0 / 191.0 / 184.4 tok/s, replies byte-identical". That decode curve is for GLM (DSA, pooled tokens, DFlash2); it shows how badly capacity-sized scratch can dominate long-context decode.

(b) Port from. Idea only (Triton in the Python patches `0009`, `0029`, `0031`, `0085`, `0086`; Zig uses captured cubins). Our code to change: `kernels/cuda/minimax_m3_index.cu` (already returns for `b > p/BLOCK`), `minimax_m3_select.cu` (loops to `nb`, early-returns the dense list when `nb <= 16`), `minimax_m3_attn.cu`, `src/cuda/minimax_m3_sparse.zig` (launch shapes), `families/minimax_m3/attn.zig` when written.

(c) Applies? Yes, cheaply, with two concrete MiniMax items:
1. `nb <= 16` (pos < 2,048): selection is the identity (all blocks), so scoring and select can be skipped entirely and attention fed an iota list (or a dense flag), bit-identical by construction since the select kernel already returns exactly that list. Today `tf_mm3_index` still runs (and `bmax` is written) for such rows. Short chats and the first 2,048 tokens of every prompt benefit; for prompt chunks it removes the quadratic-in-chunk scoring only up to 2,048 tokens, a small slice at long context. Gain is launch count and bandwidth, not a measured figure: UNVERIFIED, and probably small next to C1-C3.
2. Audit: no kernel launch grid, shared array or loop should be sized by pool capacity or `bm_stride`; they must scale with `nb`. In the merged kernels `select` loops to `nb`, `index` returns past `p/BLOCK`; but grid dims (`blockIdx.x` over blocks) are launched for the maximum, so check the host launch uses the window's max `nb`, not capacity. This is the 0085 lesson, and it is where a long-context decode cliff would come from.
Also from 0031 `_scores_rows`: when decoding R rows in a verify window, the grid is (rows, G) with each program looping pool blocks g, g+G, ... and loading the row's query once, with G chosen so enough programs exist (GLM fixed G = 128 in Zig patch `zig/patches/0006` because only that G had a captured cubin, not for speed). For MiniMax decode R = 1..16, nb up to 1,024 at 128k, 4 index heads (2 a rank): grid nb x 2 x R CTAs is plenty parallel, so G is not an issue; keep the one-CTA-per-block design for decode.

(d) Effort / risk. Low effort, low risk, bit-safe.

(e) Dependency. T5 (kernels and `attn.zig`), T0 reference (dense case already in tests), T8 for graph shapes (a branch on `nb <= 16` inside a captured graph needs a device-side flag or two graph variants; the early return in the kernel avoids needing it).

(f) Test without a GPU. Host twins already exist (`minimax_sparse_twin.zig` with mutants): add the "dense skip" path as a twin and assert equality with the full path for pos in {0, 127, 128, 2047, 2048 (nb = 17), 2049}; add a launch-record test that no grid or scratch size depends on capacity (gate-5 launch records in `core/contract/registry.zig`).

Provenance verdict: 0085, 0086: **OK** (BadAd84, credited in GLM `CREDITS.md`). 0043: credits mikolaj92's TensorFold PR #140 for the idea; "this patch's bound ... is our own". 0009/0029/0031: MiaAI-Lab. OK.

### C6. Decode EXL3 kernel loads, launch order, shared-expert side stream (rank 6, decode, GPU-gated)

(a) What / claim.
- `TF_GLM_EXL3_LOADS` (0047, GLM Python `exl3.cu`; Zig `glm_exl3.cu` in `zig/patches/0003`): trellis words as 16-byte `ld.global.nc.L1::no_allocate.v4`, PD k-steps in flight through a 1 KB staging area per warp; same bits. Combined measured gain with L2PF: +2.7% / +3.3% (see C2).
- `TF_GLM_EXL3_DEC_ORDER` (0090, Lukasz Raczylo, PR #95): decode expert kernel block grid mapped so the eight 128-column blocks of one k slice run together and read contiguous memory; same bits. Claim: "one request +2.2% tok/s (4 at once noisy) ... not yet measured here" (author's run).
- `TF_GLM_SIDE` (0105): shared expert on a second stream beside the routed experts in decode windows, fork/join inside the CUDA graph. Claim: one MoE layer 281.0 -> 262.9 us at one row (two Sparks' shapes), 135.0 -> 121.3 (four's), "about even at 4 to 16 rows" (CHANGELOG; graph replays on one GB10). GLM keeps it off at TP=2.
- `0020` prompt expert launch order and `0009` `rot_rows` (rotate each row once when gate and up share `suh`): prompt-side, mcg 4-bit `exl3_prompt` path.

(b) Port from. Python: `patches/0047-glm-exl3-decode-loads.patch`, `0090-glm-expert-launch-order.patch`, `0105-glm-decode-side-stream.patch`; Zig: `glm_exl3.cu` (3,553 lines, header "Port of TensorFold families/glm5_next/cuda/exl3.cu; authored by Ash Hart and TensorFold contributors"), `cuda_side.zig` (113 lines, pure parse + stream/event plan). Our kernels: `kernels/cuda/exl3_experts.cu` (`tf_exl3_group`, `tf_exl3_grouped_{0,1,2}`), `exl3_decode.cuh`, `exl3_linear.cu`. I checked: none of these uses `ld.global.nc`/`L1::no_allocate` today (grep count 0), so the loads variant is open.

(c) Applies? The ideas transfer (same EXL3 trellis tile layout, 16x16 tiles, 32 words), but all three are low single-digit-percent claims of a bandwidth-bound kernel, tuned on GB10 by others for 4-bit `mcg`; for `mul1` 3-6 bit (K2 6..12) the per-tile word counts differ and the decode body is `exl3_decode.cuh`'s mul1 path (dp4a), so the best prefetch distance (PD) and ordering must be re-tuned. Side stream: MiniMax's shared expert is one 5-bit expert of the same shape as a routed one, so by bytes it is about 5/(4 x 3.5) = 0.36 of the routed experts' bytes (derived from 3.55 bpw averages, rough); bandwidth-bound kernels on one LPDDR5x pool overlap poorly, which matches GLM's "about even at 4-16 rows". Rank it after C1-C5 because it cannot be validated or tuned without the GPU.

(d) Effort / risk. Medium effort (kernel variants behind env flags and kernel-contract records), low numerical risk (bits unchanged by construction; prove with the contract gates), moderate chance of zero gain.

(e) Dependency. T4 decode path working and exact first; T8 graph capture for the side stream.

(f) Test without a GPU. Kernel-contract records and row-invariance twins; compile with nvcc for sm_121 to catch register spills; `cuobjdump`/SASS inspection of the load instruction (GLM's `l2pf.cu` copy in our tree mentions `sass_check.sh`). Speed: GPU only.

Provenance verdict: 0090 (Lukasz Raczylo): **OK**. 0105: idea of BertholomusAI (commit `f14e7f7`), "written anew here, no code of theirs": **OK**. **0047: UNCLEAR** (adapted from jayleaton/glm53-tensorfold-spark patch 0580, Apache-2.0; that repo not inspected). Base kernels descend from vcruz305's EXL3 module merged into TensorFold (#42): allowed by the owner's exception only because they are TensorFold history; take the base from our TensorFold tree, never from vcruz305's own repos.

### C7. Sampling of a 200k vocabulary at TP2 (rank 7, decode, check not port)

(a) What / claim. 0034 `TENSORFOLD_NUCLEUS_UNION`: top_p draw from both ranks' candidates together with fewer whole-shard reads. Claim: "default sampling (temperature 1, top_p 0.95) decodes at greedy's pace, not a third of it" (README "Sampling"); CHANGELOG (v1.9): a whole-vocabulary CPU sort per sampled step had made sampled ~15 tok/s against greedy's ~58 and the fix gave 44.1 tok/s greedy vs 43.5 sampled at T = 1, top_p 0.95 (512 tokens, thinking off).
(b) Port from. `docs/zig-engine.md` says the Zig sampler needs none of it ("NUCLEUS_UNION | not needed (the Zig sampler has its own)"). Zig: `cuda_nucleus.zig` (262 lines: fixed-point mass, per-rank candidates, host keyed draw), `cuda_sampler.zig` (470 lines). Ours: `zig/src/cuda/sample.zig` (`margin = 8`, keyed draws, `top_p` in `Rule`).
(c) Applies? MiniMax's `generation_config.json` defaults to T = 1.0, top_p 0.95 (plan T9), so **sampled decoding is the default path** and the head is vocab-split 782/781 units at TP2 (plan section 5). Do not borrow code; verify that our sampler never needs the full 200,064-wide row on the host and that a rank whose nucleus runs past its candidates falls back correctly (GLM's rule: "sends its whole shard instead").
(d) Effort low (review plus tests), risk low. (e) T7, T8, T9. (f) CPU property tests: nucleus from per-rank candidate lists equals the nucleus from the full row for random logits and for planted flat distributions (nucleus wider than the candidates), at vocab 200,064 split 782/781 units; replay GLM's `fixtures_cuda_nucleus.json` idea with our own fixtures.
Provenance: **OK** (TensorFold + MiaAI-Lab).

### C8. Prompt reuse and kept-state policy (rank 8, time to first token on agent workloads)

(a) What / claim. Resume from kept prompt states, shared system prompt (0015, 0071 by ezoushen, PR #44), compact-before-evict (0074 by ezoushen), kept-state byte budget and per-chat quota (0089, Thomas Wade, PR #65), superseded-first eviction (0063, Alexbob0, PR #32), recency for shared cuts (0077, issue #75 by meleesciony), spill tier to local disk (0088, Robert Wojciechowski "wojo", PR #78), take-over decide-then-copy (0078, m-naoki-m). Claims (CHANGELOG): `tools/prompt_reuse.py` 5% -> 99% of a ~33k-token turn resumed, 16.5 s -> 1.0 s to first token (0071, measured by ezoushen on two Sparks); spill: a 20,469-token prompt prefill 10.96 -> 0.21 s restored after a restart; 200k prompt back while 3 streams decode: first token 199-202 s -> 2.2-2.4 s. All GLM measurements.
(b) Port from. The policy lives in shared code: `zig/patches/0001-zig-shared-engine.patch` -> `zig/src/core/prompt_cache.zig` (287 lines; its doc comments credit Thomas Wade, meleesciony, jdecker76, wojo as "the recipe's authors"), `core/lane_host.zig`, `core/lanes/stream.zig`. Python for reference: `patches/0063`, `0071`, `0074`, `0077`, `0088`, `0089`, `0097`.
(c) Applies? Yes and it is easier for MiniMax: no recurrent or conv state, so a kept snapshot is page references (plan T6). It does not change decode or prefill speed per token, but for agent traffic it removes most prefill. The prompt-cache policy is engine-level (`core/prompt_cache.zig` already in TensorFold; GLM's additions are a delta on top of v1.0.7, so diff against our v1.1.2 file before porting, because upstream may already have parts of it: I did not do that diff).
(d) Effort medium; risk medium (pool accounting bugs are silent). (e) T6, T9. (f) Host tests exist as a pattern: `core/kv_pool_store_test.zig`, `kv_pool_host_test.zig`, `tools/pool_pressure.py`-style scenarios as Zig tests (two conversations alternating at a nearly full pool).
Provenance: **OK** for the listed people; their work is Apache-2.0 (patches) or credited as contributed. `0088` carries design credit to gabewillen's idea (no code from an AGPL project), jayleaton's patch 0250 design and jakejharris' JSpark3 conventions; `0087` (dispram) is GPL/AGPL-adjacent (see section 5), skip.

### C9. Arithmetic identity and rank agreement hygiene (rank 9, correctness enabler)

0107: the prompt kernels' choices that set a prompt's bits go into the ranks' startup comparison and into kept states' identity. If C1 (mma attention), any bf16 partial option, or EXL3 load variants are switchable by env, the choice must be part of `families/glm/cuda_ranks.zig`-style `Fingerprint` and the snapshot key, or two ranks (or a resumed prompt) silently disagree. Effort low; (f) host test: flipping any knob changes the fingerprint and invalidates kept states. Provenance: OK (MiaAI-Lab). Also 0109 (prompt stop: client left -> stop at next chunk with a rank vote) is TensorFold upstream (Ash Hart, commit `05dfb7b`) + Mia's multi-rank vote; low effort, OK.

### C10. Indexer keys in FP8 (rank 10, speculative, decode IK scan)

GLM stores the DSA latent cache *and the indexer's pooled keys* as FP8 rows (0038 `KV=fp8`, "the 1M window with 4 requests fits"). For MiniMax the IK scan is the context-growing decode cost (plan 5: about 1.9 GiB a token at 128k, about 9 ms, both estimates). FP8 IK would halve it but perturbs block scores and therefore selection; it also conflicts with the exactness checks in `adding-a-zig-family.md` ("no lossy cache"), so only as an opt-in after a quality study. Port from **our tree**, not GLM: `zig/src/families/glm/cuda_kv8_ref.zig`, `kernels/cuda/kv_fp8.cuh`. Capacity benefit for the 9-10 GiB KV budget is the more likely use. (f) CPU: selection agreement rate between bf16 IK and FP8 IK on random and structured data (a statistic, not a speed). Provenance: TensorFold **OK**; GLM `kv8.py` credit: Ash Hart + contributors.

## 3. Candidate summary table

| # | Technique | Phase | Claimed gain (GLM, source) | Effort | Risk | Needs GPU to tune | Verdict |
|---|---|---|---|---|---|---|---|
| C1 | mma prompt attention + scoring tiles | prefill | msa 9.5->7.1 ms (32 heads, 32k) CHANGELOG; scoring 36% faster, 226k cold prefill 138.3->126.6 s (0086) | High | Medium (bits, selection near-ties) | Yes | OK |
| C2 | RoCE small gathers policy + L2 prefetch | decode | 11 us vs 45 us per 16 KiB gather, +6% (README Link); L2PF+loads +2.7/+3.3% (README) | Low / Medium | Low / Medium | Yes | OK (TensorFold copy) / UNCLEAR (GLM 0046 text) |
| C3 | Copy-draft policy (`reply_match`, `miss_most`, pad, 16-row windows) | decode | edits 84->119 tok/s at COPY_MAX 15; 23% vs 74% kept (0032 header) | Low | Low | Cost side only | OK |
| C4 | Chunk rows for experts; overlap; row split | prefill | experts +23-73% per row at 1,024 vs 2,048 rows (0104); 50k prefill ~1,270->~1,730 (mixed patches) | Low / Medium / High | Low / Medium / Medium | Yes | OK |
| C5 | Dense skip at `nb <= 16`; context-bounded sparse path | both | decode 226k 111->191 tok/s (0085, GLM) | Low | Low | No | OK |
| C6 | EXL3 decode loads, launch order, shared-expert side stream | decode | +2.2% (0090, author), 281->263 us/layer (0105) | Medium | Low | Yes | OK / UNCLEAR (0047) |
| C7 | Candidate-union sampling check | decode | sampled at greedy's pace (README) | Low | Low | No | OK |
| C8 | Kept-state / shared-prefix / spill policy | TTFT | 16.5->1.0 s resume (0071 CHANGELOG) | Medium | Medium | No | OK |
| C9 | Arithmetic identity in rank/snapshot fingerprints | correctness | n/a | Low | Low | No | OK |
| C10 | FP8 indexer keys | decode | n/a for MiniMax | Medium | Medium (quality) | Yes | OK (TensorFold) |

## 4. Provenance audit (hard rule)

Method: (1) `git grep -i -E 'r0b0tlab|vcruz305'` over `zig-engine`, `main` and `tp4-ring` of the GLM repo, and over every `*.patch` blob (zig and python); (2) `git log --all` author/committer/co-author scan and `--grep` on commit messages; (3) per-patch author via `git log --follow`; (4) the same grep over our `tf-wt-int` tree and the sibling local clones in `~/newMod` (glm-5.3 recipe on `tp4-ring`, `tf-upstream`, others); (5) file headers of every file named above.

Findings:
- **r0b0tlab: no hit anywhere in the GLM repo (any branch), the recipe repo `glm-5.3`, or our tree.** Verdict for all candidates: no r0b0tlab code (no exception exists; none needed).
- **vcruz305 in the GLM repo: three Zig file headers in `zig/patches/0002-zig-flashnext-cuda.patch`** ("authored by Ash Hart (ashhart) with commits by vcruz305 inside TensorFold, which the owner allows for this port"). That patch is the Flash Next Zig CUDA engine; **no candidate above uses it** (C1-C10 touch `glm_*`, `core/*`, `deepseek_v41/*`, `families/glm/*` files). If a later candidate needs anything from `0002` (for example Flash Next graph replay or lane code), confirm the specific commits are in TensorFold's own history first.
- **vcruz305 in the recipe repo (`~/newMod/glm-5.3`, branch `tp4-ring`)**: `CREDITS.md:30` and `README.md:814` credit "TensorFold's EXL3 kernels by vcruz305 ... merged upstream by Ash Hart (v0.3.6)". That is the TensorFold-merged code the owner's exception covers. The GLM patches 0001, 0009, 0020, 0047, 0090 and the Zig `glm_exl3.cu` all *modify or port* those merged kernels; fine as long as the base we use is TensorFold's merged version (our `kernels/cuda/exl3_*`), never vcruz305's own repos, gists, forks or PRs elsewhere.
- **vcruz305 in our tree**: `CHANGELOG.md`, `THIRD_PARTY_NOTICES.md`, `README.md`, `RELEASE-NOTES-1.0.0.md`, `docs/credits.md`, `docs/recipes/{exl3,cuda,exl3-universal-experts,qwen3.8-flash-next}.md` credit vcruz305 for the shared EXL3 module (#42), grouped experts (#42), KV caches and `--mtp-confidence` ("Thanks to @vcruz305 (#47), (#69)"). All TensorFold-merged: allowed. **Limit of this audit: our local TensorFold checkouts are squashed (`tensorfold-1.1.2` and `tf-wt-int` base have one upstream commit by ashhart), so `git blame`/`git log --follow` cannot show per-file authors.** Per-file verdicts for TensorFold files therefore rest on file headers and the credits pages, and I mark them OK, not "verified by history". Before importing a specific TensorFold-derived file whose header names no author, check upstream https://github.com/ashhart/TensorFold history.

BANNED: none found.

UNCLEAR (need a from-scratch implementation or the owner's decision before use):
1. GLM `0046-glm-l2-prefetch` and `0047-glm-exl3-decode-loads` text: adapted from `jayleaton/glm53-tensorfold-spark` patches 0460 / 0580 (Apache-2.0, Jay Leaton). The handle is neither banned account, but I could not inspect that repository's contributors or history (not available locally, no network access used), so I cannot clear it for the two banned handles. Mitigation: for L2PF use TensorFold's own `deepseek_v41/l2pf.*` (OK); for the non-coherent trellis loads write the load pattern from scratch from the description (`ld.global.nc.L1::no_allocate.v4`, PD steps in flight), which is a well-known PTX idiom, not a creative expression.
2. Anything from `zig/patches/0002-zig-flashnext-cuda.patch` (see above): allowed in principle, needs per-commit confirmation of TensorFold-history membership.
3. `zig/aot/glm-aot-sm121.tar.xz` (183 captured cubins): binary, no source provenance; not usable for MiniMax anyway (GLM kernels). Do not vendor.
4. GLM patches `0098-0109` are stated as "ported from the GLM-5.3 recipe's patches 0142-0168 (Mia's AI Lab)". I read the recipe repo's credits (48 commits, all one author, one vcruz305 credit as above) but did not diff its patches; the ports themselves in the GLM repo grep clean.
5. External designs referenced without code: drowzeys (`drowzeys/TensorFold`, Apache-2.0) and bertholomus (`bertholomus/TensorFold`, Apache-2.0). GLM states no code copied. We should do the same: take the design only; I did not inspect those forks, so do not copy from them.

Licensing cautions (not provenance, but they gate reuse): the DFlash2 drafter is CC BY-NC-ND 4.0 (not applicable); `0087` dispram uses a GPL-3.0 client and AGPL-3.0 daemon of kindling spark-os (not bundled upstream, skip); `DISPLAY_KV` (0072) changes capacity, not speed.

## 5. Credits to carry into CREDITS.md (people credited in the GLM repo whose work these candidates borrow)

Keep every name already in our CREDITS.md (NVIDIA, turboderp, etc.); add, worded as authorship:

- **Ash Hart (ashhart) and TensorFold contributors** (https://github.com/ashhart/TensorFold): the engine, `glm5_next` Python engine (copy drafts, L2 prefetch kernel, `kv8`, sampling), upstream prompt stop (commit `05dfb7b`), wider copy windows (PR #115).
- **vcruz305** (https://github.com/vcruz305): the shared EXL3 module and grouped experts for TensorFold (#42, shipped 0.3.6), KV caches and `--mtp-confidence` (#47, #69): TensorFold-merged work only, per the owner's 2026-10-07 exception.
- **Jürgen Schmied (jschmied)** (https://github.com/jschmied): the Zig CUDA serving path and CUDA family registry, TensorFold PR #443, commit `59e77e8f4b875ce0e863a8c896fc8e424bc539ac`; "Zig CUDA launch pattern authored by jschmied" is in the GLM kernel headers.
- **turboderp** (exllamav3, MIT): the EXL3 format.
- **local-inference-lab / b12x** (Apache-2.0): RoCEnante one-shot RoCE all-gather.
- **BadAd84** (https://github.com/BadAd84): patches 0085 (split-selection last pass), 0086 (prompt scoring one row x 32 blocks), 0087 (dispram backend; not borrowed).
- **Lukasz Raczylo (lukaszraczylo)** (https://github.com/lukaszraczylo): 0090 expert decode launch order, PR #95.
- **Robert Wojciechowski (wojo)** (https://github.com/wojo): 0088 spill tier, PR #78.
- **Alexbob0** (https://github.com/Alexbob0): 0063 superseded-first eviction (PR #32) and 0057 thinking alias (PR #25).
- **ezoushen** (https://github.com/ezoushen): 0071 shared-prefix copy (PR #44), 0072 display KV (PR #56), 0074 compact-before-evict (PR #62).
- **Thomas Wade (ThomasWadeZ)** (https://github.com/ThomasWadeZ): 0089 kept-state budget and quota (PR #65), 0079, 0080.
- **meleesciony** (issue #75 diagnosis, 0077), **jdecker76** (issue #84 diagnosis, 0097), **m-naoki-m** (0078, also TensorFold PR #421), **desy0305** (0073, PR #51), **johnwhited** (0081-0083, PR #48), **kky42** (0060), **abhicnv007** (0054), **mikolaj92** (TensorFold PR #140, the bound on selection that 0043 follows).
- **drowzeys** (https://github.com/drowzeys/TensorFold, commit `e91312a`; sequence-parallel prompt design `compute_prompt_sp`, `SP_SELECT`, `PROMPT_OVERLAP`) and **BertholomusAI / Albert Lee** (https://github.com/bertholomus/TensorFold, commits `8f649f1` prompt sparse attention, `f14e7f7` decode side stream): design credit only; GLM states no code was copied, and we must not copy from their forks unreviewed.
- **Jay Leaton (jayleaton)** (https://github.com/jayleaton/glm53-tensorfold-spark, Apache-2.0): L2 prefetch (patch 0460) and EXL3 decode loads (patch 0580) that GLM 0046/0047 adapt; credit if and only if we use ideas from those (see UNCLEAR 1).
- **Hector Isaza (hecisaza)**: Zig port fixes `zig/patches/0006`, `0007` and ENGINE=zig commits (not borrowed by any candidate).
- **plotarmordev / netrunner**: appear as release-v1.8 merge committers and as measurer of the `0071` resume figures (`@ezoushen and @plotarmordev`, CHANGELOG) in the GLM history; no authored code found among candidates. The task lists plotarmordev as a credited name; add them for the measurement/release role only if the owner confirms the wording, and do not describe them as authors of code I could not find.
- **MiaAI-Lab (Mia's AI Lab)**: the GLM recipe patches (copy-draft refinements, overlap, split, micro-batch, `msa.cu`, ring and RoCE changes) and the Zig port; "developed with Claude Code" is the GLM repo's own wording.
- Kindling AI / Matt Mastracci (spark-os `dispram`): only if display-reservation KV is ever adopted.

Re-read check for the later editor of CREDITS.md: names above must sit in the section the owner names, worded as "authored", and the words "informed", "inspired", "based on ideas from", "lineage" with no author must not be used as the credit.

## 6. Not applicable to MiniMax (engine-level vs GLM-specific)

GLM-specific, do not port: KDA linear attention and chunked kernels (0012, 0014, 0039, `glm_kda*.cu`); hyper-connection glue (`glm_hc.cu`, 0010 as written); MLA latent / DSA pooled-token top-512 selection (`_select_*`, `_scores` Triton); `glm_qmm*.cu` 4-bit affine dense matmuls (0005, 0016, 0019; MiniMax dense weights are EXL3, not bf16-to-q4); FP8 dense weights (0002); vision (0003, 0050, 0054, 0056, 0079, 0080); DFlash2 drafter and its ring/policy (0021, 0025, 0027; also the CC BY-NC-ND drafter); MTP; `exl3_prompt` mcg-4-bit-only prompt experts (0001, 0020, 0009 `rot_rows` apply to the `mcg` 4-bit path; ours uses `x3gm`/`pfdense`); GLM tool-call parsers and thinking handling (0036, 0051, 0053, 0055, 0060, 0091); TP3/TP4 and ring (0066-0068, 0098-0102); Anthropic API and server fixes (shared server code that TensorFold also ships).

Engine-level and transferable (all in the list above): collective policy (RoCE), L2 prefetch, graph-friendly side streams, copy-draft proposer policy, chunk-size planning for grouped experts, overlap/split exchange schedules, context-bounded sparse kernels, sampling-union, kept-state policy, arithmetic-identity fingerprints.

Deferred, not in v1 scope: multi-stream decode (`PARALLEL` > 1; 0026-0035, 0049, 0069: claim 8 vs 4 requests +27% prose / +32% code on two Sparks, README) and sliced fill with smooth streaming (0061, 0062). Plan T9 serializes `--parallel` at first; revisit after T8.

## 7. Test strategy without a GPU (what can and cannot be shown)

Shown on CPU: algorithm order, equality of alternative schedules (dense skip, split sums in rank order, chunking), table/plan correctness (L2PF sites, collective schedules, kept-state policy), proposer behavior and acceptance statistics on recorded token traces, fingerprint coverage, selection agreement statistics for approximate variants, sampler candidate-union equivalence.
Mechanisms already in the tree: `zig/tests/minimax_sparse_twin.zig` (host twins plus planted mutants; extend with each new kernel variant), `core/minimax_ref.zig` (float64 reference), kernel-contract gates 1-5 (`docs/kernel-contract.md`, launch records without a GPU), `zig build test-minimax-host` (T10), `core/lanes/fake.zig` (fake backend), nvcc compile for sm_121 (toolkit only) for register/shared-memory budgets.
Not shown: any speed, any L2 effect, overlap efficiency, RoCE latency, drafting wall-clock benefit. Do not quote the GLM numbers as expectations; record a GPU measurement plan instead: per candidate, before/after tokens-per-second at fixed prompts (short, 8k, 32k, 128k) with greedy and T = 1 top_p 0.95, one stream first, bits compared (`gate` records) before speed.

## 8. UNVERIFIED list (consolidated)

- Every GLM figure in sections 2-3 (author-run, different model/engine; the Zig engine itself is "not qualified").
- That the README's L2PF+loads figure can be split between 0046 and 0047.
- The claimed 11 us vs 45 us gather applies to our 24 KB gathers; GB10 L2 size vs per-layer working set; link bandwidth for the prefill exchange.
- Whether TensorFold v1.1.2 `core/prompt_cache.zig` and lanes already contain parts of the GLM kept-state deltas (not diffed).
- The bodies of GLM `glm_msa.cu`, `glm_exl3.cu`, `cuda_l2pf.zig` beyond headers/doc comments (not read in full).
- Whether jayleaton's repository involves either banned handle (not inspected).

## Appendix: how I extracted the Zig port files

```sh
R=~/newMod/GLM-5.3-Flash-EXL3-TensorFold
git -C $R show zig-engine:zig/patches/0004-zig-glm-engine.patch > $SCRATCH/0004.patch
awk '/^diff --git/{f=$4} /^\+/ && !/^\+\+\+/ {print f "\t" substr($0,2)}' $SCRATCH/0004.patch | grep '^b/zig/src/families/glm_cuda/cuda_l2pf.zig'
```

Python patches are plain unified diffs: `git -C $R show zig-engine:patches/0046-glm-l2-prefetch.patch`.
