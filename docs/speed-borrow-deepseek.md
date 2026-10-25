# Borrowing decode and prefill speed from the DeepSeek-V4.1 recipes (MiniMax-M3.1-Flash, TensorFold v1.1.2)

Status: planning only. Nothing here was built, run or measured; the local GPU driver is down. No number below is ours:
every figure is quoted with its source, and anything that cannot be traced to a file is marked UNVERIFIABLE.
Written 2026-10-22. Read-only on all DeepSeek directories.

Conventions: **VERIFIED** = read in a file named here. **CLAIM** = a number or effect stated by a recipe or third party;
the source is given. **UNVERIFIABLE** = the claim's source is not on this machine. **UNVERIFIED** = my inference.
Provenance verdicts: **OK** / **BANNED** / **UNCLEAR** (section 1).

---

## 0. What was actually read (and two corrections to the brief)

1. **The brief's path is an older snapshot.** `~/newMod/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-TensorFold` (not a git
   repo, 2026-10-18) has only `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE`, `docs/zig-engine.md`, `zig/Dockerfile`,
   an **empty** `zig/patches/` (README + `.gitkeep`), `scripts/` (no `memguard.sh`, no `jl-*`) and no `tools/bench.py`. There is
   no `docs/jl-engine.md` there. The files the brief names (`docs/jl-engine.md`, `zig/patches/0001..0006`, `tools/bench.py`,
   `scripts/memguard.sh`, `scripts/jl-env.sh`) are in the **newer git checkout**
   `~/TensorFold/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-TensorFold` (branch `main`, 24 commits, 2026-10-20). I read
   both; "the recipe" below means the newer one unless stated.
2. **Neither recipe contains engine source.** Both are scripts, a Dockerfile, pins and (newer one) six small patches. The
   engines they build are fetched at build time: BertholomusAI's fork `bertholomus/TensorFold` @ `629c524` (newer recipe;
   `3ff3b2a` in the older), and for `ENGINE=jl` Jay Leaton's `jayleaton/deepseek-v41-tensorfold-spark` @ `853310d`. **Neither
   repository is on this machine.** So the x3gm/pfdense/DSpark/L2PF code cannot be "ported from the recipe". What can be
   borrowed from the recipes is (a) settings, tuning methodology and test tools, (b) six patches, (c) pointers to the same
   code where it already sits **inside TensorFold v1.1.2** (`~/TensorFold/tf-wt-int/zig`, `families/deepseek_v41`,
   `kernels/cuda/exl3_block/`, `kernels/cuda/deepseek_v41/`, `src/fabric/tp/`), which is TensorFold-internal and which the
   MiniMax port already reuses (plan section 3.2).
3. Other directories checked: `~/newMod/DeepSeek-v4.1-Flash-1071-TensorFold` (4-Spark TP4, MXFP4/FP8 checkpoint; its
   `docs/speed-work-2026-10-17.md` and `docs/optimization-plan.md` say outright that its speed patches are not merged and
   show intermittent nondeterminism; its patch `0001` is a Python checker) and `~/newMod/DeepSeek-v4.1-Flash-DGX-Sparks`
   (SGLang + Python adapters, 3x/4x Spark, **AGPL-3.0-or-later**, zurih). They add nothing portable to a Zig/CUDA engine
   except the items marked in sections 2.6, 3 and 4.

MiniMax context used throughout (from `docs/engine-port-plan.md` and `tf-wt-int`, branch `mm3-int`, head `b0beded`): merged
so far are T0, T1, T2 (loader, host only), T3, T4 (swigluoai epilogues in `exl3_experts.cu`, `exl3_block/x3gm.cu`,
`exl3_block/x3ld_epi.cu`, `minimax_m3_act.cu`, router plan in `families/minimax_m3/ffn.zig`), T5 (sparse attention kernels
`minimax_m3_index/select/attn.cu`), T6 (KV pool), T7 (TP contract). **T8 (engine, graphs, lanes) is not merged**, so every
speed item below that touches the round loop waits for it.

---

## 1. Provenance: what the DeepSeek directories claim, and verdicts

### 1.1 What the mentions of the banned handles say, exactly

Searched all four DeepSeek directories (and `git log` authors of the newer recipe) for `r0b0tlab`, `vcruz305`, `Leaton`.

| Where | What it says | Meaning |
| --- | --- | --- |
| **r0b0tlab**: all four DeepSeek directories | **no occurrence.** (The only `r0b0tlab` hits under `~/newMod` are in `glm-5.3/work/HANDOFF-cursor.md` and `DeepSeek-v4.1-Flash-1071-TensorFold/docs/optimization-plan.md:181`, both *prohibitions*, not sources.) | nothing from r0b0tlab is claimed anywhere |
| **vcruz305**, older recipe `README.md:24` and `:217`, `CREDITS.md:27` | "(the EXL3 module and the Flash Next CUDA KV work include commits by [vcruz305](https://github.com/vcruz305) merged into TensorFold)" | credit for commits **merged into TensorFold** = inside the owner's exception |
| **vcruz305**, `1071` recipe `README.md:263`, `CREDITS.md:23` | same sentence | same |
| **vcruz305**, newer recipe | **no occurrence at all** in README, CREDITS, NOTICE, CHANGELOG, scripts (dropped when the recipe was rewritten) | the newer recipe does not credit vcruz305, though it builds on the same TensorFold EXL3 module |
| **vcruz305**, TensorFold v1.1.2 itself (`tf-wt-int`: `docs/credits.md:148,161`, `docs/recipes/exl3.md:14`, `exl3-universal-experts.md:5`, `THIRD_PARTY_NOTICES.md:148,174`) | "@vcruz305 wrote the shared EXL3 module in #42 ..." | the code the MiniMax plan reuses (`exl3.zig`, `exl3_experts.cu`, grouped experts) is vcruz305's **inside TensorFold** = allowed |
| **Jay Leaton (jayleaton)**, newer recipe `CREDITS.md:49,69,75-76`, `NOTICE` ("The CUDA DeepSeek-V4.1 Zig engine of ENGINE=jl"), `docs/jl-engine.md`, `jl/Dockerfile`, `scripts/jl-env.sh`, `scripts/jl-prepare.sh`, `scripts/jl-fixup-pack.py` | Authored "the CUDA DeepSeek-V4.1 Zig engine of `ENGINE=jl`" (`engine/zig` of `jayleaton/deepseek-v41-tensorfold-spark` @ `853310d`: decode and prefill kernels, DSpark drafting, split KV, 4K prefill pieces, CED replay, RoCE plan mailbox, sessions, vision path), its serving profile `config/prod-zig-flash17.env.example`, the dense-prefill tuning table `config/pfdense-table.json`, `docker/zig/Dockerfile`, `scripts/run-zig.sh`, `scripts/pack_engram.py`. Apache-2.0 (his `dsv41` family files MIT), "a TensorFold 1.0.2 source export". He is also credited for a JSON-writer fix in BertholomusAI's branch (`0937cc1`) and TensorFold PR #218. | not a banned account. It is a **separate third-party repo**, not on disk; its code is not in TensorFold's history *except* the parts TensorFold merged (next row) |
| Jay Leaton inside TensorFold v1.1.2 (`THIRD_PARTY_NOTICES.md:328-340`) | "The DeepSeek-V4.1 config and checkpoint planning combine @jayleaton's chain (#727) ...; shared Engram host code ... #668 ...; The CUDA sources and loading combine @jayleaton's #671, #674 and #727 contributions." | the `deepseek_v41` family in v1.1.2 already carries his merged work, as TensorFold's own code |
| **BobClawblaw**, newer recipe `CREDITS.md:84` | authored the NCCL cuMem fix (`NCCL_CUMEM_ENABLE=0`, `NCCL_CUMEM_HOST_ENABLE=0`) in `scripts/serve.sh` of Jay Leaton's repo, commit `2cd03c8` | credit needed if that setting is copied |
| Newer recipe git authors | `MiaAI-Lab` (21 of 24 commits), `ezoushen` (2), `sunlinhua-dotcom` (1). No banned account. | |

There is **no** "kit from r0b0tlab" and **no** "file from vcruz305 outside TensorFold" in any DeepSeek directory. The only
`jl` material is Jay Leaton's, who is not banned. Nothing in the DeepSeek directories contradicts the hard rule.

Limits of this check: I verified credits text, file headers, patch `From:` lines and git authors. I could **not** read the
`bertholomus/*` repos, `jayleaton/*`, or TensorFold's commit-level history (`tf-wt-int` has 14 commits by MiaAI-Lab/Claude/ashhart on
top of one squashed import, so `git blame` says nothing about per-file authors). So "no banned author in code I cannot see" is
**UNVERIFIABLE** for those; section 4 turns that into UNCLEAR verdicts where a candidate depends on such code.

### 1.2 Per-source verdicts

| Source | Author(s) (by evidence) | Evidence | Verdict |
| --- | --- | --- | --- |
| TensorFold v1.1.2 in-tree `deepseek_v41` family, `exl3_block/*.cu`, `deepseek_v41/*.cu`, `fabric/tp/*` (x3gm, x3ld, x3seg, pfdense, dense3, lin2, l2pf, planlink, serve_knobs) | ashhart/TensorFold contributors; `THIRD_PARTY_NOTICES.md` names @jayleaton (#668 #671 #674 #727), @akol1 (#824-826); EXL3 base by vcruz305 #42 (allowed), turboderp (MIT) | `THIRD_PARTY_NOTICES.md:328-340`, `docs/credits.md:148`; file headers "Copy of src/tensorfold/families/deepseek_v41/cuda/... (blob ...)"; grep of the tree for `r0b0tlab`: 0 hits | **OK** (inside TensorFold) |
| `dense3.cu` load geometry | header comment: "Lanes load geometry after antirez/ds4 (cuda/mmq/ds4_mmq.cu ...), which measured it on GB10" | `kernels/cuda/exl3_block/dense3.cu` header; no matching line in `THIRD_PARTY_NOTICES.md`/`credits.md` (grep: 0 hits) | **UNCLEAR** (attribution and licence of antirez/ds4 not in TensorFold's notices; not a banned account). Reusing `dense3` inherits that; add antirez/ds4 to MiniMax `CREDITS.md`/`NOTICE` after the licence is checked |
| `l2pace.cu` / b12x RoCEnante | Luke Alonso and b12x contributors (Apache-2.0) | `THIRD_PARTY_NOTICES.md:187-190`, `docs/roce-gather.md` | **OK** |
| Recipe patch `0005-prefill-interleave.patch` | Coinupbtc (`From: Coinupbtc <coinupbtc@users.noreply.github.com>`, PR #5 of the MiaAI-Lab recipe, Apache-2.0) | patch header | **OK** |
| Recipe patches `0001`, `0002`, `0004`, `0006` | `From: TensorFold recipe <noreply@invalid>` = Mia's AI Lab; 0004's call gate is a port of TensorFold `call_gate.py`; 0001's idea credited to drowzeys (no code copied) | patch headers, `CREDITS.md` Scripts section | **OK** (0004/0006 are DeepSeek-lane specific anyway) |
| Recipe scripts (`start.sh`, `nodes.sh`, `config.sh`, `memguard.sh`) and tools (`bench.py`, `smoke.py`, `exact.py`, `prompt_reuse.py`) | Mia's AI Lab; `nodes.sh` rails by webzone, host names by Alexbob0, rendezvous by john (GLM recipe commits); `end_of_turn.py` Alexbob0; `anthropiccheck.py` Eduardo Florencio; `THINKING_BUDGET` (PR #8) and `MIN_AVAIL_WAIT_S` re-read (PR #7) by ezoushen | `CREDITS.md`, `git log` | **OK** |
| BertholomusAI fork (`dsv41` family, `lanes.zig`, patch 0005's base) | BertholomusAI (Albert Lee) plus sayyidfareed, Lucas Peters, lepetitprince716-prog, LeetJoe, Manrique (hashd1ve), Jay Leaton, Nathan McBride | newer `CREDITS.md` Engine section | **OK as credited**, but **not on disk**: only the recipe's description is available. Anything ported from it needs the fork read first (section 4) |
| `ENGINE=jl` source: Jay Leaton's repo (`engine/zig`, `config/*`, `docs/DECODE.md`, `ZIG-SERVE.md`, `ZIG-RESULTS.md`, `patches/`) | Jay Leaton (Apache-2.0; MIT for dsv41 files) | `NOTICE` | **UNCLEAR for code**: not banned, but not on disk and (beyond #668/#671/#674/#727) not in TensorFold's history; his repo credits further parties I cannot see. Use only through section 4's procedure |
| `DeepSeek-v4.1-Flash-DGX-Sparks` (SGLang adapters) | zurih; knapcio (AGPL), rhys101 (MIT), 0xSero (MIT), sparkring (FujitsuPolycom) | its `NOTICE` | **UNCLEAR / do not copy**: AGPL-3.0-or-later and Python/SGLang; env-var names and ideas only |
| Any r0b0tlab code | | none found, none proposed | **BANNED** (n/a) |
| Any vcruz305 code outside TensorFold | | none found, none proposed | **BANNED** (n/a) |

---

## 2. Ranked candidates

Ranking weighs: plausible effect on a bandwidth-bound 2-GPU MoE (decode) or compute-bound prompt pass (prefill), the share that
is already in-tree and exact, risk, and dependency. No ranking is a measured gain.

Summary (details in 2.1-2.10):

| # | Candidate | Helps | Provenance | Effort / risk | Needs |
| --- | --- | --- | --- | --- | --- |
| 1 | In-tree mul1 prompt path: x3gm (+ v2/v3, GU2) and pfdense fused with a **MiniMax-shaped tuning table**, 4K segments opt-in | prefill | OK | M / M | T4 done, T7, T8, GPU to tune |
| 2 | In-tree mul1 decode path: x3seg (one launch for q/k/v), x3ld expert loads, dense3 repack, lin2 (17-32 rows) | decode, verify windows | OK; dense3 UNCLEAR (antirez/ds4) | M-L / M | T4, T2 (repack), T8 |
| 3 | Speculative decoding without a drafter: suffix-lookup and recall copy drafts, depth/cost rule, plus the recipes' lessons | decode | OK (in-tree, not DeepSeek-specific) | S-M / L | T8, T5 |
| 4 | Paced L2 prefetch beside the TP exchanges (`l2pf`, `l2pace`) | decode, some prefill | OK (in-tree) | M / L | T7, T8 |
| 5 | Prompt pass yields to decode (recipe patch 0005, Coinupbtc) | latency fairness under mixed load, not throughput | OK | M / M | T8 (lane host) |
| 6 | Keyed round-graph capture, same settings on both ranks | decode | OK (in-tree common runtime) | M / M | T8 |
| 7 | Round-plan link over RDMA, plan-thread pin and spin (`TF_TP_PLAN_*`) | decode | OK (in-tree) | S if MiniMax used `fabric/tp`, otherwise N/A | T7/T8 design |
| 8 | Session / prefix tiers (RAM + NVMe) | repeated long prompts | OK (in-tree, DeepSeek-named knobs) | M / L | T6, T8 |
| 9 | Methodology and tools: `bench.py`, `exact.py`, `smoke.py`, `prompt_reuse.py`, nsys/ncu recipe, same-clock rule | both, and trust in numbers | OK | S / none | recipe scripts only |
| 10 | `ENGINE=jl` wholesale, `EXPERT_TOPP` pruning, DSpark kernels | - | UNCLEAR / lossy / N/A | - | not recommended |

### 2.1 Candidate 1 - mul1 prompt kernels with a MiniMax tuning table (prefill)

(a) **What**: the DeepSeek family's prompt path for EXL3 mul1: `x3gm.cu` grouped routed-expert GEMM (`gm_k2s` includes K2 6, 8 and
10 = our 3, 4 and 5 bit), `pfdense_k8/k10/k12/k16.cu` fused dense prompt GEMM with a config chosen per (K, N) from a **tuning
table**, 4,096-row prompt pieces (`PF_4K`), a shared prefill workspace (`PF_WS_SHARE`), and gate/up at a 128-member tile for 4K
segments (`GM_GU2` with `GM_V2`).
Claimed gain: the recipe's *measured* prompt speed is a result, not a delta: ~2,140-2,180 tok/s at 8K-64K and 2,000 at 256K on
real prompts, ~1,960-2,000 on random words, DeepSeek on this pair (older and newer recipe `README.md` Status / `CHANGELOG.md`;
sparkDash and `bench.py`; VERIFIED as stated, DeepSeek only). The recipe's own A/B of the two engines found **prefill about
equal** (`docs/jl-engine.md`, "Measured here": 2,007 / 1,957 tok/s at 8K / 32K for `jl` vs ~1,960 / ~2,000 for `zig`), i.e.
**no measured gain from `jl`'s 4K pieces and fused dense GEMM on that checkpoint**; the same doc says the engine was built for
Jay Leaton's own 2.8 bpw pack with expert pruning. Jay Leaton's own claim for 4K pieces and the pfdense table is in his
`docs/ZIG-RESULTS.md` (UNVERIFIABLE, repo not on disk). The 1071 recipe's "32k-token prompt 29.8 -> 11.7 s" is MXFP4 on four
nodes and unmerged (`speed-work-2026-10-17.md`): not applicable.

(b) **Port from**: nothing outside the tree. Already in `tf-wt-int/zig`: `kernels/cuda/exl3_block/{x3gm,x3gm3,x3gm_plan,x3pf,
pfdense_k8,pfdense_k10,pfdense_k12,pfdense_k16}.cu`, `pfdense.cuh`, `pfdense_args.h`, `src/cuda/exl3_block_x3gm.zig`,
`families/deepseek_v41/serve_knobs.zig` (`TF_DSV41_PF_DENSE=fused`, `_CFG`, `_TABLE`, `PF_4K`, `PF_4K_MEMORY_OK`,
`PREFILL_CHUNK/ROWS`, `GM_GU2`, `PF_OVERLAP`), `serve_knobs_test.zig:91` (table format `{"4096,1024": {"cfg": 2, "group": 8}}`),
`forward_prefill.zig` (`PF_WS_SHARE`). Jay Leaton's numeric table `config/pfdense-table.json` is for DeepSeek shapes
(UNVERIFIABLE, not on disk) and must not be reused for MiniMax anyway: write our own from tuning runs.

(c) **MiniMax applicability**:
- T4 already added `act_mode 2` (swigluoai) to `x3gm.cu` and `x3ld_epi.cu` (verified: both include `minimax_m3_act.cuh`), so the
  routed-expert prompt GEMM has the right activation once wired. The shared expert and layers 0-2 are plain dense: they go
  through pfdense plus the pointwise `minimax_m3_act.cu`.
- Shapes to qualify (derived from plan section 5, per rank, TP2; UNVERIFIED until T7/T8 fix the cuts): q 6144x4096 (6-bit, K2
  12), k and v 6144x256 (6-bit), o 4096x6144 (row cut, 6-bit), dense MLP gate/up 6144x6144 and down 6144x6144 (5-bit, K2 10),
  shared expert 6144x1536 and 1536x6144 (5-bit), routed experts K=6144, I=1536 at K2 6 or 8 (layer 17: 8, 8, 6). pfdense has K2
  8/10/12/16, so K2 12 (6-bit) and 10 (5-bit) matter. Plan 3.4 item 1 already flags that x3gm/pfdense shape constraints for
  N=256/512 and TP cuts are UNVERIFIED.
- Differences from DeepSeek: swigluoai instead of silu (done), sigmoid router with bias (outside the GEMM), no Engram/mHC/CED
  replay (so `MHC_*`, `ENGRAM_*`, `CED` knobs are irrelevant), and the 4-index-head / 16-block selection costs O(T^2) scoring per
  prompt layer (plan risk 9) which these kernels do **not** cover: at long contexts the prompt pass may be attention-bound, and
  pfdense/x3gm tuning then helps only the weight-GEMM share.
- 4K pieces: DeepSeek's `PF_4K` needs about +0.66 GiB workspace (warning text in `serve_knobs.zig`) and falls back to 2K on a
  memory shortfall. MiniMax's budget is tight (plan section 5: ~9-10 GiB for KV) and the block-score chunk scales with piece size.
  Default to 2,048-row pieces; make 4K opt-in with its own memory receipt.

(d) **Effort / risk**: M. Wiring is T8 (prefill driver picking x3gm/pfdense by bit width); tuning needs GPU time on both Sparks;
risk is shape-constraint failures for N=256 and TP cuts (a fallback exists: `exl3.Dense` runs prompt rows as 128-row blocks of
the decode kernel, slower but row-invariant) and bit drift between kernels (the `x3gm.cu` header says "each output is one mma
chain over ascending k from +0, any tiling", i.e. tile choice does not change bits - VERIFIED in the header).

(e) **Dependencies**: T4 (done), T5 for the attention share, T7 (cuts fix K/N), T8 (prefill driver). Tuning table: GPU.

(f) **Test without GPU**: (1) host test that enumerates the MiniMax (K, N, K2) list from `config.json` plus the TP cut code and
asks the pfdense/x3gm planners (`pfdense_args.h` config list, `x3gm_plan` tile arithmetic, the Zig wrappers under `src/cuda`)
whether a config exists for each and that no tile reads past a trellis row (integer arithmetic only); (2) a parse test for our
own `pfdense-table.json` through `serve_knobs.parseTable`; (3) `kernel-contract` gate-5 launch records for every new
instantiation (computable without a GPU, `docs/kernel-contract.md`); (4) compile all new instantiations with nvcc for sm_121
(`zig build` already emits fatbins into `.zig-cache`, so compile errors are checkable now); (5) bit-exactness and speed: GPU gate.

### 2.2 Candidate 2 - mul1 decode kernels: x3seg, x3ld, dense3, lin2 (decode)

(a) **What**: Jay Leaton's decode work as merged into TensorFold's dsv41 family: `x3seg` several 1-128-row projections in **one
launch** (segment tables, graph-capture safe), `x3ld` routed-expert GEMV with a faster load path (`TF_DSV41_EXPERT_LOADS`;
prologue, ring, PDL), `dense3` dense EXL3 over a "lanes" 16-byte-coalesced repack of the trellis (`TF_DSV41_DENSE_V3`), `lin2`
dense at 17-32 rows in one weight pass (`TF_DSV41_LIN2`; matters for verify windows), router as a GEMV kernel. Their headers
state "same bits" / row-invariant.
Claimed gain: `docs/jl-engine.md` ("Why a second engine") says his notes list what moved his 1-row window from **35.0 to ~21.8
ms** (router GEMV, mHC boundary in one launch, CSA2 core, dense EXL3 repack, RoCE plan mailbox, priority streams, L2 prefetch,
slot-agnostic row graphs). Source: his `docs/DECODE.md` G13-G17 (**UNVERIFIABLE**; measured on his pack with expert pruning on).
Counter-evidence in the recipe: on Mia's EAQ checkpoint the full `jl` engine decoded **slower** than the BertholomusAI engine
(45 vs 55-61 tok/s single stream, 93 vs 113-117 at 4 streams, random-word prompts, different days; `docs/jl-engine.md` "Measured
here", VERIFIED as stated and flagged there as "not a matched A/B"). Treat 35.0 -> 21.8 ms as an upper-bound hypothesis for
DeepSeek, not an expectation for MiniMax.

(b) **Port from**: in-tree only: `zig/kernels/cuda/exl3_block/{x3seg,x3ld,x3ld_epi,dense3,lin2,linear}.cu`,
`src/cuda/exl3_block_dense.zig`, `exl3_block_lin2.zig`, `families/deepseek_v41/kernels.zig` (variant resolution). The generic
grouped kernel `exl3_experts.cu` (what plan T4 uses for decode) is the baseline.

(c) **Applicability**: highest-value pieces for MiniMax:
- `x3seg`: q, k, v share an input, so one launch instead of three per layer, 60 layers a token. The plan notes q/k/v rotate
  separately unless their `suh` match (UNVERIFIED). The two index projections are fp16, not EXL3, so x3seg does not cover them.
- `x3ld_epi.cu` already carries `act_mode 2` from T4, so `x3ld` is ready for swigluoai at the epilogue level.
- `dense3` needs a load-time **repack** of every dense trellis (q, k, v, o, dense MLP, shared experts, lm_head). That touches T2:
  the loader streams per-rank byte runs by direct I/O from NFS, and a device repack needs scratch (the memory budget is tight)
  and must not double-count unified memory. Plus the UNCLEAR antirez/ds4 attribution.
- `lin2` (17-32 rows) matters only if verify windows reach that width; with copy drafts the window is small, so low priority.
- MiniMax decode is probably weight-bandwidth-bound plus TP exchanges (plan section 5: ~7.3 GB per rank per token, ~33 ms floor
  estimate, UNVERIFIED), so kernels that already reach the floor gain nothing; `docs/recipes/exl3.md` measured 221 GB/s for 6-bit
  EXL3 on GB10, which gives the first-GPU-day yardstick.

(d) **Effort / risk**: M-L. Zig wrappers exist but are wired to dsv41 block descriptors; MiniMax needs its own `ffn.zig`/`attn.zig`
calls. Risk: shape constraints, and a race: the 1071 recipe records that its merged decode kernels gave an intermittent
nondeterminism (two replies for one request, gone with `TF_DS_DECODE_DEC=0`; "most likely a programmatic dependent launch";
`speed-work-2026-10-17.md`, "The four merged"). Require the T10 exactness harness (replay equality, window-width invariance)
before enabling any PDL-using kernel.

(e) **Dependencies**: T4 (done), T2 (repack), T8 (round driver, windows), T10 harness.

(f) **Test without GPU**: host tests of segment-table construction (integers only; `x3gm_plan.cu` header: "integers only, so
exact"), repack index arithmetic round trip on synthetic trellises (T0's `exl3_format.dequantize` accepts any int16 pattern, so
repack then un-repack must be the identity), descriptor tests in `ffn.zig`, compile-only nvcc checks. Bit-exactness against the
baseline kernel: GPU gate (row invariance for windows 1, 2, 3, 16, 17, 64, 128 per `docs/recipes/cuda.md`).

### 2.3 Candidate 3 - speculative decoding when the pack has no drafter

(a) **What**: the DeepSeek recipes run DSpark, DeepSeek's *own* draft blocks shipped in the checkpoint (`DRAFTS=1`, `--no-drafts`,
`tf-dsv41-lanes --drafts`, `TF_DSV41_DRAFT_*`, `ROWS_CAP`, `SPEC_DRAFT`). **None of that applies**: the MiniMax pack has no
`dspark`, MTP or `eh_proj` tensors (plan section 1), so `dspark_*.zig`, `draft/`, the confidence and Markov heads,
`DRAFT_CHAIN`, `DRAFT_OVERLAP` and `CALIB` are out. What transfers:
1. machinery that is **not** DSpark-specific and is already in `core/lanes`: `proposer.zig` `SuffixLookup` (n-gram suffix copy of
   the history; `ngram` 1-8, `min_match`), `context_copy.zig`, `root_copy.zig`, `windows.zig`, `depth.zig`, `cost_rule.zig`,
   `lone_costs.zig`, and `recall*.zig` (`--recall`, `docs/recall.md`: token ids of earlier requests, verified by the target, "the
   same tokens with Recall on, off or read-only"). This is what plan T8 calls "the shared copy-draft proposer";
2. the recipes' *measured lessons*: (i) drafting can be a net loss: the 1071 recipe measured 8.1 tok/s undrafted vs 7.3 drafted
   at one stream, 16% acceptance, traced to drafter loading bugs found only by checking against a reference
   (`optimization-plan.md` section 0; `speed-work-2026-10-17.md`); (ii) random words understate real drafting: 55-61 tok/s on
   random words vs 74.9 (prose) / 109.9 (code) with sparkDash's real prompts (newer `CHANGELOG.md`); (iii) a wider verify round is
   not free (1071: forward 34.2, 41.7, 47.5, 53.0, 58.4, 63.8 ms for 1-6 rows on four MXFP4 nodes, not ours), which
   `cost_rule.zig` prices; (iv) keep the "drafted replies equal undrafted" check (`tools/exact.py`, token hashes);
3. a trained drafter for MiniMax is a separate project: `docs/drafter-release.md` defines a "block-drafter" release format
   (`core/drafter_release.zig`, `lanes/drafter.zig` with `Facts{depth, step_ms, ...}`); the only worked example is Kolibri-1.
   It needs a trained head and the target's tapped rows (`Absorb`, `Features`). No MiniMax gain claim exists anywhere; DSpark's
   DeepSeek numbers (prose about 1.85 tokens a step, code about 4, 1071 `speed-work`) are for DeepSeek's blocks and do not transfer.

(b) **Port from**: in-tree `core/lanes/*` (generic). From the recipe: `tools/bench.py` suites `decode` and `multi` to compare
`--no-drafts` against copy drafts, and the `DRAFTS` plumbing in `start.sh` (MiniMax `config.sh:108` already has `DRAFTS="${DRAFTS:-0}"`).

(c) **Applicability**: the gain depends on workload repetition (tool-call JSON, code edits, agent transcripts). The plan says the
pack is strongest on tool-calling and agentic strata, which repeat a lot - UNVERIFIED hypothesis. Copy drafts add verify rows, so
the verify path (T8 window kernels, MiniMax's block-sparse attention over a window of rows) must be row-invariant (plan T5/T8
already require it).

(d) **Effort / risk**: S-M for copy drafts (flag plumbing plus the T8 vtable `verify`/`draft`), L for a trained drafter. Risk:
wasted verify rows lower throughput when acceptance is low; the depth/cost rule exists to avoid that.

(e) **Dependencies**: T8 (verify windows), T5 (window-width invariance), T10.

(f) **Test without GPU**: the lane engine runs against `core/lanes/fake.zig`; `proposal_external_test.zig`, `recall_*_test.zig`,
`drafted_*_test.zig`, `cuda_cost_rule_test.zig` are host tests already; add a MiniMax fixture that replays an agent transcript
through the fake backend and counts proposals and accepts.

### 2.4 Candidate 4 - paced L2 prefetch beside the TP exchanges

(a) **What**: while a layer waits on its two TP exchanges, a side-stream kernel issues `cp.async.bulk.prefetch.L2` for upcoming
weights, **paced** (`TF_DSV41_L2PF_PACE_GBPS`, `L2PF_MB` budget, `L2PF_CHUNK_KB`) so it does not starve the RoCE traffic; it only
reads weight memory ("writes nothing", `l2pace.cu` header), so exactness is untouched. In BertholomusAI's engine it is the served
setting `TF_DS_L2_PREFETCH=1` / `--prefetch 1` (recipe `config.sh` `PREFETCH`); in Jay Leaton's profile `TF_DSV41_L2PF=1`,
`L2PF_MB=12`, `L2PF_PACE_GBPS=150` (`scripts/jl-env.sh`).
Claimed gain: part of the 35.0 -> 21.8 ms list above (UNVERIFIABLE share). The recipes never ran an isolated A/B (every run has
`PREFETCH=1`); the 1071 plan lists "prefetch contention" as a hypothesis to test with `PREFETCH=0`. **No isolated measurement exists.**

(b) **Port from**: in-tree `families/deepseek_v41/l2pf.zig` (settings, tables filled at boot), `kernels/cuda/deepseek_v41/l2pf.cu`
and `l2pace.cu`.

(c) **Applicability**: `l2pf.zig` builds its tables from DeepSeek's `block.Options` and decode "sites" (all-gathers, after
attention projections, active experts' down weights). For MiniMax the analogous static sites are the next layer's q/k/v/o and
shared-expert tiles during the FFN reduction, and the selected experts' down matrices after the router (known only then, so that
prefetch can overlap only the gate/up GEMV). Per-rank weights are ~91 GiB against a small L2, so the benefit window is the
exchange latency (plan estimate 20-40 us per exchange, UNVERIFIED) times 120 exchanges a token. MiniMax's GLM-style shell uses
NCCL/RoCE small gathers (`tp_group.zig`), not `fabric/tp`, so the pacing must be recalibrated against that transport.

(d) **Effort / risk**: M / L (read-only kernel). Risk: starving the RoCE small gather (why pacing exists), and mismatched
`L2PF_*` values between ranks (`l2pf.zig` refuses values other than the emitter's fixed ones).

(e) **Dependencies**: T8 (a decode graph to hang the side stream on), T7 (transport), GPU time to measure.

(f) **Test without GPU**: table-building arithmetic (sites, budgets, spans) as host tests in the style of `l2pf.zig`'s error set
(`BadL2pf`, `L2pfTable`, `L2pfSpan`); a static check that the kernel only reads. Effect: GPU.

### 2.5 Candidate 5 - prompt pass yields to decode (recipe patch 0005, Coinupbtc)

(a) **What**: between two 2,048-row chunks of a long prompt, rank 0 runs the other live streams' decode rounds for up to N ms
and takes their cancels (`TF_DS_PREFILL_INTERLEAVE_MS`, recipe `PREFILL_INTERLEAVE_MS`, default 250).
Measured on the pair, DeepSeek (newer `CHANGELOG.md`): a 120K-token cold prompt used to leave a decoding stream without a token for
90.1 s; with 250 ms the longest gap was 1.3 s; the prompt itself took 99 s instead of 91 s (about 10 % longer); a cancel was
answered 0.74 s after the client left; `tools/exact.py` 31 of 31 cases identical. VERIFIED as written by the recipe; DeepSeek only.
This is a **latency-fairness** change, not a throughput win (the long prompt gets slower).

(b) **Port from**: `~/TensorFold/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-TensorFold/zig/patches/0005-prefill-interleave.patch`
(Coinupbtc). Six hunks: `core/lane_host.zig`, `core/lanes/engine.zig`, `core/lanes/fake.zig`, `core/lanes/stream.zig` (generic)
and `families/dsv41/lanes.zig`, `families/dsv41/model.zig` (BertholomusAI's fork paths; v1.1.2 calls the family
`families/deepseek_v41`). It is written against the 1.0.2-based fork and **will not apply to v1.1.2 unchanged**. Only the four
`core/` hunks are relevant; the family hunks must be redone for MiniMax's backend (the chunk loop in T8's `prefill.zig`).

(c) **Applicability**: relevant with `--parallel` > 1 and long agentic prompts (MiniMax's ~130K-token ceiling). The Engram
read-ahead join in the patch is DeepSeek-only. First check whether v1.1.2's `core/lanes` already schedules decode between prefill
pieces (it has `prefill_many`, `fair_turns_test.zig`, `pieces.zig`); my grep for "interleave/yield" in `core/lanes` and
`lane_host.zig` found nothing, so probably not - UNVERIFIED.

(d) **Effort / risk**: M / M (touches the shared lane host, so every family is affected and upstream review applies).
(e) **Dependencies**: T8. (f) **Test without GPU**: the patch's own host test with the fake backend is the model: decode rounds
occur between chunks, cancels are taken, token streams are bit-identical with and without interleave.

### 2.6 Candidate 6 - keyed round graphs, and the same settings on both ranks

(a) **What**: CUDA-graph capture of decode rounds keyed by (rows, bucket): `TF_DSV41_ROUND_GRAPH=1`, `GRAPH_FLOOR=hold`,
"slot-agnostic row graphs for multi-stream rounds" (`docs/jl-engine.md`). The recipes' cautionary story: the follower took the
round's settings as flags only, so it ran every round ungraphed until fixed (1071 `speed-work`, 7.3 -> 29.0 tok/s together with
other fixes, four nodes, not ours); the newer recipe's `start.sh` repeats the warning for rank 1's `--graphs/--side/--prefetch/--aio`.
(b) **Port from**: in-tree common runtime ("a keyed graph cache and a device arena", TensorFold `CHANGELOG.md:169`);
`deepseek_v41/graphs.zig` as an example. (c) Plan T8 puts capture after eager exactness. With block-sparse attention the selected
page lists change every token, so the graph must take lists and page tables as device inputs (T5 kernels read lists from buffers;
capture-safety UNVERIFIED). (d) M / M. (e) T8. (f) Host: key/bucket arithmetic, and a rank-fingerprint field for the graph
setting (`ranks.Fingerprint` in T7) so one rank cannot run ungraphed while the other graphs.

### 2.7 Candidate 7 - round-plan link over RDMA, plan-thread pin and spin

(a) **What**: `TF_TP_PLAN_LINK=rdma` (the follower spins `TF_TP_PLAN_SPIN_US`, default 5000 us, before blocking on the plan
socket) and `TF_TP_PLAN_PIN` (`auto` or a CPU number; the plan thread pins itself and avoids the RoCE proxy's CPU)
(`fabric/tp/config.zig:66-67,125`, `planlink.zig`). Jay Leaton's profile has `PLAN_LINK=rdma`, `PLAN_PIN=auto`, `PLAN_TIMEOUT=1800`
(`jl-env.sh`). Claimed gain: part of "the round plan through a RoCE host mailbox" (UNVERIFIABLE share).
(b) **Port from**: in-tree `src/fabric/tp/{planlink,config}.zig`. (c) Only if MiniMax adopts `fabric/tp`'s leader/follower plan
protocol. The T7 rank contract (`docs/rank-contract.md`) follows GLM: both ranks start the same `serve`, `TF_TP_*` are not read,
transport is NCCL plus `TF_ROCE` small gathers. Then there is no plan link to tune, and this is N/A unless T8 builds a follower
loop. (d) S or N/A. (e) T7/T8 decision. (f) Parsing tests already exist in `config.zig`.

### 2.8 Candidate 8 - session / prefix tiers

(a) **What**: `TF_DSV41_SESSIONS=1`, `SESSION_RAM_MIB=256`, `SESSION_DISK=<dir>`, `SESSION_DISK_GIB` (recipe 32; his profile 128),
`SESSION_DISK_MIN=1024`; TensorFold 1.1.1 notes "Paged-KV families can use shared session memory with prefix reuse, a bounded RAM
residency budget and a streamed disk tier on Linux" (`CHANGELOG.md:171`). Gain: skips re-reading long histories. Recipe data for
DeepSeek's *in-engine* prompt cache: `tools/prompt_reuse.py` resumed 99-100% of 159K and 261K-token histories in 1.5-1.9 s a turn
(newer `CHANGELOG.md`). Not a kernel speed-up.
(b) in-tree session code; the knobs are `TF_DSV41_`-named, so MiniMax needs its own names or must opt in (how generic the store
is: UNVERIFIED). (c) MiniMax's KV is 75 KiB/token with no recurrent state (plan section 2), so a kept prefix is just page
references: a good fit; an NVMe tier competes with the NFS/page-cache budget (plan addendum). (d) M / L. (e) T6, T8.
(f) `kv_pool_store_test.zig`-style host tests (T6).

### 2.9 Candidate 9 - methodology, tools and the clock rule

Not engine code; adoptable without a GPU, and they make every number above checkable:
- `tools/bench.py` (suites prefill, decode, multi, needle; `--reps`, `--out`, compare mode), `tools/exact.py` (record and compare
  token hashes across images and settings), `tools/smoke.py`, `tools/prompt_reuse.py`: all Mia's AI Lab. The MiniMax recipe has
  **no `tools/` directory** yet. They hard-code DeepSeek model ids and reply fields, so port them with the model name and
  MiniMax think handling (`<mm:think>`, adaptive thinking) changed.
- Measurement discipline the recipe records: real prompts vs random words for decode; medians of 1-2 repetitions are noise within a
  few percent (`README` Status); compare only at the same GPU clocks on both Sparks (section 3.2).
- The nsys/ncu recipe in `1071/docs/optimization-plan.md` section 0 (nsys launch inside the container, `ncu -k regex`, per-kernel
  sums for one steady round, `--profile 1/2`). The `RANK1_PROFILE` setting of the newer recipe (`start.sh:274`) is the
  recipe-side switch; MiniMax's engine needs its own T8 profile.

### 2.10 Not recommended

- **`ENGINE=jl` wholesale**: not faster on the one checkpoint it was A/B'd on, needs a 95 GB prepared weight image per node and an
  AOT Triton set (about an hour of compiling), and its source is not on disk (UNCLEAR, section 4).
- **`TF_DSV41_EXPERT_TOPP=0.85` (routed-expert pruning at decode)**: Jay Leaton's own measurement calls it lossy (top-1 0.9944 vs
  unpruned, quoted in `docs/jl-engine.md`); the recipe turns it off for exactness. MiniMax's pack is already a lossy quantization
  whose routing agrees with the original model only 50-70% on late layers (plan risk 8): do not add loss.
- **DSpark, Engram (`ENGRAM_AIO`, `HEAD_ENGRAM_DIR`, `WORKER_ENGRAM_DIR`, token map, rank cache, kit), mHC, CED replay,
  `REPLAY_FLOOR`, image prompt cache (0006), call gate (0004)**: DeepSeek-family features with no MiniMax counterpart (enforcing
  `tool_choice` could matter for MiniMax's custom tool markup, but that is T3, not speed). The kit is captured Triton binaries,
  useless here. The indexer split (0001) and second Engram AIO ring (0002) showed no measurable gain or a 1.5-2.6% loss
  (`README` Status), so they are not worth porting even if they applied.
- **The SGLang recipe's adapters** (`spark_prefill_dense.py`, `l2_prefetch.py`, `roce_gather.py`, `verify_cap.py`): Python, AGPL, wrong engine.

---

## 3. Non-engine runtime knobs the recipe scripts should adopt

Source recipe = newer `~/TensorFold/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-TensorFold` unless stated. "MiniMax today" =
`~/TensorFold/MiniMax-M3.1-Flash`. Values are the DeepSeek recipe's defaults, to be re-checked on MiniMax; none is
measured on MiniMax.

### 3.1 NCCL and transport

| Knob | Where in the DeepSeek recipe | MiniMax today | Recommendation |
| --- | --- | --- | --- |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` (both rails), `NCCL_IB_GID_INDEX` derived from the link | `scripts/nodes.sh:262-291` (`nccl_env`, `rank_nccl_env`); rail discovery `nodes.sh:114-250` (`rails`, `rail_info`, `rail_ping`, `probe_rails`: full-MTU unfragmentable ping per rail; a rail that cannot reach the peer's twin is dropped because it ends in `vendor_err 0x81`) | present (`nodes.sh` contains `probe_rails` and `rail_ping`) | keep; check the probe result also feeds the `TF_ROCE_HCA` list |
| two rails = both PCIe twins of the cabled QSFP port (one x4 ~112 Gb/s of 200) | `README.md` Requirements; `nodes.sh:247` comment "a chunk's all-gather is ~1.8x faster on two" (CLAIM from the GLM lineage, UNVERIFIED here) | present | keep; document that both ends need the same MTU and an address on each twin |
| pass-through of any other exported `NCCL_*` (values `[A-Za-z0-9_.,:=/-]` only) for A/B | `nodes.sh:264-296` (comment examples: `NCCL_NET_PLUGIN=none`, `NCCL_MAX_NCHANNELS=4`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_DEBUG_SUBSYS=INIT,NET`) | present (MiniMax `start.sh` header) | keep as the A/B hook; no defaults beyond the three derived ones. The recipe has no measured gain for any of them |
| `NCCL_CUMEM_ENABLE=0`, `NCCL_CUMEM_HOST_ENABLE=0` | `start.sh:223-232` (jl branch; credit BobClawblaw, commit `2cd03c8`): on a GB10 NCCL's cuMem allocations exhausted the driver pool at boot (`NV_ERR_NO_MEMORY`) and failed later starts | the engine is to set them itself with `overwrite = 0` when unset (rank-contract.md; T8/T9 must copy GLM) | set them explicitly in the rank env as belt-and-braces (an exported value wins); credit BobClawblaw if copied |
| `--init` | `start.sh` jl branch RUN_ARGS | MiniMax `start.sh:222` has `--ipc=host`, `--ulimit memlock=-1`, `--ulimit stack=67108864`, `--cap-add IPC_LOCK`, `--device /dev/infiniband`, but not `--init` | add `--init` (reaps helper children; benefit UNVERIFIED, cost nil) |
| RoCE selection variables | `TF_RDMA_DEVICES`, `TF_RDMA_GID_INDEX`, `TF_TP_ROCE_HCA` | MiniMax uses `TF_ROCE_HCA`, `TF_ROCE_GID_INDEX`, `TF_ROCE`, `TF_ROCE_MAX_KB` (rank-contract.md; DeepSeek's names are not read by this family) | do not copy DeepSeek names |
| SGLang recipe's NCCL set: `NCCL_NET=IB`, `NCCL_SHM_DISABLE=1`, `NCCL_BUFFSIZE`, `NCCL_PROTO` (`^LL128` in `.env.example`, `LL,LL128,Simple` in `start.sh`: the two files disagree), `NCCL_MAX_NCHANNELS` (8 vs 32), `NCCL_CROSS_NIC=1`, `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_SUBNET_AWARE_ROUTING=1` | `DeepSeek-v4.1-Flash-DGX-Sparks/.env.example:13-33`, `start.sh:72-89,347-378` (TP3/TP4, SGLang) | not set | a **GPU-day A/B list only**, not defaults. Names are facts, but the file is AGPL: do not copy it |

### 3.2 GPU clocks, CPU, memory

| Knob | Where | MiniMax today | Recommendation |
| --- | --- | --- | --- |
| **Same GPU clock ceiling on both Sparks**, `nvidia-smi -lgc 0,2200`, check with `nvidia-smi -q -d CLOCK` | `README.md` Requirements (~lines 133-135) and `CHANGELOG.md`. CLAIM (rigmark, same prompts): one Spark at 1,800 MHz vs 2,200 on the other cost ~3-6 % of prefill and ~3 % of decode; GPUs peaked at 85/81 C, nothing throttled. Jay Leaton's README suggests a 2,100-2,200 MHz cap (`nvidia-smi -lgc 300,2100`) for temperature (quoted in `docs/jl-engine.md`; UNVERIFIABLE). | not mentioned | document it; add a **read-only preflight** that reads both nodes' clocks (`nvidia-smi` over ssh) and warns on a mismatch. Do not run `-lgc` from `start.sh` (needs root and outlives the container). MiniMax's TP2 meets 120 times a token (plan section 5), so the rule applies at least as much |
| CPU pinning | **none in the recipe scripts** (no `taskset`, `cpuset` or `isolcpus` anywhere; searched). Engine side only: `TF_TP_PLAN_PIN` (`fabric/tp/config.zig`) and `jl-env.sh` `TF_DSV41_PLAN_PIN=auto`; `planlink.zig` says the engine's CPU isolation keeps off the pinned plan thread | none | nothing to copy at recipe level. If T8 adds helper threads (RoCE proxy, direct-IO loader), design pinning in the engine; do not invent `docker --cpuset-cpus` defaults without a measurement |
| `MIN_AVAIL_GIB` + `MIN_AVAIL_WAIT_S` | `config.sh:237-238` (112, 30); `start.sh:177-186`; `nodes.sh:327-340` (re-reads while the page cache is written back; PR #7 by ezoushen: 111.1-111.6 GiB read seconds after a build vs 119 once settled) | present (100, 30) | keep; derive MiniMax's number from ~91 GiB weights per rank plus reserve once measured |
| `MEMGUARD_GIB` / `MEMGUARD_HOLD_S` / `MEMGUARD_HARD_GIB` = 2.5 / 4 / 1 and `scripts/memguard.sh` (0.25 s sampling; kill the container; a dip that recovers is logged) | `config.sh:239-241`, `memguard.sh`, `nodes.sh:343-361` (copies it to the worker). History: floor was 4 GiB; a 120K cold prompt took the head to 3.3 GiB (`CHANGELOG`) | present (same values) | keep. Remember the recipe's finding that nothing holds MemAvailable above the floor at run time (`README` Memory and safety), so MiniMax's KV sizing (`--context`, `--prompt-cache-gib`) is what protects it |
| post-load headroom report, warn within 3 GiB of the floor | `start.sh:343-350` | present (`start.sh:289-298`) | keep |
| `TENSORFOLD_MEMORY_RESERVE_GIB` (12.2 default) "do not lower it" | `README` Memory and safety: a Spark froze once with the reserve at 10 GiB | mentioned in MiniMax `memguard.sh` header | keep the warning, expose no setting |
| head hygiene: desktop and services cost ~6 GiB; `sudo systemctl set-default multi-user.target` | `README` Memory and safety | not mentioned | add to MiniMax README (documentation only) |

### 3.3 Prompt cache, context, reply controls

| Knob | Where | MiniMax today | Recommendation |
| --- | --- | --- | --- |
| prompt cache size (DeepSeek: 512 MiB noted in `README` Memory and safety) | engine default; no recipe setting | `PROMPT_CACHE_GIB` -> `--prompt-cache-gib` (empty = engine default) | keep empty until T6/T8 define it; MiniMax's kept state is page references, so size it in pages |
| `CONTEXT` bounded by the head's memory | `README` Status: a head with ~6 GiB of other software ran out of margin around 464K tokens; "keep CONTEXT at 524288 or less" (DeepSeek) | `CONTEXT` empty | MiniMax's budget is ~130K tokens bf16 (plan section 5): set a conservative default once T6 gives a receipt |
| `PREFILL_INTERLEAVE_MS` (250) | `config.sh:161`, patch 0005 | absent | only with candidate 5 |
| `THINKING_BUDGET` -> `--thinking-budget` (ezoushen, PR #8) | `config.sh`, `start.sh`; README warns a small budget can make a model repeat a tool call | absent | adopt as an optional setting once MiniMax's `capabilities --json` lists the flag (`check_serve_flags` already validates flags); keep `<mm:think>` semantics (T3) |
| `ROUND_ROWS` (16..48), `PARALLEL` 4 ("the family was measured at 4") | `config.sh` | `PARALLEL` empty | measure before choosing a default |
| `RANK1_PROFILE` | `start.sh:274` | absent | define with T8 |
| `API_KEY=1`, `FOREGROUND=1`, `LOG_KEEP` with gz logs in `stop.sh` | | present | no action |
| Engram, rank-cache, kit knobs (`HEAD_ENGRAM_DIR`, `WORKER_ENGRAM_DIR`, `TF_DS_*`) | | n/a | do not copy. One lesson that does transfer: **small random reads over NFS are slow** (4 KiB random read ~940 us over NFS vs ~90 us local; prefill ~690-890 vs ~2,000 tok/s; DeepSeek `README`/`CHANGELOG`). MiniMax loads weights once by direct I/O from NFS (plan addendum), so this bites only if T8 adds runtime table reads from `/hf`; keep none |

---

## 4. BANNED / UNCLEAR candidates (need a from-scratch implementation or the owner's decision)

No candidate is BANNED. These are UNCLEAR and must not be taken as code without a decision:

| Item | Why UNCLEAR | Options |
| --- | --- | --- |
| Anything from `jayleaton/deepseek-v41-tensorfold-spark` that is **not** already merged into TensorFold (`engine/zig` @ `853310d`, `config/pfdense-table.json`, `config/prod-zig-flash17.env.example`, `docs/DECODE.md`, `ZIG-SERVE.md`, `ZIG-RESULTS.md`, `patches/`) | Not on disk, so per-file authorship cannot be inspected; his repo itself credits others (b12x; `scripts/serve.sh` by BobClawblaw). Jay Leaton is not on the banned list and Apache-2.0 permits reuse with credit, but nothing here proves no banned-account code sits inside it. | Owner decision. If wanted: clone it, list `git log --format=%an` per file to be used, grep for `r0b0tlab` and `vcruz305`, and only then plan the port; otherwise implement from scratch from the formulas and from the in-tree TensorFold versions (OK) |
| BertholomusAI fork code (`bertholomus/TensorFold` @ `629c524`, kit `bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10` @ `abe07c7`) | Not on disk; several contributors named. The kit is **captured Triton binaries**, not source, and cannot serve MiniMax at all. | Same procedure; for MiniMax prefer the in-tree v1.1.2 equivalents |
| `dense3.cu` lanes geometry "after antirez/ds4" | Third-party attribution in a comment only; licence unchecked; absent from TensorFold's notices | Check ds4's licence, then credit it; or write the 16-byte coalesced repack from scratch from the `exl3_decode.cuh` layout |
| `ENGINE=jl` image, `jl-prepare.sh`, assets | Depends on the first row | not planned |
| SGLang recipe (`DeepSeek-v4.1-Flash-DGX-Sparks`) code | AGPL-3.0-or-later (zurih), incorporating knapcio (AGPL), rhys101 (MIT), 0xSero (MIT), sparkring | Do not copy. Env-var names and quoted numbers only, as A/B candidates |
| Any measured number from Jay Leaton's docs (`35.0 -> ~21.8 ms`, pruning quality, published tok/s) | Quoted second-hand in `docs/jl-engine.md`; his pack, expert pruning on | Keep marked UNVERIFIABLE; never put in the MiniMax README |

If any of these is later ported, re-check it against the hard provenance rule using the actual file headers and `git log` author
lists, and record it in `patches/README.md` (who wrote it, what was run).

---

## 5. Credits to carry in `CREDITS.md` / `NOTICE` for what is borrowed

Only people whose work is actually borrowed (idea, text, code or setting). Existing credits (NVIDIA, turboderp, TensorFold, Mia's AI
Lab lineage, webzone, Alexbob0, john) stay as they are.

| Who | For what | Link / evidence | Needed when |
| --- | --- | --- | --- |
| Ash Hart (ashhart) and the TensorFold contributors | the engine; every in-tree kernel and loader reused (x3gm, pfdense, x3seg, dense3, lin2, x3ld, l2pf, planlink, lanes, recall) | https://github.com/ashhart/TensorFold | already; keep |
| vcruz305 | the EXL3 module and grouped experts merged into TensorFold (#42), used by every EXL3 path here | TensorFold `docs/credits.md:148`; MiniMax `CREDITS.md` currently has **no** vcruz305 line (grep) | **now**: the port already reuses `exl3.zig` and `exl3_experts.cu`; name them as the author of that merged work |
| Jay Leaton (jayleaton) | DeepSeek-V4.1 CUDA contributions merged into TensorFold (#668, #671, #674, #727; PR #218), the source of the dsv41 kernels in candidates 1, 2, 4, 7; plus his serving-profile values if any is copied (`L2PF_MB=12`, `L2PF_PACE_GBPS=150`, `PLAN_LINK=rdma`) | TensorFold `THIRD_PARTY_NOTICES.md:328-340`; newer recipe `CREDITS.md:69` | when those kernels or settings are used |
| akol1 | Engram host code and DeepSeek admission (#824-826) | `THIRD_PARTY_NOTICES.md:333` | only if Engram code is touched (not planned) |
| Luke Alonso and the b12x contributors | RoCEnante all-gather behind the RoCE small gather | `THIRD_PARTY_NOTICES.md:187-190` | the T7/T8 RoCE path uses it |
| turboderp (exllamav3) | EXL3 format and kernels' origin | plan section 6 item 13 | keep |
| antirez (antirez/ds4) | load geometry of `dense3` | `dense3.cu` header | only if `dense3` is adopted, after a licence check |
| Coinupbtc | `0005-prefill-interleave` (PR #5) | patch `From:` | if candidate 5 is ported |
| ezoushen | `MIN_AVAIL_WAIT_S` re-read (PR #7), `THINKING_BUDGET` (PR #8) | newer `CHANGELOG.md` | MiniMax already has `MIN_AVAIL_WAIT_S`; credit if copied from this recipe, and for `THINKING_BUDGET` |
| BobClawblaw | NCCL cuMem fix | newer `CREDITS.md:84` | if set explicitly in the recipe |
| BertholomusAI (Albert Lee), with sayyidfareed, Lucas Peters, lepetitprince716-prog, LeetJoe, Manrique (hashd1ve), Nathan McBride | the dsv41 family on the fork and its served settings (CUDA graphs, side stream, paced L2 prefetch) | newer `CREDITS.md` Engine section | name BertholomusAI if the served-settings list is adopted; the others only if their code is |
| drowzeys | idea of splitting the prefill indexer across ranks (no code) | recipe patch 0001 header | only if tried (not recommended) |
| Mia's AI Lab | recipe scripts and tools adapted (`bench.py`, `exact.py`, `smoke.py`, `prompt_reuse.py`, `memguard.sh`, `nodes.sh`) | MiniMax `CREDITS.md` already names them | add the tools when ported |
| webzone, Alexbob0, john, Eduardo Florencio, Steve Darlow | script and tool lineage through the GLM recipe | newer `CREDITS.md` | when `end_of_turn.py`, `anthropiccheck.py`, the Hermes page are ported |

Wording rule (user's attribution rule): credits name the person and say what they authored; no "informed by" or "inspired by".

---

## 6. How to test all of this with no GPU (summary)

1. **Host `zig build test` steps** (CPU only): tile-plan and segment-table arithmetic, table parsers (`serve_knobs.parseTable`),
   descriptors in `ffn.zig`, KV/page budgets, rank fingerprints, the lane engine against `core/lanes/fake.zig` (copy drafts,
   recall, interleave, depth/cost rule).
2. **Compile-only CUDA**: nvcc for sm_121 of every new instantiation (fatbins land in `.zig-cache`); proves syntax, register use
   (`-Xptxas -v`) and that no unsupported intrinsic is used, not speed or exactness.
3. **Kernel-contract records** (gate-5 launch records) on the host; planted controls for new kernels.
4. **Float64 reference (T0)** for expected vectors now; the GPU gate compares bits later.
5. **Synthetic checkpoints**: T2's synthetic safetensors tests; a repack round-trip test for dense3; a check that the loader never
   issues small random reads from `/hf`.
6. **Script tests**: `bash -n`, `DRY_RUN=1 ./start.sh` (prints both docker commands), fake `/proc/meminfo` plus stub `docker` tests
   (the DeepSeek recipe's `scripts/test_await_memory.sh` and its memguard tests are the model), and a fake `nvidia-smi` for the
   clock-mismatch preflight.
7. **Not testable without a GPU (so no claim)**: any speed figure, bit-exactness of a new kernel against the baseline, tuning-table
   values, L2-prefetch pacing, graph-capture safety, RoCE behaviour, NCCL variable effects, memory headroom at long context.

## 7. Suggested order once T8 exists

1. Bring up eager, exact decode and prefill with the generic kernels; collect the roofline (candidate 9 tools first).
2. Candidate 1 (prefill table) and 2 (x3seg for q/k/v, then x3ld) one at a time, each with `exact.py` equality and a before/after
   at identical GPU clocks on both Sparks.
3. Candidate 3 (copy drafts, `--recall`) measured on real agent transcripts, not random words.
4. Candidates 6, 4, 5 in that order; 7 and 8 only if the design calls for them.
5. The recipe-level knobs of section 3 (clock preflight, `--init`, explicit cuMem, `THINKING_BUDGET`, tools) can land now with
   no GPU, carrying the credits in section 5.
