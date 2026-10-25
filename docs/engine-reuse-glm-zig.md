# What T8 and T9 can take from the GLM recipe's `zig-engine` branch (and from TensorFold v1.1.2's own `glm` family)

Status: research only, read-only (2026-10-22). No GPU, nothing built or run. The GLM repo
(`~/newMod/GLM-5.3-Flash-EXL3-TensorFold`, branch `zig-engine`) was read with `git show` / `git log` and never checked out.
The one throwaway worktree of v1.1.2 (`~/TensorFold/tf-wt-scratch`) was used for `git apply --check` and is removed.
A second, separate scratch clone of TensorFold v1.0.7 (the branch's real base) lives only in the session scratchpad.

Conventions: **VERIFIED** = read in the files named or produced by a command named here. **UNVERIFIED** = not read, inferred, or an estimate.

## 0. Result in six lines

1. The branch does **not** build on v1.1.2. Its `TF_REF` is **TensorFold v1.0.7** (`e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`, `scripts/config.sh` `ZIG_TF_REF`; `zig/Dockerfile` fetches that commit and `git apply`s `zig/patches/*.patch` in filename order). All seven patches apply cleanly in order to a fresh v1.0.7 tree (VERIFIED, scratch clone). Against v1.1.2: 0001 fails on 75 files, 0002 on 5, 0003 on 4, 0005 on 2, 0006 and 0007 need files of 0004 and 0002; only 0004 applies, and it would not compile there (section 1).
2. The branch's GLM engine (`families/glm_cuda`, 81 files, 23,368 lines) is a **Triton-cubin-replay port** of the Python engine, built on the Flash Next Zig port (patch 0002) and its TCP "mirror". v1.1.2 meanwhile carries a **different, native** GLM CUDA engine (`families/glm/cuda_*.zig`, NVFP4 and EXL3, NCCL control link, kernels of its own) that upstream merged after v1.0.7. For T8/T9 the right skeleton is **v1.1.2's `glm` family**, with Nemotron for pooled KV pages, Qwen3.5's HIP backend for the backend-owned radix prompt cache, and Qwen3.8-27B's CUDA `laneFacts` for copy-draft-only windows. The branch adds ideas and a few generic fixes, no code we can apply.
3. Of the branch's fixes only two are generic engine logic MiniMax needs in some form: the **required-call restart** (call gate; v1.1.2 has the `CallGate` type and `Info.call_gates` but no implementation, and takes the grammar route instead) and the **think/tool wiring** idea. The **TP link deadlock for long prompts** (patch 0007) is a bug of the branch's TCP mirror (`flashnext/cuda_mirror.zig`); v1.1.2's GLM control link is `ncclBroadcast`, which cannot block this way, so MiniMax inherits no such bug if it copies v1.1.2's `follow.zig` (section 3).
4. Everything else (TF_GLM_* knobs, MTP/DRAFTER choice, 64-row grouped expert launch, seven-draft ask, spill, kept-state policy, `_scores_rows` G=128) is GLM-only or about head drafting, which the MiniMax pack does not have.
5. Provenance: nothing recommended is from r0b0tlab (zero hits anywhere). vcruz305 appears in the branch only in patch 0002 (3 header lines) and in v1.1.2 only in docs, via the EXL3 module he contributed inside TensorFold (#42), which the owner's exception allows. Section 6.
6. Estimate (section 7): the v1.1.2 skeleton removes roughly 35-45 percent of the T8 code to be typed and 50-60 percent of T9; the branch itself adds under 10 percent. Kernels, numerics and every GPU gate stay.

---

## 1. Which TF_REF, and what the patches are against v1.1.2

### 1.1 The pin

| Item | Value | Source |
| --- | --- | --- |
| TF_REF | `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` = TensorFold **v1.0.7** | `git show zig-engine:scripts/config.sh` (`ZIG_TF_REF`, `TAG_BASE="zig-v1.0.7"`), `docs/zig-engine.md` |
| Port ref | `zig/PORT_REF` = `dee65c75b1bf... a10c8647af52... r107-int` (TensorFold base, port head, branch of the Zig GLM port) | `git show zig-engine:zig/PORT_REF` |
| Build | Zig 0.17.0, nvcc sm_121, `zig build ... install native`, libtfgrammar over xgrammar v0.2.8 (commit pinned), NCCL wheel 2.30.7 replacing the image's, libibverbs, cuDNN for the vision tower | `zig/Dockerfile` |
| Extra payload | `zig/aot/glm-aot-sm121.tar.xz`: Triton cubins captured from the Python engine on a GB10 (aot.json + 183 cubins); the engine replays them | `docs/zig-engine.md` |
| Patch authors | commits are MiaAI-Lab's, except 0006 and 0007 (and the DRAFTER-selection doc change) by **Hector Isaza (hecisaza)** | `git log zig-engine -- zig/` |

Our tree is TensorFold **v1.1.2** (`4ff784410f70b95767684d337e59dfd6d94fec8b`, a shallow clone: one commit, so no per-file `git log`/`blame` exists there). The branch is two releases behind it.

### 1.2 `git apply --check` of each patch against v1.1.2 (VERIFIED, throwaway worktree)

| Patch | Files | Against v1.1.2 | What it is | What v1.1.2 has instead |
| --- | --- | --- | --- | --- |
| 0001 shared engine | 80 modified | **fails**: 150 errors (75 files do not apply; README, build.zig, `zig/build/cuda.zig`, `engine_api.zig`, `lane_host*.zig`, `lanes/*`, `prompt_cache.zig`, `cuda/abi.zig`, `native/cuda.zig`, most of `server/*`). only 5 files apply: `.gitignore`, `cli/cuda_main.zig`, `server/errors.zig`, `server/http_conn.zig`, `server/metrics.zig` | lane host and lane core changes, `native/cuda.zig` registry plus `follow`/`--tp`, server changes, build wiring (section 1.3) | most of the "server" lines are the one-line `@import("json.zig")` to `@import("json")` module change, which upstream already made; the rest is the recipe's GLM policy plus a call gate |
| 0002 Flash Next CUDA | 71 added | **fails**: 5 files already exist (`flashnext/cuda_config.zig`, `cuda_lanes.zig`, `cuda_mtp.zig`, `cuda_prompt.zig`, `tests/cuda/flashnext/capture.py`) | a Zig port of the *Python* Flash Next CUDA engine (decode, mirror, wire), the base of the branch's GLM engine. Three headers say `vcruz305` (section 6) | upstream has its own, different Flash Next CUDA engine under the same directory |
| 0003 GLM kernels | 11 added | **fails**: 4 same-named files exist (`glm_kda.cu`, `glm_ops.cu`, `glm_qmm_frag.cuh`, `glm_vision.cu`; v1.1.2 versions are 434/51/117/181 lines against 535/245/83/704) | `glm_exl3.cu` (3,552 lines), `glm_qmm*.cu`, `glm_kda_chunk.cu`, `glm_msa.cu`, `glm_hc.cu`, `glm_vision.cu`, `glm_foundations.cu`; ports of the Python engine's kernels | v1.1.2's own `glm_glue*.cuh`, `glm_mm.cu`, `glm_quant.cu`, `glm_dflash.cu`, `glm_kda.cu` ... |
| 0004 GLM engine | 81 added | **applies cleanly** (all paths are new) but would **not compile**: module name `glm_cuda` is already taken by `families/glm/cuda.zig` (`zig/build/cuda.zig` `glmCuda`), it imports `flashnext.mirror` / `flashnext.decode` from 0002 (11 imports in 7 files), and it needs 0003 | `families/glm_cuda/*`: `cuda_native`, `cuda_engine`, `cuda_lanes` (1,516 lines), `cuda_link`, `cuda_comm`, `cuda_mirror`, `cuda_spill`, `cuda_triton*`, `cuda_dflash*`, `cuda_vision*` ... | `families/glm/cuda_*.zig`: a separate native implementation |
| 0005 tests + rest | 1,118 added | **fails**: `zig/build/vision.zig` and `zig/src/cuda/roce.zig` exist | 1,054 vision test files, the branch's 2-4 rank RoCE gather (`cuda/roce.zig` + `roce_proxy.c`), the recipe server layer (`server/recipe.zig`, `glm_calls`, `glm_prompts`, `glm_kept`, `glm_reply`, `playout.zig`, `admission.zig`), `lanes/loop_recipe.zig`, `prompt_cache_policy_test.zig` | upstream's own two-rank `cuda/roce.zig` and `cuda/tp_group.zig`, `server/http_admission.zig` |
| 0006 scores G=128 | 1 modified | fails: needs 0004's `glm_cuda/cuda_prefill.zig` | launch the prompt-tail `_scores_rows` Triton kernel at G=128 (the captured set has no G=88) | n/a (Triton replay is not in v1.1.2) |
| 0007 record before long batch | 1 modified | fails: needs 0002's `flashnext/cuda_mirror.zig` | order fix in the TCP mirror (section 3) | n/a (NCCL control link) |

Consequence: none of the patches can be applied; any reuse is reading and re-writing. The branch's `tools/make-zig-patches.sh` (group files into 0001..0005, check base + patches == port branch) is a good method for our own patch series but MiniMax already ships numbered patches `0001..0007` against v1.1.2.

### 1.3 What 0001 adds that v1.1.2 lacks, and what upstream merged by its own route

Presence tests were run as symbol greps in the v1.1.2 tree (VERIFIED):

| 0001 addition | In v1.1.2? | Generic for MiniMax? |
| --- | --- | --- |
| `Backend.VTable.prefill_begin` / `fill_step` / `FillState`, `Engine.beginStream`, `FillHook` (fills in slices between rounds) | no. v1.1.2 has `prefill_step`, `prefill_many`, and `begin`/`piece`/`finish` (`Rounds`) | no: use `prefill_step` as GLM and HIP do |
| `Backend.prefill_many`, `BurstRefused` | `prefill_many` yes (vtable field, `lane_host_many_test`), `BurstRefused` no | optional, later |
| `Config.head_stops`, `join_batch`, `Model.draft_stops`; `depth.zig` "seven-draft ask" (`if p.len == 0 return 1; return @max(1, @min(most, 7))`, `stopsNow`) | no (v1.1.2 `depth.zig` still has the 1/2/3 ladder) | no: head-draft policy, MiniMax has no head |
| Call gate: `stream.zig` `CallSpec`, `Gate`, `gateWatching`, `gateCut`, `absorbIntoPrompt`; `lane_host.zig` `continueGate`; `open.zig` first-draw check and `reprefill`; `engine_api.CallGate.blank` | **partly**: `engine_api.CallGate {opener,lead,tail,names,think_open,think_end}`, `Request.call`, `Info.call_gates` are declared, but nothing sets or consumes them (grep: only `lane_host_lone.zig:18` reads `r.call != null`). v1.1.2 serves required/named tools through grammar (`server/tool_grammar.zig`, `tool_required_*`, `first_masked`) | yes in principle, see section 3 |
| `loop_recipe.zig`, `Request.loop_recipe` | no | no (GLM recipe's loop rule) |
| `prompt_cache.zig` `Policy`, `Stored`, spill tier, `KeptStats`, `recipeHistory`, `capVictim`, `lcpPoint`; `Open.spill_*`; `Status.pool_tokens/kept` | no | no: GLM recipe policy (caps by ThomasWadeZ, recency by meleesciony, pool share by jdecker76, disk tier by wojo) |
| `Media`/`MediaItem`, `Open.vision`, `vision_zig` | no (v1.1.2's vision is per family: `Options.vision`, `Info.vision`) | no (no tower in the pack) |
| `native/cuda.zig` `follow()`, `--tp/--rank/--master` serve flags, `openOwn` | no: v1.1.2 serves no `--tp` flag; GLM ranks come from `TF_GLM_RANK/WORLD` + `NCCL_COMM_ID` | no: T7 already chose the env route |
| `cuMemcpy2DAsync`, `copyRows`, `ncclSend/Recv/CommAbort` in `cuda/abi.zig`, `nccl.zig`, `memory.zig` | **yes**, all present in v1.1.2 (merged upstream) | already available |
| `cuProfilerStart/Stop`, `aot.zig` public structs (Triton replay) | no | no |

---

## 2. T8 and T9 skeletons: the exact files and functions in v1.1.2

All paths relative to `zig/src/` of v1.1.2 unless stated. Signatures were copied from the files (VERIFIED, read in the scratch worktree).

### 2.1 T8 (engine: prefill, decode, verify windows, lanes)

| MiniMax part | Skeleton to copy | Exact anchors | What the branch adds on top | Take it? |
| --- | --- | --- | --- | --- |
| Lane backend (`backend.zig`) | `families/glm/cuda_lanes.zig` (535 lines): `pub const Lanes = struct { x: follow.Leader, whole, alone, lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane), drawn: [1024]u32, next: u64, ... }`; `pub fn backend(self: *Lanes) be.Backend` returning one of two vtables (`whole_vt`, `steps_vt` which sets `prefill_step`); `facts(self: *const Lanes) lanes.Model`; `claim`, `claimUsing`; fns `prefillFn`, `prefillManyFn`, `prefillStepFn`, `firstFn`, `firstMaskedFn`, `heldFn`, `queueFn`, `readFn`, `verifyFn`, `verifyShared`, `keepFn`, `draftFn`, `releaseFn`; handle ring `take(token) u64` | `glm_cuda/cuda_lanes.zig` (1,516 lines) is the same shape plus spill hooks, display KV, DFlash2/MTP selection, `TF_GLM_PROMPT_*` | skeleton yes, branch no |
| `lanes.Model` for copy drafts only | `families/qwen3_5/cuda_native.zig`: `pub fn laneFacts(window: u32, verify: u32) lanes.Model` = `{ .exact_width = window, .hidden_rows = true, .batch_rows = verify, .max_streams = E.max_segs, .streams_exact = true, .copy_match = 5, .context_copies = true, .lone_cost_tile = 16 }`. GLM's `facts()` has `exact_width = if (drafting) max_rows else 1`, which would disable windows for a head-less model, so do not copy that line | n/a | Qwen's |
| Engine shell (`engine.zig`) | `families/glm/cuda_engine.zig` (558) + `cuda_engine_types.zig`: `Engine.init(gpa, io, d, ctx, checkpoint, lib, comm, rank, world, o: Options) !*Engine`; `bind(slot)`, `release(slot)`, `seqBytes()`, `pos()`, `begin(sampling)`, `beginFollowing`, `prefill(ids, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32`, `prefillChunk(part, start, len, from, cancel, marks) !?u32`, `gate(cancel)`, `verify(ids, out, bits) !void`, `verifyShared(parts: []const Part)`, `keep(n)`, `draw(logits, rows, first, bits, out)`, `checkGathers()`. Types: `Part {slot, ids, out, bits}`, `Seq {st, sampling, ...}`, `Resume {id, at}`, `Marks {at, ctx, call}`, `Cancel {ptr, check}` | `glm_cuda/cuda_engine.zig` (1,170): same API plus Triton, DFlash2, display KV | skeleton yes |
| Forward (`forward.zig`) | `families/glm/cuda_forward.zig`: `Modules.load(d, ctx)`, `Forward {d, m, stream, ranks, exchange, pieces, ...}`, `Forward.run(f, ids: []const i32) !void`, `commit(keep)`, `Exchange` union (NCCL or RoCE `Group`), `Seg {st, row0, rows}` for shared windows, `row_step = 128`, `max_pieces = 16` | `glm_cuda/cuda_forward.zig` | structure only; layer bodies are GLM's KDA/MLA/HC and are replaced by section 4 |
| Prefill (`prefill.zig`) | `Engine.prefill` / `prefillChunk` above (2,048-row chunks, mark callback at chunk starts, `gate(cancel)` between chunks, rows of a chunk = `@min(len - start, 2048)` plus the next id so the last row's draw has its token), `cuda_prompts.zig` `fits`, `prefillMany`, `pass` | `glm_cuda/cuda_prefill*.zig` (Triton prompt grid; `cuda_prefill_grid`, `cuda_prefill_split`, own-front, microbatch) | GLM-only |
| Rank 1 follower | `families/glm/cuda_follow.zig` (461): `Kind` enum (`begin, prefill, verify, keep, draft, end, drop, release, chunk, shared, image, block, many, redraw, idle, snapshots`), 64-byte `Header`, `Link {send, rows, receive}` over **`ncclBroadcast`**, `Leader` (one method per step), `pub fn follow(e: *Engine, link: *Link) !void` | `glm_cuda/cuda_link.zig` + `cuda_mirror.zig`: TCP control link and record wire | skeleton yes; do not copy the branch's |
| Sampler | `families/glm/cuda_sampler.zig` (166): `Sampler.init(d, vocab, rows)`, `draw(stream, exchange, spans, mine, rows, first, r: Rule, out)`, `mask`, `logprobRows`; `Rule` from `Sampling`; kernel `sample.cu` `tf_draw`. Generic apart from the type it takes; `cuda_structured.zig` `drawWhole(e: anytype, ...)`, `redrawFirst` | `glm_cuda/cuda_sampler.zig` (469) | import or copy v1.1.2's |
| TP exchange / sums | `cuda/tp_group.zig` `Group.allGather(s, mine, all, bytes)`, `send/recv`, `openRoce(gpa, io, d, s, max_bytes)`; `cuda/rank_scatter.zig` `RankScatter.run(lib, comm, world, rank, s, partial, rows, width, scratch, out)` and `rankSum(s, peers, own, world, rank, n, out)` (kernel `rank_sum.cu`) | the branch has its own `cuda_comm.zig`, 2-4 rank `roce.zig` | upstream's |
| Pooled KV arena | `families/nemotron/cuda_pages.zig`: `layout(e) kv.Layout`, `Pages.init(gpa, e, streams, pages)` (one `DeviceBuffer` arena of `pages * pageBytes`, `kv.Pool`, a `Lane` per extra stream), `fits(prompt, max_tokens)`, `open(prompt, max_tokens, untaught)`, `close(seq)`; `cuda_native.attach(p, streams, pages)` / `fits(p, prompt, max_tokens)` | not in the branch | skeleton for T6 on device |
| Backend-owned radix prompt cache | `families/qwen3_5/backend/hip/engine/hip_prefix.zig` (`Prefix {store: radix.Store, ...}`, `begin(prompt, history_len, shared_prefixes, owner, matched)`, `keep(owner, prompt, to)`) and `hip_lanes.zig` `begin(s)` (restore, promise pages, `s.cached = at`, marks into a `Fill`), `promptStep(h, s)` (chunk ends at the next mark; `prefix.keep` there) | not in the branch | matches T6's `state.Cache.admit/keep/end` exactly |

### 2.2 T9 (native.zig, registration, server, ranks, pool hooks, cache, tool/think)

| MiniMax part | Skeleton | Exact anchors |
| --- | --- | --- |
| `native.zig` contract | `families/glm/cuda_native.zig` | `model_type`, `formats`, `default_context`, `max_segments`, `prompt_rows`, `explains_open`, `Options`, `Loaded`, `place()`, `weightShare(a, io, dir, total) u64`, `capacity(context)`, `open(gpa, io, ctx, dir, kernels, o: Options) !Loaded`, `openRank`, `warmPeers(d, g)`, `roceBytes(text)`, `snap_*`, `close`, `explain(ctx, err) ?[]const u8`, `linkCapacity` |
| What `native/cuda.zig` reads from a family | `native/cuda.zig` (`openWith`), `cuda_refusal.zig`, `cuda_family_memory.zig`, `cuda_kv_pool.zig`, `cuda_snaps.zig`, `cuda_options.zig` | decls the host tests with `@hasDecl(F, ...)`: `model_type`, `formats`, `Options` (fields set by `family_options.set`: `lanes`, `lanes_fixed`, `kv8`, `vision`, `rank`, `drafter`...), `open`, `explain` (+ `explains_open`), `default_context`, `max_segments`, `weightShare` / `placementWeightBytes` / `tensorWeightBytes`, `rankRoom`, `healthy`, `fits`, `window_rows`, `attach`, `snap_*`, `snap_slots`, `snap_place`, `warm_quiet`; `Loaded` fields `backend, facts, rows, stream_bytes, free_streams, ctx, deinit, lone, target, peer_room, layout, snap_bytes, vision, follow, policy` |
| Registration | `native/cuda.zig` line ~28 `const registry = .{ nemotron.native, nemotron.omni, glm.native, qwen35.native, qwen35.native_moe, flashnext.native, kolibri1.native };` plus `const glm = @import("glm_cuda");`; the count test at the bottom derives from it | |
| Build module | `zig/build/cuda.zig`: `fn glmCuda(b, target, optimize, cuda, core, lanes, heat) *std.Build.Module` (root `zig/src/families/glm/cuda.zig`, imports `cuda, core, canon, lanes, heat`); used at three places: `runner.addImport("glm_cuda", ...)`, `engines()` imports, `hostTests`/`glm_family_host.zig` | add `minimaxCuda(...)` the same way; our `minimax_*.zig` build steps compile files individually (`minimax_load.zig`, `minimax_pool.zig`, `minimax_tp.zig`, `minimax_ref_host.zig`) and there is no family module yet |
| Hub title | `cli/hub.zig` line 15 `.{ .model_type = "glm5_next", .title = "GLM-5.3-Flash" }` | add `.{ .model_type = "minimax_m3_vl", .title = "MiniMax-M3.1-Flash" }` |
| Refusals | `native/cuda_refusal.zig` `family(F, o)`: `--vision` is refused for a family whose `Options` has no `vision` field (so MiniMax is refused for free), `--tp` refused unless `Options` has `rank` (GLM has none and uses env, so does T7), `--drafter` refused unless `takes_drafter`, `--kv-cache fp8` unless `kv8`; `backend(o)`: `--mtp-drafts` refused for non-Nemotron | |
| Rank startup | `glm/cuda_native.zig` `open`: `place()`, `roceBytes(TF_ROCE_MAX_KB)`, `NCCL_COMM_ID` check, `ranks.Settings.read(&ranks.environment, &problem)`, `cuda.tp_watch.start(io, rank, world)`, `openRank`: `setenv NCCL_RUNTIME_CONNECT=0, NCCL_CUMEM_HOST_ENABLE=0, NCCL_CUMEM_ENABLE=0` (overwrite 0), `nccl.Library.open`, `ncclGetUniqueId` (from `NCCL_COMM_ID`), `ncclCommInitRank`, `warmPeers`, `Engine.init`, `group.openRoce`, `follow.Link.init`, `ranks.gather` of rooms, `tp_watch.ready(io, digest)`, follower loop then `exit(0)`, rank 0 returns `Loaded` | T7 `ranks.zig` supplies `place`, `admit`, `Settings`, `Fingerprint`, `agree`; `rank-contract.md` section 2 lists the variables |
| Pool hooks | `native/cuda_kv_pool.zig` and Nemotron: `Loaded.layout: ?kv.Layout`, `Loaded.snap_bytes`, `F.attach(ctx, streams, pages)`, `F.fits`, `F.window_rows`, `F.snap_slots/snap_place` | T6 `pool.Dims.layout(max_pages)` is the `Layout` |
| Prompt cache | `native/cuda.zig`: a `pc.Store` over `snap_*` is made only `if (@hasDecl(F, "snap_save") and cache > 0)`, and `Info.prompt_cache = h.store != null`; the HIP family owns a `radix.Store` in its backend instead | see mismatch M2 |
| Tool / think | `server/server.zig` `Server.init`: `srv.markers` (default `think_markers` close `</think>`, switched only when the tokenizer has `<channel|>`), `thinking_switch` (already true for `thinking_mode`), `think_close`/`think_close_end` (only for `</think>`); `server/reply_text.zig` `calls` table (T3 added MiniMax's) and `mm_think_markers` / `mm_think_prompt_markers` (T3 constants, **used only in tests**); `server/tool_grammar.zig` `Style = enum { xml, json, gemma, glm }`, `style(rendered)` | gaps in section 5 (M8, M9) |

---

## 3. The branch's fixes, one by one: generic or GLM-only

| Branch change (commit) | What it does | Generic engine/server logic MiniMax needs? | v1.1.2 / MiniMax equivalent |
| --- | --- | --- | --- |
| **Record before long batch** (8466409, patch 0007, by hecisaza) | In `flashnext/cuda_mirror.zig` `Records.send`: `try r.wire.put(&r.rec)` then `try r.link.send(.round, bytes)`. Before, a batch larger than the socket buffers (a few hundred thousand tokens) was written to the TCP link first, rank 1 reads the link only after the record, so rank 0 blocked in the write while rank 1 waited for the record. The test uses 64 KiB socket buffers, a 4,000,000-id prompt and a 180 s watchdog | **No, for the code. Yes, for the rule.** The bug is in the branch's TCP mirror. v1.1.2's `glm/cuda_follow.zig` `Link.send` posts header and ids as `ncclBroadcast`s in a fixed order and `Link.receive` posts the matching ones, so there is no socket to fill. Keep the rule in T8: *never block on an unbounded write before the peer is guaranteed to read it*; ids per step are bounded (`Link.init(..., asked_ids)` caps at the context, `TooManyIds` otherwise) and prompts go in 2,048-row chunks | covered by copying v1.1.2's `follow.zig`; steal the test idea only if a TCP link is ever added |
| **`_scores_rows` at G=128** (1442104, patch 0006, hecisaza) | launch the Triton prompt-tail scores kernel with the grid `G=128`, since the captured cubin set has no `G=88` (`MissingTritonVariant`); G only changes parallelism, not the result | **No.** Triton cubin replay does not exist in v1.1.2; T5's kernels are ours and take any grid. Lesson only: parameterise launch grids so they do not depend on prompt length | none needed |
| **MTP head via `DRAFTER`, not `--drafter mtp`** (3a79196, hecisaza) | `cuda_mtp_choice.zig` reads `DRAFTER` (`dflash2` default) and `TF_GLM_MTP`; `tensorfold-native serve` resolves every `--drafter` value as a model dir, so "mtp" was rejected | **No.** The MiniMax pack has no MTP head or `dspark` module (plan section 1); `refusals.family` already refuses `--drafter` and `--mtp-drafts` for a family that does not take them | none |
| **64-row grouped expert launch** (60626dc, r107-int a10c8647) | launch of the GLM EXL3 4-bit mcg expert decode kernel (`glm_exl3.cu`) with 64-row groups for verify windows | **No** (GLM mcg 4-bit kernel). MiniMax uses `exl3_experts.cu` (all codebooks, any K2) and `x3gm` for prompts; its window-width invariance is a T4/T8 GPU gate | none |
| **Seven-draft ask** (a52dcb5, r107-int c50af40f) | `depth.zig`: when the family has no timed cost table, ask for 7 drafts, the drafter's confidence product cuts the chain | **No.** Head drafting only (DFlash2/MTP). MiniMax drafts by suffix copies | none |
| **MTP kernel set and required-call restart** (9f08cf6) | the MTP kernels in the AOT set; plus the call gate in lanes (`Gate`, `continueGate`, `reprefill`, `engine_api.CallGate.blank`, `server.zig` building `request.call`) | **The restart: partly yes.** `tool_choice` required/named needs *some* enforcement. Gate design: after the prompt, the first non-blank, non-think token that is not the opener is replaced by the opener and the prompt+reply is prefilled again, so the next token is a prefill draw (Python `generate_gated`). Limits for MiniMax: the opener is a single token in `CallSpec.opener: u32`, MiniMax's call opens with the special token `]<]minimax[>[` (200058) followed by the text `<tool_call>`; the gate was only exercised for GLM | v1.1.2 route: grammar (`tool_grammar.structure`) with `first_masked`/`held`/`Window.masks`, which the GLM CUDA and Qwen CUDA backends already implement; MiniMax needs a `Style` for its markup (M9). Port the gate only if the grammar cannot express the markup |
| **Record the ranks' env** (3e11e17, `zig_env()` in `start.sh`): every `TENSORFOLD_*`, `TF_GLM_*`, `TF_ROCE_*` the recipe exports, plus `TF_GLM_ROCE` from `COMM`, `TF_GLM_L2PF`, `TF_CUDA_DEVICE`, `TF_GRAPH_LOG`, `TF_ROUND_LOG`, `TF_TP_*`, `DRAFTER`; "a variable the engine does not read is ignored, a malformed one it reads fails the start" | the passthrough rule and the flag preflight | **Generic as a rule.** MiniMax's equivalent is `rank-contract.md` section 3 (`TF_MM3_*`, `TF_ROCE_*`, `NCCL_*`, `TENSORFOLD_*`). In v1.1.2 `TF_ROCE` (0 keeps NCCL), `TF_ROCE_HCA`, `TF_ROCE_GID_INDEX` are read by `cuda/roce.zig`; `TF_ROCE_MAX_KB` by the family's `roceBytes`. The branch's own `cuda_comm.zig` also reads `TF_ROCE_WAIT_S`, `NCCL_IB_TC` for its 2-4 rank RoCE, which upstream's 2-rank `roce.zig` does not | `scripts/zig_flags.py` (generic, 30 lines, by MiaAI-Lab): checks `tensorfold-native capabilities --json` flags and enumerated values before anything is stopped. Reusable verbatim for the MiniMax `start.sh` |
| **Spill tier** (`cuda_spill.zig` 855 lines, `Open.spill_*`, `prompt_cache.Stored`) | kept GLM states written to disk by rank | **No** for v1 (GLM states are large; MiniMax snapshots are page references and a position). A disk tier for pages would be new work | none |
| **Kept-state policy, prompt `Policy`, `KeptStats`** | caps per conversation/bytes, recency, shared cuts | **No** (GLM recipe) | radix store rules in `state.rules` |
| **Prompt scores rows** | see G=128 | No | |
| **VISION=1 native tower, cuDNN in the image** | | No (no tower) | |
| **Dockerfile pieces** (libtfgrammar over xgrammar v0.2.8 pinned by commit and checked, NCCL wheel replacing the CUDA image's libnccl, `libibverbs-dev` in the build, `libibverbs1 ibverbs-providers rdma-core` in the run image, `tf.patches` label from the patch hash, `tensorfold-native --version` smoke) | packaging | **Generic.** Our `zig/Dockerfile` (2,874 B, not read here) should be compared against it for the xgrammar and NCCL steps; libtfgrammar is what structured output and tool grammars need | |

TP link deadlock for long prompts, stated once more: **MiniMax must not copy `flashnext/cuda_mirror.zig`**; it must copy `glm/cuda_follow.zig`. If MiniMax's `follow.zig` ever sends a whole-prompt `prefill` step, `Link.init(asked_ids)` must be at least the context (GLM's `linkCapacity(cap, media)`), else `TooManyIds` ends the request.

---

## 4. Ordered T8/T9 recipe for MiniMax (files under `zig/src/families/minimax_m3/`)

The MiniMax patch series is `0001..0007` against v1.1.2; T8 and T9 become `0008-minimax-m3-engine.patch` and `0009-minimax-m3-native.patch` (or one patch; the build steps below can be split either way). Everything in step 0 is the "1-day contract step" the plan asks for.

### Step 0. `iface.zig` (contract, no GPU)

Pin the cross-task types, taken from what exists:

```zig
pub const prefill_rows: u32 = 2048;        // chunk; a multiple of pool.page_tokens (128), GLM's fw.prefill_rows
pub const window_rows: u32 = 16;           // verify rows; plan T4 gate widths 1,2,3,16,17,64,128. UNVERIFIED value
pub const shared_rows: u32 = 64;           // rows of one shared forward over streams (GLM: shared_rows = 32)
pub const max_shared: u32 = 16;
pub const Part = struct { slot: u32, ids: []const u32, out: []u32, bits: u64 = 0 };   // as glm cuda_engine_types.zig
pub const Resume = struct { at: usize };   // no kept-state id: the radix store hands back pages, not a copy
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };
```

and the per-layer buffers contract (rows, hidden, q/k/v/iq scratch, `bmax`/`sel` scratch from `minimax_m3_sparse.Scratch {bmax, sel, bm_stride}`, router logits/picks/weights, expert `Scratch`, fp32 partial slots `tp.sumSlotBytes(c, rows)`). Also record which `core.minimax_ref.Switches` the engine runs with, and whether they enter the rank `Fingerprint` (M5).

### Step 1. `weights.zig` (the missing link between T2's `Sink` and the kernels)

Model: `glm/cuda_weights.zig` (`Weights`, `Layer`, `load(gpa, io, d, stream, tools, dir, c, share, layers, ...)`) and `glm/cuda_exl3.zig` (`Experts`, `load(L, i, count, dims, inter, cut)`), `cuda/exl3_upload.zig`.

- Open `load.Catalog.open(gpa, io, dir)`, build `load.Plan.build(gpa, &cat, &cfg, share, .{ .layers = ... })` with `share = shard.Share.of(&cfg, rank, world)`, allocate one `DeviceBuffer` per `Plan.slots[i]` (`Slot.bytes`), and stream with `load.Streamer.init(gpa, &cat, 8 << 20).run(&plan, sink)` where `sink.put(ctx, slot, offset, bytes)` is `DeviceBuffer.uploadAsync(offset, bytes, stream)` plus a periodic sync. Do **not** use `exl3_upload.Uploader.dense(parts, g)` (it takes host `exl3_load.Parts`); build `exl3.Weight` directly from `Plan.Matrix`: `trellis = slot_base + trellis_off`, `suh`, `svh`, `k`, `n`, `halves`, `.codebook = .mul1`, `stride_k/stride_nb = exl3.strides(k, n, halves, false)`.
- Routed experts: for each `Plan.MoeLayer` build `exl3_experts.Layer` (fields in v1.1.2: `gate/up/down` int64 [E] pointer tables, `gate_k2/up_k2/down_k2` int32 [E], `suh_g/suh_u/svh_g/svh_u/suh_d/svh_d`, `experts, dims, width, codebook, uniform_k2, shared_input, gu_stride, limit, act_mode, alpha, up_bias`). The pointer arrays are `slot_base + MoeProj.trellis_off[e]` uploaded as one small buffer each; `uniform_k2` from `MoeProj.histogram`; then call `ffn.configureLayer(&layer, switches)`. `shared_input` needs a data comparison of w1/w3 `suh` that T2's plan does not do (M7).
- Plain tensors (`Plan.plain`): norms are already fp32 `1 + w` (`Streamer.fold` / `foldGemma`), which is what `tf_rms_rows(x, x_kind, w: *const f32, y, y_kind, width, eps)` and `QkW{.bf16 = 0}` take; fp16 `index_q/index_k` and fp32 router/bias stay as stored.
- Rank agreement before any weight bytes: `ranks.Fingerprint.of(place, settings, capacity, sha256(config.json), sha256(quantization_config.json), sha256(own exe), weights_digest)`; the weights digest is computed from headers and sizes (addendum of the plan: rank 1 reads over NFS).
- Drop the page cache after each shard run (plan addendum): `posix_fadvise(DONTNEED)` is not in `load.Streamer` (UNVERIFIED, not grepped in `core/direct_io.zig`).

### Step 2. New kernels the forward needs (nothing in T0-T7 provides them)

Listed in forward order; each needs a `contract/registry` entry and a float64 twin like T4/T5 did.

1. embedding gather: `kernels/cuda/embed_rows.cu` exists (UNVERIFIED that it fits a plain bf16 table without scaling; read `cuda/embed_rows.zig`).
2. gnorm: `norms.cu` `tf_rms_rows` / `tf_add_rms_rows` with the folded fp32 weights. Residual dtype and rounding points are UNVERIFIED (plan section 6).
3. q/k/v: `exl3.Dense.decode(s, w, x, x_dtype, rows, y, y_dtype)` / `.prompt(...)` three times (separate input rotations unless `suh` are bit-identical).
4. **qk-norm + partial NeoX rope + paged write of K, V (and IK)**: `kernels/cuda/qk_rows.cu` writes to contiguous `kc`/`vc` caches by `slot`/`len` (signature read), not to pages `K[kvh][128][128] | V | IK`; it has no index-head variant. New kernel `minimax_m3_qk.cu` (plan T5 listed "qk_rows.cu additions"; T5 delivered only index/select/attn).
5. **index projection**: fp16 `index_q_proj` (256 rows a rank) and `index_k_proj` (128 rows, replicated), norm, rope, IK page write. No fp16-weight GEMV exists (`dense_rows.cu` is bf16 weights); fuse with 4.
6. `minimax_m3_sparse.Sparse.run(st, iq, q, rows, p: Paged, sc: Scratch, rules: geom.Rules, max_pos, out)` (T5, done).
7. o_proj: `Dense` with `y_dtype = .f32` into this rank's partial slot, then `RankScatter.rankSum` (decode, `tp.formFor(rows) == .gather_sum`) or `run` (prompt chunks, `scatter_gather`), then residual add + gnorm.
8. **router**: logits `[rows,128]` fp32 = fp32 gate `[128,6144]` times the layer input cast to fp32. No such kernel: `route_rows.cu` `tf_route_logits(X bf16, x_ld, W bf16, OUT f32, M, D, E)` and `dense_rows.cu` take bf16 weights. Then `MiniMaxAct.route(s, logits, bias, r: Route, pick, wts, rows)` (T4, done; **not** `tf_moe_topk_sigmoid`, which ranks by logit + bias).
9. routed experts: `exl3_experts.Experts.run(s, layer, x, x_stride, x_dtype, pick, wts, rows, out, scratch)` (decode) and the `x3gm` prompt path (`exl3_block_x3gm.zig`) with `act_mode` 2; the `exl3_prompt.cuh` mcg-4-bit-only path is not usable (plan section 3.4). `exl3_block/exl3_experts.cu` has no swigluoai edit (grep: 0 hits); confirm it is off the MiniMax path.
10. shared expert + dense layers 0-2: two `Dense` (5-bit), `MiniMaxAct.swigluoai(...)`, `Dense` down (row cut, fp32), added into the same partial as the routed output (one exchange per layer, `tp.layerPlan`).
11. final gnorm, `lm_head` `Dense` (K2 12, 782/781 128-units a rank), `tp.LogitsGather` (`merge`) or `tp.mergePicks`, `Sampler.draw`.

### Step 3. `forward.zig`

Copy the shape of `glm/cuda_forward.zig` (`Modules`, `Forward.run(ids)`, `commit(keep)`, `Seg` for shared windows) and write the layer loop of plan section 2. Unlike GLM there is no state to roll back: `commit(keep)` is only "advance `pos` by `keep`" because KV rows past the kept ones are overwritten on the next write and the kernels mask by `pos`. Per call it needs: the stream's `Tables` row (device copy of `pool.Tables.rows`, `row_tab` = `slot * max_pages` entries or 0), `Rows{n, pos, pos0}`, `Paged{pool, page_bytes, layer_off = layer * layerBytes, kvh = 2, tables, row_tab}`.

### Step 4. `prefill.zig`

`prefill(ids, from: ?Resume, cancel, marks) !u32` and `prefillChunk(part, start, len, from, cancel, marks) !?u32` as in `Engine.prefill*` (v1.1.2), with: chunk starts at multiples of 2,048 (also 128), `marks.call` at the mark rows given by `state.Cache.admit`, `gate(cancel)` between chunks, the last chunk draws the first token. A resumed prompt starts at `Cache.admit`'s `plan.from` (pages adopted from the store). Gate (plan T8): resumed prompt equals fresh prompt; row invariance of `Dense`/experts is what makes this exact, `snap_grid` is not needed.

### Step 5. `engine.zig`

`Engine.init(gpa, io, d, ctx, checkpoint, lib, comm, rank, world, o)`: rank agreement, weights (step 1), `Modules`, buffers, the page arena, `Sparse.load(d)`, `MiniMaxAct.load(d)`, `Sampler.init`. Public API as `glm/cuda_engine.zig` (section 2.1) minus snapshot ids (`keepState`, `dropState` are replaced by `Cache`), minus drafter/MTP. One `Seq` per slot = `state.Stream {lease, slot, pos}`.

### Step 6. `follow.zig` (not in the user's list, but required)

`Kind = enum(u32) { begin, prefill, chunk, verify, shared, keep, release, drop, end, idle }`, `Header` 64 bytes as GLM's, `Link`, `Leader`, `follow(e, link)`. Copy `glm/cuda_follow.zig` and delete `draft, block, image, redraw, many, snapshots`. Keep the `tp_watch.sleep/wake` idle pattern (#1053: ranks sleep between requests).

### Step 7. `backend.zig`

Copy `glm/cuda_lanes.zig`: `Lanes`, `Lane {slot, pending_rows, first, fill}`, vtable with `prefill, prefill_step, first, queue(NotPipelined), read, verify, keep, draft (no-op or error), release, first_masked, held, prompt_rows`. `facts()` = Qwen's `laneFacts(window_rows, shared_rows)`. For the cache follow `hip_lanes.begin`: `Cache.admit(a, &s, prompt, s.history_len, s.max_new, window_rows, table, max_pages)` returns the marks; `s.cached = @intCast(plan_from)`; at each mark `Cache.keep(&st, prompt, at, table)`; at release `Cache.end(&st, &tables)`.

### Step 8. `native.zig` (T9)

Copy `glm/cuda_native.zig` section by section:

```zig
pub const model_type = "minimax_m3_vl";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 65536;           // plan: 65,536-131,072; budget.check decides the real one
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = iface.prefill_rows;
pub const window_rows: u32 = iface.window_rows;   // kv_pool.plan(.window = window + window_rows)
pub const explains_open = true;
pub const owns_prompt_cache = true;                // new, see M2
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 1, lanes_fixed: bool = false, room: ?Room = null };
pub fn weightShare(a, io, dir, total) u64;         // from load.Plan.slotBytes() of this rank (needs rank: ranks.place)
pub fn capacity(context: usize) usize;             // align to pool.page_tokens, + window_rows
pub fn open(gpa, io, ctx: *const cuda.Context, dir, kernels: ?[]const u8, o: Options) !Loaded;
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8;   // ranks.refusal(err), budget.refusalText, RankMismatch ...
pub fn attach(p: *anyopaque, streams: u32, pages: usize) !void;
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool;
```

`Loaded` = GLM's fields plus `layout: ?kv.Layout = pool.Dims.layout(max_pages)` and `snap_bytes = pool.snap_bytes`; `free_streams = 0` if every stream is paged (M1). `open` order as GLM: `ranks.admit(&ranks.environment)` (refusal text `ranks.refusal`) -> `Settings.read` -> `tp_watch.start` -> `setenv NCCL_*` (overwrite 0) -> NCCL init -> `warmPeers` -> `Engine.init` (fingerprint all-gather through a `gather` closure for `ranks.agree(ctx, gather, mine, out)`; GLM's version is `cuda_ranks.gather(T, d, lib, comm, stream, rank, mine, out)`) -> `group.openRoce(gpa, io, d, stream, roce_bytes)` -> `Link.init` -> room all-gather -> `tp_watch.ready(io, digest)` -> rank 1 follows and exits, rank 0 returns `Loaded`.

### Step 9. Registration (T9)

1. `zig/build/cuda.zig`: `fn minimaxCuda(b, target, optimize, cuda, core, lanes) *std.Build.Module` rooted at `zig/src/families/minimax_m3/cuda.zig` (a file like `glm/cuda.zig` that re-exports the parts); `runner.addImport("minimax_cuda", ...)`; `engines()` import `.{ .name = "minimax_cuda", ... }`; a `test-minimax-engine-host` step beside the existing `minimax_*` steps; `zig/build/cuda_kernels.zig`: the new kernel names from step 2.
2. `native/cuda.zig`: `const minimax = @import("minimax_cuda");` and `minimax.native` appended to `registry`; the test "every registered family ... are listed" derives its count.
3. `cli/hub.zig` titles; `cuda_refusal.zig` tests (add `"minimax_m3_vl"` to the vision-refusal list at line 76 and the MTP list at line 140).
4. `native/cuda.zig` `serve()` `.prompt_cache = h.store != null or (@hasDecl(F, "owns_prompt_cache") and F.owns_prompt_cache)` (M2).
5. `server/server.zig` `Server.init`: select `reply_text.mm_think_*` when the tokenizer has `</mm:think>` (id 200060) and set `think_close`/`think_close_end` for it (M8). `tool_grammar.style` gets a MiniMax style (M9).
6. Recipe: `start.sh` / `nodes.sh` `rank_env` as in `rank-contract.md` section 3; `scripts/zig_flags.py` preflight.

### Step 10. Order, gates

Order: 0, 2 (kernels, each with its host twin and contract record), 1, 3+4 (single rank, thin model via `TF_MM3_LAYERS`), 5, 8 (single rank serving, `--prompt-cache-gib 0`), 6+7 (two ranks), 7 with the radix cache, 9. GPU gates are the plan's (eager vs graph, window-width invariance, resumed equals fresh, concurrent equals solo); none can be run here.

---

## 5. Interface mismatches visible now

| # | Mismatch | Where read | Resolution suggested |
| --- | --- | --- | --- |
| M1 | **Lane 0 / pool model.** `core/kv_pool.zig plan(l, b)`: `extra = lanes - 1`, each costing `fixed_bytes + pageBytes`, `most = fit * l.pages(window)`; `native/cuda.zig` calls `F.attach(ctx, p.lanes - 1, p.pages)` and treats the first stream as the engine's own full-window sequence (outside the pool, counted in `model`). T6's `pool.zig`/`state.zig` make every stream paged (`Tables.init(slots, max_pages)`, `Cache.admit`). Lane 0's pages would come from a pool that was not sized for it | Nemotron solved it with `paged_only` (`TF_KV_PAGES=all`: its own sequence stays idle and every stream takes pages). For MiniMax the engine's own sequence should be a paged one too, with a small `Loaded.stream_bytes`; but `plan()` counts only `lanes - 1` streams against the pool, so lane 0's fixed part and pages are unaccounted. Options: a thin wrapper in `attach` that grows the pool by one stream's worth, or a flag in `Budget` for an engine that is itself paged (a small host patch). Decide in step 0 |
| M2 | **Who owns the prompt cache.** The CUDA host builds `pc.Store` (token-keyed, `Snapshots{bytes,save,restore,drop}`) only for families with `snap_save`; T6's `state.Cache` wraps a `radix.Store` (`api.prompt_radix`) that only the HIP family uses today (`hip_prefix.zig`). With `snap_save` absent the host reports `Info.prompt_cache = false`, `h.store = null`, and `kv_pages.kept` gives `copied = 0` ("no prompt cache" in the startup line) | one host patch (step 9.4). The lane host needs nothing else: `emit(.prefilled = job.stream.cached)` reads the backend-set `s.cached`; `history_len`/`shared_prefixes` come from `Stream` (server fills them). /health's kept numbers stay empty (`KeptStats` is branch-only). Bring-up first with `--prompt-cache-gib 0`, as the GLM branch qualified |
| M3 | **T5 kernels vs T6 tables.** `Paged.tables` is an int32 array and `row_tab` int32 *entry* offsets into it (per `minimax_m3_sparse.cuh`: "tables[row_tab[r] + block]"); `Tables.rows` is a host `[]i32` of `slots * max_pages` with `-1` past a stream's pages and an optional `Upload` hook. Nothing creates the device copy; a prompt chunk of one stream wants `row_tab = 0` with `tables` pointing at that slot's row, a shared window wants per-row `slot * max_pages` offsets | the engine owns a device table and the `Upload` callback (`Tables.upload`), plus a helper to fill `row_tab` per window; `Tables.write` already rolls back a failed upload |
| M4 | **T5 page addressing vs T6 `PageMap`**: consistent (page = layers x layerBytes; layer region `K[kvh][128][128] | V | IK`; `Paged.layer_off = layer * layerBytes`, `page_bytes = Dims.pageBytes()`, 76,800 B/token at kvh=2). Two checks: `kv.Layout.pageBytes()` must equal `Dims.pageBytes()` (T6 test covers it, UNVERIFIED here), and `Dims.layout().fixed_bytes = tableBytes(max_pages)` is charged per extra lane by `kv_pool.plan` while T6's tables are one array: either allocate per lane or lower `fixed_bytes` | |
| M5 | **T4 router.** `ffn.routePlan` returns `error.Unsupported` when `Switches.router_weights_use_bias`; `moeGlueRoute` ranks by logit + bias (wrong whenever the bias is non-zero, so unused); the fp32 logits GEMV is missing; the route kernel wants contiguous `[rows, experts]` fp32. Also: if any `Switches` field becomes runtime-selectable it must enter `ranks.Fingerprint` (today: `layers`, `reduce`, `capacity`, `config`, `quant`, `program`, `weights`, so only compile-time defaults are covered by `program`) | add the GEMV (step 2.8); freeze `Switches` at compile time until T0's UNVERIFIED items are settled |
| M6 | **`exl3_experts.Layer` additions** (`alpha`, `up_bias`, `act_mode` 2/3) exist in the merged tree; `exl3_block/exl3_experts.cu` was not edited (0 swigluoai hits) although the plan listed it. The mixed layer 17 (w1/w3 4-bit, w2 3-bit) is representable (`MoeProj.halves` per projection) but `uniform_k2` must be 0 there | check in the GPU gate |
| M7 | **`Layer.shared_input`** (identical gate/up `suh` for every expert, lets the kernel rotate once) is verified by the *loader* in v1.1.2's GLM path; T2's `Plan` has no such check, and neither does it compare q/k/v `suh` | compute on first upload by comparing device bytes, or leave 0 (correct, slower) |
| M8 | **Think markers.** T3 added `reply_text.mm_think_markers` (`<mm:think>`/`</mm:think>`) and `mm_think_prompt_markers` (close only), but `Server.init` only switches `srv.markers` for `<channel|>` and only fills `think_close`/`think_close_end` from `</think>`; the constants are used in tests only. Reasoning split, `think_budget`, loop guard and answer limits therefore see no think block for MiniMax | in `Server.init` choose by tokenizer (`</mm:think>` present) and by the prompt's tail (adaptive: reply writes `<mm:think>`; enabled: prompt ends in it) |
| M9 | **Tool grammar.** `tool_grammar.style(rendered)` recognises gemma, qwen xml, json and glm only; for MiniMax it returns null and `structure()` falls back to `.json`, which would force the wrong envelope for `tool_choice` required/named. The streaming parser (`tool_minimax`, `tool_stream` hooks) exists | add a MiniMax `Style` (its structural-tag form is UNVERIFIED: xgrammar may have an XML style for it; not found in this tree) or refuse required/named tools for this family until done |
| M10 | **Budget twice.** `budget.zig` (T6: `Terms`, `check(d, in, context) Verdict`, `refusalText`) predicts the largest context before load; `native/cuda.zig` then computes streams and pages itself after `F.open` from `pool()` / `room`. Two calculators can disagree by the workspaces | call `budget.check` with the real `load.Plan.slotBytes()` inside `open` to pick `capacity` and to refuse, and let the host's `kv_pool.plan` have the last word on pages; log both |
| M11 | **T7 vs GLM's `ranks`.** `minimax_m3.ranks.agree(ctx, comptime gather, mine, out)` takes a closure, GLM's `agree(d, lib, comm, stream, mine, out)` does the gather itself; `Group.open` vs the manual `ncclCommInitRank` GLM does; `Settings.read(get, problem)` signature differs by name only. `TF_ROCE` / `TF_ROCE_MAX_KB` must be equal on both ranks and are in neither fingerprint | wrap `cuda_ranks.gather`; add them to `Settings` when `open` reads them (already an open item in `rank-contract.md` section 6) |
| M12 | **Window width.** GLM's `window_rows = 8`, `shared_rows = 32`; the plan's gate list goes to 128; kv_pool adds `window_rows` pages of slack per stream via `F.window_rows`; `exl3` row invariance is claimed for 1-128 rows (CHANGELOG) while `Dense.prompt` runs 128-row blocks of the decode kernel | pick 16/64 and prove widths 1,2,3,16,17,64,128 on a GPU |
| M13 | **Vocabulary.** `tp.LogitsGather` pads each rank's slot to 782 units while `Sampler.draw(..., spans, mine, ...)` takes `shard.Span`s per rank (`families/glm/cuda_shard.zig` `unit = 64`, MiniMax `shard.unit = 128`); the fused `tf_draw` expects the same span form | reuse `Sampler` with MiniMax spans, or the `mergePicks` path for greedy |

---

## 6. Provenance (the hard rule) and credits

Method: `grep -rIi 'vcruz305|r0b0tlab'` over the whole v1.1.2 tree (`zig/`, docs, everything) and `git grep -i` over the whole `zig-engine` branch (all files, including the Python `patches/`), plus each patch's `From:`/header lines. v1.1.2 is a single-commit shallow clone, so per-file `git log`/`blame` is unavailable there; authorship below comes from `CHANGELOG.md`, `docs/credits.md`, `THIRD_PARTY_NOTICES.md` and file headers. The branch's own history (`git log zig-engine -- zig/`) names MiaAI-Lab and Hector Isaza only.

Findings (VERIFIED):

- **r0b0tlab: zero occurrences** in v1.1.2 (all files) and in the branch (all files). No exception applies; nothing to plan around.
- **vcruz305 in v1.1.2**: no source (`.zig`, `.cu`, `.cuh`) file mentions him; ten doc files do (`README.md`, `CHANGELOG.md`, `THIRD_PARTY_NOTICES.md`, `RELEASE-NOTES-1.0.0.md`, `docs/credits.md`, `docs/recipes/{cuda,exl3,exl3-universal-experts,qwen3.8-flash-next,README}.md`). They say: he wrote the Python EXL3 module (#42), the Flash Next Python CUDA path (`qwen3.8-flash-next.md` line 257), int8/int4 KV caches and `--mtp-confidence` (#47, #69). The native Zig EXL3 operators "port the Python engine's EXL3 module (#42). Thanks to @justincasey, @BHCC2025 and @vcruz305" (CHANGELOG, "The native CUDA build has EXL3 operators..."). That is code he contributed **inside TensorFold** and merged into its history: allowed by the owner's exception (2026-10-07) and already in use through T4.
- **vcruz305 in the branch**: only `zig/patches/0002-zig-flashnext-cuda.patch`, three file headers (`flashnext/cuda_engine.zig`, `cuda_forward.zig`, `cuda_mtp.zig`): "Python source (TensorFold ..., authored by Ash Hart (ashhart) with commits by vcruz305 inside TensorFold, which the owner allows for this port)". The Zig file is a port by MiaAI-Lab of TensorFold-merged Python; by its own words it falls under the exception, but it lives in the recipe repo, not in TensorFold's history, and 0004 builds on it (the branch's `glm_cuda/cuda_native.zig` says "modeled on flashnext/cuda_native.zig"; 11 `@import("flashnext")` lines). Verdict: **UNCLEAR, do not use.** Not needed anyway.

Verdict per source recommended in this document:

| Source | Verdict | Evidence |
| --- | --- | --- |
| v1.1.2 `families/glm/cuda_native.zig`, `cuda_follow.zig`, `cuda_engine.zig`, `cuda_engine_types.zig`, `cuda_lanes.zig`, `cuda_forward.zig`, `cuda_state.zig`, `cuda_sampler.zig`, `cuda_structured.zig`, `cuda_prompts.zig`, `cuda_shard.zig`, `cuda_ranks.zig`, `plain_rounds.zig` | **OK** | in TensorFold's tagged tree; no banned handle in the files; CHANGELOG credits for the GLM CUDA engine: @Bizuayeu (TP over hosts #804, idle sleep #1053, prompt logprobs #1016, structured output #1017, tool calls #1018, ring/probe #899/#902), @BHCC2025 (vocabulary cut in EXL3's 128-column unit #593), @BobClawblaw (EXL3 checkpoints, vision, rank watch #855/#976, fresh-prompt batching #1014, rings #984, packed dense #979), @ss-cong (snapshot pages #993), @jschmied (the CUDA serving path and registry #443), @taussoe (#43), @MiaAI-Lab (#128-#134), @mikolaj92 (#140), @plotarmordev (#222, #225, #1059, #1060), @akol1 (#152), @liumorrisclaw (four ranks #465), Ash Hart |
| v1.1.2 `native/cuda.zig`, `cuda_kv_pool.zig`, `cuda_refusal.zig`, `cuda_family_memory.zig`, `cuda_options.zig`, `core/lanes/*`, `core/kv_pool.zig` | **OK** | same; native CUDA server and registry authored by Jürgen Schmied (jschmied, PR #443, commit `59e77e8`) per the branch's own CREDITS and TensorFold's changelog |
| v1.1.2 `families/nemotron/cuda_pages.zig`, `cuda_native.zig` | **OK** | CHANGELOG: pooled pages "Thanks to @jschmied and @cwschroeder" (#935) |
| v1.1.2 `families/qwen3_5/backend/hip/engine/hip_lanes.zig`, `hip_prefix.zig`, `core/prompt_radix*.zig` | **OK** | CHANGELOG: HIP build #626/#617/#621 "Thanks to @BlivionIaG and @ThinkOffApp"; paged prompt cache / radix tree #625 "Thanks to @BlivionIaG, @jschmied and @BobClawblaw" |
| v1.1.2 `families/qwen3_5/cuda_native.zig` `laneFacts` | **OK** (authors not itemised in the changelog line read; no banned handle) | |
| v1.1.2 `cuda/exl3*.zig`, `kernels/cuda/exl3_*.cu(h)`, `exl3_block/*` | **OK by exception** | vcruz305 (#42) inside TensorFold; also @justincasey, @BHCC2025; format by turboderp (exllamav3, MIT, per file headers) |
| Branch `scripts/zig_flags.py`, `docs/zig-engine.md` settings-table pattern, `zig/Dockerfile` steps, `tools/make-zig-patches.sh` method | **OK** (MiaAI-Lab; zero banned strings; no copied third-party code in them) | `git log` |
| Branch patches 0001, 0003, 0004, 0005, 0006, 0007 | **OK as provenance, not recommended for use** | zero hits for either banned handle; authors MiaAI-Lab, hecisaza for 0006/0007; 0003 headers: "Port of TensorFold ...; authored by Ash Hart and TensorFold contributors; Zig CUDA launch pattern authored by jschmied"; credited third-party design inside 0001's `prompt_cache.zig` `Policy`: ThomasWadeZ (caps, PR #65), meleesciony (recency, issue #75), jdecker76 (pool share, issue #84), Robert Wojciechowski (wojo; disk tier, patch 0088) |
| Branch patch 0002 | **UNCLEAR, not used** | see above |
| MiniMax-side third-party references (SGLang, vLLM docs, NeMo-Automodel) | read for semantics only per the plan | unchanged |

Credits to carry into `CREDITS.md` if the corresponding code or design is used:

- Always (skeleton from TensorFold v1.1.2): **Ash Hart (ashhart)** and TensorFold contributors; **Jürgen Schmied (jschmied)** for the CUDA serving path, registry and pooled pages (#443, #935); **Bizuayeu** for the GLM two-rank CUDA engine and its follow/idle/structured/tool-call work; **BobClawblaw**; **BHCC2025** (128-unit vocabulary cut, EXL3 ops); **ss-cong** only if snapshot pages are adapted (we do not plan to); **BlivionIaG**, **ThinkOffApp** for the HIP radix-cache pattern; **vcruz305** (EXL3 module #42, inside TensorFold) and **justincasey**; **turboderp** (exllamav3, MIT) for the EXL3 format and kernels' origin.
- Only if their ideas are ported from the branch: **hecisaza (Hector Isaza)** for the record-before-batch ordering test pattern and the DRAFTER documentation; **MiaAI-Lab** for the Dockerfile/preflight pieces; **wojo** only if the spill tier idea is ported; **ThomasWadeZ / meleesciony / jdecker76** only for kept-state policy; **BadAd84** (`_scores_rows` design, patches 0085/0086) and **Alexbob0** (`chat_template_kwargs.thinking`, PR #25; kept-cap ordering PR #32) and **plotarmordev** (upstream #1059/#1060 deeper verify row, NCCL minimum) are not needed by T8/T9 as planned.
- Per the user's attribution rule, none of the above is wording to paste here; each is a name plus the work they authored.

---

## 7. How much T8/T9 work this removes (estimate, with reasoning)

UNVERIFIED estimate from line counts, not a measurement. v1.1.2's GLM files listed in section 2 total 4,945 lines (native 351, follow 461, lanes 535, engine 558+38, forward 533+114, state 409, ranks 591, sampler 166, structured 47, prompts 62, snapshots 114, shard 107, weights 527, exl3 206, plain_rounds 89, hold/wire 37). About 2,250 lines (native, follow, lanes, engine shell, sampler, structured, prompt glue) are structurally reusable for MiniMax; forward, state, weights, snapshots and plain_rounds (about 1,800) are model-specific and are rewritten; ranks and shard are already done by T7 and T1. I expect MiniMax's T8+T9 to be 4,000-5,500 lines of Zig plus the five new kernels of step 2. On those figures the v1.1.2 skeleton removes about 35-45 percent of the T8 typing and 50-60 percent of T9 (T9 is mostly the `native.zig` glue, registration and refusals, which is formulaic), and removes most of the *design* risk in rank lifecycle (watch lines, idle sleep, follower exit, error text), step protocol, handle ring, shared-window bookkeeping and the host contract (`@hasDecl` surface). It removes none of the numerics: new kernels (qk/rope/paged writes, fp16 index projection, fp32 router GEMV), eager-vs-graph, window-width invariance and every GPU gate remain, and those are where the schedule risk sits. The `zig-engine` branch adds at most 5-10 percent beyond that (the call-gate design, the record-ordering test idea, the flag preflight, Dockerfile packaging steps) because its engine is a Triton-replay architecture on v1.0.7 that cannot be applied or compiled against v1.1.2. Net: the T8/T9 effort is reduced by roughly a third overall, not halved.

## 8. Not read / limits

- `~/TensorFold/MiniMax-M3.1-Flash/zig/Dockerfile` and `README.md` were not compared with the branch's Dockerfile.
- The branch's `glm_cuda/cuda_lanes.zig`, `cuda_engine.zig`, `cuda_spill.zig` and 0001's lane-core hunks were read by heads, signatures and comments, not line by line; v1.1.2's `glm/cuda_forward.zig`, `cuda_state.zig`, `cuda_weights.zig`, `plain_rounds.zig` were read by signature only.
- `core/direct_io.zig` was not read for `fadvise`; `embed_rows.cu` fit and `kv.Layout.pageBytes()` equality (M4) are unchecked.
- Whether xgrammar has a MiniMax XML structural-tag style is unchecked (no xgrammar source here).
- Nothing in this document was compiled or run; no performance claim is made.
