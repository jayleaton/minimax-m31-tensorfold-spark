# patches

Patches to TensorFold v1.1.2 (`TF_REF` in `scripts/config.sh`, the commit `v1.1.2` points to), applied in filename order by
`zig/Dockerfile` (`git apply` in the checkout's root). The image's `tf.patches` label is a hash of this folder, the Dockerfile
and the pins, so adding or changing a `.patch` file makes `scripts/prepare.sh` rebuild the image (use `--rebuild` to force it).

**The applied series is `0001-minimax-m3-family.patch`** (128 files, +22,567/-69 lines; `git diff --binary` of TensorFold
v1.1.2 (`4ff7844`) against the integration branch `mm3-int`, which merges every task in `tasks/`). Checked on a clean v1.1.2
worktree: `git apply` succeeds, `zig build native` builds, `tensorfold-native capabilities --json` lists `minimax_m3_vl` (`exl3`),
and `test-minimax-engine` (77), `-forward` (117 of 118, 1 skipped) and `-proj` (10) pass. That is all that has run for the family
as a whole: no forward pass has run on a GPU, `open()` stops at `ForwardNotBuilt` until the engine's last pieces are proven,
and the model's semantics (swigluoai, RoPE layout, sparse block forcing, router order, norm folds) are UNVERIFIED guesses held as
switches. Only the individual kernels' contract gates 1-4 have run on a GPU (spark2), listed in the entries below.
`tasks/` keeps the per-task patches for history; they are NOT a linear series and `scripts/config.sh` and the Dockerfile read
only `patches/*.patch`, never `tasks/`. Regenerate the series after any change with
`git -C <tf-wt-int> diff --binary 4ff784410f70b95767684d337e59dfd6d94fec8b mm3-int > patches/0001-minimax-m3-family.patch`.

When a patch lands here, record in this file, for each one: its name, what it changes, who wrote it, and what was run against it
(nothing is claimed that was not run). Add credit for its authors to `CREDITS.md` and `NOTICE`.
Never run `scripts/prepare.sh` to rebuild with a partial set of patches.

0009-minimax-m3-rows-kernels.patch (branch mm3-t8k, one commit on mm3-int): minimax_m3_rows.cu (qk-norm and partial RoPE writing
paged K/V/IK, fp32 router GEMV), Zig wrapper, block-table device mirror, records, host twins. Run: test-minimax-rows (16), -ref, -sparse,
-ffn, test-kernel-contract, test-exl3-cuda-host, compile-cuda-tests and the sm_121 fatbin build. No GPU run.

0010-minimax-m3-forward.patch (branch mm3-t8a, one commit on mm3-int): the device forward and prompt driver of T8a.
forward_plan.zig (pure planner: embed, per layer qkv, index projection, qk-norm/rope, score/select/attend, o_proj, rank-order
exchange, residual+norm, dense or MoE FFN with routed+shared in one partial, head with padded logits gather or pick merge),
forward_exec.zig (Modules, Buffers, Device.emit over the exl3, norms, rows, sparse, act and rank_scatter wrappers),
forward.zig (Call/Out API, commit), prefill.zig (page chunks, marks, resume, first token), forward_test.zig, step
test-minimax-forward (additive hook in zig/build/minimax_family.zig), docs/interface-decisions.md "T8a requests". Run on the
build machine: test-minimax-forward (108 pass, 1 skip), test-minimax-ref/-sparse/-load/-pool/-tp/-ffn/-rows/-template/
-engine-host, test-kernel-contract (72), test-exl3-cuda-host (575), test-cuda-host, check-native-cuda-host,
compile-cuda-tests -Dsm=121, zig fmt and lean_check clean for the new files. No GPU run. Open: the fp16 index-projection
kernel and the per-rank pick kernel do not exist (the executor returns an error for those ops).

0012-minimax-m3-proj-pick-kernels.patch (branch mm3-t8c, one commit on mm3-t8a): the two kernels T8a's forward needed.
minimax_m3_proj.cu: tf_mm3_proj_bf16/f32 (fp16 index_q_proj and index_k_proj times the normed hidden state, fp32 fma sums in the
router's fixed lane/chain/xor-tree order so a row's bits do not depend on the row count, one rounding to fp16 or bf16 as a launch
parameter) and tf_mm3_pick (per-rank (logit, id) over the vocabulary slice of the padded logits: larger logit, ties to the lowest id,
NaN last). Zig wrapper cuda/minimax_m3_proj.zig, launch records, Device.emit wired for index_proj, local_pick and pick_gather
(NoIndexProjKernel/NoPickKernel removed), tp.betterPick now ranks NaN last, float64-reference contract entries
(tests/cuda/contract_minimax_proj.zig, `tf-cuda-test mm3-proj-contract` runs only them), host twins with planted mutants, step
test-minimax-proj. Run on the build machine: test-minimax-proj, all test-minimax-* steps, test-kernel-contract, test-exl3-cuda-host,
check-native-cuda-host, compile-cuda-tests -Dsm=121, zig fmt and lean_check clean for the new files. GPU run (spark2, GB10 sm_121): `tf-cuda-test mm3-proj-contract` only, 4 kernels pass gates 1-4 against the float64 references; the forward itself has not run on a GPU.

0011-minimax-m3-engine.patch (branch mm3-t8b, one commit on mm3-int): the engine loop of T8b. engine.zig (slots over the pooled
KV pages, chunked prompts that resume from adopted pages and cut at the radix cache's marks, windows, shared windows, commit,
cancellation gate across ranks), follow.zig (rank 0's Leader and rank 1's follow loop on the ncclBroadcast control link with
iface.zig's 64-byte header; a lost rank ends rank 0 with RankLost, a failed follower tells the others), backend.zig (the
lanes.backend vtable: stepped prefill, copy drafts only, no vision, radix keep and restore), draw.zig (rank 0's host draw from
the gathered fp32 logits, bounded candidate heap tested equal to lanes.gpu_rule.sample), engine_dev.zig (NCCL collectives,
arena and block table, fingerprint byte gather with ranks.agree, GLM's NCCL defaults, Engine.init), engine_ops.zig (the seam to
forward.zig and prefill.zig), native.open() completed (tp_watch, NCCL, engine, link, rooms, follower exit, Loaded), tests
engine_test.zig and backend_test.zig on recording fakes and core/lanes/fake.zig, step test-minimax-engine (additive hook in
zig/build/minimax_family.zig), docs/interface-decisions.md "T8b needs". Run on the build machine: test-minimax-engine,
test-minimax-ref/-sparse/-load/-pool/-tp/-ffn/-rows/-template/-engine-host, test-cuda-host, test-server-cpu,
check-native-cuda-host, compile-cuda-tests -Dsm=121, zig fmt and lean_check clean for the new files. No GPU run. Until
0010 is merged and 0011b applied, open() stops with ForwardNotBuilt after the ranks agree.

0011b-minimax-m3-engine-glue.patch (on a merge of mm3-t8a and mm3-t8b): forward_ops.zig, the adapter from engine_ops.Ops to
T8a's Forward.run(Call), the engine_ops switch (available() true) and the cuda.zig lines. Order: 0010, 0011, 0011b. Built and
tested on a scratch merge only (test-minimax-engine, -engine-host, -forward, check-native-cuda-host, compile-cuda-tests
-Dsm=121). No GPU run; T8a's missing fp16 index-projection kernel still stops the first forward.

0013-minimax-m3-rows-fix.patch (branch mm3-t8d, one commit on mm3-int; tests only, no kernel change): spark2's contract run failed tf_mm3_rows
decode ("v was not stored as it came" at 8x32x1) and gate 4 for both rows forms ("declares fp32 sums but its test simulates none").
Root cause of the first: a test bug. The decode window put positions (i * 7919) % k, so at k 1 all 8 rows sat at position 0 behind one
shared table and wrote one (page, slot); the CTAs race and one survives, so the earlier rows' v could not match. The kernel is correct.
Fix: each decode row owns its table and page (row_tab, the engine's shared-window path); the prompt chunk keeps one stream. The host
twin was blind because it runs rows in order (last wins, deterministic) and no case put two rows on one slot; a new test plants that
alias and the check must flag it. The second: the kernel does have fp32 sums (the norm's sum of squares), so the spec stays accum f32 and
Writer.simulate now runs those sums in the given accumulator and order for gate 4 (fp16/bf16 misses remain declared unprovable: integer
inputs keep fp16 sums of 128 squares exact). New command `tf-cuda-test mm3-contract`: act, select, attn, rows, router, proj and pick through
gates 1-4 (no controls, no limits). GPU run (spark2, GB10 sm_121), only `./tf-cuda-test-t8d mm3-contract`: SUMMARY mm3-contract 15 kernels:
15 pass gates 1-4, 0 fail; min MemAvailable 121.6 GB. Host: test-minimax-rows/-forward/-proj/-sparse/-ffn/-ref, test-kernel-contract,
test-exl3-cuda-host, check-native-cuda-host, compile-cuda-tests -Dsm=121 pass; zig fmt and lean_check clean for the touched files.

0015-minimax-m3-loader-speed.patch (branch mm3-loadspeed, one commit on mm3-int; stream.zig, tests, a bench tool and one build step): the
real TP2 start loaded in 1874 s because a rank plans 5.77 million byte runs (98% of them 4-16 KiB: column cuts of the routed trellises)
for 91 GiB and `Streamer` read each with its own O_DIRECT pread at queue depth 1 (plus 22,309 marker reads one at a time); the device
was idle, the request latency was the cost (0.32 ms a read over NFS). `Streamer.run` now plans the runs in file order into windows
(runs under 1 MiB apart share a read, at most 16 MiB), reads them on 6 threads into a ring of 8 buffers (128 MiB) while the calling
thread, the only one that touches the sink, copies finished windows in order, packs a cut tensor into few puts (at most 4 MiB), and
`verifyMarkers` checks every marker in one sorted pass with reads in flight, still before any byte is copied. Same interface, same
4 KiB alignment, same fail-closed errors; the old strategy stays as `runSerial` for the tests and the benchmark. Knobs without a rebuild:
TF_MM3_LOAD_THREADS, _CHUNK_MIB, _GAP_KIB, _PUT_MIB. Host only, no GPU. Measured on spark1's NVMe (O_DIRECT, live server running, real
plan slices): old 52-76 MiB/s, new 5.5-6.1 GB/s (about 75-110x; device ceiling 10.6 GB/s); 1 GiB strided 10 KiB-every-20 KiB pattern
42-45 MiB/s old, 4.9 GB/s new; the plan's reads go from 5.79 M to 15.6 k + 22.3 k markers, 1.64x over-read. NOT measured: NFS (spark2's
export is a root-only Docker volume, /mnt/spark1 is empty), the device copy path, a whole-rank load. Diagnosis, run-length tables and
numbers in docs/loader-speed.md. New command: `zig build mm3-loadbench` then `zig-out/bin/mm3-loadbench stats|model|pattern|raw`
(O_DIRECT only, buffers within 512 MiB, stops under 9 GiB MemAvailable, at most 8 GiB read per test). Tests: test-minimax-load (new
stream_test.zig compares every streamed byte with runSerial for world 1 and 2, 1-5 shards, chunk 16 KiB-16 MiB, 0-8 threads, gap 0-1 MiB),
-engine-host, -engine, -forward, -pool, -tp, check-native-cuda-host pass; zig fmt and lean_check clean for the touched files.

0014-minimax-m3-dspark-host.patch (branch mm3-dspark, one commit on mm3-int; NOT part of the 0001 series until merged into mm3-int and the series is regenerated): the host half of the base model's `dspark` speculative-decoding draft (the pack's `dspark/` folder: 5 layers, block 7, mxfp8 linears, bf16 fc/Markov/confidence). Files: dspark_config.zig (parse the draft's config.json, `admit` against the target with its own refusal words, byte arithmetic of weights, K/V ring and scratch, the MoE verify-window cost model), dspark_names.zig (the header's 97 tensors), dspark_load.zig (per-rank plan over load.Catalog/Item/Slot, strided runs for row cuts, mxfp8 codes and ue8m0 scales cut like the target's heads and dense MLP, nothing folded), dspark_ref.zig (float64 mxfp8 decode, taps, context state, the 5-layer sliding-window block, Markov chain, confidence, gate; every unverified semantic a named switch), dspark_twin.zig (K/V ring, two-rank block, per-rank candidate gather, each with planted mutants), dspark_test.zig and dspark_model_test.zig, two fixtures (the pack's config.json and safetensors header, data only), budget.zig (`Terms.draft`, `withDraft`, `draftBytes`, `Refusal.draft`), native.zig (`takes_drafter = false` and the draft-specific `--drafter` refusal), step test-minimax-dspark (zig/build/minimax_dspark.zig, hooked in minimax_family.zig and under test-cuda-host). Design, tensor map, memory budget (1.34 GiB of weights and about 2.03 GiB with one stream a rank), TP plan, work breakdown and the UNVERIFIED register: docs/dspark-plan.md. Written by Claude (Sonnet 5.5) for MiaAI-Lab; follows the semantics of TensorFold's DSpark/DFlash2 code (Ash Hart and contributors) without copying it. Run on the build machine: test-minimax-dspark (89 pass, 1 skip from an imported file), all other test-minimax-* steps, test-cuda-host, check-native-cuda-host, zig fmt and lean_check clean for the new files (lean_check total 636, the branch's baseline). `git apply --check` succeeds on mm3-int. NOT run: any GPU work, any kernel (none written), the draft's weights (only its header and about 150 KB of small tensors were read), the engine. The draft's semantics are UNVERIFIED (the upstream modeling code was unavailable): the tests prove the code agrees with itself, not with the trained model.

0016-minimax-m3-kv-fp8.patch (branch mm3-kv8, one commit on mm3-int c6996f2 (kv8 merged); design and rules in `../docs/fp8-kv.md`): `--kv-cache fp8` (`KV=fp8`) for
the family. K and V pages become e4m3 rows with one fp32 scale per token and kv head (rule of TensorFold's kv_fp8.cuh: scale amax / 448 floored at
2^-126, saturating nearest-even codes; a row with a NaN or infinity stores a NaN scale); the index keys stay bf16. A token takes 47,040 B a rank
instead of 76,800, a 128-token page 6,021,120 B instead of 9,830,400: the 784 pages of the 7.18 GiB rank budget become 1,280, the 766 of 7.02 GiB
become 1,251, and a 119 GiB host's largest context 225,920 instead of 138,880 tokens. Kernels: `tf_mm3_rows` (flags bit 64), `tf_mm3_attn` and
`tf_mm3_index` (trailing `kv8` parameter) take the mode as a launch parameter, so the bf16 paths and bytes are unchanged and the default; contract
records re-measured (cuobjdump sm_80..121). Zig: `Options.kv8` (the host's refusal passes it), `Dims.withKv8`, `PageMap` scales, `Paged.kv8`,
`Settings.kv8` in the ranks' fingerprint (a rank started without it stops the others), budget and layout follow `Dims`. Core: float64 `quantRow`,
`dequantRow`, `attendBlocksKv8` and the fp32 `quantRowF32` in minimax_ref.zig. New host step `test-minimax-kv8` (12 tests: codes, page tiling, bytes,
quantizer vs float64, window independence, NaN poison, attention twin vs float64, eleven planted mutants, budget and fingerprint). GPU contract
variants (not run): `tf-cuda-test mm3-contract` now has 19 kernels, adding fp8 `tf_mm3_rows` decode and prompt (v bit for bit, k within e4m3's half step)
and fp8 `tf_mm3_attn` decode and prompt. First GPU run found two test faults (the NaN plants poisoned every output; gate 4's integer stress did not fit e4m3), fixed in the second commit: e4m3-exact stress, plants on the last rows' own keys. Result on spark1 and spark2: `SUMMARY mm3-contract 19 kernels: 19 pass gates 1-4, 0 fail, 0 known findings`.
Host, on mm3-int at 2efd42b (after the dspark merge): every test-minimax-* step (incl. -kv8, -dspark, -engine, -forward), test-kernel-contract,
test-exl3-cuda-host, test-cuda-host, check-native-cuda-host and compile-cuda-tests -Dsm=121 pass; the `-Dnvcc -Dsm=121` install (all fatbins)
built before the rebase; zig fmt and lean_new clean. NO GPU RUN. The effect of fp8 K/V on the model's answers is NOT MEASURED
(host numbers on random N(0,1) K/V only: attention output within 5.0 % of its largest element, 3.8 % relative rms of bf16 K/V). Run on the owner's
session with the server stopped: `zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 install && ./zig-out/bin/tf-cuda-test mm3-contract`.

0017-minimax-m3-kv-fp8-contract-fix.patch (on mm3-int with 0016 merged, b0fbfd7; 0016 above is regenerated as the whole feature, fix included, relative to 2efd42b): contract-test fix only. Spark1 and spark2 (GB10), `tf-cuda-test mm3-contract` built with -Dnvcc -Dsm=121 -Doptimize=ReleaseSafe: `SUMMARY mm3-contract 19 kernels: 19 pass gates 1-4, 0 fail, 0 known findings` on both (MemAvailable at start 109 GiB and 116.5 GiB; not sampled during the run).

0017-minimax-m3-dspark-device.patch (branch mm3-dspark-dev, one commit on mm3-int c6996f2 (kv8 merged); design and the first GPU check in `../docs/dspark-plan.md` sections 13 and 14): the device half of the dspark draft. Forward: `Op.tap` after layers 3, 17, 31, 45 and 59 copies the residual rows as bf16 into a taps buffer (a prompt chunk at rows 0-2047, each stream's last window and prompt end at its own 16 rows). New kernels `minimax_m3_dspark.cu` (tap, add, norm, block heads with norm and RoPE, ring put, block attention over a 4112-row ring with a flash-decoding merge, top-64 candidates, Markov chain and confidence) with contract writers, launch records (cuobjdump sm_80..121) and host twins with planted mutants. `dspark_device.zig`/`dspark_weights.zig`: the 35 mxfp8 matrices and `fc` repacked for the lane matmul, per-stream K/V rings, absorb, the 5-layer block pass with rank-order sums, the target's embedding and head, 8 ring snapshots kept with prompt-cache marks. Engine: follower kinds `absorb`, `draft`, `snap`; lanes backend `prepare_features`/`features`/`prefill_features`; `dspark_drafter.zig` behind the core's `Drafted`; `--drafter` (value `dspark` = the checkpoint's `dspark/` folder, or a folder), budget (`native.draftCost`), the draft's digest in the ranks' fingerprint. Recipe (outside the patch): `DRAFTER`, README, `scripts/check-drafted.sh`. Steps: test-minimax-dspark-dev (drafted decode equals plain decode on the engine fakes: greedy, sampled, shared rounds, a wrong draft, rank-1 mirror, prompt-cache resume with and without the ring snapshot), test-minimax-dspark-kernels; `tf-cuda-test mm3-dspark-contract` (14 kernels, gates 1-4 passed on spark1 and spark2), `mm3-dspark-bench`, `mm3-dspark-pass [DIR]` (synthetic 2-layer draft against float64 on spark1: ring within 2.4e-3, block rows within 8.1e-3 relative L2). Host: every test-minimax-* step, test-kernel-contract, test-exl3-cuda-host, test-cuda-host, check-native-cuda-host and compile-cuda-tests -Dsm=121 pass; zig fmt and lean_check clean for the touched files. NOT RUN: the real weights, two ranks, the head and chain on the GPU in the engine, anything through the server; acceptance and speed are NOT MEASURED. Found, not fixed: the forward's head writes rank 1's logits rows at stride w.n but reads them at stride slot (dspark-plan.md section 13). Rebased on kv8 (Options and Fingerprint merged by hand); the kv8 step and all the host steps above pass on the result.
