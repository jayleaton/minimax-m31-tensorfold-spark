# FP8 KV cache for the MiniMax-M3.1-Flash family (`--kv-cache fp8`)

Status: written and tested on the host only (float64 reference, fp32 host twins with planted mutants, page geometry, budget).
**Only the contract tests have run on a GPU (section 8)** and the effect of fp8 K/V on this model's answers is **not measured**. Section 9 lists what cannot
be tested here.

## 1. What changes

`--kv-cache fp8` is accepted by the MiniMax family (`Options.kv8`, so `cuda_refusal.zig` no longer refuses it) and the K and V
pages become e4m3 rows. Without the flag nothing changes: the bf16 layout, kernels' bf16 paths and every existing test are as
before (the new modes are launch parameters of the same kernels, `bf16` is the default).

| Tensor | bf16 (default) | `--kv-cache fp8` |
| --- | --- | --- |
| K, per token and kv head: 128 values | bf16, 256 B | e4m3, 128 B, one fp32 scale (4 B) |
| V, per token and kv head: 128 values | bf16, 256 B | e4m3, 128 B, one fp32 scale (4 B) |
| IK, per token (shared by the kv heads): 128 values | bf16, 256 B | **bf16, 256 B (unchanged)** |

IK stays at its stored precision on purpose. The indexer is the part of the model whose output (block maxima, then the top-16
block choice) is a hard selection: a rounding that moves a score across a tie or a cut changes which keys attention reads at all,
not just by how much it weights them. The model card keeps the indexer at source precision (the owner's reading of it; this
tree does not re-derive it), and `docs/speed-borrow-glm.md` C10 already ruled FP8 IK out without a quality study. IK is 256 B of
the 784 B a token takes per layer, so keeping it costs 33 % of the fp8 page. (The brief calls IK "fp16": the index projections'
outputs are fp16, but `tf_mm3_rows` has always written IK to the page as bf16 after the norm and rotation, 256 B a token, and
that is what stays.)

## 2. What was reused, and a correction to the brief

The rule is the one TensorFold already ships for its fp8 caches: `zig/kernels/cuda/kv_fp8.cuh` (`quant_row<WIDTH, FINITE_SCALE>`,
used by the Qwen3.5 `qwen35_kv8.cu` and mirrored for GLM by `zig/src/families/glm/cuda_kv8_ref.zig`): a row's scale is
`amax / 448` in fp32, the codes are the saturating nearest-even e4m3 of `x * (448 / amax)`, and the scale is floored at 2^-126
(`FINITE_SCALE`). **GLM's scale is not a power of two** (the brief said "power-of-two scale each"): `glm/cuda_kv8_ref.zig` and
`kv_fp8.cuh` both use `amax / 448`. A power-of-two scale would waste up to a factor 2 of the e4m3 range on about half the rows
(the row maximum would land anywhere in 224..448), losing up to one bit of precision, in exchange for scales that dequantize
exactly. That trade is not worth it here, so this family follows the shipped rule.

`kv_fp8.cuh`'s `quant_row` assumes one thread a value and a block per row. The MiniMax writer holds a head in one warp (4 values a
lane), so `quant_row8` in `minimax_m3_rows.cuh` is a warp form of the same rule (max by xor tree, which is order-free). The
attention reader is new: Qwen's reads one key a warp; this one keeps T5's CTA shape and tile order. Nothing is taken from the
banned accounts; the TensorFold kv8 files are inside TensorFold's history.

## 3. Scale granularity and where the scales live

One fp32 scale **per token, per kv head, for K and for V** (a "row" is the 128 values of one token and one kv head). Reasons:
(a) a row's bits then depend on that row's 128 values alone, which is the exactness requirement of section 6; a coarser scale
(per page, per head) would make early rows depend on later ones, or force a rewrite of a page's rows as it fills; (b) finer
(per 32 or per 64 values) would cost 8 or 16 scale bytes a row for a modest precision gain: per-row already costs 3 % over the
plain codes (4 B per 128 B).

Layer region inside a page (per layer, `kvh` local kv heads, 128 tokens), the same for both ranks:

```
K  e4m3  [kvh][128 tok][128]            kvh * 16384 B
V  e4m3  [kvh][128 tok][128]            kvh * 16384 B
IK bf16  [128 tok][128]                 32768 B
Kscale fp32 [kvh][128 tok]              kvh * 512 B
Vscale fp32 [kvh][128 tok]              kvh * 512 B
```

The scales sit after the IK block so K, V and IK keep the 128-byte-aligned positions they have in the bf16 layout, K/V rows are
128 B (`uint4` loads), and the index kernel finds IK at `2 * kvh * 16384` B instead of `2 * kvh * 32768` (one launch parameter,
`kv8`). `geom.Pool` (`minimax_m3_geom.zig`), `pool.zig` (`Dims.kv8`, `PageMap.kScale/vScale`) and the kernels' `kv8_byte`,
`ik_byte`, `scale8_byte` (`minimax_m3_sparse.cuh`) are one layout, tested to tile a page with no gap or overlap.

## 4. Bytes and capacity

For one rank (2 of 4 kv heads, 60 layers):

| | bf16 | fp8 K/V |
| --- | ---: | ---: |
| per token and layer | 1,280 B | 784 B (K 256 + V 256 + IK 256 + 4 scales of 4 B) |
| per token, 60 layers | 76,800 B | **47,040 B** (46,080 before the scales) |
| 128-token page | 9,830,400 B | **6,021,120 B** |
| ratio | 1 | 0.6125 (1.633x the tokens in the same memory) |

(One rank holding all 4 kv heads: 138,240 B a token with bf16, 78,720 B with fp8.)

Pages the pool gets from the same room (`kv8_test.zig`, whole pages; the room is the pool's bytes, not the exact byte count the
host had, so the fp8 count can be one page higher or lower than a real run's):

| KV room | bf16 pages (tokens) | fp8 pages (tokens) |
| --- | --- | --- |
| 7.18 GiB (the rank budget of the 2026 start log: 784 pages) | 784 (100,352) | **1,280 (163,840)** |
| 7.02 GiB | 766 (98,048) | **1,251 (160,128)** |

784 bf16 pages are exactly 1,280 fp8 pages of memory. `budget.zig` is driven by `Dims`, so a refusal's "bytes a token" and the
largest context follow: a 119 GiB host with TensorFold's reserve (the test in `pool_test.zig` that gave 138,880 tokens) gets
**225,920** tokens with fp8 K/V. More pages do not raise `CONTEXT` by themselves: they raise the pool, the number of streams that
fit and the prompt cache (the radix store keeps page references).

## 5. Rounding, saturation and NaN rules (the exact rule, shared by writer, twins and reference)

For a row `x[0..128]` of fp32 values (V: its bf16 value widened; K: the rotated value rounded to bf16 first, which is what the
bf16 cache stores):

1. If any value is NaN or +-infinity: the row's scale is NaN and its codes are zero. Every value it dequantizes to is NaN.
2. `amax = max |x|` (exact in any order).
3. `amax == 0`: scale 1, codes zero (a negative zero keeps its sign code 0x80).
4. Else `inv = fp32(448 / amax)`, `scale = fp32(amax / 448)` (IEEE division); if `scale < 2^-126` then `scale = 2^-126` and
   `inv = 1 / scale` (a denormal row cannot overflow `inv`).
5. `code = e4m3(fp32(x * inv))` with `cvt.rn.satfinite`: nearest even, saturating at +-448 (never NaN from a finite value).

The float64 reference (`core/minimax_ref.zig`, `quantRow`) divides in float64; the two forms disagree by one code where `x /
scale` is an exact tie. bf16 inputs make those exact ties common: `x * inv` rounds in fp32 and breaks them the other way for about
4 in 10,000 values (the test `the fp32 twin follows the float64 reference` allows 1 in 1,000 and checks each is a neighbouring
code). Both are deterministic; the kernel's form is the specification.

NaN handling differs from bf16 in one visible way: a NaN or infinity in a bf16 K row makes that key's scores NaN (every head);
in a bf16 V row it makes only that element of the output NaN. With fp8 the poisoned V row turns **all 128** output dimensions of
every head NaN (its scale is NaN). That is a wider but equally loud failure, and a NaN in K/V means the forward is already broken.

## 6. Exactness: a row's bits do not depend on the window

Quantization happens once, at write time, in `tf_mm3_rows`, one warp per (row, head), from that row's 128 values only: the max is
a xor tree over max (order-free), the any-vote is order-free, and each code is a function of `x`, `inv`. There is no
cross-row, cross-window or cross-chunk term, so a prompt chunk of 2,048 rows, a verify window of 8 or one row write the same
bytes (a planted `batch_amax` mutant, a scale taken over the window, is caught by the host test). Restoring a kept prompt reuses
the pages as written, so a resumed stream reads the same bytes it would have read after a fresh prefill.

## 7. Kernels

* `minimax_m3_rows.cu` (`tf_mm3_rows`): flag bit 64 (`ROWS_KV8`, `Flags.kv8`). Off: the code path and the bytes are the old ones
  (a bf16 page written before and after is identical). On: K goes through `quant_row8` after the bf16 rounding, V through it from
  its bf16 value, and IK is written at the fp8 layout's offset. `qkRope` refuses a launch whose flag disagrees with `Paged.kv8`.
* `minimax_m3_attn.cu` (`tf_mm3_attn`): trailing parameter `kv8`. Same CTA (row, kv head), same 32-key tiles and order, same
  online softmax and fp32 chains. K tile: 32 rows of codes (33-word stride) plus 32 scales in shared memory (reusing the bf16
  tiles' bytes, so the static shared size is unchanged). A score is `((sequential fp32 dot of q and the codes) * key scale) *
  1/sqrt(128)`; the e4m3 -> fp32 widening is exact. V: each weight is `p * value scale`, then `o += (p * scale) * code` in key
  order, while the softmax sum takes `p` itself. "Dequant on read, fp32 accumulate" is realised by applying the scale once per
  key to the dot and once per weight: it reassociates two products against dequantizing first and is the documented order of the
  twin. A NaN-scaled row makes its score or its value row NaN.
* `minimax_m3_index.cu` (`tf_mm3_index`): trailing parameter `kv8`; only where IK sits changes. IK is read bf16 as before.
* `Paged.kv8` / `Cache.kv8` carry the mode to all three launches from one place (`forward_exec.zig` `paged()`, from `Pages.cache`);
  `forward.zig` sets the writer's flag from the same value.

The `mm3` contract records are per symbol and the same symbols serve both layouts, so there is one record for each of
`tf_mm3_rows`, `_index`, `_attn`; they were re-measured with `cuobjdump` (sm_80/86/89/120/121) after the change (rows 49/52/48/48/48
registers, attn 58/77/77/76/76, index 32/38/38/40/40; static shared unchanged). The contract tests got fp8 variants (section 8).

## 8. Plumbing and tests

* `native.zig`: `Options.kv8`; `cuda_options.set` already forwards `o.kv8` when the field exists, and `cuda_refusal.family`
  accepts it (test in `native/cuda.zig`). The host's pool text names "fp8 K/V".
* `ranks.zig`: `Settings.kv8` is set by `open` from the option and is in the rank fingerprint (`Field.kv8`), so a rank started
  without `--kv-cache fp8` stops the others with a named reason. `engine_dev.zig`, `forward_ops.zig`, `budget`/`admission` and the
  layout all use `Dims.withKv8`.
* `scripts/config.sh` already passes `--kv-cache "$KV"` when `KV` is set (`start.sh`); no script change was needed, only comments.
* Host step `test-minimax-kv8` (additive hook in `zig/build/minimax_family.zig`): e4m3 codes (every code round-trips, ties,
  saturation, NaN), page bytes and tiling, quantizer vs float64 reference vs fp32 twin, window independence, NaN/inf poison, corner
  rows, attention twin vs float64 (reading fp8 K/V), planted mutants for each rule (truncation, ties-up, wrong maximum, ignored
  NaN, missing scale floor, window-wide amax, unrounded K, missing or misplaced K/V scale, window-dependent tile), the family's
  page map, budget and the fingerprint.
* GPU contract (`tf-cuda-test mm3-contract`): four new variants, `tf_mm3_rows` decode and prompt chunk writing the fp8 pool
  (V checked bit for bit against the host rule, K against the float64 norm and rotation within e4m3's half step), and `tf_mm3_attn`
  decode and prompt chunk reading an fp8 pool built from the same bf16 data (held to attention_ref's bound over the decoded rows,
  NaN-poisoned unwritten rows and the NaN/inf plants of gate 1). Gate 4's fp8 stress input uses e4m3-exact values (table values mirrored over the keys, integer pairs 9..16 for the 1 residue, and a
+-448 in every row so its scale is exactly 1); the NaN/inf plants of gate 1 sit on the last rows' own keys, because a bad value poisons its
whole row. GPU run (spark1 and spark2, GB10): 19 of 19 kernels pass gates 1-4, 0 known findings; the fp8 attn variants show fp16 sums
would miss by 608x/512x and bf16 by 256x/3844x. The index kernel's `kv8` offset
  has no contract test (there is none for the bf16 index kernel either); the layout tests cover where it reads.

## 9. What is NOT known and cannot be tested here

* **Model quality.** Nothing measures what fp8 K/V does to MiniMax-M3.1-Flash's answers: not perplexity, not the model card's
  KL/top-1 scoring, not long context. Host numbers only show the format: on random N(0,1) K/V (six positions up to 2,049) the
  fp8 attention output differs from bf16 K/V by at most 5.0 % of the output's largest element, 3.8 % relative rms, which is what
  3 mantissa bits give and says nothing about a real model's sensitivity. The checkpoint was scored at 2,048 tokens at most; the
  context a bigger pool buys is exactly where nothing has been tested.
* **The kernels have not run.** They compile for sm_80..121 (cubin/fatbin built), the host twins and the contract tests are
  written, and a GPU run is outstanding. Expect to find what a first run finds: shared-memory bank behaviour, register use, and
  whether `cvt.f16x2.e4m3x2` is as fast on sm_121 as assumed.
* **Speed.** fp8 reads half the K/V bytes and adds a widen per value; decode attention is memory-bound so it may gain, prefill's
  writer does a little more work per row. Not measured.
* **Sparse selection is untouched** (IK is bf16), so the same blocks are selected as with bf16 K/V given the same queries; but q
  is computed from the previous layers' outputs, which fp8 K/V perturbs, so selections may still differ downstream.

## 10. Running the GPU contract (when the server is stopped)

```
cd tf-wt-kv8
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 install      # builds zig-out/bin/tf-cuda-test with the sm_121 fatbins
./zig-out/bin/tf-cuda-test mm3-contract                          # 19 kernels (15 before + 4 fp8 variants), all passing
```

Only that command, and only with the server stopped (it allocates its own test buffers; the contract tests of the family are small, but the GPU here is shared with the host). A mismatch between the writer and
reader layouts shows as the fp8 `tf_mm3_rows` "v was not stored as the e4m3 row of its bf16 value" or the fp8 `tf_mm3_attn`
gate 1/3 failing at every shape.
