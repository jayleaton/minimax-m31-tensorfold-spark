# Rank contract: how the two Sparks are told apart, what they exchange, what they must agree on

Status: written with plan task T7 (patch `patches/0007-minimax-m3-tp.patch`, files `zig/src/families/minimax_m3/ranks.zig` and
`tp.zig`, host step `zig build test-minimax-tp`). Everything below that says "read" was read in the TensorFold v1.1.2 tree
(`~/TensorFold/tensorfold-1.1.2`); the GPU behavior is **not** run (no GPU on the build machine).

## 1. What v1.1.2 does (read)

| Family | Rank source | Rendezvous | Reads TF_TP_*? |
| --- | --- | --- | --- |
| GLM (`families/glm/cuda_native.zig` `place()`) | `TF_GLM_RANK`, `TF_GLM_WORLD` (both required; world 1..16) | `NCCL_COMM_ID` (`ncclGetUniqueId` reads it; `cuda/tp_group.zig` `Group.open` refuses without it) | no |
| Nemotron (`cuda_latent.zig`) | `TF_NEMOTRON_RANK`, `TF_NEMOTRON_WORLD` | `NCCL_COMM_ID` | no |
| DeepSeek V4.1 (`deepseek_v41/cuda_model.zig` `tp.Config.fromEnv`) | `TF_TP_RANK`, `TF_TP_WORLD`, `TF_TP_DEVICE`, `TF_TP_MASTER`, `TF_TP_PORT` (`fabric/tp/config.zig`) | its own TCP bootstrap on `TF_TP_MASTER:TF_TP_PORT` | yes, it is the only one |
| `cuda_ranks.zig` (GLM) | not a rank source: the fingerprint all-gathered over NCCL before weights load | | |

The CUDA `serve` of v1.1.2 lists no `--tp`, `--rank`, `--master` flag for any of them. `TF_TP_*` belongs to `fabric/tp`, which only
DeepSeek V4.1 opens. MiniMax-M3.1-Flash is built on GLM's shell (`cuda/tp_group.zig`, `Group.allGather`, `openRoce`), so it takes GLM's
mechanism with its own names. Nothing in the MiniMax family reads `TF_TP_*`.

## 2. The contract for `minimax_m3_vl`

Every rank (the recipe starts the same `tensorfold-native serve` command on both Sparks) must have:

| Variable | Value | Read by | Notes |
| --- | --- | --- | --- |
| `TF_MM3_RANK` | `0` on the head, `1` on the worker | `ranks.place` | required; a rank outside its world is refused (`NoRank`) |
| `TF_MM3_WORLD` | `2` (1 for a one-GPU thin-model check) | `ranks.place` | required; at most 4 (4 kv heads) |
| `NCCL_COMM_ID` | `<rank 0 link address>:<port>`, IPv4 only, same text on both ranks, port 1..65534 | `tp_group.Group.open`, `cuda/tp_watch.zig` | the liveness line listens on **port + 1** on rank 0, so port + 1 must be free and the port must not be 65535 |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` | the link's netdev, both rails, RoCE v2 GID | NCCL | `nodes.sh` already derives them (`NCCL_ARGS`) |
| `TF_ROCE_HCA` (optional) | `name,name` (no `:gid`) | `cuda/roce.zig` | the small-gather RoCE path takes `TF_ROCE_HCA`, else `NCCL_IB_HCA` (colon suffixes stripped); so setting `NCCL_IB_HCA` alone is enough |
| `TF_ROCE_GID_INDEX` (optional) | RoCE v2 IPv4 GID index | `roce.zig` | else scanned; `nodes.sh` knows it per node (`NODE_GID`) |
| `TF_ROCE` (optional) | `0` keeps every gather on NCCL | `roce.wanted` | **both ranks must agree**; not in the fingerprint yet (open item) |
| `TF_ROCE_MAX_KB` (optional) | cap in KiB, default 4096 | engine `open` (as GLM) | |
| `TF_MM3_LAYERS` (optional) | first N layers only | `ranks.Settings` | in the fingerprint; thin model for one-GPU checks |
| `HF_HUB_OFFLINE` | as now | | unchanged |

The engine sets `NCCL_RUNTIME_CONNECT=0`, `NCCL_CUMEM_HOST_ENABLE=0`, `NCCL_CUMEM_ENABLE=0` itself with `overwrite = 0` when unset (GLM does;
T8/T9 must copy that). The recipe need not set them.

`TF_TP_*` and `TF_COMM_BACKEND` are not read by this family; exporting them is harmless but must not be relied on. `COMM=roce` in `config.sh`
does nothing here: the RoCE small gather is chosen by `TF_ROCE`, not `TF_COMM_BACKEND`.

### What the engine does with them (T8/T9, not in this patch)

1. `ranks.admit(&ranks.environment)` before touching CUDA; on error print `ranks.refusal(err)` (text names the variables).
2. `Group.open(rank, world)`, `cuda.tp_watch.start(io, rank, world)` (liveness lines first, as GLM).
3. Before any weight loads: build `ranks.Fingerprint.of(place, settings, capacity, sha256(config.json), sha256(quantization_config.json),
   sha256(own executable), checkpoint digest)`, all-gather it with the byte-gather of `glm/cuda_ranks.zig` (`gather`, one `ncclAllGather` of `u8`),
   and call `ranks.agree`. A mismatch prints `REFUSE rank settings: rank N's <field> differs from rank 0's (<why>)` and returns `error.RankMismatch`
   on every rank. `tp_watch.ready(io, checkpoint_digest)` already carries the checkpoint digest to rank 0 as GLM does; the fingerprint's `weights`
   field is the same value, so the check is double.
4. Rank 0 serves HTTP and lanes; rank 1 follows (`glm/cuda_native.zig` pattern). Nothing in the script should give rank 1 an HTTP port it uses.

## 3. Reconciling `scripts/nodes.sh` (not edited; proposal)

Today `rank_env` (nodes.sh lines 454-475) prints `TF_TP_WORLD/RANK/MASTER/PORT/DEVICE`, `TF_TP_ROCE_HCA`, `TF_COMM_BACKEND`, then copies every exported
`TENSORFOLD_*`, `TF_TP_*`, `TF_COMM_*`, `TF_CUDA_*`, then `EXTRA_ENV`. Gaps against section 2: no `TF_MM3_RANK/WORLD`, no `NCCL_COMM_ID`, and the exported-variable
allowlist would drop `TF_MM3_LAYERS`, `TF_ROCE*`. `EXTRA_ENV` is **not** a substitute for the rank-specific values: it is one string for both ranks, and rank
differs. The values in it are limited to `A-Z a-z 0-9 _ . , : = / -`, which `192.0.2.1:29741` satisfies.

Proposed edit to `rank_env` (replace the first `printf` and extend the allowlist; keep the `TF_TP_*` words only if a later build of the family wants them, they cost nothing):

```bash
rank_env() {
  local r=$1 x name val master=${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}
  printf '%s\n' "TF_MM3_WORLD=$TP" "TF_MM3_RANK=$r" "NCCL_COMM_ID=$master:$MASTER_PORT" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  [[ -z "${NODE_GID[$r]:-}" ]] || printf '%s\n' "TF_ROCE_GID_INDEX=${NODE_GID[$r]}"
  # no TF_ROCE_HCA: roce.zig falls back to NCCL_IB_HCA, which NCCL_ARGS already sets from the same link scan
  while IFS= read -r name; do
    [[ "$name" =~ ^(TENSORFOLD|TF_MM3|TF_ROCE|TF_CUDA)_[A-Za-z0-9_]+$ ]] || continue
    case "$name" in TF_MM3_WORLD|TF_MM3_RANK|TF_ROCE_GID_INDEX) continue ;; esac
    ...unchanged value check and printf...
  done < <(compgen -e | LC_ALL=C sort)
  ...EXTRA_ENV loop unchanged...
}
```

Also (all in the same files, listed so nothing is forgotten):

- `config.sh` lines 37 and 109-114: say `MASTER_PORT` is the `NCCL_COMM_ID` port and that `MASTER_PORT + 1` must be free; replace the `COMM` comment (`TF_COMM_BACKEND` is read only by DeepSeek V4.1) or drop `COMM` for this recipe; update the `EXTRA_ENV` comment's allowlist (`TENSORFOLD_*`, `TF_MM3_*`, `TF_ROCE_*`, `TF_CUDA_*`).
- `start.sh` lines 155-161: the message "told apart by TF_TP_* environment alone" becomes "by TF_MM3_RANK/TF_MM3_WORLD and NCCL_COMM_ID (scripts/nodes.sh, rank_env)". The `rank_flags` branch stays dead (v1.1.2 has no such flags) and could be deleted.
- Add a preflight check in `start.sh`: `MASTER_PORT` in 1..65533 and `ss -ltn` shows neither it nor `MASTER_PORT + 1` taken on the head.
- `roce_hca_list` and `TF_TP_ROCE_HCA` become unused for this family.

The pre-flight value check in `rank_env` should also reject `TF_MM3_LAYERS` that is not a positive integer (the engine refuses it anyway, `ranks.Settings.read`).

## 4. Per-layer collective plan (host-checked in `tp.zig`)

Two exchanges a layer (attention, FFN), 120 a token, each moving `rows x 6144` fp32 per rank (24 KB at one row). Derived from the cut axis of each projection
(`names.Proj.cut`): a row cut leaves an fp32 partial over the rank's input rows, a column cut leaves whole output columns.

| Tensor | Cut | Collective |
| --- | --- | --- |
| q_proj, k_proj, v_proj | columns, whole kv groups (rank r: kv heads {2r, 2r+1}, q heads 32r..32r+32) | none |
| q/k/index norms, router (fp32 gate), index_k_proj, embedding | replicated | none (same kernel, same bits on both ranks) |
| index_q_proj (fp16) | rows of the rank's 2 index heads; they select for the same kv heads, score over the full IK cache | none |
| index scores, block max, top-16, selected-block attention | per kv head, entirely rank-local | none |
| **o_proj** | input rows 4096 per rank | fp32 partial, rank-order sum (attention exchange) |
| dense gate/up (layers 0-2) | columns | none |
| **dense down** | input rows | joins the FFN exchange |
| routed w1/w3, shared gate/up | columns (1536 each) | none; routed combine and shared add are rank-local |
| **routed w2 and shared down** | input rows 1536 | routed combine (fp32, rank-local) plus shared are added into **one** partial; one exchange per layer |
| **lm_head** | 128-row units, 782 / 781 | gather of logits into slots padded to 782 units (`LogitsGather`); or, for greedy, one `(logit, id)` candidate a rank merged lowest id first (`mergePicks`, proven equal to a whole-vocabulary argmax on ties) |

## 5. The deterministic reduction

Rule: gather both ranks' fp32 partials, add `t0 + t1 (+ t2 ...)` in rank order with one rounded fp32 add each, so every rank holds identical bits; only then add the residual.

What the two existing pieces actually do (read from source; this replaces the plan's UNVERIFIED):

`kernels/cuda/rank_sum.cu`, `tf_rank_sum(peers, own, world, me, n, out)`: fp32 only. `peers` is `world` slots of `n` floats (slot r at `r*n`); slot `me` is never read. `own` is this rank's `n`
floats. `out[i]` = `t_0`, then `acc = __fadd_rn(acc, t_r)` for r = 1..world-1 where `t_r = own[i]` if `r == me` else the peer slot. No fma, no reassociation, one thread per index
(grid-stride, 256 threads), so `out` may alias `own`. It is the kernel for the all-gather form: `Group.allGather` the whole partial, then `peers` = the gathered buffer.

`cuda/rank_scatter.zig`, `RankScatter`: a **reduce-scatter**, not an all-reduce. `share(rows, world, rank)` = whole rows, the first `rows % world` ranks one more. `run(...)` posts one
`ncclSend`/`ncclRecv` group (each rank sends peer p the rows of p's share, receives its own share from every peer into `scratch` slot r at `r*n`), then runs `tf_rank_sum` on its rows.
`scratchBytes = world * ceil(rows/world) * width * 4`. Output is only this rank's rows, so a second all-gather (slots padded to the larger share) restores the whole tensor on both ranks.
It needs the fatbin (`kernels.available`).

Plan: `tp.formFor(rows)`: up to 128 rows (decode, verify windows) **gather_sum** (one `allGather` of `rows*6144*4` bytes, then `tf_rank_sum` over all rows); above that (prompt chunks)
**scatter_gather** (1/world of the bytes). Both give the bits of the host `rankSum`; the test proves the two forms and an in-place `out == own` agree bitwise on random fp32 data.

Honest finding the tests encode: for **two** ranks `a + b == b + a` in IEEE arithmetic, so a plain `own + peer` on each rank is already bit-identical. The rank order matters for what
is added around it: if a rank folds its own partial into the residual first, `(h + own) + peer`, the ranks disagree (h = 2^24, partials 1 and 2: rank 0 gets 16777218, rank 1 gets
16777220, the ordered sum gives 16777220 on both). For three or more ranks the order of the partials themselves matters (test: 2^24, 1, 1). ncclAllReduce/ReduceScatter give no
such order promise, which is why neither is used. The partials must stay fp32 until after the sum (no bf16 rounding before the exchange).

## 6. Open items

- GPU: `tf_rank_sum`, `RankScatter` and the padded logits gather have not been run for these shapes (hidden 6144, row cut K=8192 mul1 6-bit is new, plan risk 11); `tf-cuda-test exl3-tp2` needs the extension named in T7.
- `TF_ROCE` (and `TF_ROCE_MAX_KB`) must be equal on both ranks; add them to the fingerprint when the engine reads them (they are not `Settings` yet).
- The 128-row switch between the two forms is a guess to be measured; the exchange latency (20-40 us) is still the plan's UNVERIFIED estimate.
- The follower loop (rank 1 serves nothing) and the `NCCL_*` defaults belong to T8/T9.
- The byte gather for the fingerprint is `glm/cuda_ranks.zig`'s `gather`; it needs `cuda` and is wired by the engine, `ranks.agree` takes it as a function so the check stays host-testable.
