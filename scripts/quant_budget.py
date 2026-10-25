#!/usr/bin/env python3
"""EXL3 pack sizes and the per-node memory budget for MiniMax-M3.1-Flash on 2x DGX Spark (TP=2).

    python3 scripts/quant_budget.py                  every table (markdown)
    python3 scripts/quant_budget.py --image-gib 30   superseded image-model tables at one size (default: none)
    python3 scripts/quant_budget.py --serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8
                                                     one serving profile as the engine prices it, and its MEM_GATE_GIB
    python3 scripts/quant_budget.py --serve --context 1048576 --parallel 4 --kv fp8 --reserve-gib 8 --kv-tier
                                                     the same with the KV tier (pages on NVMe, TF_MM3_KV_TIER)
    python3 scripts/quant_budget.py --serve ... --drafter dflash2
                                                     with the DFlash2 draft (DRAFTS=dflash2); --drafter alone is dspark

The --serve mode follows the minimax_m3_vl family's own budget (patches/0001, budget.zig `tensorfold()`): the rank's
weights, the reserve (TENSORFOLD_MEMORY_RESERVE_GIB; unset: a tenth of MemTotal, at least 4 GiB), 4.5 GiB of CUDA
context, NCCL and graphs, 1.5 GiB of prompt workspaces, the dspark draft when it is on (1.34 GiB of weights, 2.03 GiB
with one stream, plus its 8 ring snapshots for prompt-cache resumes, 0.16 GiB: native.draftCost, docs/dspark-plan.md)
or the DFlash2 draft (2.37 GiB of weights, 0.71 GiB of runtime, scratch and ring snapshots, 21.1 MB a stream:
dflash2_cost(), the engine's dflash2_config.cost), and the KV. The language model and the draft are split over both
ranks, so a rank holds half the split weights plus the replicated ones. With image input on (VISION=1, the engine branch
mm3/vision; --vision-image-tokens here) the engine's admission (native.zig admissionWith) adds rank 0's vision tower and
its scratch (2.27 GiB) and every rank's image rows (PARALLEL x VISION_IMAGE_TOKENS x 6144 x 2 bytes) to the weights term,
so they come out of the KV pool, not out of the load's MemFree: the host admits by the smaller rank's room, so both ranks
size the pool to rank 0's. MEM_GATE_GIB stays the text-only value; the KV pool (tokens, 128-token pages) shrinks, and
with --kv-tier the slice cache does.

Primary target (2026-10-23): one ~200 GB pack, LLM only (no image model on the Sparks), routed experts at
~3.56 bpw average; the extra bits over the 160-180 GB plan all go to the routed experts (FIXED_BITS unchanged).

Arithmetic only: no GPU, network or download. The tensor classes and element counts are the release's own
(MiniMaxAI/MiniMax-M3.1-Flash at 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e, model.safetensors.index.json plus the
safetensors headers); `release_bytes()` reproduces the index's total_size to the byte. GB = 10^9 bytes, GiB = 2^30.

Node memory inputs, both measured on GB10 in our own runs and kept as named constants so they can be revised:
IDLE_AVAIL_GIB is MemAvailable on an idle Spark (our own measurement), FLOOR_GIB and DIP_GIB are the
DeepSeek-V4.1 recipe's hard floor and serve-time dip (its docs/FLASH-1017.md, "Memory"). CUDA_GIB and PREFILL_GIB are estimates.
"""

import argparse

GB = 1e9
GIB = 2**30

H, I_EXP, I_DENSE, VOCAB = 6144, 3072, 12288, 200064
LAYERS, MOE_LAYERS, DENSE_LAYERS, EXPERTS = 60, 57, 3, 128
Q_OUT, KV_OUT, IDX_Q_OUT, IDX_K_OUT = 64 * 128, 4 * 128, 4 * 128, 128

IDLE_AVAIL_GIB = 117.1
FLOOR_GIB = 4.0
DIP_GIB = 3.0
CUDA_GIB = 1.5
PREFILL_GIB = 2.5
SLACK_GIB = 2.0     # margin we require on top of FLOOR_GIB + DIP_GIB at the chosen context ("safe")

PRIMARY_GB = 200
TARGETS = (190, 195, 200)

# Retained cache bytes a token on one rank: K+V of 2 of the 4 KV heads on 60 layers, plus the one 128-wide index key
# head of every layer, which both ranks need (all 60 layers are sparse in M3.1).
KV_RANK_BF16 = 2 * LAYERS * 2 * 128 * 2 + LAYERS * IDX_K_OUT * 2
# FP8 cache: E4M3 K and V with one FP32 scale per token and local K/V head (4 a layer), while the index keys stay
# BF16: MSA's block choice is a hard top-k over index scores, so index-key rounding can flip which blocks a query
# attends (same layout as the MiaAI-Lab TensorFold patch, docs/fp8-kv.md). 1.633x the BF16 capacity, not 2x.
KV_RANK_FP8 = LAYERS * (2 * 2 * 128 * 1 + IDX_K_OUT * 2 + 4 * 4)

# The KV tier (engine docs/receipts/mm3-kv-tier/DESIGN.md): a slice is one 128-token page of one layer on one rank, kept
# whole on the GPU (K, V, IK, scales) or on NVMe (4 KiB aligned); every live page has a 512-byte summary of its IK (bf16
# per-dimension min and max) a layer; the last TF_MM3_KV_HORIZON tokens of each stream keep their IK (scored exactly).
PAGE = 128
TOPK_BLOCKS = 16
SLICE_RANK_FP8 = PAGE * (2 * 2 * (128 + 4) + IDX_K_OUT * 2)
SLICE_NVME = -(-SLICE_RANK_FP8 // 4096) * 4096
SUMM_SLICE = 2 * IDX_K_OUT * 2
IK_TOKEN_RANK = LAYERS * IDX_K_OUT * 2
OPEN_PAGES = 2048 // PAGE + 2       # pages one stream writes at once: a 2,048-row prompt chunk and the page before it
TIER_HORIZON = 32768                # TF_MM3_KV_HORIZON default
TIER_CANDIDATES = 16                # TF_MM3_KV_CANDIDATES default: cold pages a kv head reranks
TIER_SESSIONS_GIB = 64              # NVMe kept for retained sessions beyond the live pages (GLM / DeepSeek: 64)

# (name, [(rows, cols, count)], placement) for every linear EXL3 can quantize.
# placement: "split" halves across the ranks, "rep" is on both ranks.
LINEARS = {
    "routed": ([(I_EXP, H, 2 * EXPERTS * MOE_LAYERS), (H, I_EXP, EXPERTS * MOE_LAYERS)], "split"),
    "shared": ([(I_EXP, H, 2 * MOE_LAYERS), (H, I_EXP, MOE_LAYERS)], "split"),
    "dense_mlp": ([(I_DENSE, H, 2 * DENSE_LAYERS), (H, I_DENSE, DENSE_LAYERS)], "split"),
    "attn_qo": ([(Q_OUT, H, LAYERS), (H, Q_OUT, LAYERS)], "split"),
    "attn_kv": ([(KV_OUT, H, 2 * LAYERS)], "split"),
    "index": ([(IDX_Q_OUT, H, LAYERS), (IDX_K_OUT, H, LAYERS)], "rep"),
    "head": ([(VOCAB, H, 1)], "split"),
}

# Tensors copied as they are: (bytes, placement). "node1" / "node2" are held by one node only: the vision tower on
# node 1 and the DSpark draft on node 2, which balances the two nodes now that no image model shares node 1.
COPIED = {
    "embed_bf16": (VOCAB * H * 2, "rep"),
    "router_f32": (MOE_LAYERS * (EXPERTS * H + EXPERTS) * 4, "rep"),
    "norms_bf16": ((2 * LAYERS * H + H + 4 * LAYERS * 128) * 2, "rep"),
    "vision_bf16": (1_734_145_024, "node1"),
    "dspark_verbatim": (2_301_852_674, "node2"),
}

# Bits for every class but the routed experts, whose average is solved to hit a pack size.
# The MSA index projections stay FP16 (16 = unquantized): below 2048 keys MSA is exactly dense, so short-context KL
# cannot see index error at all, and FP16 costs 0.24 GB over 8 bits
FIXED_BITS = {"shared": 6, "dense_mlp": 6, "attn_qo": 6, "attn_kv": 8, "index": 16, "head": 8}


def params(name):
    return sum(r * c * n for r, c, n in LINEARS[name][0])


def overhead(name):
    """EXL3 per-linear sign vectors (suh, svh): fp16 over the input and the output dimension."""
    return sum((r + c) * 2 * n for r, c, n in LINEARS[name][0])


def release_bytes():
    """Bytes of the release's main shards, from the source formats; must equal the index total_size."""
    nvfp4 = params("routed") // 2 + params("routed") // 16 + EXPERTS * MOE_LAYERS * 3 * 2 * 4
    mxfp8 = sum(params(n) + params(n) // 32 for n in ("shared", "dense_mlp", "attn_qo", "attn_kv", "index"))
    bf16 = 2 * VOCAB * H * 2 + COPIED["norms_bf16"][0]
    return nvfp4 + mxfp8 + bf16 + COPIED["router_f32"][0] + COPIED["vision_bf16"][0]


def pack(routed_bits, fixed=FIXED_BITS):
    """Pack bytes by placement for an average routed width."""
    out = {"split": 0.0, "rep": 0.0, "node1": 0.0, "node2": 0.0}
    for name, (_, place) in LINEARS.items():
        bits = routed_bits if name == "routed" else fixed[name]
        out[place] += params(name) * bits / 8 + (overhead(name) if bits < 16 else 0)
    for size, place in COPIED.values():
        out[place] += size
    return out


def total(p):
    return p["split"] + p["rep"] + p["node1"] + p["node2"]


def solve_routed(target_gb, fixed=FIXED_BITS):
    lo, hi = 1.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(pack(mid, fixed)) < target_gb * GB else (lo, mid)
    return lo


def node_weights(p, share1=0.5):
    """Weights a node holds, bytes; share1 is node 1's part of the split weights."""
    return p["split"] * share1 + p["rep"] + p["node1"], p["split"] * (1 - share1) + p["rep"] + p["node2"]


def llm_room(image_gib):
    """Bytes a node can give the LLM's weights, KV and workspaces, after the floor, dip and image model."""
    return (IDLE_AVAIL_GIB - FLOOR_GIB - DIP_GIB - image_gib) * GIB


def fit(p, ctx, kv_rank, image_gib, share1=0.5):
    """Spare bytes on (node 1, node 2) at one context of ctx tokens; negative means it does not fit."""
    w1, w2 = node_weights(p, share1)
    fixed = (CUDA_GIB + PREFILL_GIB) * GIB + ctx * kv_rank
    return llm_room(image_gib) - w1 - fixed, llm_room(0) - w2 - fixed


def max_ctx(p, kv_rank, image_gib, share1=0.5):
    s1, s2 = fit(p, 0, kv_rank, image_gib, share1)
    return max(0, int(min(s1, s2) // kv_rank))


def safe_pack_gb(ctx, kv_rank, slack_gib=SLACK_GIB):
    """Largest pack (GB) whose tighter node keeps FLOOR + DIP + slack free with ctx tokens of KV, even split."""
    lo, hi = 100.0, 260.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = min(fit(pack(solve_routed(mid)), ctx, kv_rank, 0)) >= slack_gib * GIB
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def best_share(p, kv_rank, image_gib):
    """Node-1 share of the split weights that maximizes the context both nodes can hold."""
    return max((s / 1000 for s in range(250, 501)), key=lambda s: max_ctx(p, kv_rank, image_gib, s))


# The engine's terms (budget.zig): CUDA context + NCCL + graphs, prompt workspaces; the draft with one stream a rank.
ENGINE_RUNTIME_GIB = 4.5
ENGINE_WORKSPACE_GIB = 1.5
# the draft a rank: dspark_config.cost with one stream (budget.draftBytes, 2,180,405,250 B) plus the 8 ring snapshot
# buffers native.draftCost adds (8 x one stream's K/V ring of 4112 rows x 5,120 B), all spent before any KV
DSPARK_STREAM_BYTES = 2_180_405_250
DSPARK_SNAPSHOT_BYTES = 8 * 4112 * 5120
DSPARK_RANK_GIB = (DSPARK_STREAM_BYTES + DSPARK_SNAPSHOT_BYTES) / GIB
VISION_TOWER_GIB = 2.27          # rank 0: Tower.plannedBytes 1,901,408,256 B + first_call_bytes 512 MiB (mm3/vision, measured)
VISION_ROW_BYTES = 6144 * 2      # an image token's row (hidden 6144, bf16), on every rank, for each of PARALLEL lanes
PAGE_TOKENS = PAGE               # the family's KV page (minimax_m3/pool.zig page_tokens)

# The DFlash2 draft, incoai/MiniMax-M3.1-Flash-DFlash2 @ ac07a7c1 (its config.json; the engine's dflash2_config.incoai):
# 5 bf16 layers of 64 query and 8 kv heads (128 wide), MLP 12288, a sliding window of 2048, blocks of 8 rows (7 drafts),
# fc over 6 tapped layers, dynamic convs in groups of 16, a selector of rank 256 offering 32 candidates a row.
DFLASH2 = {"layers": 5, "hidden": H, "heads": 64, "kv_heads": 8, "head_dim": 128, "ffn": 12288, "vocab": VOCAB,
           "window": 2048, "block": 8, "taps": 6, "group": 16, "rank": 256, "top_k": 32}
DFLASH2_FILE_BYTES = 4_247_776_488   # model.safetensors at the pinned revision (its data: 4,247,767,552 bytes)
ENGINE_WINDOW_ROWS = 16              # iface.window_rows: a verify window's rows, which a ring holds beyond the draft's window
ENGINE_PROMPT_ROWS = 2048            # iface.prefill_rows: the rows one absorb reads (a prompt chunk)
ENGINE_DRAFT_SNAPS = 8               # backend.max_snaps: ring snapshots a rank keeps for the prompt cache
DRAFT_RUNTIME_ESTIMATE = 512 << 20   # dspark_config.runtime_estimate: CUDA module, graphs, NCCL buffers (UNVERIFIED planning figure)
MEMTOTAL_GIB = 127598748 / 2**20   # both Sparks' MemTotal since the 2026-10-24 firmware alignment (kB / 2^20)


def dflash2_split(c=DFLASH2, world=2):
    """A rank's query heads, kv heads and MLP rows (dflash2_config.Split.of)."""
    return c["heads"] // world, c["kv_heads"] // world, c["ffn"] // world


def dflash2_weights(c=DFLASH2, world=2):
    """Bf16 weight bytes a rank holds (dflash2_config.rankWeights): its heads and MLP rows; fc, the convs, the selector
    and the norms whole on every rank."""
    h, hd = c["hidden"], c["head_dim"]
    qh, kvh, f = dflash2_split(c, world)
    q, kv = qh * hd, kvh * hd
    dyn = 4 * (h // c["group"])
    return {
        "layers": c["layers"] * (2 * q * h + 2 * kv * h + 3 * h * f) * 2,
        "convs": c["layers"] * 2 * (dyn * h + 2 * 2 * h) * 2,
        "fc": h * c["taps"] * h * 2,
        "selector": (c["rank"] * h + 2 * c["vocab"] * c["rank"]) * 2,
        "small": (c["layers"] * (2 * h + 2 * hd) + 2 * h) * 2,
    }


def dflash2_ring(c=DFLASH2, world=2, window_rows=ENGINE_WINDOW_ROWS):
    """A stream's ring on a rank (dflash2_config.ringBytes): K and V, every layer, the rank's kv heads, window + window_rows."""
    return (c["window"] + window_rows) * c["layers"] * 2 * dflash2_split(c, world)[1] * c["head_dim"] * 2


def dflash2_scratch(c=DFLASH2, world=2, absorb_rows=ENGINE_PROMPT_ROWS):
    """The device scratch (dflash2_config.scratchSizes / scratchBytes): every field 256-byte aligned, in order."""
    h, hd, r, b, k = c["hidden"], c["head_dim"], absorb_rows, c["block"], c["top_k"]
    qh, kvh, f = dflash2_split(c, world)
    qw, kvw, dyn, own = qh * hd, kvh * hd, 4 * (h // c["group"]), -(-c["vocab"] // world)
    sizes = (r * h * 2, r * h * 2, r * kvw * 2, r * kvw * 2,
             b * 4, b * h * 4, b * h * 2, b * h * 2, b * dyn * 2,
             b * qw * 2, b * kvw * 2, b * kvw * 2, b * qw * 2, b * kvw * 2, b * kvw * 2,
             b * qw * 2, b * h * 4, b * h * 4, world * b * h * 4,
             b * f * 2, b * f * 2, b * f * 2, b * h * 2,
             b * own * 4, b * k * 8, world * b * k * 8, b * c["rank"] * 4)
    total = 0
    for size in sizes:
        total = -(-total // 256) * 256 + size
    return total


def dflash2_cost(c=DFLASH2, world=2, snaps=ENGINE_DRAFT_SNAPS):
    """What the DFlash2 draft costs a rank before any KV, bytes (dflash2_config.cost): weights and runtime once (the
    runtime estimate, the scratch, the ring snapshots), per_lane (a ring) for every stream."""
    ring = dflash2_ring(c, world)
    return {"weights": sum(dflash2_weights(c, world).values()),
            "runtime": DRAFT_RUNTIME_ESTIMATE + dflash2_scratch(c, world) + snaps * ring, "per_lane": ring}


def draft_gib(drafter, parallel):
    """GiB a rank gives the draft before any KV: none, dspark (one stream and its ring snapshots, native.draftCost) or
    dflash2 (weights + runtime + a ring a stream). ``drafter`` True is dspark (the old --drafter switch)."""
    if not drafter:
        return 0.0
    if drafter is True or drafter == "dspark":
        return DSPARK_RANK_GIB
    if drafter == "dflash2":
        cost = dflash2_cost()
        return (cost["weights"] + cost["runtime"] + max(1, parallel) * cost["per_lane"]) / GIB
    raise ValueError(f"no such drafter: {drafter!r} (dspark or dflash2)")


def engine_rank_weights(p):
    """Bytes a rank loads: half the split weights plus the replicated ones (no vision tower, no draft)."""
    return p["split"] / 2 + p["rep"]


def vision_gib(image_tokens, parallel=1):
    """Rank 0's image terms with --vision: the tower and scratch, and the rows of ``image_tokens`` image tokens for each
    lane (0: off). Rank 1 holds the rows only, so rank 0 is the tighter rank and both pools follow it."""
    if not image_tokens:
        return 0.0
    return VISION_TOWER_GIB + parallel * image_tokens * VISION_ROW_BYTES / GIB


def serving(pack_gb, context, parallel, kv, reserve_gib=None, drafter=False, available_gib=IDLE_AVAIL_GIB,
            vision_image_tokens=0):
    """One profile on the tighter rank as the engine prices it; GiB throughout. reserve_gib None: the engine's default.
    vision_image_tokens: VISION_IMAGE_TOKENS with VISION=1 (rank 0 then carries the tower), 0 for text only."""
    kv_rank = KV_RANK_FP8 if kv == "fp8" else KV_RANK_BF16
    reserve = max(4.0, MEMTOTAL_GIB / 10) if reserve_gib is None else reserve_gib
    weights = engine_rank_weights(pack(solve_routed(pack_gb))) / GIB
    draft = draft_gib(drafter, parallel)
    vision = vision_gib(vision_image_tokens, parallel)
    fixed = weights + reserve + ENGINE_RUNTIME_GIB + ENGINE_WORKSPACE_GIB + draft
    kv_need = context * parallel * kv_rank / GIB      # every slot able to hold a whole CONTEXT
    room = available_gib - fixed - vision             # the images' terms come out of the pool (admissionWith)
    pages = max(0, int(room * GIB // (kv_rank * PAGE_TOKENS)))
    return {"weights": weights, "reserve": reserve, "draft": draft, "vision": vision, "fixed": fixed, "kv_need": kv_need,
            "kv_room": room, "max_tokens": max(0, int(room * GIB // kv_rank)), "pages": pages,
            "pool_tokens": pages * PAGE_TOKENS, "spare": room - kv_need,
            # the load's MemFree: the text terms and the KV a whole CONTEXT needs; the images do not move it
            "mem_gate": int(-(-(fixed + kv_need) // 1)), "kv_rank": kv_rank}


def pages(tokens):
    return -(-tokens // PAGE)


def tier(context, parallel, kv_room_gib, horizon=TIER_HORIZON, candidates=TIER_CANDIDATES):
    """The KV tier's terms on one rank, GiB unless named: what the GPU holds, the least slice cache, the NVMe."""
    live = parallel * pages(context + PAGE)              # every slot's pages and its verify window
    summaries = live * LAYERS * SUMM_SLICE / GIB
    horizon_ik = parallel * min(horizon, context) * IK_TOKEN_RANK / GIB
    # the most slices pinned at once: one stream's prompt chunk on every layer, every stream's tail page on every
    # layer, and one layer's candidates of a whole prompt chunk (at most every page of the context)
    pinned = LAYERS * OPEN_PAGES + parallel * LAYERS * 2 + min(pages(context), 2048 * 2 * (TOPK_BLOCKS + candidates))
    cache_min = 2 * pinned * SLICE_RANK_FP8 / GIB         # twice the pinned set: last steps' slices stay to be hit
    cache = kv_room_gib - summaries - horizon_ik
    nvme_live = live * LAYERS * SLICE_NVME / GIB
    return {"live_pages": live, "summaries": summaries, "horizon_ik": horizon_ik, "cache_min": cache_min,
            "cache": cache, "device_min": summaries + horizon_ik + cache_min, "nvme_live": nvme_live,
            "nvme": nvme_live + TIER_SESSIONS_GIB, "exact_to": min(horizon, context) + PAGE * candidates,
            "ok": cache >= cache_min}


def draft_label(drafter):
    if drafter == "dflash2":
        cost = dflash2_cost()
        return (f"DFlash2 draft ({cost['weights'] / GIB:.2f} weights, {cost['runtime'] / GIB:.2f} runtime + snapshots, "
                f"{cost['per_lane'] / 1e6:.1f} MB a stream)")
    return "dspark draft"


def print_serving(args):
    if args.kv_tier:
        return print_tier(args)
    s = serving(args.pack_gb, args.context, args.parallel, args.kv, args.reserve_gib, args.drafter, args.available_gib,
                args.vision_image_tokens)
    print(f"## Serving profile: {args.pack_gb:g} GB pack, CONTEXT {args.context} x PARALLEL {args.parallel}, KV {args.kv}"
          f" ({s['kv_rank']:,} B a token a rank), draft {args.drafter or 'off'}\n")
    print("| term | GiB a rank |\n| --- | ---: |")
    for name, value in (("weights (half the split + replicated)", s["weights"]),
                        ("reserve (TENSORFOLD_MEMORY_RESERVE_GIB" + ("" if args.reserve_gib is not None else
                                                                    ", engine default") + ")", s["reserve"]),
                        ("CUDA context, NCCL, graphs", ENGINE_RUNTIME_GIB), ("prompt workspaces", ENGINE_WORKSPACE_GIB),
                        (draft_label(args.drafter), s["draft"]), ("KV for CONTEXT x PARALLEL", s["kv_need"])):
        print(f"| {name} | {value:.2f} |")
    print(f"\nAt {args.available_gib:g} GiB available before the load: KV room {s['kv_room']:.2f} GiB "
          f"({s['pool_tokens']:,} tokens, {s['pages']:,} pages of {PAGE_TOKENS}), {s['spare']:+.2f} GiB after CONTEXT x PARALLEL.")
    if args.vision_image_tokens:
        t = serving(args.pack_gb, args.context, args.parallel, args.kv, args.reserve_gib, args.drafter, args.available_gib)
        print(f"Images (VISION_IMAGE_TOKENS {args.vision_image_tokens}): rank 0's tower, scratch and rows "
              f"{s['vision']:.2f} GiB out of the KV pool (both ranks size to rank 0's): "
              f"{t['pool_tokens'] - s['pool_tokens']:,} tokens, {t['pages'] - s['pages']:,} pages fewer than text only "
              f"({t['pool_tokens']:,}).")
        if s["spare"] < 0:
            print(f"CONTEXT x PARALLEL NO LONGER FITS: the pool holds {s['pool_tokens']:,} tokens, it needs "
                  f"{args.context * args.parallel:,}.")
    print(f"MEM_GATE_GIB={s['mem_gate']}")
    return 0 if s["spare"] >= 0 else 1


def print_tier(args):
    """A profile whose pages live on NVMe: the room holds the tier's summaries, horizon and slice cache instead."""
    if args.kv != "fp8":
        print("the KV tier is priced for --kv fp8 only")
        return 1
    s = serving(args.pack_gb, 0, args.parallel, args.kv, args.reserve_gib, args.drafter, args.available_gib,
                args.vision_image_tokens)   # the images come out of the room, so out of the slice cache
    t = tier(args.context, args.parallel, s["kv_room"], args.horizon, args.candidates)
    gate = int(-(-(s["fixed"] + t["device_min"]) // 1))
    print(f"## Serving profile with the KV tier: {args.pack_gb:g} GB pack, CONTEXT {args.context} x PARALLEL "
          f"{args.parallel}, horizon {args.horizon}, {args.candidates} candidates, draft {args.drafter or 'off'}\n")
    print("| term | GiB a rank |\n| --- | ---: |")
    for name, value in (("weights (half the split + replicated)", s["weights"]), ("reserve", s["reserve"]),
                        ("CUDA context, NCCL, graphs", ENGINE_RUNTIME_GIB), ("prompt workspaces", ENGINE_WORKSPACE_GIB),
                        (draft_label(args.drafter), s["draft"]), ("summaries (every live page and layer)", t["summaries"]),
                        ("IK horizon (exact scores)", t["horizon_ik"]), ("slice cache, the rest of the room", t["cache"]),
                        ("slice cache needed (twice the pinned slices)", t["cache_min"]),
                        ("NVMe: live pages", t["nvme_live"]), ("NVMe: TF_MM3_KV_TIER_GIB (live + sessions)", t["nvme"])):
        print(f"| {name} | {value:.2f} |")
    print(f"\nKV room {s['kv_room']:.2f} GiB; {t['live_pages']:,} live pages; selection equals the resident pool's "
          f"up to {t['exact_to']:,} tokens a stream.")
    if args.vision_image_tokens:
        print(f"Images (VISION_IMAGE_TOKENS {args.vision_image_tokens}): rank 0's tower, scratch and rows "
              f"{s['vision']:.2f} GiB out of the room, so out of the slice cache.")
    print(f"MEM_GATE_GIB={gate}")
    print(f"TF_MM3_KV_TIER_GIB={int(-(-t['nvme'] // 1))}")
    return 0 if t["ok"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image-gib", type=float, action="append", help="image model resident GiB (repeatable)")
    ap.add_argument("--serve", action="store_true", help="one serving profile as the engine prices it")
    ap.add_argument("--pack-gb", type=float, default=199.9, help="pack size, GB (default: the 3.55 bpw pack's 199.9)")
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--kv", choices=("fp8", "bf16"), default="fp8")
    ap.add_argument("--reserve-gib", type=float, default=None, help="TENSORFOLD_MEMORY_RESERVE_GIB (default: engine's)")
    ap.add_argument("--drafter", nargs="?", const="dspark", default=None, choices=("dspark", "dflash2"),
                    help="the draft on both ranks: dspark (DRAFTER=dspark; the bare flag) or dflash2 (DRAFTS=dflash2)")
    ap.add_argument("--available-gib", type=float, default=IDLE_AVAIL_GIB, help="MemAvailable before the load")
    ap.add_argument("--vision-image-tokens", type=int, default=0,
                    help="VISION_IMAGE_TOKENS with VISION=1: rank 0's vision tower and the image rows, out of the KV pool "
                         "(default 0: text only)")
    ap.add_argument("--kv-tier", action="store_true", help="pages on NVMe through the engine's KV tier (TF_MM3_KV_TIER)")
    ap.add_argument("--horizon", type=int, default=TIER_HORIZON, help="TF_MM3_KV_HORIZON, tokens")
    ap.add_argument("--candidates", type=int, default=TIER_CANDIDATES, help="TF_MM3_KV_CANDIDATES")
    args = ap.parse_args()
    if args.serve:
        return print_serving(args)
    images = args.image_gib or []

    print("## Parameters and release bytes\n")
    print("| class | params | placement |\n| --- | ---: | --- |")
    for name, (_, place) in LINEARS.items():
        print(f"| {name} | {params(name) / 1e9:.3f} B | {place} |")
    print(f"\nrelease main-shard bytes from formats: {release_bytes():,}\n")

    print("## Packs\n")
    print("| target | routed avg bits | split GB | replicated GB | one-node GB | total GB | GiB "
          "| node 1 GB | node 2 GB |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for target in TARGETS:
        rb = solve_routed(target)
        p = pack(rb)
        w1, w2 = node_weights(p)
        one = (p['node1'] + p['node2']) / GB
        print(f"| {target} | {rb:.3f} | {p['split'] / GB:.2f} | {p['rep'] / GB:.2f} | {one:.2f} | "
              f"{total(p) / GB:.2f} | {total(p) / GIB:.2f} | {w1 / GB:.2f} | {w2 / GB:.2f} |")

    print(f"\nKV a token a rank: BF16 {KV_RANK_BF16:,} B, FP8 {KV_RANK_FP8:,} B; with the KV tier a slice is "
          f"{SLICE_RANK_FP8:,} B ({SLICE_NVME:,} B on NVMe)\n")
    print("## LLM only, even split: spare GiB above floor + dip on the tighter node, and max context\n")
    print("| pack GB | KV | spare at 32K | spare at 128K | max context (0 spare) | max context (keeping "
          f"{SLACK_GIB:g} GiB) |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for target in TARGETS:
        p = pack(solve_routed(target))
        for kv_name, kv in (("BF16", KV_RANK_BF16), ("FP8", KV_RANK_FP8)):
            s32, s128 = min(fit(p, 32768, kv, 0)) / GIB, min(fit(p, 131072, kv, 0)) / GIB
            m0 = max_ctx(p, kv, 0)
            ms = max(0, int((min(fit(p, 0, kv, 0)) - SLACK_GIB * GIB) // kv))
            print(f"| {target} | {kv_name} | {s32:+.2f} | {s128:+.2f} | {m0 // 1024}K | {ms // 1024}K |")
    print("\n## Safe size: largest pack keeping floor + dip + " f"{SLACK_GIB:g} GiB free, even split\n")
    print("| context | BF16 KV | FP8 KV |\n| ---: | ---: | ---: |")
    for ctx in (32768, 65536, 131072, 262144):
        bf16, fp8 = safe_pack_gb(ctx, KV_RANK_BF16), safe_pack_gb(ctx, KV_RANK_FP8)
        print(f"| {ctx // 1024}K | {bf16:.1f} GB | {fp8:.1f} GB |")
    if not images:
        return
    print(f"LLM room a node: no image {llm_room(0) / GIB:.1f} GiB; "
          + ", ".join(f"image {i:g} GiB -> {llm_room(i) / GIB:.1f} GiB" for i in images if i) + "\n")

    for kv_name, kv in (("BF16", KV_RANK_BF16), ("FP8", KV_RANK_FP8)):
        print(f"## Fit, {kv_name} KV, even split: max context tokens (one sequence) / spare GiB at 32K and 128K "
              "on the tighter node\n")
        print("| pack GB | " + " | ".join(f"image {i:g} GiB" for i in images) + " |")
        print("| ---: |" + " ---: |" * len(images))
        for target in (160, 170, 180) + TARGETS:
            p = pack(solve_routed(target))
            cells = []
            for i in images:
                m = max_ctx(p, kv, i)
                s32 = min(fit(p, 32768, kv, i)) / GIB
                s128 = min(fit(p, 131072, kv, i)) / GIB
                cells.append(f"{m // 1024}K / {s32:+.1f} / {s128:+.1f}")
            print(f"| {target} | " + " | ".join(cells) + " |")
        print()

    print("## Uneven split (node 1 holds less of the split weights), BF16 KV\n")
    print("| pack GB | image GiB | node-1 share | max context | node 1 GB | node 2 GB |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for target in (160, 170, 180) + TARGETS:
        p = pack(solve_routed(target))
        for i in [x for x in images if x]:
            s = best_share(p, KV_RANK_BF16, i)
            w1, w2 = node_weights(p, s)
            print(f"| {target} | {i:g} | {s:.3f} | {max_ctx(p, KV_RANK_BF16, i, s) // 1024}K | "
                  f"{w1 / GB:.2f} | {w2 / GB:.2f} |")


if __name__ == "__main__":
    raise SystemExit(main())
