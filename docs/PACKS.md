# The pack

`jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31` (private) at revision `5bff3d732e9dfa236a41b6f3054dd8550bef595f`
(the weights of `f76ad09a`; `5bff3d73` corrects only the card), quantized from `MiniMaxAI/MiniMax-M3.1-Flash` at
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e` with ExLlamaV3's `minimax-m3` architecture branch. `model_type` `minimax_m3_vl`.

| | |
| --- | --- |
| Files | 199.9 GB in all |
| Shards | 26, 197,555,310,554 bytes; the index's `total_size` 197,542,720,916 |
| `dspark/` | the base model's draft, 2,301,867,426 bytes, copied unchanged |
| Checks | `SHA256SUMS` (every shard); `quant-logs/` (recipe, gate report) |
| License | `LICENSE`: the MiniMax Community License |

Bits: routed experts 3.550 bpw on average, one width per layer and projection: 4 bits on the layers MiaAI-Lab's H35 pack
widened plus the DSpark taps (gate/up on layers 3-17, 31 and 45-59; down on 3-18, 31 and 45-59), 3 elsewhere. Attention
q/o 6, k/v 8; the MSA index projections FP16 (below 2,048 keys MSA is exactly dense, so short-context KL cannot see index
error, and FP16 costs 0.24 GB over 8 bits); shared expert and dense MLP 6; head 8; embedding BF16. The vision tower,
projector and `dspark/` are copied as they are.

## Per rank

`scripts/quant_budget.py` reproduces the release's own index to the byte (`release_bytes()`), and solves the routed width
for a pack size (3.551 bpw at 200 GB). What a rank holds depends on the engine's placement: the `minimax_m3_vl` family loads
the language model only (no vision tower) and splits the experts, attention and head over the ranks while the embedding,
router, norms and index projections are on both. So a rank holds half the split weights plus the replicated ones:
**92.66 GiB** (`--serve`). The draft, when `DRAFTER=dspark`, adds ~2.03 GiB a rank (1.34 GiB of weights, its ring and
scratch).

KV a token a rank (both ranks keep the 128-wide index key of all 60 layers; the four KV heads are split two and two):

| Cache | Bytes a token a rank | Tokens in 10.44 GiB |
| --- | ---: | ---: |
| BF16 K/V and index keys | 76,800 | 146,031 |
| FP8 K/V (e4m3, an fp32 scale per token and KV head), BF16 index keys | 47,040 | 238,418 |

FP8 holds 1.633x the tokens of BF16, not 2x: the index keys stay BF16 because MSA's block choice is a hard top-k over index
scores, and rounding them could flip which blocks a query attends.

## The serving profile

`python3 scripts/quant_budget.py --serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8` prints the profile in
`config/prod.env.example` and its `MEM_GATE_GIB` (113; with `--vision-image-tokens 2304`, the profile's `VISION=1`, the
KV pool is 185,984 tokens instead of 238,336, and the gate does not move). The table in [RESULTS R0](RESULTS.md#r0-calculator-arithmetic-not-measurement)
has the terms and the variants. Every number on this page is arithmetic on the release's tensor classes and the engine's
budget terms; the first start measures the real ones.

## Quality

The pack card's teacher-forced check against the source passed every gated group except tool_agentic ([RESULTS R1](RESULTS.md#r1-the-pack-its-own-card-offline),
[KNOWN-ISSUES](KNOWN-ISSUES.md#the-pack)). The calibration mix and the earlier 2.8 / 3.0 / 3.5 bpw planning tables are in this
file's history (the pre-release skeleton).
