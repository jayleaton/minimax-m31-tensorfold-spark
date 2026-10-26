# Engine source export

`ENGINE_PATCHES=engine/patches`: `zig/Dockerfile` applies these to upstream
[TensorFold](https://github.com/ashhart/TensorFold) at `TF_REF` (`4ff784410f70b95767684d337e59dfd6d94fec8b`, the v1.1.2 tag), in filename order.

| Patch | Author | What |
| --- | --- | --- |
| `0001-minimax-m3-family.patch` | Mia's AI Lab (MiaAI-Lab) | a link to `patches/0001-minimax-m3-family.patch`, unchanged: the `minimax_m3_vl` family |
| `0002-minimax-m3-engine.patch` | JJ Eaton (jayleaton) | the engine this recipe serves, on top of 0001:  276 files changed, 36346 insertions(+), 1000 deletions(-) |

Exported by `scripts/export-engine.sh` from engine commit `6c6f7b28936f04b22bc6edfedb4a64ca790f26fb` (`ENGINE_COMMIT`). Every file comes over as it is in that
commit, with its licence and attribution headers; TensorFold's `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and
`LICENSES/` apply to the patched tree ([LICENSING](../docs/LICENSING.md)). Changes made by the export: 9 file(s) left out
by `engine/export-exclude.txt`, 1 file(s) with private names replaced by placeholders. The script checks that
`TF_REF` plus the two patches reproduces the exported tree exactly.

Left out:

- `docs/receipts/mm3-dflash2/PARITY.md`
- `docs/receipts/mm3-kv-format/DESIGN.md`
- `docs/receipts/mm3-kv-format/KV-TIER.md`
- `docs/receipts/mm3-kv-format/README.md`
- `docs/receipts/mm3-kv-tier/DESIGN.md`
- `docs/receipts/mm3-kv-tier/README.md`
- `docs/receipts/mm3-prompt-kernels-2026-10-25/README.md`
- `docs/receipts/mm3-s8-loadpipe/README.md`
- `docs/receipts/mm3-speed-graphs/README.md`
