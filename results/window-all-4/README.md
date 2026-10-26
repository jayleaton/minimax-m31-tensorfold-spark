# window-all-4: TP2 final performance push (2026-10-26, 09:49-11:35 UTC)

Engine `mm3/window-all-4` @ `6c6f7b289` (window-all-3 + serve-latency, prefill prompt kernels, prefix cache fix,
PLAN_INLINE fix, KV-tier fix, tool_choice none fix, prompt prep, int8 index keys, decode glue; each behind its switch).
Two Sparks, FP8 KV, thinking off. Draft jobs at CONTEXT 114,688 (the head's budget was ~1.5 GiB tighter than in
window-all-3); context does not change short-prompt decode or reply hashes. Every job's 8 reply hashes equal the
baseline's (5015); the 70K revisit hashes equal window-all-3's.

**Recipe (job 5072):** window-all-3's (DSpark depth by confidence + RoCE switches) + prompt kernels (default on) +
`TF_SERVE_INCREMENTAL=1` (pass it with `EXTRA_ENV`: the recipe forwards only TENSORFOLD/TF_MM3/TF_ROCE/TF_CUDA names) +
`TF_MM3_GLUE_ACT=1 TF_MM3_GLUE_ADD=1`.

| | window-all-3 | window-all-4 |
| --- | ---: | ---: |
| Decode, probe median T0 / T1 (tok/s) | 31.75 / 30.5 | 31.3 / 30.6 (**unchanged**) |
| Prefill 32K (tok/s), same-window baseline → all winners | 485 (5015) | **582** (507), 565 (5072) |
| Prefill 2K (tok/s) | 677 (5015) | 694-696 |
| 70K prompt first token: fresh / repeat / follow-up (s) | 167.2 / 0.76 / 0.79 | **140.0 / 0.67 / 0.70** |
| C4 at 40,960, code T0 aggregate: drafts / plain (tok/s) | 52.0 / - | 52.7 / **65.3** (chat plain 63.2) |
| One code stream in the C4 profile, drafts (tok/s) | 35.3 | **53.6** |
| KV tier exact, CONTEXT 262,144 (TP2, drafts off) | panicked at first prompt | **works**: smoke 12/12, 19.0 tok/s, hashes = resident |
| Tools: smoke / own off, adaptive / toolcall off, adaptive | 12 / 12, 12 / 42, 42 | **12 / 12, 12 / 42, 42** (5072: 12 / 42 off) |
| PLAN_INLINE | crash at start | runs, hashes equal; kept off (no clear gain) |

A/B on the window-all-3 recipe (probe greedy median, tok/s; baseline 31.06): prompt kernels 30.92 (prefill +2 % / +15 %),
serve-incremental 31.55 (+1.6 %), glue 31.10, plan-inline 31.78 (sampled math 21.8: off), int8 index keys 30.81 (off,
128K overlap not measured here). Files: `<job>/probe.json`, `*.txt`, `concurrent.json`, `revisit.json`, `window.txt`.
