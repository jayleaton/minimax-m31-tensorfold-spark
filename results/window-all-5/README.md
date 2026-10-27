# window-all-5: decode attention split (2026-10-26, 16:15-18:28 UTC)

Engine `mm3/window-all-5` @ `6d758d7d5` (window-all-4 + decode attention split `TF_MM3_ATTN_SPLIT`, int8 scorer,
S15b prompt attention `TF_MM3_ATTN_MMA`, draft graphs `TF_MM3_DRAFT_GRAPHS`; each off by default). Two Sparks, FP8 KV,
thinking off, CONTEXT 114,688 (C4: 40,960). S15b was not measured here: the merged commit still had its local-memory
fallback bug, so job 602 ran with `TF_MM3_ATTN_MMA` unset (a second baseline); S15b moves to window-all-6.

**Recipe (job 604):** window-all-4's (5072) + `TF_MM3_ATTN_SPLIT=1`.

| | window-all-4 recipe (601 / 602) | + attention split (603 / 604) |
| --- | ---: | ---: |
| Decode, long prose (`bench.py decode`, 2 reps, tok/s) | 19.3 / 18.2 | **21.8 / 21.3** (604: 21.1 / 21.8) |
| Decode after a 32K prompt (256 tokens, tok/s; 611 / 612) | 20.0 | **22.9** |
| Decode, probe median greedy / sampled (drafted, tok/s) | 30.6-31.0 / 30.2-30.3 | 30.9-31.3 / 26.4-29.3 |
| Prefill 2K / 32K (tok/s) | 691-696 / 472-580 | 693 / 580 |
| 70K prompt first token: fresh / repeat / follow-up (s; 607) | 140.0 / 0.67 / 0.70 (window-all-4) | 133.3 / 0.63 / 0.66 |
| C4 at 40,960, code / chat T0 x4, drafts off (tok/s; 605) | 65.3 / 63.2 (window-all-4) | 64.2 / 56.6 |
| Smoke / own tools off, adaptive / toolcall off, adaptive | 12 | **12 / 12, 12 / 42, 42** |

- The split changes rounding (FP64-gated, single-Spark contract and quality pass), so 4 of 8 probe replies hash differently
  from 601; every split job hashes the same as 603. 602's hashes equal 601's.
- Plain decode (what drafts cannot help: prose, after a long prompt) is +10-14 %. The drafted probe median is unchanged;
  one sampled code reply in 604 (22.3) pulled its sampled median down, 603 and 610 read 29.3 / 30.1.
- The 601 / 602 32K prefill spread (472 vs 580) is the same image and toggles: the first start after the build.

A/B (off): draft graphs at C4 (606 vs 6081): code x4 51.6 vs 49.6, but code x1 18.1 vs 53.8. int8 index keys on the 70K
prompt (609): fresh 156.7 s vs 133.3, other block choices (reply differs). Plan-inline (610): probe 30.35 / 30.11, decode
21.1 / 21.2, no gain.

Heat: the worker reached its device emergency (93 C for 30 s) at 18:28:41 in 612's release step (MMLU after the 32K decode);
its guard stopped the job. 612's structured / MMLU lines are not valid. Job 608 did not start (the worker held 116 GiB free
after the window-all-6 image load, gate 117); 6081 is the same job at gate 116.

Files: `<job>/probe.json`, `*.txt`, `concurrent.json`, `revisit.json`, `decode32k.json`, `window.txt`.
