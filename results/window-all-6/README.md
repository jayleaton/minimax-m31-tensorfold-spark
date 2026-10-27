# window-all-6: tensor-core prompt attention (2026-10-26, 20:22-21:10 UTC)

Engine `mm3/window-all-6` @ `0f591bc41` (window-all-5 + S15b's fallback fix: prompt attention on bf16 tensor cores,
`TF_MM3_ATTN_MMA=1`, off by default). Built, gated and staged on both Sparks during window-all-5's gaps. Two Sparks, FP8
KV, thinking off, CONTEXT 114,688 (C4: 40,960).

**Recipe (job 704):** window-all-5's (window-all-4's + `TF_MM3_ATTN_SPLIT=1`) + `TF_MM3_ATTN_MMA=1`.

| | window-all-5 recipe (701 / 703) | + S15b (702 / 704) |
| --- | ---: | ---: |
| Prefill 32K (tok/s) | 578.6 / 580.3 | **720.5 / 719.7** (+24 %) |
| Prefill 2K (tok/s) | 689.1 / 692.2 | **783.1 / 786.0** (+14 %) |
| 70K prompt first token: fresh / repeat / follow-up (s; 705) | 133.3 / 0.63 / 0.66 (window-all-5 607) | **109.4 / 0.62 / 0.65** |
| Decode, long prose (2 reps, tok/s) | 21.8 / 21.3 (window-all-5 603) | 21.7 / 22.6 |
| Decode, probe median greedy / sampled (drafted) | 25.5 / 27.5 (701) | 31.7 / 26.8 (702), 30.2 / 28.3 (704) |
| C4 at 40,960, code / chat T0 x4, drafts off (706) | 64.2 / 56.6 (window-all-5 605) | 57.7 / 57.5 |
| Smoke / own tools off, adaptive / toolcall off, adaptive | 12 | **12 / 12, 12 / 42, 42** |

- S15b changes prompt attention's summation order (FP64-gated on one Spark: contract 0 of 255 failures, golden = base), so
  replies whose prompt passes through it hash differently from 701 (5 of 8 probe replies); 702 and 704 hash the same.
- 701's greedy median (25.5) is one reply: math greedy at 19.2 tok/s with the same hash as window-all-5's 31.3 (the
  draft depth rule's boot calibration); 702 / 704 read 33.3 / 29.5 on it.
- C4 4-stream code reads 57.7 here, 64.2 in window-all-5 and 65.3 in window-all-4; one rep each, 1-2 streams are flat
  (21.7 / 40.2). Not resolved in this window.

Files: `<job>/probe.json`, `*.txt`, `concurrent.json`, `revisit.json`, `window.txt`.
