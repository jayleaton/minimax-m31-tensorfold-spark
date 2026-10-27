# window-all-6b: C4 with and without tensor-core prompt attention (2026-10-27, 01:02-01:27 UTC)

Image as window-all-6 (`mm3/window-all-6` @ `0f591bc41`). C4 plain: `PARALLEL=4`, CONTEXT 40,960, drafts off, window-all-5's
recipe (`TF_MM3_ATTN_SPLIT=1`); `TF_MM3_ATTN_MMA` off / on / off / on, one `bench_concurrent.py --alone` run each.

| Aggregate tok/s at 4 streams | 711 off | 712 on | 713 off | 714 on |
| --- | ---: | ---: | ---: | ---: |
| code T0 | 67.6 | **57.1** | 69.2 | - |
| code T1 | 46.0 | 45.6 | 46.9 | - |
| chat T0 | 55.0 | 54.9 | 56.7 | - |
| chat T1 | 43.5 | 44.1 | 44.2 | - |

- With `TF_MM3_ATTN_MMA=1`, greedy code at 4 streams is about 16 % lower (window-all-6's 706 read 57.7 the same way);
  sampled code, chat and 1-2 streams are unchanged. So S15b suits the one-stream profile; leave it off with `PARALLEL=4`
  until this is understood.
- 714 is not a result: the worker reached its device emergency (94 C held 30 s) at 01:27:36 and its guard stopped rank 1
  during the run (its thermal-stop record stays until an operator clears it).

Files: `<job>/concurrent.json`, `concurrent.txt`, `window.txt`.
