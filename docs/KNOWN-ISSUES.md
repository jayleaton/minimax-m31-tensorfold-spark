# What is not known or not solved

What is open after the latest measured runs ([results/window-all-6](../results/window-all-6/README.md),
[-6b](../results/window-all-6b/README.md), [-5](../results/window-all-5/README.md)): `TF_MM3_ATTN_MMA=1` costs greedy code
~16 % at four streams (not understood; off in the four-slot profile); the worker reached its device emergency limit three
times under sustained load (the guard stopped each job); MMLU-200 has not been re-measured since window-all-3. Fixed
since window-all-3: `tool_choice: "none"` (`bench/structured.py` tools 15/15, once), the KV tier (exact 262,144 works,
opt-in), `TF_MM3_PLAN_INLINE=1` (starts; off, no gain). From window-all-3 and still open: prose gains nothing from the draft; plain decode is ~5 % below window-all-2's; at four
streams drafts decode slower than plain decode; tool-eval-bench is not run. The run hit both Sparks' thermal limits and
stopped early, so the S11-off and S8-off checks are missing. Window-all-2's prefill regression is fixed. Some notes below
predate these runs and say so.

## The engine

- **The family is not this repository's.** The `minimax_m3_vl` family (TensorFold v1.1.2 plus a patch series) lives on the
  engine branch (jayleaton/TensorFold); this recipe builds whatever `ENGINE_PATCHES` names (default `patches/`, MiaAI-Lab's series at fd1d379,
  unchanged) and refuses to start a build that does not list the family. That series has passed host tests and per-kernel
  GPU contract runs only: no forward pass of the whole model, no two-rank run and no generated token is recorded for it
  (patches/README.md). Its model semantics (swigluoai, RoPE layout, sparse block forcing, router order, norm folds) are
  held as switches with the likeliest defaults.
- **The engine's memory lines are unknown.** The guard needs `boot_floor_log_regex` (the engine's boot memory line) and
  cannot load until the first start shows it; a slot-count check after the load (GLM's and DeepSeek's `check_slots`) needs
  the family's slot line the same way. Until then the production profile runs without the guard and without a slot check.
- **Fail-fast inside the engine is not verified.** With `mm3/robust` the family ends both ranks with exit 70 (a coherent
  stop); without it, the family's follower ends rank 0 with `RankLost` when rank 1 is lost
  (per its design notes); whether that happens, how fast, and with which exit code is not measured. `scripts/pairwatch.sh`
  stops the survivor from outside within ~5 s either way; the watchdog heals on its next tick (an exit 70 only once).
- **The pool is the engine's.** The family sizes its KV pool from the memory left after the reserve; the recipe sets the
  reserve (`TENSORFOLD_MEMORY_RESERVE_GIB=8`), not the pool. Whether `PARALLEL` slots share the pool or each reserves a
  whole `CONTEXT` is not confirmed; the calculator assumes the latter (conservative).
- **`/health` fields.** `scripts/watch.sh` reads TensorFold's `/health` (`ok`, `status`, `stall_reason`, `inflight`,
  `requests_running`, `streams.max`, `rounds.last_age_s`) and rechecks one startup-stall shape seen on the DeepSeek engine.
  Whether the family reports the same fields is not checked.
- **RoCE.** The RoCE failure marker (`TF_ROCE_MARK`) and the memory floor (`TF_MM3_FLOOR_*`) are read only by an engine
  that includes `mm3/robust` (work in progress); until then a RoCE setup failure falls back to NCCL inside the engine with
  no marker, and nothing pauses a prompt near the floor but memguard. `TF_ROCE` must be equal on both ranks and is not in
  the ranks' fingerprint yet; the recipe always passes the same value to both.
- **Request features wait for the engine branch `mm3/api-tools`.** The family as imported refuses `tool_choice`
  "required" and a named function and a grammar on the first token (so `response_format`), and the pack's always-adaptive
  template ignores `enable_thinking`. That branch adds all of them plus `thinking_budget` and the five effort levels
  ([PORT, Engine features](PORT.md#engine-features-the-plan)). Until `ENGINE_COMMIT` includes it, the harnesses report
  those cases SKIP (`ENGINE_API_TOOLS=0`); the profile's engine has it (`ENGINE_API_TOOLS=1`). How requests beyond `PARALLEL=1` are handled (queued or refused) is not checked.
- **`TF_CUDA_SCHED`** (the DeepSeek profile's host wait mode) is not confirmed for v1.1.2's native server and is not set.

## The pack

- **The tool-agentic quality gate failed.** The pack card's teacher-forced check (assistant tokens, rows up to 2,048):
  tool_agentic KL 0.0416 against a bar of 0.0127 and top-1 95.2 % against 97.1 % on 18 conversations; all, code, chat /
  multilingual / STEM, math, science and reasoning passed. 14 of 32 tool conversations were dropped (no assistant token in
  2,048). A re-test on 66 tool conversations is in progress. Tool calling through this server is to be measured (`bench/tooleval`, `bench/toolcall.py`) before relying on it.
- **Every shard is checked.** The pack card's old "upload incomplete" note was removed at `5bff3d73` (only README.md
  changed); all 26 shards are in the repository and each one's LFS sha256 equals its `SHA256SUMS` line (checked
  2026-10-24). `scripts/prepare.sh` checks the downloaded shards against `SHA256SUMS` once per revision.
- **Long context is untested.** The pack was scored at 2,048 tokens at most; nothing has exercised the sparse-attention
  indexer at 131,072 tokens on this pack.
- **FP8 KV fidelity is not measured.** K and V as e4m3 with the index keys bf16 (MSA's block choice is a hard top-k over
  index scores); `KV=bf16` is the fallback (the calculator: 146,000 tokens of room instead of 238,000).
- **Images are off by default.** Measured once (`VISION=1`, CONTEXT 23,680): smoke 9/9, 10 visual questions 10/10. The pack carries the vision tower and projector unchanged; the engine
  branch `mm3/vision` loads them on rank 0 (`VISION=1`). Its memory (2.27 GiB of tower and scratch on rank 0, plus
  12 KiB an image token on every rank, out of the KV pool) is the engine work's estimate, and `bench/vision.py` has run against the stand-in server only. Video is not served: a
  video part gets HTTP 400. An engine without `mm3/vision` refuses `--vision` at start (`VISION=0`).
- **Images are decoded as Hugging Face does, not as a photo viewer does.** The engine reproduces the reference
  preprocessing exactly (PIL `convert('RGB')`, then transformers 5.20.2's MiniMax processor): the alpha channel is
  dropped, not composited over white, so a transparent pixel shows whatever color is stored under it (often black);
  and no EXIF orientation is applied, so a phone photo taken rotated is seen as stored, sideways or upside down. Rotate
  and flatten images on the client when it matters.
- **Images are off by default (`VISION=0`)**, because of the next item; an all-resident run with images uses a smaller
  `CONTEXT` (38,400 at 110 GiB available, docs/OPERATIONS.md section 3a). With the KV tier (the production profile) the
  images come out of the slice cache instead: 5.26 GiB left at 117.1 GiB available (1.82 needed), 0.16 at the 112 GiB
  gate (`scripts/quant_budget.py --serve --context 1048576 --parallel 4 --kv fp8 --reserve-gib 8 --kv-tier
  --vision-image-tokens 2304`); not checked with an engine carrying both mm3/vision and mm3/kv-tier.
- **Images shrink the KV pool, and at the gate's edge the context no longer fits.** All-resident (131,072 x 1, the smoke
  profile), with `VISION=1` the KV pool is 2.30 GiB smaller on both ranks (52,352 tokens, 409 pages). At the idle 117.1
  GiB available that leaves 185,984 tokens, above `CONTEXT`. With less available the margin goes first: at 113 GiB
  available, 144,768 tokens text only but 92,288 with images (short of 131,072 by 1.69 GiB); at 110 GiB, 76,288 and
  23,808 (neither holds `CONTEXT`). A node idling at 110-113 GiB needs a smaller `CONTEXT` or `VISION=0` for a
  full-length request (`scripts/quant_budget.py --serve --available-gib <its MemAvailable> --vision-image-tokens 2304`).
  How the engine answers a pool short of `CONTEXT` (a shorter window or a refused load) is to be seen on the first run.
- **License.** The pack ships the MiniMax Community License (LICENSE), including its Prohibited Uses; commercial use needs
  the attribution and notice of its clause 2. This repository's Apache-2.0 covers the recipe only.

## Memory

- **The profile is the calculator's.** `MEM_GATE_GIB=113`, the 8 GiB reserve and the 131,072-token context come from
  `scripts/quant_budget.py --serve` with the family's own budget terms; the idle MemAvailable (117.1 GiB) and the 3 GiB
  serving dip are our earlier GB10 measurements (an idle Spark, and the DeepSeek recipe). The real numbers come from the first start.
- **Equal nodes are assumed.** Both Sparks have the same MemTotal since the worker's BIOS and firmware were brought to the
  head's versions (2026-10-24); `preflight` warns when they differ by more than 0.5 GiB. One value for every floor, checked
  on both ranks.
- **earlyoom is not under the recipe's control.** On our nodes it runs `-m 2,1` (SIGTERM at 2.43 GiB) with
  `--avoid ^(...|tensorfold-|...)`, which lowers a rank's score but does not exempt it. The recipe's floors (memguard 3 /
  2.75 GiB, the guard's 4 GiB) sit above it; `preflight` warns when a node's earlyoom level reaches `MEMGUARD_HARD_GIB`.
- **The memory gate's slack is a choice.** The gate passes on MemAvailable after a drop at `MEM_GATE_GIB` with MemFree
  at most `MEM_GATE_SLACK_GIB` (4) under it. On GB10 the page cache is GPU memory too: whether the engine's KV pool at a
  MemFree 4 GiB short of the gate is still the calculator's (or the load reclaims that cache first) is to be read from the
  pool line of the first starts that pass this way. A smaller slack waits longer; 0 is the old MemFree-only rule.
- **NFS weights** (`WEIGHTS=nfs`, MiaAI-Lab's default): rank 1's load speed over NFS and the worker's page-cache cost are
  not measured; the production profile copies the checkpoint to the worker instead.

## Operations

- **Prepared weights are not faster yet.** On one GPU a prepared read (21.0 s) lost to the checkpoint (14.9 s) for the
  same 73 GiB, and the one-time write took 102 s (OPERATIONS 3b). The TP2 case, where the checkpoint path over-reads,
  is not measured; `PREPARED` stays off by default.
- **No prompt-cache hit observed.** Every logged request on the TP2 profiles shows `cached=0`, including two identical
  prompts in a row (window-all run 206). Whether the family's kept prompt states get any memory on these profiles
  (`PROMPT_CACHE_GIB` is unset, and the engine's default is what 70 % of RAM leaves past the loaded server) or whether
  resumes are off for TP2 is not established. docs/CLIENTS.md's stable-prefix guidance is written for when it works;
  measure with the server log's `cached=` before promising revisit speedups.

- **MiaAI-Lab's speed survey sets GPU limits; do not apply it.** `docs/speed-borrow-deepseek.md` (her engine notes,
  kept as imported) suggests one ceiling for both Sparks with `nvidia-smi`; the operator's setting on both nodes is
  stated in [OPERATIONS section 4](OPERATIONS.md#4-temperature-the-devices-limits) and is never changed by this recipe or by hand.
- **Thermal handling is checked on the CPU only.** memguard's reactions (device `emergency`, its own 93 °C / 30 s hold) and
  the guard's pause on `throttled` are tested against fake state files and readings; the device services are the
  operator's and were verified on both nodes by them. The first two-node run records the real peaks per rank. Pausing new
  work on `throttled` needs the guard, which waits for the engine's boot line (above): until then a throttled node keeps
  serving the requests it has (at its lower GPU limit), start refuses, and the stop on `emergency` still protects the pair.

- **The watchdog's probe and the guard with `API_KEY=1`.** The watchdog sends the key from its file; the guard's own calls
  to the engine (`/tokenize`) send none, so the guard needs `API_KEY=0` behind an authenticating proxy.
- **Drafting.** `DRAFTS=1` (copies), `DRAFTER=dspark` and `DRAFTS=dflash2` are verified token for token by the target, but
  their acceptance and speed on this MoE are not measured; a verify window reads ~3.4x a one-token step's bytes
  (docs/dspark-plan.md). The measurements are planned for the next TP2 window ([DRAFTS](DRAFTS.md):
  `config/drafts.env.example`, `bench/drafts.py`; [DFLASH2](DFLASH2.md): `scripts/measure-drafts.sh`); until they have
  run, production keeps `DRAFTS=0`. The DFlash2 draft's depth prior in the engine is GLM-5.3's survival, a placeholder
  until the TP2 window measures this pack's ([DFLASH2](DFLASH2.md) section 5); its 0.5 GiB runtime term is an estimate.
  RadixArk's DFlash2 draft is not supported (the engine refuses its causal block; its card declares no license).
- **Prepared weights are the engine's, and new.** `PREPARED=1` passes `TF_MM3_PREPARED=/prepared` and
  `TF_MM3_PREPARED_WRITE=1` to an engine that reads them (another work stream's); the recipe's side (folders, free space,
  list / clean, the per-rank `read|wrote|none|refused` lines in the boot record) is checked on the CPU only. The file
  size (~94 GiB a rank), the time a write adds to the first boot and the time a read saves are NOT MEASURED; so is what
  the engine does with a partial file left by a killed write (a `refused` line is expected).
- **The boot start and the user units need a reboot to verify.** `mm31-serve.service`'s ExecStop runs `serve.sh stop`
  while the user manager stops; whether docker (a system service) and the link to the worker are still up then is not
  under a user unit's ordering, so the stop at shutdown may only save logs and leave the ranks to docker's own stop. The
  serving marker survives either way, so the boot start still starts the pair. `systemctl is-system-running` reading
  `stopping` at that moment (which keeps the marker) is checked on the CPU only.
- **Upgrade and rollback are checked on the CPU only.** The refusals, the last good record and tag, the pin and its
  re-run are tested with fakes; a real failed upgrade rolling back by itself is to be run once on the pair (docs/PORT.md,
  item 17). The pair is down from the stop until the new start, or the rollback, is LIVE.
- **`ready` needs the canary.** A pair started with `CANARY=off`, or whose rank 0 was started by hand, is never `ready`
  (no canary record for that container's start time); `health` still reports it alive.
- **Log rotation copies, then truncates.** A line written between the copy and the truncation is lost (the copytruncate
  caveat); the window is one `cp` of a file of at most `LOG_ROTATE_MIB` MiB.
- **Imported scripts keep their shellcheck notes.** MiaAI-Lab's scripts carry style and warning findings (intended word
  splitting); `tests/run.sh` fails on shellcheck errors only, and the recipe's own changes add no new finding.
