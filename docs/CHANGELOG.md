# Changelog

Every change to this recipe, newest first. Nothing here has been run on two Sparks with the model: see the README's Status.

## Unreleased

- **The measured profile (window-all-3).** `config/prod.env.example` is the profile measured on the pair
  (results/window-all-3, 2026-10-26): the dspark draft with each round's depth sized by its own confidences
  (`TF_MM3_DSPARK_CHANCES=1`, `TF_MM3_CALIB=auto`; `TF_MM3_DRAFT_DEPTH` no longer set), otherwise as window-all-2. Median
  decode 31.75 / 30.5 tok/s (T0 / T1) against 19.75 / 19.1 with every switch off; prefill 687 / 531 tok/s at 2K / 32K,
  window-all-2's prefill regression gone; a repeated 70K prompt's first token in 0.76 s. `config/drafts.env.example`
  measures that depth rule; `config/slots4.env.example` (four streams, drafts at depth 3, as measured) builds
  `engine/patches`. `scripts/config.sh` checks both new switches. The engine is `engine/patches`, exported from
  `mm3/window-all-3` @ `0547afe80`. The run hit the Sparks' thermal limits and stopped early: the README's Limits say
  which checks are missing.

- **Operations at parity with the DeepSeek-V4.1 recipe.** `PREPARED=1`: each node's `HEAD_PREPARED` / `WORKER_PREPARED`
  folder at `/prepared` and `TF_MM3_PREPARED` / `TF_MM3_PREPARED_WRITE=1` on both ranks (an engine without the feature
  ignores them); preflight and start report each node's file or want `PREPARED_FREE_GIB` free; `serve.sh prepared list /
  clean [all]`. `serve.sh start` (and its alias `up`) prints each phase's seconds and appends a JSON line to
  `STATE_DIR/boot-times.jsonl` (image, content id, revision, each rank's prepared outcome); `serve.sh boot-times`. The
  memory gate asks for what the profile needs: MemAvailable after a drop >= `MEM_GATE_GIB` and MemFree >= `MEM_GATE_GIB -
  MEM_GATE_SLACK_GIB` (4), logging which node waits and why (it waited 4-6 minutes on the first TP2 starts). `serve.sh
  ready [--json]` (and `health --json`) for supervisors and load balancers. `scripts/boot-start.sh`,
  `mm31-boot-start.service` and `mm31-serve.service`: after a reboot the pair starts only if it was serving.
  `scripts/logrotate.sh` and `mm31-logrotate.timer`: size-based copytruncate rotation and an hourly line a node of its
  device thermal state and the limits its own `/etc/spark/thermal.env` states. `serve.sh upgrade` / `rollback`: the last
  good image tagged and recorded after every start whose canary passed, and restored by itself when an upgrade fails.
  Checked on the CPU only (tests/test_ops.py).

- **The measured profile (window-all-2).** `config/prod.env.example` is the profile measured on the pair
  (results/window-all-2): 131,072 x 1, FP8 KV, CUDA graphs at TP2, the dspark draft at depth 3, the RoCE switches
  (`TF_MM3_ROCE_SUM`, `TF_MM3_SHARE_INPLACE`, `TF_ROCE_FAST`), `ENGINE_API_TOOLS=1`, `MEM_GATE_GIB=115`, the KV tier and
  images off. The engine is `engine/patches`, exported from `mm3/window-all-2` @ `4e6390030`. The README's results and
  limits come from that run; AGENTS, OPERATIONS, RESULTS and KNOWN-ISSUES no longer say the model was never served.

- **Release docs.** A README in the format of the DeepSeek-V4.1 recipe (summary, results, quick start, configuration,
  rollback, troubleshooting), [CREDITS](CREDITS.md) naming everyone whose work this uses, [LICENSING](LICENSING.md) for
  the recipe, TensorFold, the family patch, the drafts, ExLlamaV3, xgrammar and the model's license, and NOTICE entries
  for the MMLU data and the components built into the image. MiaAI-Lab's `CREDITS.md` and `CHANGELOG.md` moved to
  `docs/` (CREDITS-MIAAI-LAB.md, CHANGELOG.md).

- **Tool-call harness fixes from the triage of the 9/12 own and 36/42 toolcall run** (thinking off, temperature 0).
  The first-turn prompts of all 27 cases and scenarios have the token counts the checkpoint's chat_template.jinja
  gives with Python Jinja2 and the checkpoint tokenizer, and every call parsed cleanly, so the misses were the harnesses' own:
  `chains.py`'s simulated files now keep what `write_file` writes (edit-config read its write back, saw `workers: 4`
  and rewrote it until the 8-turn limit); retry-after-error asks for the corrected call outright (the old "use the tool
  exactly with the city I wrote first" made stopping after the error a valid answer). `toolcall.py`: edit_replace_all
  (read before edit), long_reasoning_tool (todowrite before read) and followup_answer (glob before read, which was
  answered "ok") answer up to 2 preparatory calls with canned results, then check the reply; `direct` counts the passes
  made without one, so the stricter first-call rate stays visible. New mode `adaptive` (enable_thinking true, no effort:
  the template's only mode) in chains.py, toolcall.py and run.sh; `--temperature -1` / `PY_TEMP=-1` leaves the
  temperature out, so the server samples with generation_config.json's settings (1.0, top_p 0.95). Tested against the stand-in server.

- **The drafts measurement's profile, bench and plan** (NOT MEASURED: the plan for the next TP2 window).
  `config/drafts.env.example`: the smoke profile with `DRAFTS=1 DRAFTER=dspark`, `TF_MM3_DSPARK_STATS=1` on both ranks (rank 0 logs the draft's
  acceptance by position for each finished stream; documented and checked in scripts/config.sh) and `MEM_GATE_GIB=115`;
  `config/smoke.env`, unchanged, is its matched drafts-off control (tests check the files and both ranks' commands differ
  only in the draft). The calculator's draft term now includes the 8 ring snapshots the engine's admission prices
  (2.19 GiB a rank instead of 2.03; the gate stays 115). `bench/drafts.py`: C1, thinking off, 256 tokens, four fixed
  classes (code, prose, structured, agent with a tool turn and its follow-up), greedy and keyed-sampled reps; client and
  server tok/s, TTFT, `speculative.*`, `token_sha256`; with rank 0's log each request's acceptance line (conditional and
  cumulative acceptance by position, tau per class); both nodes' device state and readings around each prompt; `compare`
  (speedup, token identity: a greedy mismatch fails) and `log` (a saved log's totals). `docs/DRAFTS.md`: the ordered steps,
  stop conditions and what to record. Tested against a stand-in server.

- **Image input.** With an engine that includes the branch `mm3/vision`, `VISION=1` (off by default: with images the
  all-resident pool does not hold 131,072 tokens at 110-113 GiB available, 38,400 at 110, and with the KV tier the slice
  cache is short at the 112 GiB gate) passes `--vision`, `--vision-max-images` (`VISION_MAX_IMAGES`, 4) and
  `--vision-image-tokens` (`VISION_IMAGE_TOKENS`, 2,304 = 4 x 576) to both ranks, and `VISION_URLS=1` adds
  `--vision-urls` on rank 0; the flag check names `VISION=0` for an engine without it. The image carries libjpeg-turbo
  3.1.4.1 and libwebp 1.6.0's decoder, built from their release tarballs (sha256-checked, version-checked) and named by
  `TENSORFOLD_LIBTURBOJPEG` / `TENSORFOLD_LIBWEBP`. The engine takes rank 0's vision tower and scratch (2.27 GiB) and
  every rank's image rows out of the KV pool; the calculator's `--vision-image-tokens` prices that (185,984 pool tokens
  instead of 238,336 all-resident; with `--kv-tier` out of the slice cache), and `MEM_GATE_GIB` does not move.
  `bench/vision.py smoke` (OpenAI and Anthropic routes, an image with a tool call, clean 4xx refusals of an over-large
  image, too many images, video, GIF and an https URL) and `bench` (TTFT, image tokens at three sizes, a 10-question
  synthetic VQA set), tested against the stand-in server. docs/OPERATIONS.md section 3a. Not yet run on two Sparks.

- **The DFlash2 draft profile and its TP2 measurement** (docs/DFLASH2.md). `DRAFTS=dflash2`: copy drafts plus the DFlash2
  draft `incoai/MiniMax-M3.1-Flash-DFlash2` @ `ac07a7c1` (Apache-2.0) on both ranks, for an engine built from `mm3/dflash2`
  (which reads the draft's kind from its `config.json`; no new flag). `DFLASH2_DIR` / `WORKER_DFLASH2_DIR` (default beside
  each node's pack) are mounted read-only at `/dflash2`, so both ranks pass `--drafter /dflash2`; `DRAFTER` defaults to
  `dflash2` and any other drafter is refused with it; a folder whose `config.json` names no DFlash2 class is refused.
  `DRAFTS=0`, `1` and `DRAFTER=dspark` are unchanged. `scripts/prepare.sh --dflash2` (and a full prepare with
  `DRAFTS=dflash2`) fetches the pinned revision without a token, copies it to the worker (both resumable) and checks every
  file's sha256 on both Sparks; preflight checks both folders. `scripts/quant_budget.py --serve --drafter dflash2` prices it
  as the engine does: 2.37 GiB of weights a rank, 0.71 GiB of runtime estimate, scratch and 8 ring snapshots, 21.1 MB a
  stream (gates 116 smoke, 115 production). `bench/draft_positions.py` and `scripts/measure-drafts.sh`: C1 greedy
  requests in four classes, drafts on against `DRAFTS=0` and `"draft": false`, with a_j, survival, tau, tok/s, the speed ratio and a hard
  token_sha256 equality gate, and a depth prior from the measured survival. RadixArk's DFlash2 draft is not supported (the
  engine refuses it; no license on its card). Not run at TP2.

- **The KV tier, opt-in.** `KV_TIER_DIR` turns on the engine's KV tier (mm3/kv-tier): every page on each Spark's NVMe,
  mounted at `/kvtier` (`TF_MM3_KV_TIER`, `TF_MM3_KV_TIER_GIB`); exact with `TF_MM3_KV_HORIZON` empty, approximate below
  the context (4 x 1M: `TF_MM3_KV_TIER_GIB=252`, `TF_MM3_KV_HORIZON=32768`, `TF_MM3_KV_CANDIDATES=16`). The profile stays
  131,072 x 1, all resident. `scripts/quant_budget.py --serve
  --kv-tier` prices the tier's summaries, horizon and slice cache and gives `MEM_GATE_GIB=112`; start.sh refuses unless
  both disks hold the tier plus `KV_TIER_KEEP_GIB`. The floors are unchanged. Not run at TP2 yet.

- **A worker reached as root.** A worker may be reached as `WORKER=root@<link address>` (the worker's own
  account has no docker or sudo). Every worker path already resolves in the ssh user's home (`$HOME` there: memguard,
  saved logs, thermal-stop records) and `WORKER_MODEL_DIR` takes an absolute path; `config/smoke.env.example` says when
  to give one, preflight's unreadable-folder line says a `~/` is the ssh user's home, and a test covers both.

- **The TP2 smoke profile and `WEIGHTS=dir`.** `config/smoke.env.example` (`CONFIG=config/smoke.env`): the production
  profile on an engine built from the integration branch `mm3/integrate` (`ENGINE_COMMIT` a placeholder until that branch has
  a head; `ENGINE_PATCHES` the series cut from it), `ENGINE_API_TOOLS=1`, `TF_MM3_GRAPHS_TP=0` on both ranks for the first
  load. `WEIGHTS=dir`: each Spark's own plain folder of the pack (`MODEL_DIR`, `WORKER_MODEL_DIR`, `~/` per node), mounted
  read-only at `/model`; prepare.sh checks it (shards, total_size, LICENSE, SHA256SUMS) and never downloads or copies;
  preflight and prepare.sh compare the worker's folder with the head's by size. `ENGINE_COMMIT` must be a commit hash;
  `ENGINE_PATCHES` may be `~/...`.

- **jayleaton's port onto this recipe** (docs/PORT.md has every feature and its verdict). The production profile in
  config/prod.env.example (jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31 at f76ad09a, 131,072 tokens, one slot, FP8 KV,
  `TENSORFOLD_MEMORY_RESERVE_GIB=8`, `MEM_GATE_GIB=113` from scripts/quant_budget.py --serve), read by scripts/config.sh for
  every script. `scripts/serve.sh` (build, prepare, preflight, start, stop, restart, status, health, canary, xid, watch). The
  MemFree gate with page-cache drops replaces `MIN_AVAIL_GIB`; memguard's floors raised above earlyoom's. `scripts/pairwatch.sh`
  stops the surviving rank when one exits; the watchdog (scripts/watch.sh) heals the pair. The rank environment is the family's
  contract only (`TF_MM3_*`, `NCCL_COMM_ID`, `TF_ROCE*`). `WEIGHTS=local` (default) copies the checkpoint to the worker;
  `WEIGHTS=nfs` keeps the NFS volume. LICENSE and SHA256SUMS checked by prepare.sh. HOST 127.0.0.1, PORT 8000. Engine patches
  unchanged; `ENGINE_PATCHES` / `ENGINE_COMMIT` select the family's series. Credential checks, bench harnesses, the canary,
  AGENTS.md, docs/OPERATIONS.md, KNOWN-ISSUES.md, RESULTS.md, PACKS.md. Nothing served on two Sparks yet.

- **FP8 KV cache (`KV=fp8`) in the family patches** (`patches/tasks/0016-minimax-m3-kv-fp8.patch`; design in `docs/fp8-kv.md`).
  K and V pages as e4m3 rows with an fp32 scale per token and kv head (47,040 B a token a rank instead of 76,800); the index keys
  stay bf16; `start.sh` already passes `--kv-cache` when `KV` is set. Host-tested against a float64 reference with fp32 twins and
  planted mutants; the kernels have not run on a GPU and the quality impact is NOT MEASURED.

- **First version of the recipe** (private; nothing is released). `start.sh`, `stop.sh` and `scripts/` for serving
  `Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw` (revision `fd3466a016989e0e1fceb901453046fdd7247dec`) on two DGX Sparks
  with TensorFold v1.1.2's native Zig binary, built from `zig/Dockerfile` (TensorFold at commit
  `4ff784410f70b95767684d337e59dfd6d94fec8b`, Zig 0.17.0, `sm_121`) plus `patches/`. The start and stop behaviour follows
  the GLM-5.3-Flash recipe: both Sparks set up, rank 1 started over ssh and then rank 0 with the API, loading progress, a
  smoke test and an `is now LIVE` line; `./start.sh restart`, `./stop.sh`, `FOREGROUND=1`, `PREPARE=`, `DRY_RUN=1`,
  `scripts/config.sh` with the environment, `scripts/local.sh` and `./.env` in that order, `WORKER`, `FABRIC_PEER`, and
  gzipped logs saved on both Sparks at stop.
- **A family gate.** TensorFold v1.1.2 has no `minimax_m3_vl` family and the checkpoint has never been loaded in any engine.
  `start.sh`, `scripts/prepare.sh` and `scripts/publish-image.sh` ask the image's `tensorfold-native capabilities --json` for
  the family and refuse with "MiniMax-M3.1-Flash family not in this TensorFold build; apply patches/" before anything is
  stopped, downloaded or started; `start.sh` also runs `tensorfold-native info` on the checkpoint. `ALLOW_UNSUPPORTED=1`
  overrides the gate for engine development.
- **No invented defaults.** `CONTEXT`, `PARALLEL`, `MAX_TOKENS`, `KV`, `PROMPT_CACHE_GIB` and `THINKING` are empty (the engine's
  own default), `DRAFTS=0` (the pack has no draft module), vision is off (the pack has no vision tower). Performance, KV sizing
  and quality through the server are NOT MEASURED.
- **The private checkpoint.** `scripts/prepare.sh` needs a Hugging Face token on the head (`hf auth login` or `HF_TOKEN`); it only
  tests that one exists and never prints, stores or forwards it. The worker reads the checkpoint over a read-only NFS volume.
- **Taken from the sibling recipes, changed:** the memory guard and its preflight (`MIN_AVAIL_GIB` now 100, a guess), the serve
  flags check against `capabilities --json` (now inside `scripts/nodes.sh`), the API key file, the node and link discovery. Left out
  on purpose: the DeepSeek recipe's alternative engine, overlay checkpoint, kernel kit, Engram tables and per-rank weight files,
  which belong to that model.
