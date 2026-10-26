# Operations

> The profile is measured on one pair ([results/window-all-2](../results/window-all-2/README.md); open items:
> [KNOWN-ISSUES](KNOWN-ISSUES.md)). This page is how the pair is run;
> [AGENTS.md](../AGENTS.md) is the step-by-step bring-up.

Terms: **head** = the Spark that runs the scripts, rank 0, the API, the pair watch, the guard and the watchdog;
**worker** = the other Spark, rank 1, reached from the head over ssh (`WORKER`).

## 1. The commands

`scripts/serve.sh` is the entry point (the GLM-5.3 and DeepSeek-V4.1 recipes' interface); the launch itself is MiaAI-Lab's
`./start.sh` / `./stop.sh`, which `serve.sh` calls.

| Command | What it does |
| --- | --- |
| `build [--rebuild]` / `prebuild` | the image (zig/Dockerfile: TensorFold v1.1.2 + `ENGINE_PATCHES`), copied to the worker, compared by content, family gate. The kernels compile into the binary at build time, so nothing compiles beside loaded weights; refuses while the server runs |
| `prepare [--dflash2]` | build, then the checkpoint here (license, `SHA256SUMS` once per revision) and rank 1's copy (`WEIGHTS`); with `DRAFTS=dflash2` also the DFlash2 draft on both Sparks, sha256-pinned. `--dflash2`: only that draft ([DFLASH2](DFLASH2.md)) |
| `preflight [static]` | read-only checks of both nodes (AGENTS.md section 4); `static` skips ports, GPUs and MemFree |
| `start` / `up` / `restart` | preflight (`PREFLIGHT`), then `./start.sh [restart]` (below); at the end one line of each phase's seconds, kept in `STATE_DIR/boot-times.jsonl` (section 6) |
| `stop` | `./stop.sh`: the pair watch stops, the stop marker is written, each rank's log is saved (gzipped, `LOG_KEEP`), both containers removed |
| `status` / `logs 0\|1` / `canary` / `xid [SINCE]` / `args` | inspection; `args` prints both ranks' `docker run` without starting anything |
| `health [--json]` / `ready [--json]` | liveness and readiness (below), for people or as one JSON object |
| `boot-times [N]` | the last N starts' phase timing (section 6) |
| `prepared [list \| clean [all]]` | the prepared per-rank weights' files on both nodes (section 3b) |
| `upgrade` / `rollback [--clear]` | a new image or pack revision with an automatic way back, and the way back by hand (section 8) |
| `watch [--once]` | the watchdog (section 5) |

**Health and readiness.** `scripts/serve.sh health` is liveness: both ranks running, `/health` 200, `/v1/models` lists
`SERVED_NAME`. `scripts/serve.sh ready` is readiness, for a supervisor or a load balancer: everything `health` checks, and
the last start's canary passed for the rank 0 container that runs now (start.sh writes `STATE_DIR/canary-last` with the
container's start time), both nodes' device thermal state is `normal`, and no thermal-stop or fail-fast record exists.
Both exit 0 only when every check passes; `--json` prints one object, `{"check": "ready", "ok": true, "utc": ...,
"checks": {"rank0": {"ok": true, "detail": "running"}, ...}}` (checks `rank0`, `rank1`, `health`, `models`, and for
`ready` also `canary`, `thermal_head`, `thermal_worker`, `thermal_stop`, `failfast`). A pair started with `CANARY=off` is
never ready; a container started by hand (not by start.sh) neither.

What a start does, in order: one start at a time (`STATE_DIR/start.lock`; the watchdog stands down while it is held); the
guard's lifecycle decision when `GUARD_CONFIG` is set; setup (`scripts/prepare.sh`) when the prepared marker differs; the
family gate and `tensorfold-native info`; the link and `LINK_NETDEV` / `LINK_HCAS`; the serve flags against the build's
capabilities; any old server stopped; `PORT`, `MASTER_PORT`, `MASTER_PORT + 1` free; no CUDA process on either Spark; page
cache dropped and the memory gate on both (section 3); the prepared folders with `PREPARED=1` (section 3b); memguard on
both; rank 1, then rank 0; the stop marker cleared; the pair watch; `/v1/models` within `WAIT_TIMEOUT`; caches dropped;
smoke test; canary (`CANARY=strict` stops both ranks on a failure and saves their last logs in
`STATE_DIR/canary-fail-r{0,1}.log`; the verdict goes to `STATE_DIR/canary-last`); caches dropped again; the serving
marker (`STATE_DIR/serving`, section 5a). After a start whose canary passed, `serve.sh` records the image and revision as
the last good (section 8).

## 2. Configuration

`config/prod.env` (from `config/prod.env.example`, which documents every setting) through `scripts/config.sh`. Order of
precedence: the environment, `scripts/local.sh`, `./.env`, `config/prod.env`, the defaults. A placeholder left in the
config stops every script. Every `TENSORFOLD_*`, `TF_MM3_*`, `TF_ROCE_*` and `TF_CUDA_*` line reaches both ranks; a
credential-named one is refused (it would show in `docker inspect`).

`config/smoke.env.example` is the first TP2 load's profile: the production values on the `mm3/integrate` engine
(`ENGINE_COMMIT`, `ENGINE_PATCHES`, `ENGINE_API_TOOLS=1`), `TF_MM3_GRAPHS_TP=0` on both ranks, and `WEIGHTS=dir` (each Spark's
own plain folder of the pack, `MODEL_DIR` / `WORKER_MODEL_DIR`, mounted at `/model`, compared by size, never copied). Copy it
to `config/smoke.env`, fill in its two placeholders, and run every command with `CONFIG=config/smoke.env`.

`config/drafts.env.example` is the drafts measurement's profile ([DRAFTS](DRAFTS.md)): the smoke profile with
`DRAFTS=1 DRAFTER=dspark`, `TF_MM3_DSPARK_STATS=1` on both ranks (rank 0 logs the draft's acceptance by position for each
request) and `MEM_GATE_GIB=115` (the draft's 2.19 GiB a rank, spent before any KV). `config/smoke.env`, unchanged, is its
drafts-off control: the two files hold the same `WORKER`, `ENGINE_COMMIT` and `ENGINE_PATCHES`, and nothing but the draft
differs (tests/test_repo.py checks both the files and the ranks' commands).

The rank contract is the family's (docs/rank-contract.md), not the DeepSeek engine's. The skeleton's planned names map as:

| Skeleton / DeepSeek name | This recipe | How it reaches the engine |
| --- | --- | --- |
| `TF_TP_WORLD`, `TF_TP_RANK` | (computed) | `TF_MM3_WORLD=2`, `TF_MM3_RANK=0\|1` |
| `TF_TP_MASTER`, `TF_TP_PORT` (29571) | `MASTER_ADDR` (default: the head's link address), `MASTER_PORT` (29620) | `NCCL_COMM_ID=<addr>:<port>`; the liveness line on port + 1 |
| `TF_TP_DEVICE`, `TF_TP_ROCE_HCA` | none | not read by the family |
| `TF_COMM_BACKEND=roce` | `ROCE=1\|0` | `TF_ROCE=0` when 0 (else the engine's RoCE with NCCL fallback) |
| `GLM53_TF_ROCE_MAX_KB` | `ROCE_MAX_KB` | `TF_ROCE_MAX_KB` |
| `GLM53_TF_ROCE_FAST` | none | not read by the family |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | detected (`LINK_NETDEV`, `LINK_HCAS` pin them) | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` (both rails), `NCCL_IB_GID_INDEX`, `TF_ROCE_GID_INDEX` |
| `TF_MINIMAX_CONTEXT`, `TF_MINIMAX_SLOTS` | `CONTEXT`, `PARALLEL` | `--context`, `--parallel` (both ranks) |
| `TF_MINIMAX_KV_DTYPE` | `KV` | `--kv-cache fp8\|bf16` (both ranks; in the ranks' fingerprint) |
| `TF_MINIMAX_POOL_TOKENS`, `KV_BUDGET_GIB` | `TENSORFOLD_MEMORY_RESERVE_GIB` | the engine sizes its pool from what is free less this reserve |
| `TF_MINIMAX_PF_WS_FLOOR_GIB` | none (the engine prices 1.5 GiB of prompt workspaces) | |
| `TF_MINIMAX_PLAN_PIN`, `CPUSET_CPUS` | none | the family has no plan thread to pin |
| `ENGINE_BIN`, `ENGINE_FOLLOW_ARGS` | none | the same `tensorfold-native serve` on both; rank 1 serves no HTTP |
| `MODEL_DIR`, `STAGE_DIR`, `PACK_*` | `MODEL_ID`, `MODEL_REVISION`, `HF_CACHE`, `WORKER_HF_CACHE`, `WEIGHTS` | the snapshot mounted read-only at `/hf` |
| `TF_CUDA_SCHED` | not set | not confirmed for v1.1.2's native server |

## 3. Memory

Both Sparks have the same MemTotal (127.6 GB) since the worker's BIOS update to 5.36_1.8.0 (2026-10-24) brought it to the
head's BIOS and firmware (it gained ~2 GiB), so either rank can bind: every level below is one value, checked on **both**
ranks (the DeepSeek recipe's rule, PR #79); `preflight` warns when the two MemTotal differ by more than 0.5 GiB. Highest
first:

| Level | Value | What happens |
| --- | --- | --- |
| `MEM_GATE_GIB` | 115 GiB MemAvailable, MemFree >= 111 | before a load, on both, after a page-cache drop (below); the start stops after `MEM_GATE_TIMEOUT` (600 s) |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 GiB | what the engine leaves free when it sizes the KV pool. Unset, the engine takes a tenth of MemTotal (12.2 GiB) and this pack gets ~6 GiB of KV a rank |
| `TF_MM3_FLOOR_GIB` | 5 GiB MemAvailable, either rank | the engine (mm3/robust): a new prompt waits while other streams run |
| `TF_MM3_FLOOR_HARD_GIB` | 4 GiB, either rank | the engine (mm3/robust): no prompt chunk starts on either rank; `TF_MM3_FLOOR_TRANSIENT_GIB` (0.5) is what one chunk may take beyond its scratch. All three are in the ranks' fingerprint: unequal values refuse the start |
| the guard's hard STOP | 4 GiB MemAvailable | when the guard runs (section 4a) |
| `MEMGUARD_GIB` | 3 GiB for 4 s | `scripts/memguard.sh` kills that rank's container; the pair watch stops the other |
| `MEMGUARD_HARD_GIB` | 2.75 GiB, two samples | the same, at once |
| earlyoom (if the nodes run it) | 2.43 GiB (`-m 2,1`) | SIGTERM to the process its score picks. Its `--avoid ^(...\|tensorfold-\|...)` lowers a rank's score, it does not exempt it. The recipe's floors sit above it so the recipe picks what stops; `preflight` reads earlyoom's arguments on both nodes |

The arithmetic: `python3 scripts/quant_budget.py --serve --context 131072 --parallel 1 --kv fp8 --reserve-gib 8
--vision-image-tokens 2304` (the family's own budget terms: a rank's weights 92.66 GiB, the reserve, 4.5 GiB CUDA / NCCL /
graphs, 1.5 GiB workspaces, KV 47,040 B a token a rank in 128-token pages). At the 117.1 GiB an idle Spark has
available, that leaves 10.44 GiB (238,336 fp8 tokens, 1,862 pages) for the pool text only, and 8.15 GiB (185,984
tokens, 1,453 pages) with `VISION=1`: the engine takes rank 0's vision tower, its scratch and the image rows (2.30 GiB)
out of the pool, and both ranks size their pool to rank 0's (section 3a). `MEM_GATE_GIB` is 113 either way. Change `CONTEXT`, `PARALLEL`, `KV`, `DRAFTER` or the reserve and run it again: `MEM_GATE_GIB` follows
(tests/test_repo.py checks the two agree). The page cache is dropped (`echo 3`) before the load, after the server
answers and after the canary; while the gate waits it drops it again (`echo 3`) every 10 s (scoped sudo, AGENTS.md).

### The memory gate

The gate asks for what the profile needs, not for MemFree a node does not reach. After the drop, on **both** nodes:

- **MemAvailable >= `MEM_GATE_GIB`**: MemFree plus what the kernel can still reclaim, which after `echo 3` is the page
  cache a drop could not free at once. `MEM_GATE_GIB` stays `scripts/quant_budget.py --serve` for the profile (112 for
  production, 113 for the all-resident smoke profile, 115 / 116 with a draft).
- **MemFree >= `MEM_GATE_GIB - MEM_GATE_SLACK_GIB`** (default 4 GiB of slack: 108 for production). On GB10 the page cache
  is the GPU's memory too, so the slack bounds how much of the profile may still sit in reclaimable cache when the load
  starts.

The earlier rule (MemFree alone at `MEM_GATE_GIB`) waited minutes for memory the node did not have free yet: on the first
TP2 starts MemFree after the drop sat at 110-113 GiB and rose about 1 GiB a minute, so two starts at a 115 / 116 GiB gate
spent 4-6 minutes in it (312 s and 425 s to LIVE, against 66 s for a start that passed it at once). While it waits the gate
logs which node waits and why, every 10 s (`Waiting for memory (40s of 600s): <worker>: MemFree 110.2 GiB < 112
(MEM_GATE_GIB - MEM_GATE_SLACK_GIB 4)`), and a start that runs out of `MEM_GATE_TIMEOUT` stops with the last reason.
`preflight` reports the same rule's verdict now, as a warning. The rule is `mem_gate_check` in scripts/nodes.sh (tested
with fake meminfo values in tests/test_ops.py).

### The KV tier (opt-in, off by default)

`KV_TIER_DIR` empty is today's all-resident pool. With it set and `TF_MM3_KV_HORIZON` empty, the tier is exact: the whole
context's index keys stay on the GPU and only K/V pages go to NVMe (contexts to ~250K tokens a rank at one stream). With
`TF_MM3_KV_HORIZON` below the context it is approximate past the horizon, which is what 4 x 1M needs (below).

At 47,040 B a token a rank, 4 x 1,048,576 tokens need ~188 GiB of KV on each rank, and the room holds ~10 GiB. With
`KV_TIER_DIR` set, the engine (mm3/kv-tier, `docs/receipts/mm3-kv-tier/DESIGN.md` in TensorFold) keeps every page on that
folder's NVMe, one O_DIRECT file a rank (no page cache), mounted at `/kvtier`. The GPU holds a 512-byte summary of every
page's index keys a layer (0.94 GiB at 4 x 1M), the index keys of each stream's last `TF_MM3_KV_HORIZON` tokens (1.88
GiB), and a cache of whole page slices (the rest of the room). Every layer of every step scores the horizon exactly, ranks
the older pages by their summaries' bound, rereads the best `TF_MM3_KV_CANDIDATES` of them exactly, and attends the
selected pages from the cache, reading missing ones from NVMe first.

- `python3 scripts/quant_budget.py --serve --context 1048576 --parallel 4 --kv fp8 --reserve-gib 8 --kv-tier` prices it:
  `MEM_GATE_GIB=112`, `TF_MM3_KV_TIER_GIB=252` (187.5 GiB of live pages and 64 GiB for retained sessions), and refuses
  a room whose slice cache would hold less than twice the slices pinned at once.
- start.sh makes both folders and refuses unless each disk has `TF_MM3_KV_TIER_GIB + KV_TIER_KEEP_GIB` (300) GiB free
  (the tier's own files count as free).
- The selection equals the all-resident pool's while a stream's pages past the horizon number at most
  `TF_MM3_KV_CANDIDATES` (34,816 tokens with the defaults). Past that it is a candidate search whose effect on answers is
  NOT MEASURED (the engine's receipt has its recall on real pack layers); qualify it as section 7 says.
- The floors are unchanged: the tier's device bytes are made at boot inside the room, and its runtime extra is 8 MiB of
  pinned staging. Decode at 1M is NOT MEASURED at TP2 (the receipt has one node's numbers).

Never build an image beside loaded weights (`serve.sh build` refuses while the server runs). Keep other workloads off both
nodes.

## 3a. Images

With an engine that includes the branch `mm3/vision`, the server reads images with `VISION=1`. **The profile's default
is `VISION=0`**: the images' memory comes out of the KV pool, and on a node with 110-113 GiB available the all-resident
pool then no longer holds the 131,072-token context (table below); with the KV tier it comes out of the slice cache,
which is short at the 112 GiB gate (docs/KNOWN-ISSUES.md). The settings, in
`config/prod.env`:

| Setting | Default | Serve flag | What it does |
| --- | --- | --- | --- |
| `VISION` | 0 | `--vision` (both ranks) | 1 loads the vision tower on rank 0 (lower `CONTEXT` with it, below); 0: text only, an image part gets HTTP 400. An engine without `mm3/vision` refuses the flag at start |
| `VISION_MAX_IMAGES` | 4 | `--vision-max-images` (both ranks) | images a request at most; one more gets HTTP 400 |
| `VISION_IMAGE_TOKENS` | 2304 | `--vision-image-tokens` (both ranks) | a request's images together: 4 x 576, so every image keeps its full size |
| `VISION_URLS` | 0 | `--vision-urls` (rank 0 only) | 1: an image part may name a public https URL, which the head downloads (10 MB an image, 20 MB a request; private, loopback and link-local addresses refused); 0: `data:` URLs only, an https part gets HTTP 400 |

**Requests.** OpenAI chat: a `messages[].content` part `{"type": "image_url", "image_url": {"url":
"data:image/png;base64,..."}}`. Anthropic `/v1/messages`: an `image` block with `"source": {"type": "base64",
"media_type": "image/png", "data": "..."}`. JPEG, PNG and WebP are read; the engine decodes JPEG with libjpeg-turbo 3.1.4.1
and WebP with libwebp 1.6.0, both built into the image from their release tarballs (zig/Dockerfile,
`TENSORFOLD_LIBTURBOJPEG`, `TENSORFOLD_LIBWEBP`). An image is resized to at most 672 x 672 pixels (its aspect kept) and
cut into 28 x 28-pixel tokens: at most 576 tokens an image.

**Refused, with an HTTP 4xx and a message:** more than `VISION_MAX_IMAGES` images; an image over 10 MB (20 MB a request);
other formats (GIF, animated pictures); video parts; an https URL without `VISION_URLS=1`; bad base64. A refusal answers
before any token and does not touch the ranks.

**Memory.** The tower loads and runs on rank 0 only (the head): 2.27 GiB with its scratch and its first call's plans. Every rank holds a request's
image rows (`PARALLEL` x `VISION_IMAGE_TOKENS` x 6144 x 2 bytes: 27 MiB at the profile). The engine adds both to its
weights term, so they come out of the KV pool, not out of the load's MemFree, and since the host admits by the smaller
rank's room both ranks size their pool to rank 0's. `scripts/quant_budget.py --serve ... --vision-image-tokens 2304`
prints it: 2.30 GiB, 52,352 tokens (409 pages of 128) fewer, 185,984 instead of 238,336 at 117.1 GiB available, which
still holds the 131,072-token context; `MEM_GATE_GIB` stays 113. With less memory available it does not:

| MemAvailable before the load | text only: pool tokens (pages) | `VISION=1`: pool tokens (pages) | holds 131,072? |
| ---: | ---: | ---: | --- |
| 117.1 GiB (idle Spark) | 238,336 (1,862) | 185,984 (1,453) | both |
| 113 GiB | 144,768 (1,131) | 92,288 (721) | text only |
| 110 GiB | 76,288 (596) | 23,808 (186) | neither |

Every row loses 2.30 GiB, 52,352 tokens, 409 pages to the images (`scripts/quant_budget.py --serve --available-gib A
[--vision-image-tokens 2304]`). **To run with images on such a node**, lower `CONTEXT` to the pool less the engine's
one-page verify window: at 110 GiB that is `CONTEXT=23680` (185 pages), with `MEM_GATE_GIB=110` (the load's terms with
the images counted, rounded up), as in the TP2 window's vision run (`config/smoke.env.example`, `VISION_RUN`):

```bash
VISION=1 CONTEXT=23680 MEM_GATE_GIB=110 CONFIG=config/smoke.env scripts/serve.sh restart
```

The tower's 2.27 GiB is measured on one Spark (the engine's GPU parity run); the pair's pool is NOT MEASURED: record
the engine's pool line and rank 0's MemAvailable after the load with `VISION=1` and with `VISION=0` (section 7).

**Decoding** follows the reference exactly (PIL `convert('RGB')`, transformers 5.20.2's MiniMax processor): alpha is
dropped (not composited over white) and EXIF orientation is not applied (docs/KNOWN-ISSUES.md).

**Check it** on a running server (bench/README.md): `python3 bench/vision.py smoke` (a red circle described over the OpenAI
route, a blue square over `/v1/messages`, an image with a `record_color` tool call, the refusals above, and the server
still answering afterwards), then `python3 bench/vision.py bench` (time to the first token and the image's prompt tokens
at three sizes, and 10 synthetic visual questions with exact answers).

## 3b. Prepared weights (`PREPARED=1`)

The engine side is the branch `mm3/boot` (`3a8093c6f`, on `mm3/window-all` `3fdc50c67`): an engine without it ignores
`TF_MM3_PREPARED` and loads the checkpoint every time. Its key covers the pack's metadata digest, not the shard bytes:
after replacing shards in place under the same headers, run `serve.sh prepared clean all`.

**Measured (one Spark, one GPU, a 24-layer thin model, world 1, 73 GiB, 2026-10-25):** the checkpoint loaded in 14.9 s;
the writing boot took 102.4 s (the file written at 0.81 GB/s); the reading boot 21.0 s (the file read in 15.1 s, 5.2 GB/s).
The 32 greedy tokens were byte-identical across the three. So on one rank a prepared file is **slower** than the
checkpoint: the checkpoint path already streams at the drive's speed there. The case it was built for is TP2, where each
rank cuts its columns out of every routed tensor and the checkpoint path over-reads ~1.64x (docs/loader-speed.md); that is
not measured. Keep `PREPARED=0` unless a TP2 measurement (`serve.sh boot-times`, read against plain) shows a gain.

An engine that reads `TF_MM3_PREPARED` can keep each rank's share of the weights, laid out as that rank holds them on the
GPU, in one file, and read it instead of the pack on later boots. Off by default (`PREPARED=0`); an engine without the
feature ignores the variables.

- **Switch and folders.** `PREPARED=1` in `config/prod.env` mounts a folder of each node at `/prepared` in its rank:
  `HEAD_PREPARED` here (default `~/.cache/tensorfold-minimax-m3/prepared`, under `CACHE_DIR`) and `WORKER_PREPARED` on the
  worker (the same path in the worker's home by default; absolute or `~/`). Both ranks get `TF_MM3_PREPARED=/prepared` and
  `TF_MM3_PREPARED_WRITE=1` (0: read a file that is there, never write one).
- **Files.** A boot that finds no file for its key writes `mm3-r<rank>of<world>-<key16>.tfprep`, ~94 GiB a rank (the
  head's folder holds `mm3-r0of2-*`, the worker's `mm3-r1of2-*`); later boots read it. Each rank logs one line,
  `[tensorfold] prepared: read|wrote|none|refused ...`, which the boot timing record keeps per rank (section 6).
- **Invalidation.** The key covers the pack revision, the engine build and the layout settings. A new revision, a new
  image or a changed layout setting makes a new key: the next boot writes a new file and the old one is stale, not wrong
  (it is still the right file for the old build, which a rollback reads again, section 8).
- **Both ranks.** The switch must be the same on both ranks, on or off; `rank_env` gives both the same variables, so it
  is. The files themselves are each rank's own. A boot where one node has its file and the other none reads on one rank
  and writes on the other, and is as slow as the writing rank (`preflight` warns).
- **Disk.** ~94 GiB a rank a key. `preflight` and start want `PREPARED_FREE_GIB` (100) free on a node whose folder has no
  file yet, and report which files exist on each node. `scripts/serve.sh prepared` lists them (size, age, newest or stale);
  `prepared clean` removes the stale ones (every file but the newest of each rank), `prepared clean all` every one (with
  the server stopped). Only file names the engine writes are removed. After a rolled-back upgrade the newest file can be
  the failed build's: clean once the upgrade has settled, or the last good build's file goes and its next boot writes it
  again.

The time a read saves and the cost of a write are NOT MEASURED (docs/KNOWN-ISSUES.md).

## 4. Temperature: the device's limits

Heat and GPU limits are settings of each machine, not of this recipe. The recipe reads them; it never defines them, and
`config/prod.env` may not set them (config.sh refuses `GUARD_PAUSE_C`, `GUARD_RESUME_C`, `GUARD_KILL_C`,
`GUARD_KILL_HOLD_S` there).

On each Spark, identical on both:

| Piece | What it does |
| --- | --- |
| `/etc/spark/thermal.env` | the source of truth: `GUARD_PAUSE_C=92`, `GUARD_RESUME_C=86`, `GUARD_KILL_C=93`, `GUARD_KILL_HOLD_S=30`, the GPU limits (`SPARK_CLOCK_*`) and the sampling interval. The reading is the hottest of the GPU temperature (`nvidia-smi`) and the ACPI thermal zones; the Sparks run in the high 80s and degrade at 94 °C. No inline comments: systemd reads it too |
| `spark-thermal-guard.service` | every 2 s: at or above 92 °C the node is `throttled` (lower GPU limit) until it is at or below 86 °C; at or above 93 °C held 30 s it is `emergency` (the lowest limit) until 86 °C. Publishes `/run/spark-thermal/state` and `/run/spark-thermal/temp`. It never kills a process: stopping a job is the job's own guard's decision |
| `dgx-gpu-clock-cap.service` | sets the normal GPU limit from the file at boot (drop-in `20-from-thermal-env.conf`) |

Both nodes cap the GPU clocks at 300-2100 MHz by `dgx-gpu-clock-cap.service` (the operator's setting); this recipe never changes it.

What the recipe does with them:

| Where | Reads | Does |
| --- | --- | --- |
| `scripts/config.sh` | `/etc/spark/thermal.env` (`SPARK_THERMAL_ENV`) | the four limits, over anything the environment says; refuses limits out of order (resume < pause <= kill < 94) |
| `start.sh` / `serve.sh start` | both nodes' file and state | refuses while the file is missing on either node, differs between them, or either node is not `normal` |
| `serve.sh preflight` / `status` | the same | reports the file (identical, its sha256), each node's state, its reading |
| `scripts/memguard.sh` (both nodes, guard or not) | its own node's file and state | on `emergency`: stops its rank (`docker stop`), the pair watch stops the other, and a `thermal-stop` record holds the pair. Its own hold too (the hottest reading at or above 93 °C for 30 s kills the rank), for a node whose `spark-thermal-guard` is not running. Refuses to start without the file |
| the guard (section 4a, when it runs) | both nodes' file (sha256 compared), state and readings | pauses admission on `throttled` (or a reading at or above 92 °C) until both nodes are `normal` and at or below 86 °C; a hard STOP of both ranks on `emergency` |
| the watchdog | both nodes' state and `thermal-stop` records | does not heal while either node is `throttled` / `emergency` or a record exists |

A thermal stop holds the pair: read `memguard.log` on that node, then `scripts/serve.sh thermal-clear`, which removes the
records once both nodes are `normal` and at or below 86 °C.

**Installing the device limits** on a fresh pair: `scripts/install-device-limits.sh` (the plan: what differs on each node,
nothing changed), then, after asking the user (it changes system files with sudo on both machines),
`scripts/install-device-limits.sh --apply`. It installs `device/` (the operator's files, verbatim: `thermal.env`,
`spark-thermal-guard`, its unit, the clock-cap drop-in, and the base `dgx-gpu-clock-cap.service` only where a node has
none), renames any other drop-in that would override the cap to `<name>.disabled-<date>`, reloads systemd, enables and
restarts both units, and reads back both units active and a fresh state file. An existing `/etc/spark/thermal.env` that
differs is shown and kept unless `--force`. Rollback on a node: `sudo systemctl disable --now spark-thermal-guard`, remove
`20-from-thermal-env.conf`, restore a renamed drop-in, `sudo systemctl daemon-reload && sudo systemctl restart
dgx-gpu-clock-cap`.

## 4a. Guard (optional)

The temperature and memory guard (`scripts/guard`, ported from the DeepSeek-V4.1 recipe's pull request #48) is a front on
`127.0.0.1:8000` with the engine moved to `PORT=8001`. It needs `boot_floor_log_regex`, the engine's boot memory line,
which only the first two-node start can show: until then the guard refuses to load, so the production profile runs
without it.

- **Policy.** A two-node sampler (every 2 s; the worker over ssh) records MemAvailable, the hottest of the GPU and ACPI
  readings, the device state (`/run/spark-thermal/state`) and the sha256 of each node's `/etc/spark/thermal.env`. The
  limits are the device's (section 4; guard config `device_limits`): admission pauses on `throttled` or a reading at or
  above 92 °C until both nodes are `normal` and at or below 86 °C; `emergency` on either node, a reading at or above
  93 °C for 30 s, or a fresh sample under 4 GiB is a hard STOP (in-flight requests cancelled, both containers stopped, the
  watchdog held). Stale or missing telemetry, a stale state file or different limit files on the two nodes block
  admission. The recipe config may not define the limits.
- **Admission.** FIFO, one prefill at a time, each admitted only when fresh samples from both nodes cover 4.25 GiB plus
  every retained reservation plus its own 0.5 GiB.
- **Identity.** Generation is forwarded only while `config/prod.env`'s SHA-256 matches a snapshotted profile and both
  containers have that profile's image, mounts and engine environment: `TF_MM3_*`, `TF_ROCE*`, `TENSORFOLD_*`, and
  `MM31_SERVE_PROFILE` (start.sh puts the serve flags there, so a `CONTEXT` or `KV` change is a new profile). A profile
  whose `TENSORFOLD_MEMORY_RESERVE_GIB` is under 4 is refused. `limits.max_inflight` must equal `PARALLEL`.
- **Device keys.** `device_limits` (default `/etc/spark/thermal.env`) and `device_state` (default
  `/run/spark-thermal/state`) name where the guard reads the device's limits and state on each node; `policy` restates
  only the fixed 4 GiB memory floor. The guard refuses a limits file that is missing, out of order or sets a key twice.
  `/__guard/status` shows the limits in force and, under `invalid_samples`, why a node's sample was not used (a stale
  state, a different limits file, a missing reading).
- **Setup.** Copy `config/guard.example.json` to `~/.config/tf-guard/guard.json`; set the paths, `worker_ssh`,
  `production_config`, `boot_floor_log_regex`; set `PORT=8001` and `WATCH_PORT=8000`; start the pair; snapshot it with
  `python3 scripts/guard/profile.py --config <guard.json> --write`; install `tf-guard-sampler.service` and
  `tf-guard.service`; set `GUARD_CONFIG` in `config/prod.env`. With the guard, keep `API_KEY=0` (the guard's own calls to
  the engine carry no key) and put authentication in a proxy in front of 8000.
- **Upgrading from the earlier policy.** A guard state file written under the old thresholds is refused ("invalid durable
  policy; no automatic reset"), on purpose. Once, with the pair stopped: archive and remove the file named by
  `state_path` (and `samples_path`), then start the guard again.
- **After a STOP.** Read the evidence first (`curl -s 127.0.0.1:8000/__guard/status`, `events.jsonl`). The same profile is
  not restarted: change what the evidence points at (memory: a smaller `CONTEXT`; temperature: a shorter context or
  fewer slots), start it, snapshot it, then `control.py prepare-profile --sha <new>` and `authorize-profile --sha <new>`.

## 5. Watchdog, pair watch and restarts

- **Pair watch** (`scripts/pairwatch.sh`, `PAIR_WATCH=1`): every 5 s from the head. When one rank has exited it stops the
  other and writes `STATE_DIR/failfast` with both states and exit codes. A lone rank would otherwise wait forever in its
  next collective holding most of its Spark's memory. Exit 70 is the engine's coherent stop (mm3/robust: both ranks end on
  purpose together, `tp_watch`); the record then says "coherent stop". This is the DeepSeek recipe's fail-fast (exit 70, ~1 s) done from
  outside the engine; a worker out of reach is not counted as an exit.
- **Watchdog** (`scripts/watch.sh`, the timer runs `scripts/serve.sh watch --once` every minute): a tick is bad when a rank
  is not running or `/health` fails. `WATCH_FAILS` (3) bad ticks in a row restart both ranks (`WATCH_HEAL=1`; command
  `scripts/serve.sh restart`, or `WATCH_RESTART_CMD`) at most every `WATCH_MIN_HEAL` s (1800). The pair watch's record, an
  exit 70, the engine's own `stalled` verdict and `WATCH_PROBE_FAILS` failed idle probes heal on that tick, at most every
  `WATCH_FF_MIN_HEAL` s (120). Every heal saves evidence (health, both logs and states, the fail-fast record) under
  `STATE_DIR/watch/heal-evidence/`.
- **Idle-only probe**: every 180 s one `max_tokens: 1` completion (header `X-TF-Probe: watchdog`, the API key from its
  file when `API_KEY=1`) proves a `/health` 200 is not a wedged engine; skipped while every slot is busy.
- **It stands down** while a start or a heal holds its lock, during the boot in which `scripts/serve.sh stop` wrote the
  stop marker (after a reboot a gone pair heals), while `WATCH_LEASE` is fresh (touch it while you benchmark or maintain
  the pair), and while the guard holds the pair. A rank 0 younger than `WATCH_GRACE` (2400 s) that does not answer yet is
  loading.
- Install: copy `scripts/systemd/tf-watchdog.{service,timer}` to `~/.config/systemd/user/`, set `WorkingDirectory`,
  `systemctl --user enable --now tf-watchdog.timer`. Before benchmarks, stop the timer or use `WATCH_LEASE`.

- **Exit 70 heals once.** A coherent stop (exit 70 on either rank, or the pair watch's "coherent stop" record) heals on the
  next tick, once. The same stop again within `WATCH_MIN_HEAL` s (1800) of that heal is not healed: the watchdog alerts
  and leaves the pair down for an operator (`STATE_DIR/watch/last_heal_70`), so a stop the engine keeps choosing does not
  become a restart loop.
- **RoCE failure marker** (`TF_ROCE_MARK=/cache/roce-failed`, mm3/robust): a failed RoCE gather writes it in that node's
  cache volume (`CACHE_VOL`, mounted at `/cache` in each rank) and the next start of both ranks serves on NCCL. `preflight`
  and `status` report it; after the link is fixed (AGENTS.md section 2), `scripts/serve.sh roce-reset` deletes it on both
  nodes and the next start tries RoCE.

## 5a. Unattended: systemd user units

All user units on the head (`~/.config/systemd/user/`), `WorkingDirectory` set to this checkout, then
`systemctl --user daemon-reload`. `loginctl enable-linger <user>` lets them run without a login session.

| Unit | What it does | Enable |
| --- | --- | --- |
| `mm31-serve.service` | oneshot, `RemainAfterExit=yes`: start runs `scripts/serve.sh start`, stop runs `scripts/serve.sh stop` (`TimeoutStartSec=4500`: gate, load and canary) | **not** for every boot: `mm31-boot-start.service` starts it |
| `mm31-boot-start.service` | once at boot, `scripts/boot-start.sh`: starts the pair only if it was serving before the reboot | `systemctl --user enable mm31-boot-start.service` |
| `tf-watchdog.timer` | the watchdog (section 5), the fallback | `systemctl --user enable --now tf-watchdog.timer` |
| `mm31-logrotate.timer` | hourly `scripts/logrotate.sh` (section 6) | `systemctl --user enable --now mm31-logrotate.timer` |

**Was it serving?** start.sh writes `STATE_DIR/serving` when the pair is LIVE. A deliberate stop (`scripts/serve.sh stop`,
`./stop.sh`, `systemctl --user stop mm31-serve`) removes it; a stop while the system is shutting down
(`systemctl is-system-running` says `stopping`: `mm31-serve.service`'s ExecStop at a reboot) keeps it. A power loss
leaves it too. So after a reboot `scripts/boot-start.sh` starts the pair when the marker is from an earlier boot, and
does nothing when it is missing (stopped on purpose, or never started), when it is from this boot, when rank 0 already
runs, or while `BOOT_LEASE` (a file you touch) is younger than `BOOT_LEASE_MIN` minutes. Before the start it waits, at most
`BOOT_WAIT` s (300), for docker here, the worker over ssh with docker up, and the link: the route to the worker over a CX7
port with a RoCE v2 GID (the `LINK_NETDEV` / `LINK_HCAS` pins apply) and every RoCE device of it `ACTIVE` on both nodes.
Then `systemctl --user start mm31-serve.service` when that unit is installed (`BOOT_UNIT`; its ExecStop then stops the
pair cleanly at the next shutdown), else `scripts/serve.sh restart`. `BOOT_DRY_RUN=1 scripts/boot-start.sh` decides and
waits but starts nothing; its log is `STATE_DIR/boot-start.log` (`BOOT_STDOUT=1`: the terminal). A start it gives up on
is left to the watchdog (`WATCH_HEAL=1`).

## 6. Logs and state

- `scripts/serve.sh logs 0|1`; each rank's whole log is saved (gzipped) at every stop and before a stopped container is
  removed: `~/.cache/tensorfold-minimax-m3/logs/<date>-<time>-rank<N>.log.gz` on its own node (the newest `LOG_KEEP`).
- `LOG_DIR` here also holds `memguard.log`, `pairwatch.log` and `thermal.log`; the worker keeps its own `memguard.log`.
- **Rotation.** `scripts/logrotate.sh` (hourly by `mm31-logrotate.timer`) rotates `memguard.log`, `pairwatch.log`,
  `thermal.log`, `STATE_DIR/watch/heal.log` and `STATE_DIR/boot-start.log` here, and the worker's `memguard.log`, once one
  is over `LOG_ROTATE_MIB` (16) MiB: copied to `<file>.1.gz` and truncated in place (copytruncate: the writers append with
  `>>` and go on writing; only a line written between the copy and the truncation can be lost), older generations shift
  up, `LOG_KEEP` (10) are kept. `--dry-run` says what it would rotate.
- **The thermal note.** Every run first appends one line a node to `LOG_DIR/thermal.log`: the device thermal state, the
  hottest GPU / ACPI reading, and the limits as that node's own `/etc/spark/thermal.env` (`SPARK_THERMAL_ENV`) states them,
  with the file's sha256. A limit it cannot read is written as `limits unreadable (no whole GUARD_KILL_C)`, never as a
  number; the script holds no temperature and no GPU value.
- **Boot timing.** Every `scripts/serve.sh start` / `up` / `restart` that reaches the memory gate prints, at the end, one
  line (`boot times (start): checks 10.2 s, gate wait 0.4 s, load 24.6 s, smoke 1.0 s (first reply 0.6 s), canary 30.1 s,
  total 66.3 s; prepared r0 read, r1 read`) and appends one JSON line to `STATE_DIR/boot-times.jsonl`: `utc`, `mode`,
  `ok`, `failed_at` (the phase a failed start stopped in), `image`, `image_id` (its content id, 12 hex), `model_revision`,
  `prepared` (`r0` / `r1`: `read`, `wrote`, `none`, `refused` from each rank's log, `-` without a line) and `seconds`
  (`checks`: preflight to the gate plus the checks after it, `gate`: the drop and the memory gate, `load`: the launch to
  `/v1/models`, `smoke` and `first_reply`, `canary`, `total`). `scripts/serve.sh boot-times [N]` prints the last N. A
  `FOREGROUND=1` start records none.
- `STATE_DIR` (`~/.local/state/tensorfold-minimax-m3`): `start.lock`, `prepared`, `stopped`, `serving` (section 5a),
  `failfast`, `canary-last`, `boot-times.jsonl`, `last-good.env` and `rollback.env` (section 8), the API key file (mode
  600) and its header file, canary-failure logs, `boot-start.log`, `watch/` (counters, heal log, heal evidence).

## 7. Qualifying the profile

The production profile is the calculator's, not a measurement. The first two-node run qualifies it, in this order, and
records each number in [RESULTS](RESULTS.md); stop at the first failure and add it to [KNOWN-ISSUES](KNOWN-ISSUES.md):

1. `scripts/serve.sh preflight`, then `start`: record the load time, both ranks' MemAvailable after the load, the
   engine's boot memory and pool lines (they give `boot_floor_log_regex` for the guard), and the canary.
2. Exactness: greedy replies repeat exactly across two runs; `scripts/check-drafted.sh` plain against drafted, if drafting
   is tried. The drafts measurement (drafts off against the dspark draft, speed, acceptance by position, token
   identity) has its own ordered steps and stop conditions in [DRAFTS](DRAFTS.md); it is NOT MEASURED yet. The DFlash2
   draft's window ([DFLASH2](DFLASH2.md) section 4: `DRAFTS=0`, then `DRAFTS=dflash2`, optionally `DRAFTER=dspark`, each
   measured with `scripts/measure-drafts.sh`) records acceptance by position and speed at C1; token equality with plain
   decode is its hard gate.
3. The harnesses (bench/README.md): tool calling (`bench/tooleval/run.sh`, `bench/toolcall.py`), structured output,
   quality (`bench/quality.py` against the pack card's offline numbers).
4. Memory: a prompt near `CONTEXT` with MemAvailable sampled on both nodes the whole time; both must stay above 4 GiB.
   Images (section 3a): `bench/vision.py smoke`, then `bench/vision.py bench`; rank 0's MemAvailable after the load with
   `VISION=1` against `VISION=0`, and under four full-size images in one request.
5. Failure paths: `docker kill` one rank (the pair watch stops the other and the watchdog heals it); `scripts/serve.sh stop`
   (the watchdog stands down).
6. Only then raise `PARALLEL` (after concurrent replies equal solo ones) or `CONTEXT`, each with
   `scripts/quant_budget.py --serve` first, and soak for several hours with the watchdog on.

## 8. Upgrade and rollback

**The last good.** After every start whose canary passed (for the rank 0 container that runs now), `serve.sh` tags the
running image `<repo>:last-good` on both nodes (so a rebuild under the same tag cannot take it away) and writes
`STATE_DIR/last-good.env`: `IMAGE`, `LAST_GOOD_IMAGE`, `IMAGE_IDENT` (the content id, the same on both nodes),
`MODEL_REVISION`, `ENGINE_PATCHES`, `ENGINE_COMMIT`, `PATCHES_HASH`, `UTC` (KEY=value lines, read, never run).
`scripts/serve.sh status` shows it.

**`scripts/serve.sh upgrade`** is the safe path to a new engine image or pack revision. Change `config/prod.env` first
(`ENGINE_PATCHES` / `ENGINE_COMMIT` for a new engine, `IMAGE` for a new tag, `MODEL_REVISION` for a new pack revision),
then run it. It refuses unless the last good image is on both nodes with its recorded content; then it stops the pair
(no build beside loaded weights: the pair is down from here until the new start is LIVE), runs `scripts/prepare.sh` (the
image built or pulled and shipped, the family gate, the checkpoint of that revision on both nodes), `preflight`, and a
start with `CANARY=strict`. A start whose canary passes is the new last good: `upgrade ok`. A failure at any step restores
the last good by itself and says so: it writes `STATE_DIR/rollback.env` (the pin) and restarts on `<repo>:last-good` and
the last good revision, then exits non-zero with `ROLLED BACK to ...`.

**`scripts/serve.sh rollback`** does the same by hand: pin, then restart on the last good. While the pin exists, every
`start`, `up`, `restart`, `args` and `preflight` (the watchdog's heals too) runs on the pinned image and revision, with
`PREPARE=0` (no rebuild of the config's image), and says so; `status` shows it. Put the last good pins back in
`config/prod.env` (or fix the new ones and run `upgrade` again, which drops the pin), then
`scripts/serve.sh rollback --clear`.

Also:

- **Prepared files** (section 3b): the old build's file keeps its key, so it stays valid for the old build and a rollback
  reads it again; the new build's boot writes its own.
- **The device limits** are not the recipe's: rolling them back is section 4's rollback lines
  (`scripts/install-device-limits.sh` installed them).
- **The recipe itself**: `git checkout <previous tag>` of this repository, then `scripts/serve.sh restart`.
- The `last-good` tag keeps one more image on each node; `docker image prune` leaves it (it is tagged).
