# The port: what came from where

The base is Mia / MiaAI-Lab's MiniMax-M3.1-Flash TensorFold recipe and `minimax_m3_vl` family patch (private repo)
at `fd1d379` (Apache-2.0; her LICENSE, NOTICE and CREDITS kept). On top of it: what the GLM-5.3 recipe (origin/main
`dd5df45`) and the DeepSeek-V4.1 recipe (origin/main `60b6688`) learned about operating and serving two Sparks, and this
repository's own skeleton (the calculator, the guard and the watchdog). Verdicts: **ported** (the same behaviour),
**adapted** (the same purpose, changed for this engine or pack), **not applicable** (with the reason).

## Features

| Feature | Source | Verdict | Where, and why |
| --- | --- | --- | --- |
| `serve.sh` start / stop / restart / status / logs | GLM, DS | adapted | `scripts/serve.sh` over MiaAI-Lab's `start.sh` / `stop.sh`, which keep the launch (link discovery, family gate, serve-flag check, loading progress, smoke test) |
| `serve.sh build` (docker build, `save \| ssh load`, content key compare) | GLM, DS | adapted | `serve.sh build` = `scripts/prepare.sh --image-only`: MiaAI-Lab's zig/Dockerfile and image identity by layers; refuses while serving |
| `serve.sh prebuild` (extensions compiled with no weights loaded) | DS | adapted | the Zig engine compiles its kernels into the binary at image build, so `prebuild` is `build`; the rule it served (never compile beside loaded weights) is kept by refusing while the server runs |
| `serve.sh cache` (pfdense table, image routing bias) | DS | not applicable | DeepSeek-only data files |
| `serve.sh run MODULE` (an engine module on both ranks) | DS | not applicable | Python engine modules; the native binary has none for this family |
| `serve.sh args` | DS | ported | `DRY_RUN=1 ./start.sh` prints both ranks' `docker run` |
| `serve.sh preflight [static]` | GLM, DS | adapted | ssh, docker, image on both and identical and current, family gate, checkpoint and LICENSE on both, the link, driver / `vm.min_free_kbytes` / MemTotal parity, earlyoom level, `sudo -n`, ports, GPUs idle, MemFree |
| `serve.sh xid` | GLM | ported | Xid lines from both kernels' logs |
| `serve.sh gpucheck` / `gpuwatch.py` (GB10 power-state check) | GLM | not applicable | GLM's tool depends on host tuning details this repository keeps out (public wording rule); not requested |
| Health check | GLM, DS | adapted | `serve.sh health` (both ranks up, `/health` 200, model listed) and `status` (pair watch, memguard, fail-fast and stop records, MemAvailable, the last good, a rollback pin) |
| `serve.sh ready` (readiness for supervisors and load balancers, `--json`) | DS (canary gating), new | adapted | `scripts/ops.sh`: health plus this rank 0 container's canary passed (`STATE_DIR/canary-last`, written by start.sh with the container's start time), both nodes' device thermal state `normal`, no thermal-stop or fail-fast record; `health --json` / `ready --json` print one object |
| Canary after a start (`CANARY=strict` stops on failure, logs kept) | DS, GLM | adapted | `scripts/canary.py` for MiniMax's reasoning field and tool parser (raw `]<]minimax[>[` markup is a failure); MiaAI-Lab's smoke test stays before it |
| Memory gate (MemFree on both, timeout, page-cache drops) | GLM, DS | adapted | `mem_gate` / `mem_gate_check` / `drop_caches` in nodes.sh, the calculator's value for the profile (`MEM_GATE_GIB`, 112). The rule asks for what the profile needs: MemAvailable after a drop >= `MEM_GATE_GIB` and MemFree >= `MEM_GATE_GIB - MEM_GATE_SLACK_GIB` (4), not MemFree alone, which waited 4-6 minutes on the first TP2 starts for memory the node already had reclaimable; it logs which node waits and why |
| Drop caches before the load, after ready, after the canary | DS | ported | `DROP_CACHES=1`; `echo 3` every 10 s while the gate waits |
| Scoped sudoers line for the drops | GLM | ported | AGENTS.md section 1 |
| Slot check after the load (`check_slots`, one retry) | GLM, DS | not applicable | the family's slot line is unknown until a first start (KNOWN-ISSUES); add it then |
| Memory floors on both ranks, one value (PR #79) | DS | ported | every level in config.sh and OPERATIONS section 3 is checked on both ranks; preflight warns on a MemTotal mismatch |
| The engine's reserve as the serving floor | DS (`TF_DSV41_FLOOR_*`) | adapted | `TENSORFOLD_MEMORY_RESERVE_GIB=8` on both ranks; the engine default (12.2 GiB) would leave this pack ~6 GiB of KV |
| earlyoom notes | DS | adapted | memguard's hard floor (2.75 GiB) above earlyoom's 2.43; preflight reads each node's earlyoom `-m`; the `--avoid tensorfold-` behaviour documented |
| Fail-fast for the pair (exit 70, both ranks end in ~1 s) | DS | adapted | `scripts/pairwatch.sh` from outside the engine: stops the survivor within 5 s, writes `STATE_DIR/failfast`; the engine's own coherent stop (exit 70, `mm3/robust`) is recorded as such and healed once, not in a loop |
| Watchdog restart (timer, `WATCH_*`, lease, stop marker, lock) | DS (PR #48), GLM | adapted | the skeleton's `scripts/watch.sh` on MiaAI-Lab's contract: one container name, `serve.sh restart` heals, stop marker scoped to the boot, the start lock, the fail-fast record heals at once, the probe carries the API key |
| Start after a reboot (`boot-start.sh`) | DS | adapted | `scripts/boot-start.sh` and `mm31-boot-start.service`: starts only if the pair was serving before the reboot (start.sh's `STATE_DIR/serving`, removed by a deliberate stop, kept by a stop during a system shutdown), after docker, the worker and the link's RoCE ports are up; `BOOT_LEASE`, `BOOT_DRY_RUN=1`; the watchdog stays the fallback |
| User units for the pair (`mm31-serve.service`) | DS (systemd units) | adapted | oneshot, `RemainAfterExit=yes`: `serve.sh start` / `serve.sh stop`, a start budget of 4500 s; not enabled for every boot (the boot start starts it); the `tf-watchdog` and `tf-guard` units kept as they were |
| Prepared per-rank weights (`HEAD_PREPARED` / `WORKER_PREPARED` at `/prepared`) | DS | adapted | `PREPARED=1` mounts each node's folder and passes `TF_MM3_PREPARED=/prepared` and `TF_MM3_PREPARED_WRITE=1` to both ranks (an engine without the feature ignores them); preflight and start report each node's file or want 100 GiB free; `serve.sh prepared list / clean [all]` (docs/OPERATIONS.md section 3b) |
| Boot timing (per-phase seconds of a start) | DS | adapted | start.sh marks its phases; `serve.sh start` prints checks, gate wait, load, smoke (first reply), canary, total, and appends a JSON line (image, content id, revision, each rank's prepared read / wrote / none) to `STATE_DIR/boot-times.jsonl`; `serve.sh boot-times [N]` |
| Log rotation with a thermal note | DS (`--log-opt` caps), new | adapted | `scripts/logrotate.sh` and `mm31-logrotate.{service,timer}`: memguard, pair watch, heal, boot-start and thermal logs here and the worker's memguard log, by size, `LOG_KEEP` gzipped generations, copytruncate; one line a node with its device state, hottest reading and the limits its own `/etc/spark/thermal.env` states (never a value of its own) |
| `serve.sh upgrade` / `rollback` (last good image, automatic way back) | DS (image content key), new | adapted | the last good image tagged `<repo>:last-good` on both nodes and recorded in `STATE_DIR/last-good.env` after every start whose canary passed; `upgrade` stops, builds / prepares, preflights, starts with the canary required, and on a failure pins and restarts on the last good by itself; `rollback [--clear]` (docs/OPERATIONS.md section 8) |
| tokens-a-round alert (`tpr_check`) | GLM | not applicable | drafting is off by default and the family's `/metrics` names are unverified |
| Container log cap (`--log-opt`) | DS | ported | `LOG_MAX_SIZE` / `LOG_MAX_FILE`; MiaAI-Lab's saved logs at stop kept |
| NCCL cuMem off | DS | ported | already in MiaAI-Lab's nodes.sh |
| `config/prod.env.example` documenting every variable | GLM, DS | ported | read by config.sh for every script; tests check every setting the scripts read is in it, and that `MEM_GATE_GIB` is the calculator's |
| A caller export wins over the config file | GLM, DS | ported | config.sh precedence: environment, local.sh, .env, config/prod.env, defaults |
| Placeholder refusal | GLM | ported | any `<...>` left in the config stops every script |
| RoCE / NCCL link checks (AGENTS.md section 2) | GLM | adapted | MiaAI-Lab's discovery and rail probe kept; added: `LINK_NETDEV` / `LINK_HCAS` pins, every RDMA port ACTIVE, a full-MTU unfragmented ping, `MASTER_PORT + 1` free |
| RoCE failure marker (`roce-failed`) | GLM, DS | adapted | `TF_ROCE_MARK=/cache/roce-failed` in a per-node cache volume (`CACHE_VOL`, new); preflight and status report it, `serve.sh roce-reset` deletes it; read by the engine from `mm3/robust` |
| `scripts/check-public.sh` | GLM, DS | adapted | the skeleton's scan merged with DS's: credentials moved to check_secrets.py, `--history`, allow-lists for MiaAI-Lab's patch authors (patches stay unedited) |
| `scripts/check_secrets.py`, `check-secrets-allow.txt` | DS (PR #77) | ported | tree and history; values printed masked |
| `scripts/redact.py` | DS (PR #77) | ported | byte-identical; used by bench/tooleval |
| No keys in argv or saved JSON | DS (PR #77) | adapted | `start.sh` refuses `--api-key`; rank_env refuses credential-named variables; the watchdog probe and check-drafted read the key file; MiaAI-Lab's `--api-key-file` mount kept |
| `bench/tooleval` (run.sh, chains.py) | DS, GLM | adapted | MiniMax's `reasoning_content`, tool parser and raw-markup leak count (also when tool calls are present); `tool_choice` "auto"; the key from its file to the benches by environment, never argv; DS's redaction and scrub kept; DSML checks dropped (bench/README.md) |
| `bench/quality.py`, `structured.py`, `stress.py`, `soak.py`, `data/mmlu200.jsonl` | DS | adapted | this model's fields and refusals ("required" / named `tool_choice` and `response_format` report UNSUPPORTED); stress ids clear of MiniMax's special tokens, defaults for 131,072 tokens and `PARALLEL`; dropped: the CED-replay framing, DSML, the image soak kind (images have their own harness, `bench/vision.py`); the key masked in replies (bench/README.md) |
| Image input (`TF_DSV41_IMAGES`, its limits, the image soak kind) | DS | adapted | the engine's `--vision` flags (`mm3/vision`) as `VISION`, `VISION_MAX_IMAGES`, `VISION_IMAGE_TOKENS`, `VISION_URLS`; the tower on rank 0 priced out of the KV pool by the calculator; `bench/vision.py` smoke (OpenAI and Anthropic routes, a tool call, clean refusals) and bench (TTFT, image tokens, VQA); no routing-bias file (DeepSeek-only) |
| `bench/toolcall_harness.py` | GLM | adapted | `bench/toolcall.py`: MiniMax's XML tool format, nested arguments, parallel calls, tool-result turns |
| GLM draft / drafter / lookup research harnesses (`draftsim`, `divergence`, ...) | GLM | not applicable | GLM's MTP and drafter studies |
| RigMark | GLM | not applicable | a GLM benchmarking suite; the bench harnesses cover the first run |
| Docker compose and the Python image | GLM, DS | not applicable | the native Zig image (MiaAI-Lab's zig/Dockerfile) |
| Request log, session NVMe tier (`HEAD_STATE`) | DS | not applicable | the family has no such settings |
| AGENTS.md bring-up guide | GLM | adapted | written for this recipe: equal-memory and earlyoom checks, the link pins, the private pack, the failure table |
| OPERATIONS.md, KNOWN-ISSUES.md, RESULTS.md | DS | adapted | rewritten for this engine and pack; RESULTS holds only computed and offline numbers until the first run |
| The guard (temperature / memory front) | DS (PR #48), skeleton | adapted | identity keys moved to `TF_MM3_*`, `TF_ROCE*`, `TENSORFOLD_*`, the reserve and `MM31_SERVE_PROFILE`; optional until the engine's boot line is known |
| Thermal limits (pause 92 / resume 86 °C, stop at 93 °C held 30 s, hottest of GPU and ACPI) | the operator's device settings | adapted | read, never defined: `/etc/spark/thermal.env` and `/run/spark-thermal/state` from `spark-thermal-guard.service` on each node; config.sh reads the file over the environment and refuses it in the recipe config; start refuses without it or on a node not `normal`; memguard stops its rank on `emergency` (plus its own 93 °C / 30 s hold), the guard pauses on `throttled`; `scripts/install-device-limits.sh` installs `device/` (the operator's files, verbatim) on a fresh pair; the guard's sustain and pause-storm soft STOPs and its 78 / 70 °C admission cooldown are retired |
| Calculator | skeleton | adapted | `quant_budget.py --serve`: the family's budget terms and placement, `MEM_GATE_GIB` for a profile |
| `scripts/run.sh` (one rank, planned `TF_MINIMAX_*` / `TF_TP_*` names) | skeleton | not applicable | replaced by `start.sh` (both ranks, the family's real contract); `serve.sh args` prints one rank's command |
| `scripts/prepare.sh` fetch / verify / budget / stage (pre-release) | skeleton | adapted | MiaAI-Lab's prepare.sh is the base; the skeleton's license refusal and two-node copy compare live on in it (`WEIGHTS=local`, LICENSE, SHA256SUMS) |
| `config/m3-reference.json`, `docs/RELEASE-DAY.md` | skeleton | not applicable | pre-release planning against MiniMax-M3; the release and the pack exist now |
| NFS weights for the worker | MiaAI-Lab | adapted | kept as `WEIGHTS=nfs`; the default is a local copy on the worker (GLM, DS), so rank 1's load does not go over NFS |
| HOST / PORT default | MiaAI-Lab (0.0.0.0:8888) | adapted | 127.0.0.1:8000 (GLM, DS: no auth on the API); `API_KEY=1` for other machines |
| The engine patches | MiaAI-Lab | ported | `patches/` byte-identical (a test pins its hash); `ENGINE_PATCHES` / `ENGINE_COMMIT` point at the family's series at a pinned commit |

## Engine features: the plan

Some request features are refused by the family as imported and are being added in the engine branch `mm3/api-tools`
(the engine's; commits `839b48b3f`, `cea776ee5`, `175cb9da5` on top of MiaAI-Lab's family patch). The recipe
does not edit the engine: each feature is **supported once `ENGINE_COMMIT` includes `mm3/api-tools`**, which means
pointing `ENGINE_PATCHES` at a series cut from that branch, `scripts/serve.sh build`, then `ENGINE_API_TOOLS=1`.

| Feature (request field) | Today (`patches/` as imported) | With `mm3/api-tools` | Recipe side |
| --- | --- | --- | --- |
| `tool_choice: "required"` | refused (`minimax_refusal`) | one `]<]minimax[>[<tool_call>` block of offered invokes, through an xgrammar grammar (`minimax_m3_xml`); untyped values read as text, objects keep their named properties, string patterns / formats / lengths dropped, a tool without parameters refused by xgrammar | bench case: >= 1 call, an offered name, arguments valid against the schema, no raw markup; streamed argument deltas equal the final arguments |
| `tool_choice: {"type": "function", ...}` (named) | refused | that function only | bench case: exactly that function; one call with `parallel_tool_calls: false` |
| `response_format` (`json_object`, `json_schema`) | refused (a grammar on the first token: `StructuresUnsupported`) | served: with thinking off the think close is forced first and the grammar starts after it | bench case: content parses and validates, with and without thinking, no reasoning markup |
| Thinking off (`chat_template_kwargs.enable_thinking: false`) | passed to the template, which is always adaptive: not honoured | the prompt ends in `</mm:think>`; no reasoning; an adaptive reply's lone close no longer leaks into content | bench and canary case: no `reasoning_content`, no `</mm:think>` in content, also with tools and `response_format` |
| `thinking_budget` (integer; the Anthropic route's `thinking.budget_tokens`) | not checked for this family | caps the reasoning, closing with `</mm:think>` | bench case: reasoning within budget + slack, the answer after the close |
| `reasoning_effort` (low, medium, high, xhigh, max) | passed verbatim (high / max sent as xhigh) | the five levels as given; `minimal` runs as the nearest (low); a name outside none / minimal / low / medium / high / xhigh / max is a 400 | bench case: each level accepted and reported as asked, and `minimal` accepted |

The switch is `ENGINE_API_TOOLS` (config/prod.env.example). At 0, the default until the engine includes the branch, a
refusal of any of these is reported `SKIP (engine lacks mm3/api-tools)` and does not fail a run, and the canary does not
probe them. At 1, a refusal fails the harnesses and the canary adds a required-tool probe and a thinking-off probe. In both
modes a feature the engine accepts must behave as above, or the case fails. The cases and their stand-in tests are in
`bench/` (bench/README.md, "Engine features"); the first run with an engine that includes the branch is item 11 below.

## Needs `ENGINE_COMMIT` including `mm3/robust`

The engine branch `mm3/robust` (the engine's; work in progress at `5fe050953`) adds the family's memory floor and
a coherent transport failure. The recipe sets everything it reads now; an engine without the branch ignores the variables.

| Setting or behaviour | Value here | What the branch does with it | Recipe side |
| --- | --- | --- | --- |
| `TF_MM3_FLOOR_HARD_GIB` | 4 | no prompt chunk starts while either rank's MemAvailable is under it (each chunk gathers both ranks') | config/prod.env.example; equals the guard's 4 GiB STOP; above memguard's 3 / 2.75 |
| `TF_MM3_FLOOR_GIB` | 5 | a new prompt waits under it while other streams run | config/prod.env.example; under the 8 GiB reserve the pool is sized with |
| `TF_MM3_FLOOR_TRANSIENT_GIB` | 0.5 | what one chunk may take beyond its preallocated scratch (wait or refuse) | config/prod.env.example |
| the three floors in the ranks' fingerprint | equal on both | unequal values refuse the start | rank_env passes the same `TF_MM3_*` to both ranks; the guard's identity check pins them |
| `TF_ROCE_MARK` | `/cache/roce-failed` | a failed RoCE gather writes it; while it exists that rank asks for NCCL | the per-node cache volume, preflight / status, `serve.sh roce-reset` |
| exit 70 | | a coherent stop of both ranks (`tp_watch`) | the pair watch records it; the watchdog heals it once, then alerts |

Live checks with the branch built in: a prompt that would cross 4 GiB on either rank waits or is refused instead of the
rank dipping under memguard's floor; both ranks' fingerprints with the floors; a forced RoCE failure writes the marker and
the next start serves on NCCL; `roce-reset` brings RoCE back; a coherent stop exits 70 on both ranks and the watchdog
heals it exactly once.

## Changes to MiaAI-Lab's files

`start.sh`, `stop.sh`, `scripts/config.sh`, `scripts/nodes.sh`, `scripts/prepare.sh`: the rows above. `scripts/check-drafted.sh`:
the key read as `api_header_file` reads it, port 8000. `scripts/local.sh.example`: points at config/prod.env, the
removed settings dropped. `scripts/banner.sh`, `config.sh`, `nodes.sh`: a shellcheck shell
directive. `docs/*.md` (her engine notes): private paths and one private address replaced by placeholders, nothing else.
`scripts/memguard.sh`: the device's limits and state (stop on `emergency`, its own 93 °C / 30 s hold) and the
thermal-stop record, beside the memory floors. `device/`: the operator's device files, verbatim (hashes pinned in
tests/test_repo.py). `patches/`, `zig/Dockerfile`, `scripts/publish-image.sh`, `CREDITS.md` (kept as `docs/CREDITS-MIAAI-LAB.md`):
unchanged.

## What only a live two-node run can verify

1. The image builds with `ENGINE_PATCHES` and lists `minimax_m3_vl`; `tensorfold-native info` accepts the pack.
2. Both ranks start with `TF_MM3_*` / `NCCL_COMM_ID`, agree on their fingerprint, and load the pack (time, both ranks'
   memory after the load).
3. The memory profile: MemFree passes `MEM_GATE_GIB=113` after a drop; the engine's KV pool with the 8 GiB reserve; both
   ranks above 4 GiB MemAvailable under a prompt near 131,072 tokens; the engine's boot memory line (for the guard's
   `boot_floor_log_regex`) and its slot line (for a slot check).
4. RoCE gathers on both rails, or the NCCL fallback; `ROCE=0`.
5. The smoke test and the canary: chat, separated `reasoning_content`, a tool call parsed with no raw markup, JSON.
   Whether `enable_thinking` / `reasoning_effort` change anything with the pack's always-adaptive template; whether the
   parser's argument typing matches what the model writes; the refusal texts the harnesses read as UNSUPPORTED;
   `"draft": false`; `/v1/completions` with token ids, `ignore_eos` and `min_tokens` (stress); a second request while
   `PARALLEL=1` is busy; the real tool-eval-bench and spark-bench command lines (tests use stand-ins).
6. Greedy repeatability; drafted equal to plain (`scripts/check-drafted.sh`) if drafting is tried.
7. FP8 KV against BF16 on the same prompts; long-context retrieval through the indexer.
8. The harnesses' numbers: tool calling (the pack's failed tool_agentic gate), structured output, quality, decode and
   prefill speed.
9. Failure paths: one rank killed (the pair watch stops the other; the watchdog heals on its next tick), memguard's floors
   against earlyoom on a real dip, `serve.sh stop` (the watchdog stands down), a reboot (the watchdog heals).
10. `/health`'s fields for the watchdog's probe and stall recheck; whether the engine's own `RankLost` ends rank 0, and
    with which exit code.
11. With `ENGINE_COMMIT` including `mm3/api-tools` and `ENGINE_API_TOOLS=1`: every engine-feature case above passes (none
    skipped), on the real model rather than the stand-in: required and named calls, `response_format`, thinking off, the
    thinking budget, the five effort levels, streamed tool deltas.
12. Thermal handling under real load (the device's limits, docs/OPERATIONS.md section 4): `serve.sh status` and preflight
    read both nodes' `/etc/spark/thermal.env` and state; memguard stops its rank when its node turns `emergency`; the guard
    pauses on `throttled`; a start refuses on a node that is not `normal`. Record the peak reading and any state change per
    rank during the first two-node run. `scripts/install-device-limits.sh --apply` on a fresh pair (with the user's go).
13. Images (`VISION=1 CONTEXT=23680 MEM_GATE_GIB=110`, docs/OPERATIONS.md section 3a): `bench/vision.py smoke` passes on the real model (descriptions,
    the Anthropic route, an image with a tool call, clean 4xx refusals, the server alive afterwards); `bench/vision.py
    bench` gives the image tokens MiniMax's geometry predicts; rank 0's memory with the tower against the estimate.
14. The memory gate's rule: a start passes at once when MemAvailable after the drop covers `MEM_GATE_GIB` and MemFree is
    within `MEM_GATE_SLACK_GIB`; the engine's KV pool (its pool line) is the calculator's at that MemFree, not smaller.
15. Prepared weights (`PREPARED=1`, an engine that reads `TF_MM3_PREPARED`: `mm3/boot` `3a8093c6f`): the first boot writes one file a rank and logs
    `prepared: wrote`, the next reads it (`prepared: read`); whether that is faster than the checkpoint at TP2 (`serve.sh boot-times`; on one GPU it was not: 21.0 s against 14.9 s); a new image makes a new
    key; `prepared clean` keeps each rank's newest.
16. `serve.sh ready` / `ready --json` against a live pair; `mm31-serve.service` start and stop; a reboot of the head with
    `mm31-boot-start.service` enabled starts the pair once (and not after a deliberate stop).
17. `serve.sh upgrade` to a deliberately broken image rolls back to the last good by itself and serves; `rollback --clear`.
18. `scripts/logrotate.sh` under the timer while memguard and the pair watch write.
