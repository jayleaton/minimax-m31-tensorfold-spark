# AGENTS.md: setting this up on 2x DGX Spark

Instructions for AI assistants (and people) bringing up MiniMax-M3.1-Flash on TensorFold across two DGX Sparks. The goal
is to run **the production profile** (`config/prod.env.example`): jayleaton's EXL3 3.55 bpw pack, TP=2 over the CX7 link,
a 131,072-token context, one request at a time, FP8 KV. Follow the steps in order; run each check and stop on a failure.
Ask the user before anything that changes system state outside this repo (installing packages, editing network config,
`sudo`, sudoers, deleting files, starting a server on a pair someone else may be using).

> **Measured on one pair of Sparks** ([results/window-all-2](results/window-all-2/README.md)). The engine is built from
> `ENGINE_PATCHES` (`engine/patches`: TensorFold v1.1.2 plus the `minimax_m3_vl` series). A changed profile is qualified
> as [docs/OPERATIONS.md section 7](docs/OPERATIONS.md#7-qualifying-the-profile) describes; [docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md)
> lists what is open.

Terms: **head** = the Spark you run commands on (rank 0, serves the API). **worker** = the other Spark (rank 1), reached
from the head over ssh. The worker needs no copy of this repository.

## 1. Check both nodes

Run on the head; run the `ssh <worker>` lines to check the worker too.

| Check | Command | Expect |
| --- | --- | --- |
| GPU, driver | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` (both) | `NVIDIA GB10`, the **same** driver on both; 580.x is the known-good line (590.x: a CUDA-graph deadlock is reported on GB10) |
| Nothing else on the GPUs | `nvidia-smi --query-compute-apps=pid,name --format=csv,noheader` (both) | empty. Stop any other stack first (ask the user): only one fits |
| Docker + NVIDIA runtime | `docker info --format '{{.Runtimes}}'` (both) | contains `nvidia`; runs without `sudo` |
| Passwordless ssh | `ssh -o BatchMode=yes <worker> docker ps` | no password prompt; the worker user can run docker |
| Same memory on both | `grep MemTotal /proc/meminfo` (both) | equal within 0.5 GiB (127.6 GB on ours since both run the same BIOS and firmware). Every floor here is one value for both ranks |
| Free memory | `grep -E 'MemFree\|MemAvailable' /proc/meminfo` (both) | MemFree >= 113 GiB once the page cache is dropped (`MEM_GATE_GIB`). Close desktop sessions, browsers, other containers |
| Passwordless sudo (cache drops) | `sudo -n true` (both) | exit 0 with the scoped sudoers line below; without it the gate cannot drop the page cache |
| earlyoom | `systemctl is-active earlyoom; ps -o args= -C earlyoom` (both) | if active, `-m 2` or lower (SIGTERM under 2% of MemTotal, 2.43 GiB): `MEMGUARD_HARD_GIB` (2.75) must stay above it. `preflight` checks |
| Disk | `df -h ~/.cache/huggingface` (both) | ~220 GB free on the head and on the worker (`WEIGHTS=local`: a copy on each) |
| rsync | `command -v rsync` (both) | present (`WEIGHTS=local` copies the checkpoint with it) |
| Device heat limits | `cat /etc/spark/thermal.env /run/spark-thermal/state; systemctl is-active spark-thermal-guard` (both) | the same file on both, `normal`, `active`. Missing: install them (below); the recipe never starts without them |

**Device heat limits.** Heat and GPU limits are each machine's settings (docs/OPERATIONS.md section 4), not the
recipe's: `/etc/spark/thermal.env`, `spark-thermal-guard.service` and the GPU limit's drop-in. On a fresh pair, show the
plan, then **ask the user** before installing them (sudo on both machines, like any system change here):

```bash
scripts/install-device-limits.sh            # what differs on each node; changes nothing
scripts/install-device-limits.sh --apply    # after the user agrees: sudo asks for a password on each node
```

Never edit the installed file or the GPU limit for this recipe; the operator owns them.

**Scoped sudo for the page-cache drops.** The scripts need root for exactly three commands: `sudo -n true` (preflight) and
`sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'` / `... echo 3 ...` (the memory gate and the drops around a start).
Instead of blanket passwordless sudo, allow only those, on both nodes (ask the user first; this edits sudoers):

```bash
sudo visudo -f /etc/sudoers.d/minimax-tf
# one line; replace <user> with the account that runs the scripts (and the ssh user on the worker)
<user> ALL=(root) NOPASSWD: /usr/bin/true, /usr/bin/sh -c echo 1 > /proc/sys/vm/drop_caches, /usr/bin/sh -c echo 3 > /proc/sys/vm/drop_caches
```

Check: `sudo -n true && sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' && echo ok` on each node. sudo matches the
arguments literally, so the line must stay exactly as written; if `command -v sh` is not `/usr/bin/sh`, use that path.

## 2. The link

The two Sparks are cabled QSFP to QSFP on their ConnectX-7 ports. The scripts find the link themselves from the route to
the worker (`scripts/nodes.sh`: the netdev, its RoCE device, the RoCE v2 GID, and the port's PCIe twin as a second rail,
probed with a full-size unfragmented ping before use). Check it by hand on each node:

```bash
ibdev2netdev          # e.g. "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)": RDMA device ==> netdev, for the cabled port
ip -br addr           # the IPv4 address on that netdev (and on its twin, e.g. enP2p1s0f1np1, in its own subnet)
cat /sys/class/infiniband/<RDMA device>/ports/1/state    # "4: ACTIVE"
ping -c 3 -M do -s 8972 -I <netdev> <worker link address>   # with MTU 9000; -s 1472 with MTU 1500
```

- `WORKER` in `config/prod.env` is best the worker's **link** address (`user@<link address>`): the checkpoint copy and
  the rendezvous then use the link. With a LAN address, the scripts find the worker's CX7 address themselves
  (`FABRIC_PEER` sets it explicitly).
- Pin the names once they are known: `LINK_NETDEV=<netdev>` and `LINK_HCAS=<device>[,<twin device>]` (what `scripts/serve.sh
  preflight` prints as `link ...`). A start then refuses when the detected link differs (a renamed netdev after a firmware
  or kernel update, a cable on another port) instead of failing minutes into the load.
- Netdev names follow the PCI slot. To keep the address on the cabled port across re-enumeration, a NetworkManager
  profile matched by MAC helps (ask the user before changing networking):
  `nmcli con add type ethernet con-name cx7-link ifname '*' 802-3-ethernet.mac-address <MAC> ipv4.method manual
  ipv4.addresses <link address>/24 ipv6.method disabled` (MAC: `ip -br link`).
- Both ends of each rail need the same MTU and an address in one subnet per rail. A twin that cannot reach its peer is
  dropped with a warning (fewer rails, still correct); `NCCL_RAILS=1` uses one rail on purpose.
- If the port is DOWN or has no address, the link is not configured: do not reconfigure networking without the user's
  approval; tell them what is missing.
- The ranks' small gathers use RoCE when its setup succeeds and NCCL otherwise (`ROCE=1`); `ROCE=0` forces NCCL on both.
  `MASTER_PORT` (29620) and `MASTER_PORT + 1` must be free on the head (preflight checks; a firewall must allow them on
  the link).

## 3. Weights and config

The pack is private: the user must be logged in on the **head** with a token that can read it (`hf auth login`). The token
stays on the head: `scripts/prepare.sh` downloads there and copies the snapshot to the worker over ssh.

```bash
cp config/prod.env.example config/prod.env
```

Edit **only** these fields in `config/prod.env`:

| Field | Value | How to find it |
| --- | --- | --- |
| `WORKER` | `user@<worker link address>` (or an ssh alias) | the target that worked in step 1 |
| `LINK_NETDEV`, `LINK_HCAS` | optional pins | step 2 (`scripts/serve.sh preflight` prints them) |
| `HF_CACHE`, `WORKER_HF_CACHE` | only if a cache is not `~/.cache/huggingface` | `echo ${HF_HOME:-~/.cache/huggingface}` on each node |
| `ENGINE_PATCHES`, `ENGINE_COMMIT` | the engine family's patch series and its pinned commit | keep the example's: `engine/patches` and the commit it was exported from (engine/SOURCE-EXPORT.md); `patches` builds MiaAI-Lab's family alone |
| `ENGINE_API_TOOLS` | `1` once `ENGINE_COMMIT` includes `mm3/api-tools`, else `0` | the user |
| `VISION` | `0` (default); `1` only for an image run on an engine with `mm3/vision`, with the smaller `CONTEXT` of docs/OPERATIONS.md section 3a | the user |
| `WEIGHTS=dir`, `MODEL_DIR` (`WORKER_MODEL_DIR`) | only when both Sparks already hold the pack as a plain folder | `ls <folder>/model.safetensors.index.json` on each node; nothing is downloaded or copied then |

Every script refuses a config that still holds a `<placeholder>`; `scripts/serve.sh` reads `config/prod.env` by default.
The first TP2 load uses `config/smoke.env.example` instead (docs/OPERATIONS.md section 2): `CONFIG=config/smoke.env`.

## 4. Build, prepare, preflight, start

**Ask the user before starting a server**: the pair may be in use by other work.

```bash
scripts/serve.sh build       # the image (TensorFold v1.1.2 + ENGINE_PATCHES, Zig, sm_121), shipped to the worker; stops at
                             # the family gate if the build lacks minimax_m3_vl. No weights read
scripts/serve.sh prepare     # + the checkpoint here (199.9 GB, resumable), its LICENSE and SHA256SUMS, the worker's copy
scripts/serve.sh preflight   # must end with "preflight ok"; fix every line it reports first ("warning:" lines do not fail it)
scripts/serve.sh start
```

`preflight` checks, on both nodes: ssh and docker; the image on both, identical, built from the current patches, listing
the family; the checkpoint (config, index, LICENSE) here and on the worker; the link (route over a CX7 port with a RoCE v2
GID, the pins, every RDMA port ACTIVE, a full-MTU ping); the driver (and 590.x), `vm.min_free_kbytes` and MemTotal alike;
earlyoom's level under `MEMGUARD_HARD_GIB`; `sudo -n` for the cache drops; and (not with `static`) nothing on `PORT`,
`MASTER_PORT` or `MASTER_PORT + 1`, no CUDA process, MemFree against `MEM_GATE_GIB`.

What `start` does, in order: the family gate and `tensorfold-native info` on the checkpoint; the link and the pins; the
serve flags against the build's capabilities; any old server stopped; the ports; no CUDA process; the page cache dropped
and MemFree >= `MEM_GATE_GIB` on both Sparks (waiting up to `MEM_GATE_TIMEOUT`); the memory guard on both; rank 1 on the
worker, then rank 0 here; the pair watch; a line every 15 s while it loads; `/v1/models`; the caches dropped; a smoke test;
the canary (`scripts/canary.py`: chat, separated reasoning, a parsed tool call, JSON); the caches dropped again; then
`MiniMax-M3.1-Flash-EXL3 is now LIVE! on port 8000`.

Timings: NOT MEASURED for this pack. The first start reads ~93 GiB a rank; MiaAI-Lab's first TP2 load of her 157 GiB
pack took 1874 s before her loader fix (`patches/tasks/0015`, 5.5-6.1 GB/s on NVMe afterwards, docs/loader-speed.md).
`WAIT_TIMEOUT` is 2400 s. Follow a long start with `scripts/serve.sh logs 0` and `logs 1`; do not interrupt it.

## 5. Verify

```bash
scripts/serve.sh status      # both containers "Up", pair watch on, MemAvailable, /v1/models and /health JSON
scripts/serve.sh health      # one verdict: both ranks up, /health 200, the model listed
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "MiniMax-M3.1-Flash-EXL3", "max_tokens": 512,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

The reply has `choices[0].message.content`, and `reasoning_content` when the model thinks. Tool calls come back in
`message.tool_calls`; raw `]<]minimax[>[` or `<mm:think>` text in `content` is a parser failure (report it). Images
(`VISION=1`, off by default, an engine with `mm3/vision`) go as `image_url` parts with a `data:` URL, JPEG, PNG or WebP, at most
`VISION_MAX_IMAGES` (4) a request; check them with `python3 bench/vision.py smoke` (docs/OPERATIONS.md section 3a). With the
family as imported, `tool_choice` is "auto" only
("required" and a named function are refused), `response_format` is refused, and thinking cannot be turned off (the
pack's template is always adaptive). Those, `thinking_budget` and the five `reasoning_effort` levels arrive with the
engine branch `mm3/api-tools`: once `ENGINE_COMMIT` includes it, set `ENGINE_API_TOOLS=1` so the canary and harnesses
require them (docs/PORT.md, Engine features).

Then the harnesses (bench/README.md): `bash bench/tooleval/run.sh`, `python3 bench/toolcall.py`, `python3 bench/quality.py`,
`python3 bench/vision.py smoke` and `bench`.

Client settings: base URL `http://127.0.0.1:8000/v1`, model `MiniMax-M3.1-Flash-EXL3`, context window 131,072, max output
32,768. The API has no auth and listens on 127.0.0.1 only: do not set `HOST=0.0.0.0` without `API_KEY=1` or a reverse
proxy with auth in front, and ask the user first.

## 6. Common failures and fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no config/prod.env` / `set WORKER (still '<...>')` | the config was not created or filled | step 3 |
| `MiniMax-M3.1-Flash family not in this TensorFold build; apply patches/` | `ENGINE_PATCHES` lacks the family | point it at the family's series; `scripts/serve.sh build --rebuild` |
| `is a PRIVATE repository and no Hugging Face token was found` | no login on the head | `hf auth login` on the head |
| `a file does not match SHA256SUMS` | a damaged or partial download | delete that file from the cache; `scripts/serve.sh prepare` |
| `... copy in ... differs from this Spark's` | an interrupted copy | `scripts/serve.sh prepare` again (rsync resumes) |
| `the link to the worker runs over X, not LINK_NETDEV=Y` | renamed netdev, other port | step 2; fix the pin or the cabling (ask the user) |
| `a CUDA process is running on a Spark` | another stack holds a GPU | stop it (ask the user) |
| `MemFree ... still under MEM_GATE_GIB 113` | other workloads, page cache without `sudo -n` | stop them; the scoped sudoers line; a smaller `CONTEXT` needs less (`scripts/quant_budget.py --serve`) |
| `rank 1 exited` / `rank 0 exited` early, NCCL timeout in its log | link down, wrong address, firewall on `MASTER_PORT`/`+1` | `scripts/serve.sh preflight`; step 2 |
| a rank's container gone, `memguard.log` says `killed container` | MemAvailable under `MEMGUARD_HARD_GIB`, or under `MEMGUARD_GIB` for 4 s | do not lower the floors: lower `CONTEXT`, keep `DRAFTER` empty, find the other memory user |
| `scripts/serve.sh status` shows `fail-fast: rank N exited ...` | one rank died; the pair watch stopped the other | read both saved logs (`~/.cache/tensorfold-minimax-m3/logs`); the watchdog heals it with `WATCH_HEAL=1` |
| `the canary failed (CANARY=strict)` | the model loads but answers wrong (parser, template) | `STATE_DIR/canary-fail-r{0,1}.log`; report it; `CANARY=warn` only to investigate |
| `do not pass --api-key on the command line` | a key in argv | `API_KEY=1` (a key file) |
| `preflight: warning: the RoCE failure marker ... is on the head / worker` | a RoCE gather failed earlier (engine with `mm3/robust`); both ranks serve on NCCL | fix the link (step 2), then `scripts/serve.sh roce-reset` |
| `coherent stop (exit 70) again within 1800s of its heal: not healing` | the engine stopped both ranks on purpose twice in a row | read both saved logs; tell the user; `scripts/serve.sh start` once the cause is fixed |
| `no device thermal limits on this Spark` / `on <worker>` | `/etc/spark/thermal.env` not installed | section 1, "Device heat limits": show the plan, ask the user, `--apply` |
| `the <node>'s device thermal state is 'throttled'` (or `emergency`, `stale`, `missing`) | the node is hot, or `spark-thermal-guard` is not running | wait for `normal`; if `stale` / `missing`, check the service (`systemctl status spark-thermal-guard`) and tell the user |
| `a thermal hard stop is recorded` (start), `memguard.log` says `at or above 93 C for 30 s` | the hottest GPU / ACPI reading stayed at or above 93 °C for 30 s on that node | tell the user; read `memguard.log`; `scripts/serve.sh thermal-clear` once both nodes read 86 °C or less. Never raise the limits |

## 7. Do not change these knobs

The profile in `config/prod.env.example` is the one measured in window-all-2 (results/window-all-2).
Leave these unless the user asks for an experiment, and then follow docs/OPERATIONS.md section 7:

- `CONTEXT=131072`, `PARALLEL=1`, `KV=fp8`, `MAX_TOKENS=32768`
- the KV tier stays off (`KV_TIER_DIR` empty) unless the user asks for it (docs/OPERATIONS.md section 3)
- `TENSORFOLD_MEMORY_RESERVE_GIB=8` and `MEM_GATE_GIB=115` (they move together: `scripts/quant_budget.py --serve --drafter dspark`)
- `VISION=0`, `VISION_MAX_IMAGES=4`, `VISION_IMAGE_TOKENS=2304`, `VISION_URLS=0` (`VISION=1` takes the images out of the
  KV room: config/prod.env.example, Image input)
- the floors: `TF_MM3_FLOOR_GIB=5`, `TF_MM3_FLOOR_HARD_GIB=4`, `TF_MM3_FLOOR_TRANSIENT_GIB=0.5` (equal on both ranks),
  `MEMGUARD_GIB=3`, `MEMGUARD_HARD_GIB=2.75` (above earlyoom), `MEM_GATE_DROP_CACHES=1`, `DROP_CACHES=1`
- `PAIR_WATCH=1`, `CANARY=strict`, `PREFLIGHT=strict`, `HOST=127.0.0.1`
- the device's heat limits and GPU setting (`/etc/spark/thermal.env`, docs/OPERATIONS.md section 4): they are not in
  `config/prod.env` and never set by this recipe
- `DRAFTS=1`, `DRAFTER=dspark`, `TF_MM3_DRAFT_DEPTH=3` (replies equal plain decode's; code, tool calls and math +38-49 %;
  `DRAFTS=0` turns it off; [docs/DFLASH2.md](docs/DFLASH2.md) for the DFlash2 alternative)
- `TF_MM3_GRAPHS_TP=1`, `TF_MM3_ROCE_SUM=1`, `TF_MM3_SHARE_INPLACE=1`, `TF_ROCE_FAST=1`, `ENGINE_API_TOOLS=1` (as measured)

## 8. Operate

```bash
scripts/serve.sh status | health | logs 0 | logs 1 | stop | restart
scripts/serve.sh canary      # the post-load probes on demand
scripts/serve.sh xid 2h      # NVIDIA Xid events on both nodes
```

Watchdog (optional, restarts a dead pair): `scripts/systemd/tf-watchdog.{service,timer}` as user units, with
`WATCH_HEAL=1` in `config/prod.env` (docs/OPERATIONS.md section 5). `scripts/serve.sh stop` makes it stand down until the
next start. Before leaving the server unattended, watch both nodes' MemAvailable during the first long sessions.
