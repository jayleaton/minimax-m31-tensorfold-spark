#!/usr/bin/env bash
# Serve MiniMax-M3.1-Flash (MODEL_ID: jayleaton's EXL3 3.55 bpw pack by default) on two DGX Sparks with TensorFold v1.1.2's native
# Zig engine plus the minimax_m3_vl family (ENGINE_PATCHES), end to end: runs scripts/prepare.sh when the image or the
# checkpoint is not ready yet, checks the link, the ports and both Sparks' memory (page cache dropped, MemFree gate), starts
# rank 1 on the worker and rank 0 here (the API, 127.0.0.1:8000 by default), starts the memory guard and the pair watch,
# waits until the OpenAI API answers, then runs a smoke test and the canary. scripts/serve.sh is the operator's entry point
# (start / stop / restart / status / health / watch); it calls this. Stop it with ./stop.sh or scripts/serve.sh stop.
#
# STATUS: nothing here has served the model on two Sparks yet (docs/KNOWN-ISSUES.md). The family gate refuses a build
# without minimax_m3_vl before anything is stopped, downloaded or started; ALLOW_UNSUPPORTED=1 overrides it for engine work.
# The shape of this script (setup, checks, launch, load, smoke test) is MiaAI-Lab's, from her GLM-5.3-Flash and
# DeepSeek-V4.1 recipes; the memory gate, cache drops, pair watch, canary and stop marker are jayleaton's GLM-5.3 and
# DeepSeek-V4.1 recipes' (docs/PORT.md).
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (every serving value is the engine's own default until
#                               # measured); if the server already runs, says so and leaves it alone
#   ./start.sh restart          # stop both ranks (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=32768 ./start.sh restart
#   API_KEY=1 ./start.sh restart  # require an API key (generated in $STATE_DIR/api-key, mode 600; see scripts/config.sh)
#   FOREGROUND=1 ./start.sh       # stay attached to rank 0's log and exit with its code (for systemd Restart=on-failure)
#   PREPARE=0 ./start.sh          # skip scripts/prepare.sh (PREPARE=1 forces it; default auto)
#   DRY_RUN=1 ./start.sh        # print the link found and both ranks' docker commands, change nothing and start nothing
#   ALLOW_UNSUPPORTED=1 ./start.sh   # go on although the build lacks the family (engine development only)
# Extra arguments go to both ranks' `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER (an ssh target, key-based) in scripts/local.sh.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
# Settings, from the environment, scripts/local.sh, ./.env or config/prod.env (every one documented in config/prod.env.example):
#   serving  CONTEXT, PARALLEL, MAX_TOKENS, THINKING (1|0), KV (bf16|fp8), PROMPT_CACHE_GIB, DRAFTS (0|1|dflash2), DRAFTER,
#            ROCE, ROCE_MAX_KB, VISION, VISION_MAX_IMAGES, VISION_IMAGE_TOKENS, VISION_URLS, EXTRA_ENV, SERVED_NAME, HOST,
#            PORT, API_KEY, API_KEY_FILE
#   nodes    WORKER, FABRIC_PEER, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, NCCL_RAILS, NCCL_DEBUG, LINK_NETDEV, LINK_HCAS
#   files    MODEL_ID, MODEL_REVISION, HF_CACHE, HF_TOKEN / HF_TOKEN_FILE (the repository is private), WEIGHTS (local|nfs|dir),
#            WORKER_HF_CACHE, MODEL_DIR, WORKER_MODEL_DIR, NFS_PATH, NFS_SERVER, NFS_VOLUME, STATE_DIR, CACHE_DIR, LOG_DIR, LOG_KEEP
#            DFLASH2_DIR, WORKER_DFLASH2_DIR (DRAFTS=dflash2: the DFlash2 draft's folder on each Spark, mounted at /dflash2)
#   engine   ENGINE_PATCHES, ENGINE_COMMIT, TF_REPO, TF_TAG, TF_REF, ZIG_VERSION, BASE_IMAGE, SM, IMAGE, CONTAINER_NAME,
#            REQUIRED_FAMILY, ALLOW_UNSUPPORTED
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1, FOREGROUND=1, CANARY (strict|warn|off)
#   memory   MEM_GATE_GIB, MEM_GATE_TIMEOUT, MEM_GATE_SLACK_GIB, MEM_GATE_DROP_CACHES, DROP_CACHES, MEMGUARD_GIB, MEMGUARD_HOLD_S, MEMGUARD_HARD_GIB
#   KV tier  KV_TIER_DIR, WORKER_KV_TIER_DIR, KV_TIER_KEEP_GIB (with TF_MM3_KV_TIER, TF_MM3_KV_TIER_GIB, TF_MM3_KV_HORIZON, ...)
#   prepared PREPARED, HEAD_PREPARED, WORKER_PREPARED, PREPARED_FREE_GIB (each rank's prepared weights at /prepared)
#   pair     PAIR_WATCH, PAIR_WATCH_S, GUARD_CONFIG
#   ranks    every TENSORFOLD_*, TF_MM3_*, TF_ROCE_* and TF_CUDA_* variable you export goes to both ranks, and so does every
#            other exported NCCL_* variable (values of A-Z a-z 0-9 _ . , : = / - only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
for v in CONTEXT PARALLEL MAX_TOKENS; do [[ -z "${!v}" || "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count (or empty for the engine's default), not ${!v}"; done
[[ -z "$THINKING" || "$THINKING" =~ ^[01]$ ]] || die "THINKING is 1, 0 or empty, not $THINKING"
[[ -z "$KV" || "$KV" =~ ^(bf16|fp8)$ ]] || die "KV is bf16, fp8 or empty, not $KV"
[[ -z "$PROMPT_CACHE_GIB" || "$PROMPT_CACHE_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "PROMPT_CACHE_GIB is a number of GiB (0 turns kept prompt states off) or empty, not $PROMPT_CACHE_GIB"
[[ "$ROCE" =~ ^[01]$ ]] || die "ROCE is 1 (RoCE gathers when their setup succeeds) or 0 (NCCL only), not $ROCE"
[[ -z "$ROCE_MAX_KB" || "$ROCE_MAX_KB" =~ ^[1-9][0-9]*$ ]] || die "ROCE_MAX_KB is a whole number of KiB or empty, not $ROCE_MAX_KB"
[[ "$MEM_GATE_GIB" =~ ^[0-9]+$ && "$MEM_GATE_TIMEOUT" =~ ^[0-9]+$ ]] || die "MEM_GATE_GIB (GiB, 0 turns it off) and MEM_GATE_TIMEOUT (s) are whole numbers, not $MEM_GATE_GIB / $MEM_GATE_TIMEOUT"
[[ "$MEM_GATE_SLACK_GIB" =~ ^[0-9]+$ ]] || die "MEM_GATE_SLACK_GIB is a whole number of GiB under MEM_GATE_GIB, not $MEM_GATE_SLACK_GIB"
(( MEM_GATE_GIB == 0 || MEM_GATE_SLACK_GIB < MEM_GATE_GIB )) || die "MEM_GATE_SLACK_GIB is a whole number of GiB under MEM_GATE_GIB, not $MEM_GATE_SLACK_GIB"
for v in MEM_GATE_DROP_CACHES DROP_CACHES PAIR_WATCH; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$PAIR_WATCH_S" =~ ^[1-9][0-9]*$ ]] || die "PAIR_WATCH_S is a whole number of seconds, not $PAIR_WATCH_S"
[[ "$CANARY" =~ ^(strict|warn|off)$ ]] || die "CANARY is strict, warn or off, not $CANARY"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] && (( MASTER_PORT >= 1 && MASTER_PORT <= 65533 )) || die "MASTER_PORT is 1..65533 (the liveness line listens on MASTER_PORT + 1), not $MASTER_PORT"
[[ "$MEMGUARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_GIB is a number of GiB (2.5 is fine; 0 turns it off), not $MEMGUARD_GIB"
[[ "$MEMGUARD_HOLD_S" =~ ^[0-9]+(\.[0-9]+)?$ && "$MEMGUARD_HARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_HOLD_S is a number of seconds and MEMGUARD_HARD_GIB a number of GiB, not $MEMGUARD_HOLD_S and $MEMGUARD_HARD_GIB"
[[ "$API_KEY" =~ ^[01]$ ]] || die "API_KEY is 0 or 1, not $API_KEY"
[[ "$DRAFTS" =~ ^(0|1|dflash2)$ ]] || die "DRAFTS is 0 (off), 1 (copy drafts, DRAFTER optional) or dflash2 (copies and the DFlash2 draft in DFLASH2_DIR), not $DRAFTS"
[[ -z "$DRAFTER" || "$DRAFTS" != 0 ]] || die "DRAFTER=$DRAFTER needs DRAFTS=1 (drafts off means --no-drafts, and nothing would draft)"
# DRAFTS=dflash2: DRAFTER is the DFlash2 folder (DFLASH2_DIR); another drafter beside it is a contradiction
if [[ "$DRAFTS" == dflash2 ]]; then
  DRAFTER=${DRAFTER:-dflash2}
  [[ "$DRAFTER" == dflash2 ]] || die "DRAFTS=dflash2 loads the DFlash2 draft of DFLASH2_DIR: leave DRAFTER empty (or dflash2), not $DRAFTER; for another draft use DRAFTS=1 DRAFTER=<folder>"
fi
# the DFlash2 folder, when it is there already (scripts/prepare.sh fetches it during setup; step 2 requires it on both Sparks)
if dflash2_wanted && [[ -d "$(dflash2_dir_here)" ]]; then
  _cls=$(dflash2_class head) || die "DRAFTS=$DRAFTS DRAFTER=dflash2: $_cls. DFLASH2_DIR is the folder of a DFlash2 draft ($DFLASH2_REPO); scripts/prepare.sh --dflash2 fetches it"
fi
if [[ "$DRAFTER" == dspark ]]; then
  _snap=$(snapshot_dir)
  [[ ! -d "$_snap" || -f "$_snap/dspark/config.json" ]] || die "DRAFTER=dspark: the snapshot $(snapshot_rev | cut -c1-8) of $MODEL_ID has no dspark/ folder: use a revision that carries it, or leave DRAFTER empty"
fi
[[ -z "$DRAFTER" || "$DRAFTER" =~ ^[A-Za-z0-9_./-]+$ ]] || die "DRAFTER is dspark (the checkpoint's dspark/ folder), dflash2 (DFLASH2_DIR) or a folder both ranks can read, not $DRAFTER"
# a key never goes on a command line (ps, docker inspect and the logs below show it): API_KEY=1 passes a key file instead
for a in "$@"; do
  [[ "$a" != --api-key && "$a" != --api-key=* ]] || die "do not pass --api-key on the command line (ps and docker inspect show it): set API_KEY=1, which mounts API_KEY_FILE and passes --api-key-file"
done
nccl_extra_collect
check_workers

# The serve arguments. COMMON goes to both ranks (a follower must agree with rank 0 on the window, the streams and the drafts);
# HEAD_ONLY is rank 0's HTTP side. Nothing is passed for a setting that is empty: the engine's own default applies.
MODEL_ARG=$(model_in_container)
COMMON=(--backend cuda)
[[ -z "$CONTEXT" ]] || COMMON+=(--context "$CONTEXT")
[[ -z "$PARALLEL" ]] || COMMON+=(--parallel "$PARALLEL")
[[ -z "$MAX_TOKENS" ]] || COMMON+=(--max-tokens "$MAX_TOKENS")
[[ -z "$KV" ]] || COMMON+=(--kv-cache "$KV")
[[ -z "$PROMPT_CACHE_GIB" ]] || COMMON+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ "$DRAFTS" != 0 ]] || COMMON+=(--no-drafts)
# dspark = the checkpoint's own dspark/ folder: the server resolves --drafter itself, so it gets the folder's path in the container.
# dflash2 = each Spark's DFLASH2_DIR, mounted read-only at /dflash2 in its rank: both ranks pass the same path (the engine reads
# the draft's kind from its config.json)
if [[ "$DRAFTER" == dspark ]]; then COMMON+=(--drafter "$MODEL_ARG/dspark")
elif [[ "$DRAFTER" == dflash2 ]]; then COMMON+=(--drafter "$DFLASH2_IN_CONTAINER")
elif [[ -n "$DRAFTER" ]]; then COMMON+=(--drafter "$DRAFTER"); fi
if [[ "$THINKING" == 1 ]]; then COMMON+=(--thinking); elif [[ "$THINKING" == 0 ]]; then COMMON+=(--no-thinking); fi
# images (mm3/vision): the tower runs on rank 0; both ranks get the flags and limits, so they agree on a request's image rows
[[ "$VISION" != 1 ]] || COMMON+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS")
COMMON+=("$@")
HEAD_ONLY=(--host "$HOST" --port "$PORT" --name "$SERVED_NAME")
[[ "$VISION_URLS" != 1 ]] || HEAD_ONLY+=(--vision-urls)   # the downloads are rank 0's HTTP side
# The serve flags both ranks run, as one word for the containers' environment (MM31_SERVE_PROFILE, nodes.sh rank_env): the
# guard's identity check compares the containers' environment, which would not see a changed CONTEXT or KV otherwise.
SERVE_PROFILE=$(IFS=,; echo "context=$CONTEXT,parallel=$PARALLEL,max_tokens=$MAX_TOKENS,kv=$KV,prompt_cache_gib=$PROMPT_CACHE_GIB,drafts=$DRAFTS,drafter=${DRAFTER//\//:}$(! dflash2_wanted || echo ",dflash2=${DFLASH2_REVISION:0:12}"),thinking=$THINKING,vision=$VISION,vision_max_images=$VISION_MAX_IMAGES,vision_image_tokens=$VISION_IMAGE_TOKENS,vision_urls=$VISION_URLS")
[[ "$API_KEY" != 1 ]] || HEAD_ONLY+=(--api-key-file "$API_KEY_IN_CONTAINER")

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner                                             # the TensorFold ribbon and MIA AI LAB (terminals only)
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 2 x DGX Spark · TensorFold %s · %s · port %s%s\n\n' "$D" "$MODEL_ID" "$TF_TAG" \
  "$([[ -n "$CONTEXT" ]] && echo "$CONTEXT-token window" || echo "engine's default window")" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }
# mark <phase>: the phase timing scripts/serve.sh start reads (MM31_BOOT_MARKS, a file it names; unset: nothing)
mark() { [[ -z "${MM31_BOOT_MARKS:-}" ]] || echo "$1 $EPOCHREALTIME" >> "$MM31_BOOT_MARKS"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"
# the guard's lifecycle decision (scripts/guard): 75 holds the pair (cooling down, a lease, stale telemetry), 78 is a durable
# STOP that only an operator clears (docs/OPERATIONS.md, After a STOP). Asked before anything is stopped or started.
if [[ -n "$GUARD_CONFIG" ]] && (( ! DRY )); then
  python3 scripts/guard/lifecycle.py --config "$GUARD_CONFIG" check --operation start >/dev/null ||
    die "the guard's lifecycle check holds the pair (exit $?; python3 scripts/guard/lifecycle.py --config $GUARD_CONFIG status)"
fi
(( ! ALLOW_UNSUPPORTED )) || warn "ALLOW_UNSUPPORTED=1: this build is not checked for the $REQUIRED_FAMILY family; the load is expected to fail unless patches/ added it"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
if (( ! DRY )) && [[ "$MODE" == start ]] && running_here && running_worker 1; then
  [[ "$(prepared_state 2>/dev/null)" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] ||
    die "the requested configuration differs from the prepared server; run ./start.sh restart with the same settings"
  log "$CONTAINER_NAME is already running on both Sparks (port $PORT): nothing to do. ./start.sh restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image (TensorFold $TF_TAG + patches/), the private checkpoint, the worker's NFS volume"
# An image that is already here is asked first, so that a build without the family refuses before the long setup.
if (( ! DRY )); then family_gate || true; fi
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    # setup builds the image and reads $MODEL_GIB GiB: with a running rank holding most of a Spark's memory that can freeze it
    if running_here || running_worker 1; then
      log "Setup is needed and a server is running: stopping it first (the build needs the memory)"
      ./stop.sh
    fi
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image and downloads $MODEL_GIB GiB)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE, $MODEL_ID @ $(snapshot_rev | cut -c1-8)${PREPARE:+ (PREPARE=$PREPARE)}"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || (( DRY )) || die "image $IMAGE missing: scripts/prepare.sh did not build it"

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: engine family, link, previous server, port, memory"
gate=0; family_gate || gate=$?
if (( gate == 2 )); then
  (( DRY )) || die "image $IMAGE missing: run scripts/prepare.sh"
  warn "DRY_RUN: $IMAGE is not built here, so the engine's family and flags are not checked (a real start refuses without $REQUIRED_FAMILY)"
elif [[ -d "$(snapshot_dir)" ]]; then
  # what this build says of the checkpoint itself (no GPU): its family, format, window and weights
  if ! checkpoint_info; then
    (( ALLOW_UNSUPPORTED )) || die "tensorfold-native info refused $MODEL_ID (above): MiniMax-M3.1-Flash family not in this TensorFold build; apply patches/"
    warn "ALLOW_UNSUPPORTED=1: tensorfold-native info refused the checkpoint (above)"
  fi
fi
if (( DRY )) && [[ -n "${WORKER_DOWN[1]:-}" ]]; then
  # DRY_RUN with a worker that cannot be reached: the link cannot be found, the commands print with placeholders
  warn "DRY_RUN: the worker is unreachable, so the link is not detected; the commands below show <placeholders> for it"
  NODE_DEV=("<head-netdev>" "<worker-netdev>"); NODE_HCAS=("<head-roce-device>" "<worker-roce-device>"); NODE_GID=("<gid>" "<gid>")
  LINK_HEAD_ADDR[1]="<head-link-address>"; MASTER_ADDR=${MASTER_ADDR:-"<head-link-address>"}
else
  detect_links
  check_link_pins
fi
log "Link: rendezvous $MASTER_ADDR:$MASTER_PORT (liveness on $((MASTER_PORT + 1)))"
for r in 0 1; do
  log "  rank $r ($( (( r == 0 )) && echo this node || worker_host "$r")): ${NODE_DEV[r]}, RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]})$( (( r == 0 )) || case "$WEIGHTS" in nfs) echo ", weights over NFS from ${LINK_HEAD_ADDR[r]:-?}" ;; dir) echo ", weights from its own folder $(worker_model_dir)" ;; *) echo ", weights from its own HF cache" ;; esac)"
done
# the DFlash2 draft: each Spark's folder holds a DFlash2 config.json (the files' sha256 were checked by scripts/prepare.sh)
if dflash2_wanted; then
  for node in head worker; do
    if [[ "$node" == worker && -n "${WORKER_DOWN[1]:-}" ]]; then continue; fi
    if cls=$(dflash2_class "$node"); then
      log "  DFlash2 draft on the $node: $(dflash2_dir "$node") ($cls) at $DFLASH2_IN_CONTAINER"
      [[ "$cls" == DFlash2DraftModel ]] || warn "the $node's draft is a $cls: this recipe pins $DFLASH2_REPO (DFlash2DraftModel); the engine refuses a layout it does not run"
    elif (( DRY )); then warn "DRY_RUN: $cls (scripts/prepare.sh --dflash2 fetches it)"
    else die "DRAFTS=$DRAFTS: $cls. scripts/prepare.sh --dflash2 fetches $DFLASH2_REPO @ ${DFLASH2_REVISION:0:8} to both Sparks; DFLASH2_DIR / WORKER_DFLASH2_DIR name the folders"
    fi
  done
fi
# Each rank's command. tensorfold-native serve with --rank above 0 serves no HTTP and follows rank 0, when the build serves --tp
# (v1.1.2's CUDA build lists none of --tp/--rank/--master: a family that runs ranks takes them from its environment, rank_env).
rank_flags() { printf '%s\n' --tp "$TP" --rank "$1" --master "${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}" --master-port "$MASTER_PORT"; }
RANK_FLAGS=0
if [[ "$gate" != 2 ]] && serves_flag --tp; then RANK_FLAGS=1
elif [[ "$gate" != 2 ]]; then
  log "This build serves no --tp/--rank/--master: the ranks are told apart by TF_MM3_RANK/TF_MM3_WORLD and NCCL_COMM_ID (scripts/nodes.sh, rank_env)"
fi
SERVE0=(tensorfold-native serve "$MODEL_ARG" "${HEAD_ONLY[@]}" "${COMMON[@]}")
SERVE1=(tensorfold-native serve "$MODEL_ARG" "${COMMON[@]}")
if (( RANK_FLAGS )); then
  mapfile -t _rf0 < <(rank_flags 0); mapfile -t _rf1 < <(rank_flags 1)
  SERVE0+=("${_rf0[@]}"); SERVE1+=("${_rf1[@]}")
fi
# the serve flags against what this build of tensorfold-native serves: the binary refuses an unsupported flag at start, after
# the container is up and the old server stopped
if [[ "$gate" != 2 ]]; then
  check_serve_flags "${SERVE0[@]:3}" || die "rank 0's serve arguments are not ones $IMAGE serves (above); nothing was stopped"
  check_serve_flags "${SERVE1[@]:3}" || die "rank 1's serve arguments are not ones $IMAGE serves (above); nothing was stopped"
fi
if (( ! DRY )); then
  if running_here || running_worker 1; then ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"
  # the device's heat limits and state (they belong to each machine, scripts/config.sh "temperature"): both nodes must carry
  # the same /etc/spark/thermal.env and report `normal`; a throttled or emergency node takes no new load
  need_device_limits
  hsha=$(device_limits_sha head); wsha=$(device_limits_sha worker)
  [[ "$wsha" =~ ^[0-9a-f]{64}$ ]] || die "no device thermal limits on $(worker_host 1) ($SPARK_THERMAL_ENV: $wsha): install them on both Sparks with scripts/install-device-limits.sh"
  [[ "$hsha" == "$wsha" ]] || die "the device thermal limits differ between the Sparks ($SPARK_THERMAL_ENV): the same file belongs on both (scripts/install-device-limits.sh)"
  for node in head worker; do
    st=$(device_state "$node")
    [[ "$st" == normal ]] || die "the $node's device thermal state is '$st', not normal (spark-thermal-guard.service, $SPARK_THERMAL_RUN/state): wait for it to cool to GUARD_RESUME_C $GUARD_RESUME_C C, or install the device limits (scripts/install-device-limits.sh)"
  done
  # a thermal hard stop (memguard, either node) holds the pair until an operator has read it and cleared it
  if stops=$(thermal_stops) && [[ -n "$stops" ]]; then
    die "a thermal hard stop is recorded (the hottest reading stayed at or above $GUARD_KILL_C C for $GUARD_KILL_HOLD_S s): $stops. Read memguard.log on that node, then scripts/serve.sh thermal-clear (it waits for both nodes at or below $GUARD_RESUME_C C)"
  fi
  for p in "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))"; do
    if ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then die "port $p is already in use: $(ss -ltnp "sport = :$p" 2>/dev/null | tail -n +2)"; fi
  done
  # nothing else on the GPUs: the pair needs both Sparks' memory (another stack's process holds its share until it exits)
  others=$( { nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; worker 1 nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; } 2>/dev/null | grep -c '[0-9]' || true)
  (( others == 0 )) || die "a CUDA process is running on a Spark (nvidia-smi --query-compute-apps on both): stop the other stack first"
  # memory: a rank that takes all of a Spark's memory freezes the machine. Drop the page cache, then the gate on both
  mark gate
  drop_caches 3
  if ! mem_gate; then
    shm=$(du -sm /dev/shm 2>/dev/null | cut -f1)
    die "the memory gate is still shut after ${MEM_GATE_TIMEOUT}s (MEM_GATE_GIB $MEM_GATE_GIB, MemFree floor $(( MEM_GATE_GIB - MEM_GATE_SLACK_GIB ))): $MEM_GATE_WHY. Stop other workloads (containers, a desktop session, a copy in progress)${shm:+; /dev/shm holds $shm MiB here}. The value is scripts/quant_budget.py --serve for this CONTEXT x PARALLEL; a smaller CONTEXT needs less"
  fi
  mark gate_done
  if ! kv_tier_gate; then
    die "the KV tier needs TF_MM3_KV_TIER_GIB $TF_MM3_KV_TIER_GIB + KV_TIER_KEEP_GIB $KV_TIER_KEEP_GIB GiB on each Spark's disk; $(kv_tier_here) has ${TIER_HERE_GIB:-?} and $(worker_kv_tier_dir) on $(worker_host 1) ${TIER_WORKER_GIB:-?} (free plus the tier's own files)"
  fi
  [[ -z "$KV_TIER_DIR" ]] || log "KV tier: $TIER_HERE_GIB / $TIER_WORKER_GIB GiB for its $TF_MM3_KV_TIER_GIB GiB (here / $(worker_host 1))"
  # the prepared weights (PREPARED=1): each node's folder made; a rank with no file yet needs PREPARED_FREE_GIB to write one
  if [[ "$PREPARED" == 1 ]]; then
    for node in head worker; do
      p=$(prepared_node "$node" mkdir)
      case "$p" in
        file\ *) read -r _ pf ps _ <<<"$p"; log "Prepared weights on the $node: $pf ($(( ps / 1073741824 )) GiB): this boot reads it if its key matches" ;;
        none\ *) (( ${p#none } >= PREPARED_FREE_GIB )) ||
                   die "PREPARED=1: no prepared file on the $node yet and only ${p#none } GiB free in $(prepared_dir "$node") (it writes ~94 GiB; PREPARED_FREE_GIB $PREPARED_FREE_GIB): free space, move HEAD_PREPARED / WORKER_PREPARED, or PREPARED=0"
                 log "Prepared weights on the $node: none yet in $(prepared_dir "$node") (${p#none } GiB free): this boot writes one (~94 GiB, slower than a plain load)" ;;
        *) die "PREPARED=1: cannot read or create $(prepared_dir "$node" 2>/dev/null || echo the folder) on the $node" ;;
      esac
    done
  fi
  memguard_start
fi

# ---------------------------------------------------------------- 3. launch
# FOREGROUND=1: stay attached to rank 0's log and exit with its code, so that a supervisor (systemd's Restart=on-failure)
# can restart the server. Either rank ending takes the other one down: a lone rank waits for its peer forever. SIGINT and
# SIGTERM run ./stop.sh. Note the supervisor restarts a start that was killed by memguard too; give it a RestartSec.
# (From MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold's start.sh, for the one worker here.)
foreground() {
  local watch w code
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      running_here || exit 0
      worker 1 true 2>/dev/null || continue         # a worker out of reach for a moment says nothing about its rank
      running_worker 1 && continue
      warn "rank 1 on $(worker_host 1) exited: stopping rank 0"
      docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
      exit 1
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # the worker's rank failed, even if rank 0 then shut down cleanly
  worker 1 "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true
  exit "$code"
}
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="-v $CACHE_VOL:/cache --init --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband --log-opt max-size=${LOG_MAX_SIZE:-200m} --log-opt max-file=${LOG_MAX_FILE:-3}"   # docker's log capped (DS)
AUTH=()      # curl's Authorization header (a file, so the key stays out of ps) when API_KEY=1
if [[ "$API_KEY" == 1 ]]; then
  if (( DRY )); then log "DRY_RUN: API_KEY=1: the key would be read from $API_KEY_FILE (created, mode 600, when missing)"
  else api_key_ensure; AUTH=(-H "@$(api_header_file)"); fi
fi
step 3 "Launch: container $CONTAINER_NAME, rank 1 on the worker, then rank 0 here"
mapfile -t ENV1 < <(rank_env 1); mapfile -t ENV0 < <(rank_env 0)
cmd1=$(printf '%q ' "${SERVE1[@]}")
case "$WEIGHTS" in
  nfs) wmount="$NFS_VOLUME:/hf:ro" ;;
  dir) wmount="$(printf '%q' "$(worker_model_dir)"):/model:ro" ;;
  *) wmount="$(printf '%q' "$(worker_hf_cache 1)"):/hf:ro" ;;
esac
tier1=""; tier0=()   # the KV tier's folder of each node at /kvtier (KV_TIER_DIR)
if [[ -n "$KV_TIER_DIR" ]]; then
  tier1="-v $(printf '%q' "$(worker_kv_tier_dir)"):$KV_TIER_IN_CONTAINER"; tier0=(-v "$(kv_tier_here):$KV_TIER_IN_CONTAINER")
fi
draft1=""; draft0=()   # the DFlash2 draft's folder of each node at /dflash2, read-only (DFLASH2_DIR, WORKER_DFLASH2_DIR)
if dflash2_wanted; then
  draft1="-v $(printf '%q' "$(worker_dflash2_dir)"):$DFLASH2_IN_CONTAINER:ro"; draft0=(-v "$(dflash2_dir_here):$DFLASH2_IN_CONTAINER:ro")
fi
prep1=""; prep0=()   # the prepared weights' folder of each node at /prepared, read-write (PREPARED=1: the engine writes it)
if [[ "$PREPARED" == 1 ]]; then
  prep1="-v $(printf '%q' "$(worker_prepared_dir)"):$PREPARED_IN_CONTAINER"; prep0=(-v "$(prepared_here):$PREPARED_IN_CONTAINER")
fi
mark launch
remote="docker run -d --name $CONTAINER_NAME $RUN_ARGS -v $wmount $tier1 $draft1 $prep1 $(envs "${ENV1[@]}") $(rank_nccl_env 1) $IMAGE $cmd1"
if (( DRY )); then printf '[dry-run] rank 1 on %s:\n  %s\n' "$(worker_host 1)" "$remote"
else
  log "Rank 1 on $(worker_host 1): ${SERVE1[*]}"
  worker 1 "$remote" >/dev/null || die "could not start rank 1 on $(worker_host 1)"
fi
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$(head_model_mount)" "${tier0[@]}" "${draft0[@]}" "${prep0[@]}")
if [[ "$API_KEY" == 1 ]]; then here+=(-v "$API_KEY_FILE:$API_KEY_IN_CONTAINER:ro"); fi
for x in "${ENV0[@]}"; do here+=(-e "$x"); done
rank_nccl_args 0
here+=("${NCCL_ARGS[@]}" "$IMAGE" "${SERVE0[@]}")
if (( DRY )); then printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"; exit 0; fi
log "Rank 0 here: ${SERVE0[*]}"
"${here[@]}" >/dev/null
rm -f "$STATE_DIR/stopped"   # both ranks launched: a deliberate stop is over, the watchdog may heal this pair again
pairwatch_start
[[ "${FOREGROUND:-0}" != 1 ]] || foreground

# ---------------------------------------------------------------- 4. load
step 4 "Loading: the $MODEL_GIB GiB of weights split over both Sparks, then the caches (the time is not measured for this model)"
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image'
# docker logs is the background job, so killing it ends the whole pipeline (no orphaned `docker logs -f`)
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── rank 0 (here): last log lines ──%s\n' "$D" "$R"
  docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  printf '%s── rank 1 (%s): last log lines ──%s\n' "$D" "$(worker_host 1)" "$R"
  worker 1 docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
start=$SECONDS; next_beat=15
until curl -sf --max-time 5 "${AUTH[@]}" "$URL/v1/models" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running: docker logs -f $CONTAINER_NAME"
  if (( SECONDS - start >= next_beat )); then
    running_worker 1 || fail "rank 1 on $(worker_host 1) exited (code $(worker 1 docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")) before the server was ready"
    g=$(worker 1 "$(declare -f gpu_gib); gpu_gib $CONTAINER_NAME" 2>/dev/null || echo "?")
    printf '  %s⋯ %ss elapsed, %s GiB on the GPU here, %s on the worker%s\n' "$D" "$((SECONDS - start))" "$(gpu_gib "$CONTAINER_NAME")" "$g" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
mark answered
log "Server answered after $((SECONDS - start))s: $(curl -s --max-time 5 "${AUTH[@]}" "$URL/v1/models")"
drop_caches 3                                       # the load's page cache, before the smoke test and the canary

# ---------------------------------------------------------------- 5. smoke test
# Greedy and short; no reply text, or one character repeated (a collapsed model), fails the start.
step 5 "Smoke test: one chat completion through both ranks"
mark smoke
if smoke=$(curl -s --max-time 300 "${AUTH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 64, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -I -c 'import json,sys; r = json.load(sys.stdin); m = r["choices"][0]["message"]; c = m.get("content") or m.get("reasoning_content") or ""; assert c.strip(); assert len(c.strip()) < 8 or len(set(c.strip())) > 2, "one repeated character"; print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  mark smoke_done
  log "OK: $smoke"
else
  fail "the smoke test request failed (no reply text, or one character repeated: the weights load but the model does not answer); the ranks are still running"
fi
# The canary (scripts/canary.py): a greedy chat, a reply with separated reasoning, a tool call the server parses into tool_calls
# with no raw markup, a JSON reply. CANARY=strict stops both ranks on a failure and keeps their last logs (DeepSeek's rule).
# STATE_DIR/canary-last records the verdict for this rank 0 container's start time (scripts/serve.sh ready reads it).
canary_record() { echo "started=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown) result=$1 at=$(date -u +%FT%TZ)" > "$STATE_DIR/canary-last"; }
if [[ "$CANARY" != off ]]; then
  cargs=(--base "$URL" --model "$SERVED_NAME"); [[ "$API_KEY" != 1 ]] || cargs+=(--api-key-file "$API_KEY_FILE")
  if timeout 900 python3 -I scripts/canary.py "${cargs[@]}"; then canary_record pass
  elif [[ "$CANARY" == strict ]]; then
    canary_record fail
    docker logs --tail 200 "$CONTAINER_NAME" > "$STATE_DIR/canary-fail-r0.log" 2>&1 || true
    worker 1 docker logs --tail 200 "$CONTAINER_NAME" > "$STATE_DIR/canary-fail-r1.log" 2>&1 || true
    ./stop.sh >/dev/null 2>&1 || true
    die "the canary failed (CANARY=strict): both ranks stopped; their last logs are in $STATE_DIR/canary-fail-r{0,1}.log"
  else
    canary_record fail
    warn "the canary failed (CANARY=warn): still serving"
  fi
  drop_caches 3                                     # the canary's page cache out again
else
  canary_record off
fi
# Headroom: how far MemAvailable sits above memguard's floor once the weights and caches are loaded. Nothing in the engine holds
# it above that floor while it runs, and memguard kills the rank if it reaches the floor.
if awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}'; then
  mem_here=$(mem_available_gib)
  mem_there=$(worker 1 "awk '/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}' /proc/meminfo" 2>/dev/null || echo "?")
  log "Memory after the load: ${mem_here} GiB available on this Spark, ${mem_there} GiB on $(worker_host 1); memguard's floor is $MEMGUARD_GIB GiB"
  for pair in "this Spark:$mem_here" "$(worker_host 1):$mem_there"; do
    v=${pair##*:}; [[ "$v" =~ ^[0-9.]+$ ]] || continue
    awk -v a="$v" -v g="$MEMGUARD_GIB" 'BEGIN {exit !(a < g + 3)}' &&
      warn "only $v GiB available on ${pair%:*}, within 3 GiB of memguard's floor ($MEMGUARD_GIB GiB): a long prompt can take that and memguard then kills the rank. Free memory there before relying on this server"
  done
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
# the pair serves: after a reboot scripts/boot-start.sh starts it again (stop.sh removes this, except at a system shutdown)
echo "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown) $(date +%s)" > "$STATE_DIR/serving"
mark live
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED_NAME)
    Access   $( [[ "$API_KEY" == 1 ]] && echo "API key required: Authorization: Bearer \$(cat $API_KEY_FILE)" || echo "open: no key, anyone who reaches port $PORT can use it (API_KEY=1 requires one)")
    Window   ${CONTEXT:-engine default} tokens · ${PARALLEL:-engine default} at once · KV ${KV:-engine default} · drafts $(if [[ "$DRAFTS" == 0 ]]; then echo off; elif dflash2_wanted; then echo "copies + the DFlash2 draft ($DFLASH2_REPO @ ${DFLASH2_REVISION:0:8})"; elif [[ -n "$DRAFTER" ]]; then echo "copies + $DRAFTER draft"; else echo copies; fi) · $( [[ "$VISION" == 1 ]] && echo "images (JPEG/PNG/WebP, up to $VISION_MAX_IMAGES a request)" || echo "text only")
    Logs     docker logs -f $CONTAINER_NAME   (rank 1: ssh $(worker_host 1) docker logs -f $CONTAINER_NAME)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
