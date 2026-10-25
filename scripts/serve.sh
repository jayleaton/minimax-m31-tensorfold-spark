#!/usr/bin/env bash
# Start / stop / inspect the two MiniMax-M3.1-Flash TensorFold ranks. Run on the head Spark (rank 0). The operator's entry
# point, as scripts/serve.sh is in the GLM-5.3 and DeepSeek-V4.1 recipes; the launch itself is MiaAI-Lab's ./start.sh.
#
#   scripts/serve.sh build [--rebuild]   the engine image here (zig/Dockerfile: TensorFold v1.1.2 + ENGINE_PATCHES), shipped
#                                to the worker, family gate; no weights read (scripts/prepare.sh --image-only). The kernels
#                                are compiled into the binary at build time, so this is also the prebuild: nothing compiles
#                                beside loaded weights. Refuses while the server runs (a build beside the weights can freeze
#                                a Spark)
#   scripts/serve.sh prebuild    the same as build
#   scripts/serve.sh prepare     everything a first start needs: build, the checkpoint here and on the worker (WEIGHTS), checks
#                                (and the DFlash2 draft with DRAFTS=dflash2); prepare --dflash2: only that draft, on both Sparks
#   scripts/serve.sh preflight [static]   read-only checks of both nodes (AGENTS.md section 1 lists them); static skips the
#                                ones that fail while something else serves (ports, GPUs idle, MemFree)
#   scripts/serve.sh start       preflight (PREFLIGHT=strict|warn|off), then ./start.sh: page cache dropped, MemFree gate on
#                                both Sparks, rank 1 then rank 0, memory guard and pair watch, wait for /v1/models, smoke test,
#                                canary, caches dropped again
#   scripts/serve.sh restart | stop | status | logs [0|1] | health | canary | args | xid [SINCE] | watch [--once]
#   scripts/serve.sh roce-reset      delete the RoCE failure marker (TF_ROCE_MARK) on both nodes after the link is fixed: the
#                                next start tries RoCE again (while it exists both ranks start on NCCL)
#   scripts/serve.sh thermal-clear   clear a thermal hard stop (memguard's record on either node) once both nodes read at or
#                                below GUARD_RESUME_C and report `normal`; start refuses and the watchdog does not heal until
#                                then. The limits are the device's (/etc/spark/thermal.env; scripts/install-device-limits.sh)
#
# Configuration: config/prod.env (cp config/prod.env.example config/prod.env and fill in the <placeholders>; CONFIG=path for
# another). A caller export wins over the file: CONTEXT=65536 scripts/serve.sh restart.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
CONFIG="${CONFIG:-config/prod.env}"
if [[ ! -f "$CONFIG" && "${1:-}" != "" && "${1:-}" != help && "${1:-}" != -h ]]; then
  echo "[serve.sh] no $CONFIG: cp config/prod.env.example config/prod.env, then fill in its <placeholders> (AGENTS.md section 3)" >&2
  exit 2
fi
export CONFIG
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
[[ "${1:-}" =~ ^(|help|-h|--help)$ ]] || check_workers
mkdir -p "$STATE_DIR" "$LOG_DIR"
PREFLIGHT="${PREFLIGHT:-strict}"
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
BASE="http://$API_HOST:$PORT"
AUTH=()
if [[ "$API_KEY" == 1 && -s "$API_KEY_FILE" ]]; then AUTH=(-H "@$(api_header_file)"); fi

running() { # $1 = rank: true | false | absent | unreachable
  local out rc
  if [[ "$1" == 0 ]]; then out=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null) || out=absent
  else
    out=$(worker 1 docker inspect -f "'{{.State.Running}}'" "$CONTAINER_NAME" 2>/dev/null) && rc=0 || rc=$?
    if [[ ${rc:-0} == 255 ]]; then out=unreachable; elif [[ ${rc:-0} != 0 ]]; then out=absent; fi
  fi
  echo "${out:-absent}"
}
server_running() { [[ "$(running 0)" == true || "$(running 1)" == true ]]; }
gpu_busy() { # a CUDA process on either Spark (other than this pair's, when it runs): another stack is still up
  { nvidia-smi --query-compute-apps=pid --format=csv,noheader; worker 1 nvidia-smi --query-compute-apps=pid --format=csv,noheader; } \
    2>/dev/null | grep -q '[0-9]'
}
listening() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }
on_node() { local n=$1; shift; if [[ "$n" == head ]]; then bash -c "$*"; else worker 1 "$*"; fi; }

# preflight [static]: read-only; prints one line a problem ("warning:" lines do not fail it); non-zero on a problem
preflight() {
  local bad=0 static=${1:-} node h w out hca st lvl
  pf() { log "preflight: $*"; }
  worker 1 true 2>/dev/null || { pf "cannot ssh to $WORKER without a password (ssh-copy-id $WORKER)"; return 1; }
  docker info >/dev/null 2>&1 || { pf "docker does not answer here (service running? user in the docker group?)"; bad=1; }
  worker 1 'docker info >/dev/null 2>&1' || { pf "docker does not answer on $WORKER (docker group?)"; bad=1; }
  # the image: on both Sparks, the same content, and its build lists the family
  h=$(image_ident "$IMAGE"); w=$(worker_image_ident 1 "$IMAGE")
  [[ "$h" != missing ]] || { pf "no image $IMAGE here (scripts/serve.sh build)"; bad=1; }
  [[ "$w" != missing ]] || { pf "no image $IMAGE on the worker (scripts/serve.sh build ships it)"; bad=1; }
  [[ "$h" == missing || "$w" == missing || "$h" == "$w" ]] || { pf "$IMAGE differs between the Sparks (scripts/serve.sh build)"; bad=1; }
  if [[ "$h" != missing ]]; then
    [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null)" == "$(image_hash)" ]] ||
      pf "warning: $IMAGE was built from other patches or pins than ENGINE_PATCHES / zig/Dockerfile now name (scripts/serve.sh build)"
    out=$( (family_gate) 2>&1 ) || { pf "${out##*ERROR: }"; bad=1; }
    out=$(roce_marks | paste -sd, -)
    [[ -z "$out" ]] || pf "warning: the RoCE failure marker ($TF_ROCE_MARK) is on the $out: the next start serves on NCCL. After fixing the link: scripts/serve.sh roce-reset"
  fi
  # the checkpoint: here, and where rank 1 reads it
  local snap rel; snap=$(snapshot_dir); rel="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
  for f in config.json model.safetensors.index.json "$MODEL_LICENSE_FILE"; do
    [[ -f "$snap/$f" ]] || { pf "$MODEL_ID @ $(snapshot_rev | cut -c1-8): $f missing here ($snap; $([[ "$WEIGHTS" == dir ]] && echo "WEIGHTS=dir: put the pack in MODEL_DIR" || echo "scripts/serve.sh prepare"))"; bad=1; }
  done
  if [[ "$WEIGHTS" == dir ]]; then
    # each node's own folder: the worker's must hold what this one's does, file by file, by size
    h=$(model_dir_manifest head 2>/dev/null || true); w=$(model_dir_manifest worker 2>/dev/null || true)
    if [[ -z "$w" ]]; then
      pf "the pack is not readable in the worker's folder $(worker_model_dir 2>/dev/null) (WORKER_MODEL_DIR; WEIGHTS=dir copies nothing)$([[ "${WORKER_MODEL_DIR:-$MODEL_DIR}" != "~/"* ]] || echo ": a leading ~/ is the home of the ssh user ($WORKER) on the worker; give WORKER_MODEL_DIR as an absolute path when the pack is under another account's home")"; bad=1
    elif [[ "$h" != "$w" ]]; then
      pf "the worker's folder $(worker_model_dir) differs from $snap here (WEIGHTS=dir: the same pack on both): $(diff <(printf '%s\n' "$h") <(printf '%s\n' "$w") | grep '^[<>]' | head -4 | paste -sd';' -)"; bad=1
    else log "preflight: the pack: $(grep -c '\.safetensors ' <<<"$h") shards, the same files and sizes in $snap here and $(worker_model_dir) on the worker"
    fi
  elif [[ "$WEIGHTS" == local ]]; then
    worker 1 "test -f $(printf '%q' "$(worker_hf_cache 1)/$rel/model.safetensors.index.json")" ||
      { pf "the checkpoint is not in the worker's HF cache ($(worker_hf_cache 1)): scripts/serve.sh prepare copies it"; bad=1; }
  else
    worker 1 "docker volume inspect '$NFS_VOLUME' >/dev/null 2>&1" || { pf "no NFS volume $NFS_VOLUME on the worker (scripts/serve.sh prepare)"; bad=1; }
  fi
  # the DFlash2 draft (DRAFTS=dflash2): a DFlash2 config.json and the pinned model.safetensors' size in each node's folder
  if dflash2_wanted; then
    for node in head worker; do
      if ! out=$(dflash2_class "$node"); then pf "$out (scripts/serve.sh prepare --dflash2)"; bad=1
      elif ! dflash2_on "$node" "$(dflash2_verify_cmd "$(dflash2_dir "$node")" quick)"; then
        pf "the DFlash2 draft in $(dflash2_dir "$node") on the $node is not the pinned $DFLASH2_REPO @ ${DFLASH2_REVISION:0:8} (scripts/serve.sh prepare --dflash2)"; bad=1
      else log "preflight: the DFlash2 draft on the $node: $(dflash2_dir "$node") ($out)"
      fi
    done
  fi
  # the link (GLM's checks): the route to the worker over a CX7 port with a RoCE v2 GID, the pins, every RDMA port ACTIVE,
  # a full-MTU unfragmented ping over the link's netdev
  if out=$( (detect_link && check_link_pins && echo "OK ${NODE_DEV[0]} ${NODE_HCAS[0]} ${NODE_DEV[1]} ${NODE_HCAS[1]} ${LINK_WORKER_ADDR[1]}") 2>&1 | tail -1); [[ "$out" == OK* ]]; then
    read -r _ hdev hhcas wdev whcas waddr <<<"$out"
    log "preflight: link $hdev ($hhcas) <-> $wdev ($whcas), worker $waddr"
    for node in head worker; do
      [[ "$node" == head ]] && out=$hhcas || out=$whcas
      for hca in ${out//,/ }; do
        st=$(on_node "$node" "cat /sys/class/infiniband/$hca/ports/1/state" 2>/dev/null || true)
        [[ "$st" == *ACTIVE* ]] || { pf "$hca port 1 on the $node is '${st:-unreadable}', not ACTIVE (a down port shows as an NCCL timeout minutes into the load)"; bad=1; }
      done
    done
    local mtu; mtu=$(cat "/sys/class/net/$hdev/mtu" 2>/dev/null || echo 1500)
    rail_ping "$hdev" "$waddr" "$mtu" || { pf "a $mtu-byte unfragmented ping over $hdev does not reach $waddr (MTU differs between the ends? the link is down?)"; bad=1; }
  else
    pf "the link: ${out##*ERROR: }"; bad=1
  fi
  # the nodes alike (GLM): the driver (590.x: a CUDA-graph deadlock is reported on GB10), vm.min_free_kbytes, MemTotal
  h=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  w=$(worker 1 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  [[ "$h" == "$w" ]] || pf "warning: the NVIDIA driver differs (here ${h:-?}, worker ${w:-?})"
  [[ "$h $w" != *590.* ]] || pf "warning: driver 590.x (a GB10 CUDA-graph deadlock is reported on 590; 580.x is the known-good line)"
  h=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo ?); w=$(worker 1 cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo ?)
  [[ "$h" == "$w" ]] || pf "warning: vm.min_free_kbytes differs (here $h, worker $w): the ranks get different room"
  h=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo); w=$(worker 1 "awk '/^MemTotal:/ {print \$2}' /proc/meminfo" 2>/dev/null || echo 0)
  (( ${w:-0} > 0 && (h > w ? h - w : w - h) < 524288 )) ||
    pf "warning: MemTotal differs by more than 0.5 GiB (here $h kB, worker ${w:-?} kB): BIOS / firmware not aligned? Every floor here assumes equal nodes"
  # the device's heat limits (scripts/config.sh "temperature"): the same file on both, spark-thermal-guard publishing
  h=$(device_limits_sha head); w=$(device_limits_sha worker)
  [[ "$h" =~ ^[0-9a-f]{64}$ && "$w" =~ ^[0-9a-f]{64}$ ]] || { pf "device thermal limits missing ($SPARK_THERMAL_ENV: here ${h:0:12}, worker ${w:0:12}): scripts/install-device-limits.sh (asks for sudo; AGENTS.md section 1)"; bad=1; }
  [[ "$h" != "$w" || ! "$h" =~ ^[0-9a-f]{64}$ ]] || log "preflight: device limits identical on both ($SPARK_THERMAL_ENV, sha256 ${h:0:12}): pause $GUARD_PAUSE_C, resume $GUARD_RESUME_C, stop $GUARD_KILL_C C for $GUARD_KILL_HOLD_S s"
  [[ ! "$h" =~ ^[0-9a-f]{64}$ || ! "$w" =~ ^[0-9a-f]{64}$ || "$h" == "$w" ]] || { pf "the device thermal limits differ between the Sparks: the same $SPARK_THERMAL_ENV belongs on both"; bad=1; }
  for node in head worker; do
    st=$(device_state "$node")
    case "$st" in
      normal) ;;
      throttled|emergency) pf "the $node's device thermal state is $st: no start until it is normal" ; [[ "$static" == static ]] || bad=1 ;;
      *) pf "the $node's device thermal state is $st: is spark-thermal-guard.service running? (scripts/install-device-limits.sh)"; bad=1 ;;
    esac
  done
  # the memory levels: earlyoom must sit under the memory guard's hard floor on both Sparks; sudo -n for the cache drops
  for node in head worker; do
    lvl=$(earlyoom_level "$node")
    if [[ "$lvl" != - ]] && awk -v e="$lvl" -v g="$MEMGUARD_HARD_GIB" 'BEGIN {exit !(g > 0 && e >= g)}'; then
      pf "warning: earlyoom on the $node sends SIGTERM at $lvl GiB, at or above MEMGUARD_HARD_GIB $MEMGUARD_HARD_GIB: earlyoom, not memguard, would pick what dies"
    fi
    if [[ "$MEM_GATE_DROP_CACHES" == 1 || "$DROP_CACHES" == 1 ]]; then
      on_node "$node" "sudo -n true" >/dev/null 2>&1 || pf "warning: no passwordless sudo -n on the $node: the page cache cannot be dropped (AGENTS.md, scoped sudoers line)"
    fi
  done
  [[ "$static" == static ]] && return $bad
  # what fails while something else serves
  for p in "$PORT" "$MASTER_PORT" "$((MASTER_PORT + 1))"; do listening "$p" && { pf "something listens on :$p already"; bad=1; }; done
  if gpu_busy; then pf "a CUDA process is running on a Spark (nvidia-smi): stop the other stack first"; bad=1; fi
  h=$(memfree head); w=$(memfree worker)
  if (( MEM_GATE_GIB > 0 )) && (( ${h:-0} < MEM_GATE_GIB || ${w:-0} < MEM_GATE_GIB )); then
    pf "warning: MemFree ${h:-?} / ${w:-?} GiB (here / worker) under MEM_GATE_GIB $MEM_GATE_GIB: start drops the page cache and waits up to ${MEM_GATE_TIMEOUT}s"
  fi
  return $bad
}

# health: one verdict for scripts and people: both containers up, /health 200, /v1/models lists SERVED_NAME. 0 healthy
health() {
  local r0 r1 code body
  r0=$(running 0); r1=$(running 1)
  [[ "$r0" == true ]] || { log "health: rank 0 $r0"; return 1; }
  [[ "$r1" == true ]] || { log "health: rank 1 $r1"; return 1; }
  body=$(curl -s -m 10 "${AUTH[@]}" -w '\n%{http_code}' "$BASE/health" || true); code=${body##*$'\n'}; body=${body%$'\n'*}
  [[ "$code" == 200 ]] || { log "health: /health $code ${body:0:300}"; return 1; }
  curl -s -m 10 "${AUTH[@]}" "$BASE/v1/models" | grep -q "\"$SERVED_NAME\"" || { log "health: /v1/models does not list $SERVED_NAME"; return 1; }
  log "health: ok (both ranks up, /health 200, $SERVED_NAME listed)"
}

cmd_xid() { # NVIDIA Xid events on both Sparks since $1 (journalctl syntax; default the last hour); non-zero if one was seen
  local since=${1:-"-1h"} q out rc=0
  q="journalctl -k --since '$since' --no-pager 2>/dev/null | grep -E 'NVRM: Xid' || dmesg 2>/dev/null | grep -E 'NVRM: Xid' || true"
  for node in head worker; do
    out=$(on_node "$node" "$q" 2>/dev/null || true)
    if [[ -n "$out" ]]; then rc=1; sed "s/^/[$node] /" <<<"$out"; fi
  done
  (( rc == 0 )) && log "xid: none since $since on either Spark"
  return $rc
}

case "${1:-}" in
build|prebuild)
  shift
  if server_running; then die "the server is running: scripts/serve.sh stop first (a build beside the weights can freeze a Spark)"; fi
  exec ./scripts/prepare.sh --image-only "$@" ;;
prepare)
  if server_running; then die "the server is running: scripts/serve.sh stop first (the setup reads ${MODEL_GIB} GiB)"; fi
  shift; exec ./scripts/prepare.sh "$@" ;;
preflight) preflight "${2:-}" && log "preflight ok${2:+ ($2)}" ;;
start|restart)
  mode=$1; shift
  case "$PREFLIGHT" in
    off) ;;
    # a restart preflights statically: the running pair holds the ports and the GPUs until ./start.sh stops it
    strict) preflight "$( [[ "$mode" == restart ]] && server_running && echo static)" || die "preflight failed (above); not starting (PREFLIGHT=warn starts anyway)" ;;
    *) preflight "$( [[ "$mode" == restart ]] && server_running && echo static)" || warn "preflight: problems above (PREFLIGHT=warn: starting anyway)" ;;
  esac
  if [[ "$mode" == restart ]]; then exec ./start.sh restart "$@"; else exec ./start.sh "$@"; fi ;;
stop) exec ./stop.sh ;;
status)
  echo "rank 0 (here): $(docker ps -a --filter "name=^${CONTAINER_NAME}\$" --format '{{.Status}}' 2>/dev/null || echo '?')"
  echo "rank 1 ($WORKER): $(worker 1 "docker ps -a --filter 'name=^${CONTAINER_NAME}\$' --format '{{.Status}}'" 2>/dev/null || echo unreachable)"
  [[ -f "$STATE_DIR/pairwatch.pid" ]] && echo "pair watch: on (pid $(cat "$STATE_DIR/pairwatch.pid"))" || echo "pair watch: off"
  [[ -f "$STATE_DIR/memguard.pid" ]] && echo "memory guard here: on" || echo "memory guard here: off"
  [[ ! -f "$STATE_DIR/failfast" ]] || echo "fail-fast: $(cut -d' ' -f2- "$STATE_DIR/failfast") (at $(date -d "@$(cut -d' ' -f1 "$STATE_DIR/failfast")" '+%F %T'))"
  [[ ! -f "$STATE_DIR/stopped" ]] || echo "stopped on purpose (the watchdog stands down until the next start)"
  rm_=$(roce_marks | paste -sd, -); [[ -z "$rm_" ]] || echo "RoCE failure marker on the $rm_: serving on NCCL (scripts/serve.sh roce-reset)"
  ts=$(thermal_stops); [[ -z "$ts" ]] || echo "THERMAL HARD STOP recorded: ${ts//$'\n'/; } (scripts/serve.sh thermal-clear)"
  echo "device thermal state: $(device_state head) here, $(device_state worker) on the worker; hottest reading (GPU, ACPI): $(hottest_now head || echo ?) / $(hottest_now worker || echo ?) C (the device's limits: pause ${GUARD_PAUSE_C:-?}, resume ${GUARD_RESUME_C:-?}, stop ${GUARD_KILL_C:-?} C for ${GUARD_KILL_HOLD_S:-?} s)"
  echo "MemAvailable: $(mem_available_gib) GiB here, $(worker 1 "awk '/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}' /proc/meminfo" 2>/dev/null || echo ?) GiB on the worker"
  curl -s -m 10 "${AUTH[@]}" "$BASE/v1/models" || true; echo
  curl -s -m 10 "${AUTH[@]}" "$BASE/health" || true; echo ;;
logs) if [[ "${2:-0}" == 1 ]]; then worker 1 docker logs --tail "${TAIL:-80}" "$CONTAINER_NAME"; else docker logs --tail "${TAIL:-80}" "$CONTAINER_NAME"; fi ;;
health) health ;;
canary)
  cargs=(--base "$BASE" --model "$SERVED_NAME"); [[ "$API_KEY" != 1 ]] || cargs+=(--api-key-file "$API_KEY_FILE")
  exec python3 -I scripts/canary.py "${cargs[@]}" ;;
args) DRY_RUN=1 exec ./start.sh ;;
xid) cmd_xid "${2:-}" ;;
roce-reset)
  [[ "${TF_ROCE_MARK:-}" == /cache/* ]] || die "TF_ROCE_MARK is not a path under /cache (config/prod.env)"
  q="docker run --rm --network none -v '$CACHE_VOL:/cache' --entrypoint rm '$IMAGE' -f '$TF_ROCE_MARK'"
  bash -c "$q" && worker 1 "$q" || die "could not remove $TF_ROCE_MARK on both nodes"
  log "RoCE failure marker removed on both nodes: the next start tries RoCE" ;;
thermal-clear)
  ts=$(thermal_stops); [[ -n "$ts" ]] || { log "no thermal hard stop is recorded"; exit 0; }
  need_device_limits
  h=$(hottest_now head || true); w=$(hottest_now worker || true)
  [[ "$h" =~ ^[0-9]+$ && "$w" =~ ^[0-9]+$ ]] && (( h <= GUARD_RESUME_C && w <= GUARD_RESUME_C )) ||
    die "both nodes must read at or below GUARD_RESUME_C $GUARD_RESUME_C C first (here ${h:-?} C, worker ${w:-?} C); the record stays: $ts"
  [[ "$(device_state head)" == normal && "$(device_state worker)" == normal ]] ||
    die "both nodes' device thermal state must be normal first (here $(device_state head), worker $(device_state worker)); the record stays: $ts"
  rm -f "$STATE_DIR/thermal-stop"; worker 1 "rm -f $MEMGUARD_REMOTE_DIR/thermal-stop"
  log "thermal hard stop cleared ($ts; now $h / $w C): scripts/serve.sh start may run again" ;;
watch) shift; WATCH_RESTART_CMD="${WATCH_RESTART_CMD:-$PWD/scripts/serve.sh restart}" exec ./scripts/watch.sh "$@" ;;
""|help|-h|--help) usage; [[ -n "${1:-}" ]] || exit 2 ;;
*) usage; exit 2 ;;
esac
