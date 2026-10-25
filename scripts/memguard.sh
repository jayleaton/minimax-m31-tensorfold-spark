#!/usr/bin/env bash
# Memory watchdog for a DGX Spark running a rank: a Spark's GPU shares its 121 GiB with the host, so a rank that takes
# all of it does not get a clean CUDA out-of-memory error, the whole machine stops answering (and has to be power-cycled).
# This kills the rank's container (docker kill, no grace period) before the machine gets there, on either of two levels:
#   - MemAvailable under the HARD floor (default 1 GiB) in two samples in a row (0.25 s apart): at once;
#   - MemAvailable under the (soft) floor for HOLD seconds without once coming back over it (default 4 s): a long prompt's workspace
#     takes a few GiB for a moment and gives them back, which is not a rank in trouble. A dip that recovers is logged, not killed.
# It also acts on the node's heat, from the device's own settings (they belong to the machine, not to this recipe): it sources
# /etc/spark/thermal.env (MEMGUARD_THERMAL_ENV) and refuses to start without it, and every 2 s it reads
# /run/spark-thermal/state (MEMGUARD_THERMAL_RUN), which spark-thermal-guard.service publishes:
#   - emergency: the coherent stop of this rank (docker stop; the pair watch on the head then stops the other) and
#     <pid file's folder>/thermal-stop ("<epoch> <reading> C"). start.sh refuses and the watchdog does not heal while that
#     record exists on either node; scripts/serve.sh thermal-clear removes it once both nodes are `normal` again;
#   - throttled: logged (pausing new work is the guard's, which reads the same state);
#   - missing or older than 30 s (spark-thermal-guard not running): logged, and memguard's own hold below still protects.
# Its own hold, the same rule as the device's: the hottest of the GPU temperature (nvidia-smi) and the ACPI zones at or above
# GUARD_KILL_C for GUARD_KILL_HOLD_S (a reading below it, or none at all, resets the hold) kills the container and writes the
# same record.
# Self-contained: start.sh copies it to the worker.
#
# Usage: memguard.sh start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]
#                                                       (replaces a guard already running for the pid file)
#        memguard.sh stop <pid file>
# The engine keeps its own reserve (TENSORFOLD_MEMORY_RESERVE_GIB, 12.2 GiB by default) free, so a healthy rank stays well
# above a floor of a few GiB; only something else eating into the reserve (or a reserve set too low) trips this.
set -u

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
THERMAL_ENV=${MEMGUARD_THERMAL_ENV:-/etc/spark/thermal.env}
THERMAL_RUN=${MEMGUARD_THERMAL_RUN:-/run/spark-thermal}
# load_limits: this node's device limits (sourced: the operator's root-owned file); refuses without them
load_limits() {
  if [[ ! -r "$THERMAL_ENV" ]]; then
    echo "memguard: no device thermal limits on $(hostname) ($THERMAL_ENV): install them with scripts/install-device-limits.sh" >&2; return 1
  fi
  # shellcheck source=/dev/null
  . "$THERMAL_ENV"
  [[ "${GUARD_KILL_C:-}" =~ ^[1-9][0-9]*$ && "${GUARD_KILL_HOLD_S:-}" =~ ^[1-9][0-9]*$ ]] ||
    { echo "memguard: $THERMAL_ENV has no whole GUARD_KILL_C / GUARD_KILL_HOLD_S" >&2; return 1; }
}
# hottest_c: the hottest of the GPU temperature and the ACPI zones, whole degrees C; empty when nothing could be read
hottest_c() {
  local f v best=-1
  for f in ${MEMGUARD_THERMAL_GLOB:-/sys/class/thermal/thermal_zone*/temp}; do
    read -r v < "$f" 2>/dev/null || continue
    [[ "$v" =~ ^[0-9]+$ ]] && (( v > 0 && v < 150000 )) || continue
    (( v / 1000 > best )) && best=$(( v / 1000 ))
  done
  while read -r v; do
    v=${v%%.*}; [[ "$v" =~ ^[0-9]+$ ]] && (( v > 0 && v < 150 && v > best )) && best=$v
  done < <(timeout 2 "${MEMGUARD_NVIDIA_SMI:-nvidia-smi}" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
  (( best >= 0 )) && echo "$best"
}
avail_kb() { local k v; while read -r k v _; do [[ $k == MemAvailable: ]] && { echo "$v"; return; }; done <"${MEMGUARD_MEMINFO:-/proc/meminfo}"; }

stop_guard() {
  local pidfile=$1 pid
  [[ -f "$pidfile" ]] || return 0
  pid=$(cat "$pidfile" 2>/dev/null)
  if [[ "$pid" =~ ^[0-9]+$ ]] && grep -q memguard "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 50); do [[ -d /proc/$pid ]] || break; sleep 0.1; done   # its exit handler runs first
  fi
  rm -f "$pidfile"
}
# a sleep a signal ends at once (a plain sleep holds the guard's exit handler back until it is over)
nap() { sleep "$1" & wait $!; }

run_guard() {
  local c=$1 floor_kb hard_kb log=$3 pidfile=$4 hold_us a snap now since=0 low_min=0 hard_low=0 floor_gib=$2 hold=${5:-4} hard_gib=${6:-1}
  local kill_c=$GUARD_KILL_C kill_hold=$GUARD_KILL_HOLD_S tick=0 t hot_since=0 dstate dlast=unknown dage record
  floor_kb=$(awk -v g="$floor_gib" 'BEGIN {printf "%d", g * 1048576}')
  hard_kb=$(awk -v g="$hard_gib" 'BEGIN {printf "%d", g * 1048576}')
  hold_us=$(awk -v h="$hold" 'BEGIN {printf "%d", h * 1000000}')
  snap=$(mktemp "${TMPDIR:-/tmp}/memguard-snap.XXXXXX") || snap=/dev/null
  mkdir -p "$(dirname "$log")" "$(dirname "$pidfile")"
  echo $$ > "$pidfile"
  trap '[[ "$(cat "$pidfile" 2>/dev/null)" == "$$" ]] && rm -f "$pidfile"; rm -f "$snap"; exit 0' TERM INT
  echo "$(stamp) memguard: watching $c, floor $floor_gib GiB for $hold s, hard floor $hard_gib GiB (MemAvailable $(( $(avail_kb) / 1048576 )) GiB now)$( (( kill_c > 0 )) && echo ", thermal stop at $kill_c C held $kill_hold s ($(hottest_c || echo '?') C now)")" >> "$log"
  while nap 0.25; do
    # the device state and the thermal hold, every 8th tick (2 s): nvidia-smi is not cheap enough for every 0.25 s
    if (( ++tick % 8 == 0 )); then
      record="$(dirname "$pidfile")/thermal-stop"
      dstate=$(cat "$THERMAL_RUN/state" 2>/dev/null) || dstate=missing
      dage=$(( $(date +%s) - $(stat -c %Y "$THERMAL_RUN/state" 2>/dev/null || echo 0) ))
      (( dage <= 30 )) || dstate="stale (${dage}s; spark-thermal-guard not running?)"
      if [[ "$dstate" != "$dlast" ]]; then echo "$(stamp) memguard: device state $dstate" >> "$log"; dlast=$dstate; fi
      if [[ "$dstate" == emergency ]]; then
        timeout 30 docker stop -t 10 "$c" >/dev/null 2>&1
        echo "$(date +%s) $(cat "$THERMAL_RUN/temp" 2>/dev/null || hottest_c || echo 0) C" > "$record"
        echo "$(stamp) memguard: device emergency (at or above $kill_c C for $kill_hold s): stopped container $c; wrote $record (an operator clears it)" >> "$log"
        nap 5; continue
      fi
      if t=$(hottest_c); then
        if (( t >= kill_c )); then
          (( hot_since )) || { hot_since=$SECONDS; echo "$(stamp) memguard: $t C, at or above $kill_c C: the $kill_hold s hold starts" >> "$log"; }
          if (( SECONDS - hot_since >= kill_hold )); then
            timeout 20 docker kill "$c" >/dev/null 2>&1
            echo "$(date +%s) $t C" > "$record"
            echo "$(stamp) memguard: $t C, at or above $kill_c C for $(( SECONDS - hot_since )) s: killed container $c; wrote $record (an operator clears it)" >> "$log"
            hot_since=0; nap 5; continue
          fi
        elif (( hot_since )); then
          echo "$(stamp) memguard: $t C after $(( SECONDS - hot_since )) s at or above $kill_c C: the hold resets" >> "$log"; hot_since=0
        fi
      else
        echo "$(stamp) memguard: no temperature could be read (GPU and ACPI zones)$( (( hot_since )) && echo ": the hold resets")" >> "$log"; hot_since=0
      fi
    fi
    a=$(avail_kb)
    now=${EPOCHREALTIME/./}
    if (( a >= floor_kb )); then
      if (( since )); then
        echo "$(stamp) memguard: MemAvailable was under the $floor_gib GiB floor for $(( (now - since) / 1000 )) ms (lowest $(( low_min / 1024 )) MiB) and recovered: no kill" >> "$log"
      fi
      since=0; hard_low=0; continue
    fi
    if (( ! since )); then since=$now; low_min=$a; fi
    (( a < low_min )) && low_min=$a
    if (( a < hard_kb )); then (( ++hard_low )); else hard_low=0; fi
    # the hard floor needs two samples (a single read can be a glitch), the soft floor needs HOLD seconds under it throughout
    if (( hard_low < 2 && now - since < hold_us )); then continue; fi
    since=0; hard_low=0
    # Kill first, log afterwards: on a machine that is already thrashing every command ahead of the kill costs time. The
    # process snapshot runs beside it (it shows who else holds the memory); docker kill fails when the container is not running.
    ps -eo rss,pid,comm --sort=-rss 2>/dev/null | head -6 > "$snap" &
    if timeout 20 docker kill "$c" >/dev/null 2>&1; then
      wait
      {
        echo "$(stamp) memguard: MemAvailable $(( a / 1024 )) MiB (lowest $(( low_min / 1024 )) MiB) was under the $floor_gib GiB floor for $hold s or under the $hard_gib GiB hard floor: killed container $c"
        sed 's/^/    /' "$snap"
      } >> "$log"
      nap 5
    else
      wait
    fi
  done
}

case "${1:-}" in
  start)
    shift
    (( $# >= 4 && $# <= 6 )) || { echo "usage: memguard.sh start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]" >&2; exit 2; }
    load_limits || exit 2
    [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: floor is a number of GiB (2 or 2.5), not $2" >&2; exit 2; }
    [[ -z "${5:-}" || "$5" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: hold is a number of seconds, not $5" >&2; exit 2; }
    [[ -z "${6:-}" || "$6" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: the hard floor is a number of GiB, not $6" >&2; exit 2; }
    stop_guard "$4"
    # detach: its own session, no terminal, so it outlives the ssh or shell that started it. Close fd 8, start.sh's lock
    # (flock on STATE_DIR/start.lock): a guard that kept it would hold the lock for as long as it runs.
    setsid nohup "$0" run "$@" >/dev/null 2>&1 < /dev/null 8>&- 9>&- &
    for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -f "$4" ]] && exit 0; sleep 0.2; done
    echo "memguard: did not start" >&2; exit 1 ;;
  run) shift; load_limits || exit 2; run_guard "$@" ;;
  stop) stop_guard "${2:?usage: memguard.sh stop <pid file>}" ;;
  *) echo "usage: memguard.sh start|stop ..." >&2; exit 2 ;;
esac
