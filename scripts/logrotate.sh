#!/usr/bin/env bash
# Rotate the recipe's own logs by size, and note each node's device thermal state beside them. Run on the head by
# scripts/systemd/mm31-logrotate.timer (hourly), or by hand; the head reaches the worker over ssh.
#   here:       LOG_DIR/memguard.log, LOG_DIR/pairwatch.log, LOG_DIR/thermal.log, WATCH_STATE_DIR/heal.log,
#               STATE_DIR/boot-start.log
#   the worker: ~/.cache/tensorfold-minimax-m3/logs/memguard.log (its memguard's)
# A file over LOG_ROTATE_MIB MiB is copied to <file>.1.gz and truncated in place (copytruncate): the writers append with >>
# and go on writing to the same file; only a line written between the copy and the truncation can be lost. The older
# generations shift up (<file>.2.gz ...) and the oldest past LOG_KEEP is removed. The saved rank logs (*-rank<N>.log.gz) are
# stop.sh's and keep their own LOG_KEEP.
# The thermal hook: each run first appends one line a node to LOG_DIR/thermal.log: the device thermal state
# (spark-thermal-guard's SPARK_THERMAL_RUN/state), the hottest GPU / ACPI reading (memguard's reading), and the limits as
# that node's own SPARK_THERMAL_ENV states them. A limit it cannot read is written as unreadable, never as a number.
#   scripts/logrotate.sh             rotate and note
#   scripts/logrotate.sh --dry-run   say what would rotate; change and note nothing
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 2
source ./scripts/config.sh
source ./scripts/nodes.sh

LOG_ROTATE_MIB="${LOG_ROTATE_MIB:-16}"
DRY=0; [[ "${1:-}" != --dry-run ]] || DRY=1
[[ "$LOG_ROTATE_MIB" =~ ^[1-9][0-9]*$ ]] || die "LOG_ROTATE_MIB is a whole number of MiB, not $LOG_ROTATE_MIB"
[[ "$LOG_KEEP" =~ ^[1-9][0-9]*$ ]] || die "LOG_KEEP is a whole number of generations, not $LOG_KEEP"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 7>"$STATE_DIR/logrotate.lock"
flock -n 7 || { log "another logrotate.sh runs: nothing to do"; exit 0; }

# rotate_file <file> <max bytes> <keep> [dry]: copytruncate one file over the size. Self-contained: it runs on the worker too.
rotate_file() {
  local f=$1 max=$2 keep=$3 size i
  [[ -f "$f" ]] || return 0
  size=$(stat -c %s "$f" 2>/dev/null) || return 0
  (( size > max )) || return 0
  if [[ "${4:-}" == dry ]]; then echo "would rotate $f ($(( size / 1048576 )) MiB)"; return 0; fi
  rm -f "$f.$keep.gz"
  for (( i = keep - 1; i >= 1; i-- )); do [[ ! -f "$f.$i.gz" ]] || mv -f "$f.$i.gz" "$f.$(( i + 1 )).gz"; done
  if ! { cp -p "$f" "$f.rotating" && : > "$f"; }; then rm -f "$f.rotating"; echo "could not rotate $f"; return 1; fi
  gzip -f "$f.rotating" && mv -f "$f.rotating.gz" "$f.1.gz"
  echo "rotated $f ($(( size / 1048576 )) MiB) to $f.1.gz"
}
# limits_desc: stdin a device limits file (KEY=value lines); prints the four limits it states, or "unreadable (...)" naming
# what is missing. Pure: nothing is assumed, no default is filled in.
limits_desc() {
  local line pause="" resume="" kill="" hold="" miss=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*GUARD_(PAUSE_C|RESUME_C|KILL_C|KILL_HOLD_S)=([0-9]+)[[:space:]]*$ ]] || continue
    case ${BASH_REMATCH[1]} in
      PAUSE_C) pause=${BASH_REMATCH[2]} ;; RESUME_C) resume=${BASH_REMATCH[2]} ;;
      KILL_C) kill=${BASH_REMATCH[2]} ;; KILL_HOLD_S) hold=${BASH_REMATCH[2]} ;;
    esac
  done
  [[ -n "$pause" ]] || miss+=(GUARD_PAUSE_C); [[ -n "$resume" ]] || miss+=(GUARD_RESUME_C)
  [[ -n "$kill" ]] || miss+=(GUARD_KILL_C); [[ -n "$hold" ]] || miss+=(GUARD_KILL_HOLD_S)
  if (( ${#miss[@]} )); then echo "limits unreadable (no whole ${miss[*]})"
  else echo "limits pause $pause C, resume $resume C, stop $kill C held $hold s"; fi
}
# thermal_line <head|worker>: one line: the node's state, its hottest reading and its own limits
thermal_line() {
  local node=$1 st hot lim sha
  if [[ "$node" == worker ]] && ! worker 1 true 2>/dev/null; then echo "$(date -u +%FT%TZ) worker unreachable over ssh"; return 0; fi
  st=$(device_state "$node"); hot=$(hottest_now "$node" 2>/dev/null || true)
  if [[ "$node" == head ]]; then lim=$(limits_desc < "$SPARK_THERMAL_ENV" 2>/dev/null) || lim="limits unreadable ($SPARK_THERMAL_ENV missing)"
  else lim=$(worker 1 "cat $SPARK_THERMAL_ENV" 2>/dev/null | limits_desc); fi
  [[ -n "$lim" ]] || lim="limits unreadable ($SPARK_THERMAL_ENV)"
  sha=$(device_limits_sha "$node")
  [[ "$hot" =~ ^[0-9]+$ ]] && hot="$hot C" || hot=unreadable
  echo "$(date -u +%FT%TZ) $node state $st, hottest $hot, $lim ($SPARK_THERMAL_ENV sha256 ${sha:0:12})"
}

if (( ! DRY )); then
  { thermal_line head; thermal_line worker; } >> "$LOG_DIR/thermal.log"
fi
max=$(( LOG_ROTATE_MIB * 1048576 )); dry=""; (( ! DRY )) || dry=dry
for f in "$LOG_DIR/memguard.log" "$LOG_DIR/pairwatch.log" "$LOG_DIR/thermal.log" "${WATCH_STATE_DIR:-$STATE_DIR/watch}/heal.log" \
         "$STATE_DIR/boot-start.log"; do
  if out=$(rotate_file "$f" "$max" "$LOG_KEEP" "$dry"); then [[ -z "$out" ]] || log "here: $out"; else warn "here: $out"; fi
done
if out=$(worker 1 "$(declare -f rotate_file); rotate_file \"$MEMGUARD_REMOTE_DIR/logs/memguard.log\" $max $LOG_KEEP $dry" 2>/dev/null); then
  [[ -z "$out" ]] || log "worker: $out"
else
  warn "the worker's memguard.log was not rotated (unreachable, or: $out)"
fi
