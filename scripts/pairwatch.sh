#!/usr/bin/env bash
# Pair watch for the two ranks: a lone rank waits for its peer forever (its next NCCL collective never completes), holding most
# of its Spark's memory and answering /health as if all were well. This stops the surviving rank as soon as the other one has
# exited, and records why in <state dir>/failfast, so the watchdog (scripts/watch.sh) heals the pair on its next tick. It is the
# DeepSeek recipe's fail-fast (a failure on either rank ends both in about a second, exit 70) done from outside the engine
# (the minimax_m3 family's own coherent stop, exit 70, comes with the engine branch mm3/robust; it is recorded as such):
# whether the minimax_m3_vl family ends rank 0 by itself when rank 1 is lost (its RankLost path) is NOT MEASURED, and this does
# not depend on it.
# Runs on the head (rank 0): rank 1 is read over ssh. A worker out of reach is not an exit: it is logged and checked again.
# Self-contained (start.sh starts it, stop.sh stops it). A deliberate stop (stop.sh) stops this first.
#
# Usage: pairwatch.sh start <container> <worker ssh target> <interval s> <state dir> <log file>
#        pairwatch.sh stop <state dir>
set -u

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
nap() { sleep "$1" & wait $!; }   # a sleep a signal ends at once
# state <here|worker>: true | false | absent | unreachable, and the exit code after a space when not running
state() {
  local out rc
  if [[ "$1" == here ]]; then
    out=$(docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' "$C" 2>/dev/null) || { echo absent; return; }
  else
    out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$W" "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' '$C' 2>/dev/null || echo absent") && rc=0 || rc=$?
    [[ $rc == 255 ]] && { echo unreachable; return; }
  fi
  [[ "$out" == "true "* ]] && echo true || echo "$out"
}

run_watch() {
  C=$1 W=$2
  local every=$3 dir=$4 log=$5 r0 r1 why
  mkdir -p "$dir" "$(dirname "$log")"
  echo $$ > "$dir/pairwatch.pid"
  trap '[[ "$(cat "$dir/pairwatch.pid" 2>/dev/null)" == "$$" ]] && rm -f "$dir/pairwatch.pid"; exit 0' TERM INT
  echo "$(stamp) pairwatch: watching $C here and on $W every $every s" >> "$log"
  while nap "$every"; do
    r0=$(state here); r1=$(state worker)
    [[ "$r1" == unreachable ]] && { echo "$(stamp) pairwatch: $W unreachable over ssh (not counted)" >> "$log"; continue; }
    [[ "$r0" == true && "$r1" == true ]] && continue
    why="rank 0 ${r0/false /exited with code }, rank 1 ${r1/false /exited with code }"
    # exit 70 is the engine's coherent stop (tp_watch): both ranks end on purpose; the survivor, if any, is about to
    [[ "$why" != *"code 70"* ]] || why="coherent stop (exit 70): $why"
    # the survivor first (a stop takes seconds), then the record
    if [[ "$r0" == true ]]; then timeout 60 docker stop -t 10 "$C" >/dev/null 2>&1
    elif [[ "$r1" == true ]]; then timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$W" "docker stop -t 10 '$C'" >/dev/null 2>&1
    fi
    printf '%s %s\n' "$(date +%s)" "$why" > "$dir/failfast"
    echo "$(stamp) pairwatch: $why: stopped the other rank; wrote $dir/failfast for the watchdog" >> "$log"
    [[ "$(cat "$dir/pairwatch.pid" 2>/dev/null)" == "$$" ]] && rm -f "$dir/pairwatch.pid"
    exit 0
  done
}

stop_watch() {
  local pid
  pid=$(cat "$1/pairwatch.pid" 2>/dev/null) || return 0
  if [[ "$pid" =~ ^[0-9]+$ ]] && grep -q pairwatch "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 30); do [[ -d /proc/$pid ]] || break; sleep 0.1; done
  fi
  rm -f "$1/pairwatch.pid"
}

case "${1:-}" in
  start)
    shift
    (( $# == 5 )) || { echo "usage: pairwatch.sh start <container> <worker ssh target> <interval s> <state dir> <log file>" >&2; exit 2; }
    [[ "$3" =~ ^[1-9][0-9]*$ ]] || { echo "pairwatch: the interval is whole seconds, not $3" >&2; exit 2; }
    stop_watch "$4"
    rm -f "$4/pairwatch.pid"
    # detached (its own session, no terminal) and without start.sh's lock (fd 8) or prepare.sh's (fd 9)
    setsid nohup "$0" run "$@" >/dev/null 2>&1 < /dev/null 8>&- 9>&- &
    for _ in $(seq 1 20); do [[ -f "$4/pairwatch.pid" ]] && exit 0; sleep 0.1; done
    echo "pairwatch: did not start" >&2; exit 1 ;;
  run) shift; run_watch "$@" ;;
  stop) stop_watch "${2:?usage: pairwatch.sh stop <state dir>}" ;;
  *) echo "usage: pairwatch.sh start|stop ..." >&2; exit 2 ;;
esac
