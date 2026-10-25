#!/usr/bin/env bash
# Install the device's heat and GPU limits on both Sparks: the machine settings this recipe reads, not recipe settings.
# Copies device/ (the operator's files, verbatim) to each node:
#   device/thermal.env                  -> /etc/spark/thermal.env (0644): the source of truth on that node
#   device/spark-thermal-guard          -> /usr/local/sbin/spark-thermal-guard (0755)
#   device/spark-thermal-guard.service  -> /etc/systemd/system/ (0644): throttles at GUARD_PAUSE_C, restores at
#                                          GUARD_RESUME_C, emergency at GUARD_KILL_C held GUARD_KILL_HOLD_S; publishes
#                                          /run/spark-thermal/state; never kills a process
#   device/dgx-gpu-clock-cap.service    -> /etc/systemd/system/ (0644), only where the node has none
#   device/20-from-thermal-env.conf     -> /etc/systemd/system/dgx-gpu-clock-cap.service.d/ (0644): the cap from the file
# then daemon-reload, enables and (re)starts dgx-gpu-clock-cap.service and spark-thermal-guard.service, and reads back
# what it installed (file hashes, both units active, a fresh state file).
#
# It changes system state with sudo on both machines: ASK THE USER before running it with --apply (AGENTS.md section 1).
#   scripts/install-device-limits.sh             the plan: what differs on each node; nothing is changed
#   scripts/install-device-limits.sh --apply     install on this Spark, then on WORKER (sudo asks for a password on each)
#   ... --force                                  replace an existing /etc/spark/thermal.env that differs (shown first)
# Another drop-in in dgx-gpu-clock-cap.service.d that sets ExecStart would override the cap: it is renamed
# <name>.disabled-<date> (kept, not deleted) and reported. Rollback is in docs/OPERATIONS.md section 4.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

APPLY=0; FORCE=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done
check_workers
for f in thermal.env spark-thermal-guard spark-thermal-guard.service dgx-gpu-clock-cap.service 20-from-thermal-env.conf; do
  [[ -s "device/$f" ]] || die "device/$f is missing from this checkout"
done

# The per-node part, run as root on each node from a folder holding device/'s files. Self-contained (no config.sh there).
# Arguments: plan|apply, force (0|1).
NODE_SCRIPT=$(cat <<'NODE'
set -euo pipefail
mode=$1 force=$2 src=$(pwd) stamp=$(date +%Y%m%d)
dropins=/etc/systemd/system/dgx-gpu-clock-cap.service.d
same() { [ -f "$2" ] && cmp -s "$1" "$2"; }
say() { echo "  [$(hostname)] $*"; }
for pair in "thermal.env /etc/spark/thermal.env" "spark-thermal-guard /usr/local/sbin/spark-thermal-guard" \
            "spark-thermal-guard.service /etc/systemd/system/spark-thermal-guard.service" \
            "20-from-thermal-env.conf $dropins/20-from-thermal-env.conf"; do
  set -- $pair
  if same "$src/$1" "$2"; then say "$2: already this file"; elif [ -e "$2" ]; then say "$2: DIFFERS (would be replaced)"; else say "$2: missing (would be installed)"; fi
done
if [ -e /etc/systemd/system/dgx-gpu-clock-cap.service ] || systemctl cat dgx-gpu-clock-cap.service >/dev/null 2>&1; then
  say "dgx-gpu-clock-cap.service: present (kept as it is)"; base=0
else
  say "dgx-gpu-clock-cap.service: missing (would be installed)"; base=1
fi
others=$(for f in "$dropins"/*.conf; do [ -e "$f" ] || continue; [ "${f##*/}" = 20-from-thermal-env.conf ] && continue; grep -q '^ExecStart=' "$f" && echo "$f"; done; true)
for f in $others; do say "$f sets ExecStart: would be renamed $f.disabled-$stamp"; done
if [ -e /etc/spark/thermal.env ] && ! same "$src/thermal.env" /etc/spark/thermal.env; then
  say "the installed /etc/spark/thermal.env differs from device/thermal.env:"; diff /etc/spark/thermal.env "$src/thermal.env" | sed 's/^/      /' || true
  [ "$mode" = plan ] || [ "$force" = 1 ] || { say "refusing to replace it without --force"; exit 3; }
fi
[ "$mode" = apply ] || exit 0
install -D -m 0644 "$src/thermal.env" /etc/spark/thermal.env
install -D -m 0755 "$src/spark-thermal-guard" /usr/local/sbin/spark-thermal-guard
install -D -m 0644 "$src/spark-thermal-guard.service" /etc/systemd/system/spark-thermal-guard.service
[ "$base" = 0 ] || install -D -m 0644 "$src/dgx-gpu-clock-cap.service" /etc/systemd/system/dgx-gpu-clock-cap.service
install -D -m 0644 "$src/20-from-thermal-env.conf" "$dropins/20-from-thermal-env.conf"
for f in $others; do mv "$f" "$f.disabled-$stamp"; say "renamed $f -> $f.disabled-$stamp"; done
systemctl daemon-reload
systemctl enable dgx-gpu-clock-cap.service spark-thermal-guard.service >/dev/null 2>&1
systemctl restart dgx-gpu-clock-cap.service
systemctl restart spark-thermal-guard.service
sleep 6
for u in dgx-gpu-clock-cap.service spark-thermal-guard.service; do say "$u: $(systemctl is-active "$u")"; done
age=$(( $(date +%s) - $(stat -c %Y /run/spark-thermal/state 2>/dev/null || echo 0) ))
say "/run/spark-thermal/state: $(cat /run/spark-thermal/state 2>/dev/null || echo missing) (${age}s old), $(cat /run/spark-thermal/temp 2>/dev/null || echo ?) C"
say "/etc/spark/thermal.env sha256 $(sha256sum /etc/spark/thermal.env | cut -c1-16)"
[ "$(systemctl is-active spark-thermal-guard.service)" = active ] && [ "$age" -le 10 ]
NODE
)
mode=plan; (( APPLY )) && mode=apply
log "device/thermal.env sha256 $(sha256sum device/thermal.env | cut -c1-16); mode: $mode$( (( FORCE )) && echo ', --force')"
(( APPLY )) || log "Plan only: nothing is changed. --apply installs it with sudo on this Spark and on $WORKER (ask the user first)"

# this Spark
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
cp device/* "$tmp/"
log "This Spark ($(hostname)):"
as_root=(); (( APPLY )) && as_root=(sudo)   # the plan reads world-readable files only: no sudo for it
( cd "$tmp" && "${as_root[@]}" bash -c "$NODE_SCRIPT" node "$mode" "$FORCE" ) || die "this Spark: the device limits were not installed (above)"

# the worker: device/ goes over ssh into a temporary folder, then the same script runs there with sudo (a terminal for its
# password prompt)
wtmp=$(worker 1 mktemp -d) || die "cannot reach $WORKER"
tar -C device -cf - . | worker 1 "tar -C '$wtmp' -xf -" || die "could not copy device/ to $WORKER"
log "The worker ($WORKER):"
ssh -t -o BatchMode=no -o ConnectTimeout=10 "$WORKER" "cd '$wtmp' && ${as_root[*]} bash -c $(printf '%q' "$NODE_SCRIPT") node $mode $FORCE; rc=\$?; rm -rf '$wtmp'; exit \$rc" ||
  die "the worker: the device limits were not installed (above)"
(( APPLY )) && log "Done: both Sparks carry the device limits; scripts/serve.sh preflight checks them" || true
