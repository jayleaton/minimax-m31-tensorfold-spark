#!/usr/bin/env bash
# Start the pair once after a reboot, and only if it was serving before it. Run on the head by
# scripts/systemd/mm31-boot-start.service (a user unit; `loginctl enable-linger <user>` lets it run without a login); the
# watchdog (tf-watchdog.timer, WATCH_HEAL=1) stays the fallback. The DeepSeek-V4.1 recipe's scripts/boot-start.sh, on this
# recipe's markers: start.sh writes STATE_DIR/serving when the pair is LIVE; a deliberate stop (scripts/serve.sh stop,
# ./stop.sh) removes it; a stop while the system shuts down (mm31-serve.service's ExecStop) keeps it. In order:
#   - BOOT_LEASE (optional file) younger than BOOT_LEASE_MIN minutes: nothing (you are working on the pair)
#   - no serving marker, or one written during this boot: nothing (stopped on purpose, never started, or started already)
#   - docker answers here (it waits, at most BOOT_WAIT s in all); rank 0 already running: nothing
#   - the worker answers over ssh with docker up, and the link is up: the route to the worker over a CX7 port with a RoCE v2
#     GID (scripts/nodes.sh, the LINK_NETDEV / LINK_HCAS pins), every RoCE device of it ACTIVE on both nodes
#   - then the start: `systemctl --user start BOOT_UNIT` (mm31-serve.service) when that unit is installed, so its ExecStop
#     stops the pair at the next shutdown; else scripts/serve.sh restart. Either runs preflight, the memory gate and the
#     canary, and removes a rank 1 still running on the worker after a head-only reboot.
# Log: STATE_DIR/boot-start.log (BOOT_STDOUT=1: this terminal). BOOT_DRY_RUN=1 decides and waits, and starts nothing.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 2
source ./scripts/config.sh
source ./scripts/nodes.sh

BOOT_WAIT="${BOOT_WAIT:-300}"
BOOT_LEASE="${BOOT_LEASE:-}"
BOOT_LEASE_MIN="${BOOT_LEASE_MIN:-20}"
BOOT_UNIT="${BOOT_UNIT-mm31-serve.service}"
[[ "$BOOT_WAIT" =~ ^[0-9]+$ && "$BOOT_LEASE_MIN" =~ ^[0-9]+$ ]] || die "BOOT_WAIT (s) and BOOT_LEASE_MIN (minutes) are whole numbers"
check_workers
mkdir -p "$STATE_DIR"
[[ "${BOOT_STDOUT:-0}" == 1 ]] || exec >> "$STATE_DIR/boot-start.log" 2>&1
up() { awk '{printf "%.0f", $1}' /proc/uptime 2>/dev/null || echo "?"; }
blog() { echo "[$(date '+%F %T')] boot-start (+$(up)s): $*"; }
boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)

if [[ -n "$BOOT_LEASE" && -f "$BOOT_LEASE" ]] && (( $(date +%s) - $(stat -c %Y "$BOOT_LEASE") < BOOT_LEASE_MIN * 60 )); then
  blog "$BOOT_LEASE is fresh: not starting (the watchdog takes over when it ages)"; exit 0
fi
if [[ ! -f "$STATE_DIR/serving" ]]; then
  blog "the pair was not serving before this boot (no $STATE_DIR/serving: stopped on purpose, or never started): nothing to do"; exit 0
fi
read -r sb st < "$STATE_DIR/serving" || true
if [[ "${sb:-}" == "$boot_id" ]]; then blog "the serving marker is from this boot: started already, nothing to do"; exit 0; fi
end=$(( $(date +%s) + BOOT_WAIT ))
waited() { (( $(date +%s) < end )); }
until docker info >/dev/null 2>&1; do waited || { blog "docker not up after ${BOOT_WAIT}s: leaving it to the watchdog"; exit 1; }; sleep 2; done
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; then blog "rank 0 already running: nothing to do"; exit 0; fi
blog "the pair was serving before this reboot (since $(date -d "@${st:-0}" '+%F %T' 2>/dev/null || echo '?')): waiting for the worker and the link"

# link_ready: the link detected and pinned as a start would find it, and every RoCE device of it ACTIVE on both nodes
link_ready() {
  local out hcas whcas d
  out=$( (detect_link && check_link_pins && echo "OK ${NODE_HCAS[0]} ${NODE_HCAS[1]}") 2>/dev/null | tail -1)
  [[ "$out" == OK\ * ]] || return 1
  read -r _ hcas whcas <<<"$out"
  for d in ${hcas//,/ }; do grep -q ACTIVE "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null || return 1; done
  worker 1 "for d in ${whcas//,/ }; do grep -q ACTIVE /sys/class/infiniband/\$d/ports/1/state || exit 1; done" 2>/dev/null || return 1
  LINK_SEEN="$hcas / $whcas"
}
until worker 1 'docker info >/dev/null 2>&1' 2>/dev/null && link_ready; do
  waited || { blog "the worker ($(worker_host 1)), its docker or the link is not ready after ${BOOT_WAIT}s: leaving it to the watchdog"; exit 1; }
  sleep 3
done
blog "the worker $(worker_host 1) answers, docker up, the link's RoCE devices ACTIVE on both nodes ($LINK_SEEN)"
if [[ "${BOOT_DRY_RUN:-0}" == 1 ]]; then blog "BOOT_DRY_RUN=1: not starting"; exit 0; fi
if [[ -n "$BOOT_UNIT" ]] && systemctl --user cat "$BOOT_UNIT" >/dev/null 2>&1; then
  blog "systemctl --user start $BOOT_UNIT"
  systemctl --user start "$BOOT_UNIT"; rc=$?
else
  blog "scripts/serve.sh restart (CONFIG=$CONFIG)"
  ./scripts/serve.sh restart; rc=$?
fi
blog "start exit $rc; /v1/models: $(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | cut -c1-120)"
exit "$rc"
