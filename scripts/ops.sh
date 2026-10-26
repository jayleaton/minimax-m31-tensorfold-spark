# shellcheck shell=bash
# The pair's operations beyond start and stop, sourced by scripts/serve.sh after config.sh and nodes.sh (it uses serve.sh's
# running, server_running, preflight, BASE and AUTH):
#   - health and readiness verdicts, for people and (--json) for supervisors and load balancers
#   - the boot timing record of every start (STATE_DIR/boot-times.jsonl)
#   - the prepared per-rank weights' files (PREPARED=1): list and clean
#   - the last good image and pack revision (STATE_DIR/last-good.env), rollback to it, and the upgrade path that falls back
#     to it by itself
# The DeepSeek-V4.1 recipe's boot timing, prepared folders and boot start, adapted to this engine (docs/PORT.md).

LAST_GOOD="$STATE_DIR/last-good.env"        # the image and revision of the last start whose canary passed
ROLLBACK_PIN="$STATE_DIR/rollback.env"      # while it exists, start / restart run on the last good image (rollback)
BOOT_TIMES="$STATE_DIR/boot-times.jsonl"    # one JSON line a start

# ops_kv <file> <KEY>: a KEY=value line of a state file (read, never run); empty when missing
ops_kv() { [[ -f "$1" ]] && sed -n "s/^$2=//p" "$1" | tail -1; return 0; }

# ---------------------------------------------------------------- health and readiness
# ops_checks <health|ready>: one "<name><TAB><ok|fail><TAB><detail>" line a check. health (liveness): both ranks running,
# /health 200, /v1/models lists SERVED_NAME. ready adds: the last start's canary passed for this rank 0 container's start
# time (STATE_DIR/canary-last, written by start.sh), both nodes' device thermal state normal, no thermal-stop record and no
# fail-fast record.
ops_checks() {
  local r0 r1 code body started rec st n ts
  ck() { local d=${3//$'\t'/ }; printf '%s\t%s\t%s\n' "$1" "$2" "${d//$'\n'/ }"; }
  r0=$(running 0); r1=$(running 1)
  if [[ "$r0" == true ]]; then ck rank0 ok running; else ck rank0 fail "$r0"; fi
  if [[ "$r1" == true ]]; then ck rank1 ok running; else ck rank1 fail "$r1"; fi
  body=$(curl -s -m 10 "${AUTH[@]}" -w '\n%{http_code}' "$BASE/health" 2>/dev/null || true); code=${body##*$'\n'}; body=${body%$'\n'*}
  if [[ "$code" == 200 ]]; then ck health ok "/health 200"; else ck health fail "/health ${code:-000} ${body:0:200}"; fi
  if curl -s -m 10 "${AUTH[@]}" "$BASE/v1/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; then ck models ok "$SERVED_NAME listed"
  else ck models fail "/v1/models does not list $SERVED_NAME"; fi
  [[ "$1" == ready ]] || return 0
  started=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || true)
  rec=$(cat "$STATE_DIR/canary-last" 2>/dev/null || true)
  if [[ -n "$started" && "$rec" == "started=$started result=pass "* ]]; then ck canary ok "passed for this start (${rec##*at=})"
  elif [[ -n "$started" && "$rec" == "started=$started "* ]]; then ck canary fail "the canary of this start: $(sed -E 's/.* result=([a-z]+).*/\1/' <<<"$rec") (CANARY=strict passes it or stops the pair)"
  else ck canary fail "no canary record for this rank 0 container (started ${started:-?}): it was not started by start.sh, or the record is older"; fi
  for n in head worker; do
    st=$(device_state "$n")
    if [[ "$st" == normal ]]; then ck "thermal_$n" ok normal; else ck "thermal_$n" fail "device thermal state $st"; fi
  done
  ts=$(thermal_stops)
  if [[ -z "$ts" ]]; then ck thermal_stop ok none; else ck thermal_stop fail "${ts//$'\n'/; } (scripts/serve.sh thermal-clear)"; fi
  if [[ ! -f "$STATE_DIR/failfast" ]]; then ck failfast ok none; else ck failfast fail "$(cut -d' ' -f2- "$STATE_DIR/failfast")"; fi
}
# ops_verdict <health|ready> [--json]: the checks as log lines (one a failure) or one JSON object; 0 when every check passed
ops_verdict() {
  local kind=$1 lines bad
  lines=$(ops_checks "$kind")
  bad=$(grep -c $'\tfail\t' <<<"$lines" || true)
  if [[ "${2:-}" == --json ]]; then
    python3 -I -c '
import json, sys, time
checks = {}
for line in sys.stdin.read().splitlines():
    name, verdict, detail = (line.split("\t", 2) + ["", ""])[:3]
    checks[name] = {"ok": verdict == "ok", "detail": detail}
print(json.dumps({"check": sys.argv[1], "ok": all(c["ok"] for c in checks.values()),
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks}))' "$kind" <<<"$lines"
  elif (( bad == 0 )); then
    if [[ "$kind" == ready ]]; then log "ready: ok (both ranks up, /health 200, $SERVED_NAME listed, the canary passed for this start, both nodes normal, no stop record)"
    else log "health: ok (both ranks up, /health 200, $SERVED_NAME listed)"; fi
  else
    awk -F'\t' -v k="$kind" '$2 == "fail" {print k ": " $1 ": " $3}' <<<"$lines" | while IFS= read -r l; do log "$l"; done
  fi
  (( bad == 0 ))
}

# ---------------------------------------------------------------- boot timing
# The phases, from start.sh's marks (begin is serve.sh's, before preflight): checks = begin..gate plus gate_done..launch
# (preflight, setup, family gate, link, ports, the prepared folders, memguard), gate = the drop and the memory gate, load =
# launch..answered (/v1/models), smoke = answered..smoke_done (a drop and the smoke reply; first_reply is the reply alone),
# canary = smoke_done..live (the canary and its drop), total = begin..live.
OPS_BOOT_PY='
import json, sys, time
marks_file, out, mode, ok, image, ident, rev, p0, p1 = sys.argv[1:10]
m = {}
for line in open(marks_file):
    name, _, t = line.strip().partition(" ")
    try:
        m.setdefault(name, float(t))
    except ValueError:
        pass
def span(a, b):
    return round(m[b] - m[a], 1) if a in m and b in m else None
order = ["begin", "gate", "gate_done", "launch", "answered", "smoke", "smoke_done", "live"]
s = {"checks": None, "gate": span("gate", "gate_done"), "load": span("launch", "answered"),
     "smoke": span("answered", "smoke_done"), "first_reply": span("smoke", "smoke_done"),
     "canary": span("smoke_done", "live"), "total": None}
if span("begin", "gate") is not None:
    s["checks"] = round(span("begin", "gate") + (span("gate_done", "launch") or 0), 1)
last = max((n for n in order if n in m), key=lambda n: m[n])
s["total"] = round(m[last] - m["begin"], 1) if "begin" in m else None
ok = ok == "1" and "live" in m
failed_at = None if ok else {"begin": "the checks", "gate": "the memory gate", "gate_done": "the checks after the gate",
                             "launch": "the load", "answered": "the smoke test", "smoke": "the smoke test",
                             "smoke_done": "the canary", "live": "after LIVE"}[last]
rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m.get("begin", time.time()))), "mode": mode, "ok": ok,
       "failed_at": failed_at, "image": image, "image_id": ident[:12], "model_revision": rev,
       "prepared": {"r0": p0, "r1": p1}, "seconds": s}
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
def fmt(v):
    return "?" if v is None else "%.1f s" % v
print("boot times (%s): checks %s, gate wait %s, load %s, smoke %s (first reply %s), canary %s, total %s; prepared r0 %s, r1 %s%s"
      % (mode, fmt(s["checks"]), fmt(s["gate"]), fmt(s["load"]), fmt(s["smoke"]), fmt(s["first_reply"]), fmt(s["canary"]),
         fmt(s["total"]), p0, p1, "" if ok else "; FAILED at " + failed_at))
'
# ops_prepared_outcome <0|1>: what that rank's engine logged of its prepared file (read | wrote | none | refused), "-" when
# the log has no such line (PREPARED=0, an engine without the feature, or the container already removed)
ops_prepared_outcome() {
  local q="docker logs '$CONTAINER_NAME' 2>&1 | grep -F '[tensorfold] prepared: ' | tail -1" out
  if [[ "$1" == 0 ]]; then out=$(bash -c "$q" 2>/dev/null || true); else out=$(worker 1 "$q" 2>/dev/null || true); fi
  out=$(sed -nE 's/.*\[tensorfold\] prepared: ([a-z]+).*/\1/p' <<<"$out")
  echo "${out:--}"
}
# ops_boot_record <marks file> <mode> <rc>: prints the timing line and appends the start's JSON to STATE_DIR/boot-times.jsonl.
# Nothing for a start that never reached the memory gate (a refusal before it, a dry run, a server already running).
ops_boot_record() {
  local marks=$1 mode=$2 rc=$3
  grep -q '^gate ' "$marks" 2>/dev/null || return 0
  python3 -I -c "$OPS_BOOT_PY" "$marks" "$BOOT_TIMES" "$mode" "$([[ "$rc" == 0 ]] && echo 1 || echo 0)" "$IMAGE" \
    "$(image_ident "$IMAGE")" "$(snapshot_rev)" "$(ops_prepared_outcome 0)" "$(ops_prepared_outcome 1)" ||
    warn "could not write the boot timing record ($BOOT_TIMES)"
}
# ops_boot_times [N]: the last N starts (default 10), oldest first
ops_boot_times() {
  local n=${1:-10}
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "boot-times takes a count of starts, not $n"
  [[ -s "$BOOT_TIMES" ]] || { log "no start recorded yet ($BOOT_TIMES)"; return 0; }
  tail -n "$n" "$BOOT_TIMES" | python3 -I -c '
import json, sys
cols = ("checks", "gate", "load", "smoke", "canary", "total")
print("%-20s %-7s %-4s %s %-9s %s" % ("utc", "mode", "ok", " ".join("%8s" % c for c in cols), "prepared", "image / revision"))
for line in sys.stdin:
    try:
        r = json.loads(line)
    except ValueError:
        continue
    s = r.get("seconds", {})
    p = r.get("prepared", {})
    print("%-20s %-7s %-4s %s %-9s %s %s @ %s%s" % (r.get("utc", "?"), r.get("mode", "?"), "yes" if r.get("ok") else "NO",
          " ".join("%8s" % ("-" if s.get(c) is None else "%.1f" % s[c]) for c in cols),
          "%s/%s" % (p.get("r0", "-"), p.get("r1", "-")), r.get("image", "?"), r.get("image_id", "?"),
          (r.get("model_revision") or "?")[:8], "" if r.get("ok") else "  (failed at %s)" % r.get("failed_at")))'
}

# ---------------------------------------------------------------- prepared weights
# ops_age <epoch>: how long ago, e.g. "3d 4h", "2h 5m", "40s"
ops_age() {
  local s=$(( $(date +%s) - $1 ))
  if (( s >= 86400 )); then echo "$(( s / 86400 ))d $(( s % 86400 / 3600 ))h"; elif (( s >= 3600 )); then echo "$(( s / 3600 ))h $(( s % 3600 / 60 ))m"
  elif (( s >= 60 )); then echo "$(( s / 60 ))m"; else echo "${s}s"; fi
}
# ops_prepared list | clean [all]: each node's prepared files (size, age, newest or stale); clean removes the stale ones
# (every file but the newest of each rank), clean all every one. Only file names the engine writes are ever removed.
ops_prepared() {
  local cmd=${1:-list} all=${2:-} node d out stale kind name size mt n freed
  [[ "$cmd" =~ ^(list|clean)$ && ( -z "$all" || ( "$cmd" == clean && "$all" == all ) ) ]] || die "usage: scripts/serve.sh prepared [list | clean [all]]"
  [[ -z "$all" ]] || ! server_running || die "the server is running: prepared clean all would remove the files its ranks read; scripts/serve.sh stop first (plain clean keeps each rank's newest file)"
  [[ "$PREPARED" == 1 ]] || log "PREPARED=0: the ranks neither read nor write these files (config/prod.env)"
  for node in head worker; do
    d=$(prepared_dir "$node") || { warn "cannot resolve the prepared folder on the $node"; continue; }
    if ! out=$(prepared_scan "$node" 2>/dev/null); then warn "the $node's prepared folder $d cannot be read"; continue; fi
    stale=$(prepared_stale "$all" <<<"$out")
    if [[ "$cmd" == list ]]; then
      n=0
      while read -r kind name size mt; do
        [[ "$kind" == F && "$name" =~ $PREPARED_RE ]] || continue
        n=$((n + 1))
        printf '%-7s %s  %6.1f GiB  %-8s  %s\n' "$node" "$name" "$(awk -v b="$size" 'BEGIN {print b / 1073741824}')" \
          "$(ops_age "${mt%.*}")" "$(grep -qxF "$name" <<<"$stale" && echo "stale" || echo "newest of rank ${BASH_REMATCH[1]}")"
      done <<<"$out"
      (( n )) || log "the $node: no prepared file in $d ($(( $(sed -n 's/^D //p' <<<"$out") / 1073741824 )) GiB free)"
      continue
    fi
    freed=0
    while IFS= read -r name; do
      [[ "$name" =~ $PREPARED_RE ]] || continue
      size=$(awk -v n="$name" '$1 == "F" && $2 == n {print $3}' <<<"$out")
      if [[ "$node" == head ]]; then rm -f -- "$d/$name" || { warn "could not remove $d/$name"; continue; }
      else worker 1 "rm -f -- $(printf '%q' "$d/$name")" || { warn "could not remove $d/$name on the worker"; continue; }; fi
      freed=$(( freed + ${size:-0} )); log "removed $name on the $node"
    done <<<"$stale"
    log "the $node: $(( freed / 1073741824 )) GiB freed in $d"
  done
}

# ---------------------------------------------------------------- the last good image, rollback, upgrade
# image_repo <image>: the image name without its tag or digest (a registry's host:port stays)
image_repo() { local i=${1%%@*}; [[ "${i##*/}" != *:* ]] || i=${i%:*}; echo "$i"; }
# ops_last_good_record: after a start whose canary passed (STATE_DIR/canary-last for this rank 0 container), IMAGE is
# tagged <repo>:last-good on both nodes (so a rebuild under the same tag cannot take it away) and STATE_DIR/last-good.env
# records it: the tag, its content id, the pack revision and the engine pins it was built from.
ops_last_good_record() {
  local started rec tag ident tmp
  started=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null) || return 0
  rec=$(cat "$STATE_DIR/canary-last" 2>/dev/null || true)
  [[ "$rec" == "started=$started result=pass "* ]] || return 0
  tag="$(image_repo "$IMAGE"):last-good"; ident=$(image_ident "$IMAGE")
  [[ "$ident" != missing ]] || return 0
  if [[ "$IMAGE" != "$tag" ]]; then
    if ! { docker tag "$IMAGE" "$tag" && worker 1 "docker tag '$IMAGE' '$tag'"; }; then
      warn "could not tag $IMAGE as $tag on both nodes: the last good record ($LAST_GOOD) is unchanged"; return 0
    fi
  fi
  tmp=$(mktemp "$STATE_DIR/last-good.XXXXXX") || return 0
  { echo "# the last start whose canary passed (scripts/serve.sh); KEY=value lines, read, never run"
    echo "UTC=$(date -u +%FT%TZ)"
    echo "IMAGE=$IMAGE"
    echo "LAST_GOOD_IMAGE=$tag"
    echo "IMAGE_IDENT=$ident"
    echo "MODEL_REVISION=$(snapshot_rev)"
    echo "ENGINE_PATCHES=$ENGINE_PATCHES"
    echo "ENGINE_COMMIT=$ENGINE_COMMIT"
    echo "PATCHES_HASH=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null)"
  } > "$tmp" && mv -f "$tmp" "$LAST_GOOD"
  log "last good: $IMAGE (content ${ident:0:12}, kept as $tag on both nodes), $MODEL_ID @ $(snapshot_rev | cut -c1-8) ($LAST_GOOD)"
}
# ops_last_good_check: the last good image is still there, the same content, on both nodes; says what is wrong otherwise
ops_last_good_check() {
  local tag ident h w
  [[ -f "$LAST_GOOD" ]] || { echo "no last good start is recorded ($LAST_GOOD): one start whose canary passes records it"; return 1; }
  tag=$(ops_kv "$LAST_GOOD" LAST_GOOD_IMAGE); ident=$(ops_kv "$LAST_GOOD" IMAGE_IDENT)
  [[ -n "$tag" && "$ident" =~ ^[0-9a-f]{64}$ ]] || { echo "$LAST_GOOD is incomplete"; return 1; }
  h=$(image_ident "$tag"); w=$(worker_image_ident 1 "$tag")
  [[ "$h" == "$ident" && "$w" == "$ident" ]] ||
    { echo "the last good image $tag is not the recorded one on both nodes (here ${h:0:12}, worker ${w:0:12}, recorded ${ident:0:12})"; return 1; }
}
# ops_rollback [why]: pin the last good image and revision (STATE_DIR/rollback.env: every start / restart runs on it until
# `scripts/serve.sh rollback --clear`), then restart on it
ops_rollback() {
  local why=${1:-by hand} msg tag rev
  msg=$(ops_last_good_check) || die "cannot roll back: $msg"
  tag=$(ops_kv "$LAST_GOOD" LAST_GOOD_IMAGE); rev=$(ops_kv "$LAST_GOOD" MODEL_REVISION)
  { echo "# scripts/serve.sh rollback ($why); start / restart run on these until scripts/serve.sh rollback --clear"
    echo "UTC=$(date -u +%FT%TZ)"
    echo "IMAGE=$tag"
    echo "MODEL_REVISION=$rev"
  } > "$ROLLBACK_PIN"
  warn "rolling back ($why): restarting on the last good image $tag (content $(ops_kv "$LAST_GOOD" IMAGE_IDENT | cut -c1-12), was $(ops_kv "$LAST_GOOD" IMAGE)) and $MODEL_ID @ ${rev:0:8}; pinned in $ROLLBACK_PIN"
  ./scripts/serve.sh restart || die "the rollback start failed too (above): the pair is down; read both logs (scripts/serve.sh logs 0 / 1, $LOG_DIR)"
  log "rolled back: serving $tag; config/prod.env still names $IMAGE. Put back the last good pins (ENGINE_PATCHES=$(ops_kv "$LAST_GOOD" ENGINE_PATCHES), ENGINE_COMMIT=$(ops_kv "$LAST_GOOD" ENGINE_COMMIT), MODEL_REVISION=$rev) or fix the new ones and run scripts/serve.sh upgrade; scripts/serve.sh rollback --clear drops the pin"
}
# ops_upgrade: the safe path to a new engine image or pack revision (config/prod.env or the environment name it): the
# server is stopped (no build beside loaded weights), scripts/prepare.sh builds / prepares it, preflight, a start with the
# canary required (CANARY=strict); any failure restores the last good image by itself (ops_rollback) and says so
ops_upgrade() {
  local msg step=""
  msg=$(ops_last_good_check) || die "no safe way back, so no upgrade: $msg"
  if [[ -f "$ROLLBACK_PIN" ]]; then log "a rollback pin is set ($ROLLBACK_PIN): this upgrade tries config/prod.env's image instead"; rm -f "$ROLLBACK_PIN"; fi
  log "upgrade: from $(ops_kv "$LAST_GOOD" IMAGE) (content $(ops_kv "$LAST_GOOD" IMAGE_IDENT | cut -c1-12)) @ $(ops_kv "$LAST_GOOD" MODEL_REVISION | cut -c1-8) to $IMAGE @ $(snapshot_rev | cut -c1-8); the last good stays as $(ops_kv "$LAST_GOOD" LAST_GOOD_IMAGE)"
  if server_running; then log "upgrade: stopping the pair (the build and the checks need it stopped)"; ./stop.sh || step="the stop"; fi
  [[ -n "$step" ]] || ./scripts/prepare.sh || step="the build / prepare (scripts/prepare.sh)"
  [[ -n "$step" ]] || preflight || step="preflight"
  [[ -n "$step" ]] || CANARY=strict PREFLIGHT=off ./scripts/serve.sh start || step="the start or its canary"
  if [[ -z "$step" ]]; then
    [[ "$(ops_kv "$LAST_GOOD" IMAGE_IDENT)" == "$(image_ident "$IMAGE")" ]] || step="the last good record (the canary passed but $LAST_GOOD was not updated)"
  fi
  if [[ -z "$step" ]]; then log "upgrade ok: serving $IMAGE @ $(snapshot_rev | cut -c1-8), now the last good"; return 0; fi
  warn "upgrade to $IMAGE failed at $step: restoring the last good image by itself"
  ops_rollback "upgrade to $IMAGE failed at $step"
  die "upgrade to $IMAGE failed at $step; ROLLED BACK to $(ops_kv "$LAST_GOOD" LAST_GOOD_IMAGE), which serves now (pinned: $ROLLBACK_PIN)"
}
