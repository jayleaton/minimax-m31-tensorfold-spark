#!/usr/bin/env bash
# The TP2 draft measurement (docs/DFLASH2.md): C1 greedy requests in four classes (bench/draft_positions.py) against
# the server that runs now, drafts on and drafts off, then one report. Run it on the head, in the window, after each
# start:
#   DRAFTS=0 scripts/serve.sh restart                       # the baseline: plain decode
#   scripts/measure-drafts.sh run plain <dir>
#   DRAFTS=dflash2 MEM_GATE_GIB=<calc> scripts/serve.sh restart    # copies + the DFlash2 draft
#   scripts/measure-drafts.sh run dflash2 <dir>              # drafts on, then the same requests with "draft": false
#   DRAFTS=1 DRAFTER=dspark MEM_GATE_GIB=<calc> scripts/serve.sh restart   # optional: the dspark draft, same window
#   scripts/measure-drafts.sh run dspark <dir>
#   scripts/measure-drafts.sh report <dir>                   # exit 1 when any drafted reply differs from plain
# The label must match the server's profile (MM31_SERVE_PROFILE in rank 0's container): plain needs drafts=0, dflash2
# drafts=dflash2, dspark drafter=dspark. Before every request each Spark's device state (SPARK_THERMAL_RUN/state, written by
# spark-thermal-guard from /etc/spark/thermal.env) must read normal: the run waits while a node is throttled and stops on
# emergency. BENCH_ARGS passes more options to bench/draft_positions.py run (e.g. "--reps 2 --classes code,agent").
# URL defaults to http://127.0.0.1:${PORT:-8000}; API_KEY_FILE (if the server asks for a key) is read, never printed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[[ -f "$here/config.sh" ]] && source "$here/config.sh" 2>/dev/null || true

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
cmd=${1:-}
case "$cmd" in
  run) label=${2:-}; dir=${3:-}; [[ "$label" =~ ^(plain|dflash2|dspark)$ && -n "$dir" ]] || usage ;;
  report) dir=${2:-}; [[ -n "$dir" ]] || usage ;;
  *) usage ;;
esac
bench="$here/../bench/draft_positions.py"

if [[ $cmd == report ]]; then
  [[ -s "$dir/plain.jsonl" ]] || { echo "[measure-drafts] no $dir/plain.jsonl: the DRAFTS=0 run is the baseline (run plain first)" >&2; exit 2; }
  files=("$dir/plain.jsonl")
  for f in dflash2 dflash2-off dspark dspark-off; do [[ ! -s "$dir/$f.jsonl" ]] || files+=("$dir/$f.jsonl"); done
  exec python3 -I "$bench" report "${files[@]}" --json "$dir/report.json"
fi

url=${URL:-http://127.0.0.1:${PORT:-8000}}
mkdir -p "$dir"
# the server's profile, from rank 0's container (start.sh puts the serve flags in MM31_SERVE_PROFILE)
profile=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME:-minimax-m3-flash-tf}" 2>/dev/null |
  sed -n 's/^MM31_SERVE_PROFILE=//p')
[[ -n "$profile" ]] || { echo "[measure-drafts] no running server (rank 0's container ${CONTAINER_NAME:-minimax-m3-flash-tf} has no MM31_SERVE_PROFILE)" >&2; exit 2; }
case "$label" in
  plain) want=",drafts=0," ;;
  dflash2) want=",drafts=dflash2," ;;
  dspark) want=",drafter=dspark," ;;
esac
[[ ",$profile," == *"$want"* ]] || { echo "[measure-drafts] the server runs $profile, not a $label profile (${want//,/}): start it as docs/DFLASH2.md says" >&2; exit 2; }
curl -sf --max-time 10 "$url/v1/models" >/dev/null || { echo "[measure-drafts] $url/v1/models does not answer" >&2; exit 2; }
printf '%s\n' "$profile" > "$dir/$label.profile"

# each node's device state, read before every request (the limits are the device's; nothing here holds a number)
run_dir=${SPARK_THERMAL_RUN:-/run/spark-thermal}
states=(--state-cmd "cat $(printf '%q' "$run_dir/state")")
if [[ -n "${WORKER:-}" ]]; then
  states+=(--state-cmd "ssh -o BatchMode=yes -o ConnectTimeout=10 $(printf '%q' "$WORKER") cat $(printf '%q' "$run_dir/state")")
fi
common=(--base "$url" --model "${SERVED_NAME:-MiniMax-M3.1-Flash-EXL3}" "${states[@]}")
[[ -z "${API_KEY_FILE:-}" || ! -r "${API_KEY_FILE:-}" || "${API_KEY:-0}" != 1 ]] || common+=(--api-key-file "$API_KEY_FILE")
read -r -a extra <<<"${BENCH_ARGS:-}"
rc=0
python3 -I "$bench" run --label "$label" --out "$dir/$label.jsonl" "${common[@]}" "${extra[@]}" || rc=$?
if [[ $label != plain ]]; then
  # the same requests with "draft": false on this server: plain decode beside the drafted run (the engine honours it per
  # request; a reply that drafted anyway fails this pass, and the DRAFTS=0 run stays the baseline)
  python3 -I "$bench" run --label "$label-off" --draft-off --out "$dir/$label-off.jsonl" "${common[@]}" "${extra[@]}" || rc=$?
fi
exit "$rc"
