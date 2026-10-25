#!/usr/bin/env bash
# Tool-calling benchmarks against the MiniMax-M3.1-Flash endpoint: tool-eval-bench (SeraphimSerapis, MIT) and
# spark-bench TrueScore (Weschera, README says MIT) at pinned commits, installed under TOOLEVAL_CACHE (default
# ~/.cache/tooleval), plus `own` (bench/tooleval/chains.py) and `toolcall` (bench/toolcall.py), stdlib, no install.
#
#   bench/tooleval/run.sh setup                           # clone the pinned commits + venvs (no server needed)
#   bench/tooleval/run.sh teb  MODE [label]               # tool-eval-bench, 69 scenarios
#   bench/tooleval/run.sh sb   MODE [label]               # spark-bench TrueScore, 76 scenarios x 2, temp 0.3
#   bench/tooleval/run.sh both MODE [label]
#   bench/tooleval/run.sh chains MODE [label]             # quick: tool-eval-bench C (multi-step) + spark-bench agentic
#   bench/tooleval/run.sh own MODE [label]                # chains.py: 6 multi-step scenarios, simulated tools
#   bench/tooleval/run.sh toolcall MODE [label]           # toolcall.py: MiniMax markup round trips, typing, leaks
#
# Modes: off  = chat_template_kwargs {"enable_thinking": false}
#        adaptive = thinking on, no effort: the checkpoint template's own (and only) mode
#        low  = thinking on, reasoning_effort low
#        high = thinking on, reasoning_effort high. The server reads chat_template_kwargs enable_thinking /
#               reasoning_effort and the top-level reasoning_effort; the template writes the effort as <effort>.
# Env: BASE_URL (http://$HOST:$PORT/v1, else http://127.0.0.1:8000/v1), MODEL ($SERVED_NAME, else
#      MiniMax-M3.1-Flash-EXL3), API_KEY_FILE, OUT (results/tooleval), TEB_ARGS / SB_ARGS (extra flags),
#      TEB_TEMP (0.0, the bench default), SB_TEMP (0.3), SB_REPEATS (2), TEB_TIMEOUT (off: 300, thinking: 900 s),
#      PY_TEMP (own / toolcall: 0.0; -1 leaves it out, so the server samples with generation_config.json's settings),
#      PY_REPS (own: 1, toolcall: 2).
# Key: API_KEY_FILE (one key, or "label: key", a line; used when readable, required when API_KEY=1) is read into this
# shell only. curl gets it as a header on a file descriptor, tool-eval-bench as TOOL_EVAL_API_KEY and spark-bench as
# SPARK_BENCH_API_KEY in their environment, chains.py / toolcall.py read the file themselves: never argv. Command lines
# and logs go through scripts/redact.py, which also scrubs the run dir when the run ends.
# Each run writes <OUT>/<date>-<bench>-<mode>[-label]/ with the bench's own reports, the server's /health and
# /v1/models before and after, and the command line.
set -euo pipefail

CACHE=${TOOLEVAL_CACHE:-$HOME/.cache/tooleval}
TEB_REPO=https://github.com/SeraphimSerapis/tool-eval-bench.git
TEB_REV=c7b5b9550                      # v2.7.0+8 (2026-09-29)
SB_REPO=https://github.com/Weschera/spark-bench.git
SB_REV=125ba161                        # v6.8.0 (2026-08-29)
host=${HOST:-127.0.0.1}
[[ $host == 0.0.0.0 || $host == :: ]] && host=127.0.0.1   # a wildcard bind is reached on loopback
BASE_URL=${BASE_URL:-http://$host:${PORT:-8000}/v1}
MODEL=${MODEL:-${SERVED_NAME:-MiniMax-M3.1-Flash-EXL3}}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/results/tooleval}

die() { echo "run.sh: $*" >&2; exit 1; }

# the first key of $1 (one key, or "label: key", a line), as scripts/config.sh's api_header_file reads it
read_key() {
    local line
    while IFS= read -r line || [[ -n $line ]]; do
        line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
        [[ -z $line || $line == \#* ]] && continue
        if [[ $line == *:* ]]; then line=${line#*:}; line=${line#"${line%%[![:space:]]*}"}; fi
        printf '%s' "$line"
        return 0
    done < "$1"
}

KEY=""
case ${API_KEY:-0} in
    0|1) ;;
    *) die "API_KEY is 0 or 1 here; put the key itself in API_KEY_FILE (mode 600)" ;;
esac
if [[ -n ${API_KEY_FILE:-} && -r $API_KEY_FILE ]]; then
    KEY=$(read_key "$API_KEY_FILE")
    [[ -n $KEY ]] || die "$API_KEY_FILE holds no key"
elif [[ ${API_KEY:-0} == 1 ]]; then
    die "API_KEY=1 but API_KEY_FILE (${API_KEY_FILE:-unset}) cannot be read"
fi

# redact.py learns the key from its environment (a credential-named variable), never from argv
redact() { REDACT_API_KEY="$KEY" python3 -I -u "$ROOT/scripts/redact.py" "$@"; }

# curl with the key as a header read from a file descriptor, never in argv (ps) or a saved command
api() {
    if [[ -n $KEY ]]; then curl -H @<(printf 'Authorization: Bearer %s\n' "$KEY") "$@"; else curl "$@"; fi
}

setup() {
    mkdir -p "$CACHE"
    if [[ ! -d $CACHE/tool-eval-bench/.git ]]; then git clone -q "$TEB_REPO" "$CACHE/tool-eval-bench"; fi
    git -C "$CACHE/tool-eval-bench" fetch -q origin && git -C "$CACHE/tool-eval-bench" checkout -q "$TEB_REV"
    [[ -x $CACHE/tool-eval-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/tool-eval-bench/.venv"
    "$CACHE/tool-eval-bench/.venv/bin/pip" install -q -e "$CACHE/tool-eval-bench"
    if [[ ! -d $CACHE/spark-bench/.git ]]; then git clone -q "$SB_REPO" "$CACHE/spark-bench"; fi
    git -C "$CACHE/spark-bench" fetch -q origin && git -C "$CACHE/spark-bench" checkout -q "$SB_REV"
    [[ -x $CACHE/spark-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/spark-bench/.venv"
    # the golden gate renders pages (VIS scenarios): playwright + pillow and a chromium
    "$CACHE/spark-bench/.venv/bin/pip" install -q playwright pillow
    "$CACHE/tool-eval-bench/.venv/bin/tool-eval-bench" --help >/dev/null
    (cd "$CACHE/spark-bench" && .venv/bin/python spark_bench.py eval --help >/dev/null)
    echo "installed: tool-eval-bench $(git -C "$CACHE/tool-eval-bench" rev-parse --short HEAD)," \
         "spark-bench $(git -C "$CACHE/spark-bench" rev-parse --short HEAD) in $CACHE"
}

# thinking mode -> request fields. Both benches merge extra body fields into every request
kwargs_for() {
    case $1 in
        off)  echo '{"enable_thinking": false}' ;;
        adaptive) echo '{"enable_thinking": true}' ;;
        low)  echo '{"enable_thinking": true, "reasoning_effort": "low"}' ;;
        high) echo '{"enable_thinking": true, "reasoning_effort": "high"}' ;;
        *) die "mode must be off, adaptive, low or high (got '$1')" ;;
    esac
}

body_for() {       # $1 = mode: the extra request fields of a thinking mode (no top-level effort when adaptive)
    if [[ $1 == adaptive ]]; then echo "{\"chat_template_kwargs\": $(kwargs_for "$1")}"
    else echo "{\"chat_template_kwargs\": $(kwargs_for "$1"), \"reasoning_effort\": \"$1\"}"; fi
}

snapshot() {       # $1 = dir, $2 = tag: what the server was
    local server=${BASE_URL%/v1}
    api -fsS -m 10 "$server/health" -o "$1/health-$2.json" 2>/dev/null || echo '{"unreachable": true}' > "$1/health-$2.json"
    api -fsS -m 10 "$BASE_URL/models" -o "$1/models-$2.json" 2>/dev/null || true
}

preflight() {
    local models
    models=$(api -fsS -m 10 "$BASE_URL/models") || die "no server at $BASE_URL (or the key was refused)"
    grep -q "\"$MODEL\"" <<< "$models" || die "$BASE_URL/models does not list $MODEL"
}

run_teb() {        # $1 = mode, $2 = dir, $3.. = extra args (scenario / category filters)
    local mode=$1 dir=$2; shift 2
    local bin=$CACHE/tool-eval-bench/.venv/bin/tool-eval-bench
    [[ -x $bin ]] || die "tool-eval-bench not installed: bench/tooleval/run.sh setup"
    local args=(--base-url "$BASE_URL" --model "$MODEL" --no-live --temperature "${TEB_TEMP:-0.0}"
                --output-dir "$dir/teb" --json-file "$dir/teb.json")
    if [[ $mode == off ]]; then
        args+=(--no-think --timeout "${TEB_TIMEOUT:-300}")
    else
        args+=(--timeout "${TEB_TIMEOUT:-900}"
               --backend-kwargs "$(body_for "$mode")")
    fi
    # shellcheck disable=SC2206
    args+=("$@" ${TEB_ARGS:-})
    redact argv -- ${KEY:+"TOOL_EVAL_API_KEY=<redacted>"} "$bin" "${args[@]}" > "$dir/teb.cmd"
    TOOL_EVAL_API_KEY="$KEY" "$bin" "${args[@]}" 2>&1 | redact text | tee "$dir/teb.log"
}

run_sb() {         # $1 = mode, $2 = dir, $3.. = extra args
    local mode=$1 dir=$2; shift 2
    mkdir -p "$dir/sb-dump"   # spark-bench opens raw_dump.jsonl in it and silently skips the dump if the dir is missing
    [[ -x $CACHE/spark-bench/.venv/bin/python ]] || die "spark-bench not installed: bench/tooleval/run.sh setup"
    local args=(eval --endpoint "$BASE_URL" --model "$MODEL" --label "MM31-TF-$mode${LABEL:+-$LABEL}"
                --out-dir "$dir/sb" --repeats "${SB_REPEATS:-2}" --temperature "${SB_TEMP:-0.3}"
                --uncapped --timeout 0 --tier all --skip-throughput)
    local extra
    if [[ $mode == off ]]; then
        args+=(--thinking off)
        extra=""
    else
        # the bench's --thinking on sends chat_template_kwargs {enable_thinking, thinking_mode}; SPARK_BENCH_EXTRA_BODY
        # replaces that dict (shallow update), so it carries enable_thinking and the effort itself
        args+=(--thinking on)
        extra=$(body_for "$mode")
    fi
    # shellcheck disable=SC2206
    args+=("$@" ${SB_ARGS:-})
    local shown=()     # the environment the command line shows
    [[ -n $extra ]] && shown+=("SPARK_BENCH_EXTRA_BODY=$extra")
    [[ -n $KEY ]] && shown+=("SPARK_BENCH_API_KEY=<redacted>")
    redact argv -- "${shown[@]}" .venv/bin/python spark_bench.py "${args[@]}" > "$dir/sb.cmd"
    (cd "$CACHE/spark-bench" && SPARK_BENCH_EXTRA_BODY="$extra" SPARK_BENCH_API_KEY="$KEY" \
        SPARK_BENCH_DUMP_DIR="$dir/sb-dump" .venv/bin/python spark_bench.py "${args[@]}") 2>&1 | redact text | tee "$dir/sb.log"
}

run_py() {         # $1 = harness (bench/...), $2 = name, $3 = dir, $4.. = its args; the key file goes by path
    local py=$1 name=$2 dir=$3; shift 3
    local args=(--base "${BASE_URL%/v1}" --model "$MODEL" "$@" --out "$dir/$name.json")
    [[ -n $KEY ]] && args+=(--api-key-file "$API_KEY_FILE")
    redact argv -- python3 -I "$ROOT/$py" "${args[@]}" > "$dir/$name.cmd"
    python3 -I "$ROOT/$py" "${args[@]}" 2>&1 | redact text | tee "$dir/$name.log"
}

cmd=${1:-}; mode=${2:-}; LABEL=${3:-}
case $cmd in
    setup) setup; exit 0 ;;
    teb|sb|both|chains|own|toolcall) kwargs_for "$mode" >/dev/null ;;
    *) sed -n '2,29p' "$0"; exit 2 ;;
esac
preflight
dir=$OUT/$(date +%Y%m%d-%H%M)-$cmd-$mode${LABEL:+-$LABEL}
mkdir -p "$dir"
trap 'redact scrub "$dir"' EXIT    # the benches' own reports and dumps (spark-bench's raw request dump, for one)
snapshot "$dir" before
rc=0
case $cmd in
    teb)      run_teb "$mode" "$dir" ;;
    sb)       run_sb "$mode" "$dir" ;;
    both)     run_teb "$mode" "$dir"; run_sb "$mode" "$dir" ;;
    chains)   run_teb "$mode" "$dir" --categories C
              SB_REPEATS=${SB_REPEATS:-3} run_sb "$mode" "$dir" --domains agentic ;;
    own)      run_py bench/tooleval/chains.py chains "$dir" --mode "$mode" --temperature "${PY_TEMP:-0.0}" \
                     --reps "${PY_REPS:-1}" || rc=$? ;;
    toolcall) run_py bench/toolcall.py toolcall "$dir" --mode "$mode" --temperature "${PY_TEMP:-0.0}" \
                     --reps "${PY_REPS:-2}" || rc=$? ;;
esac
snapshot "$dir" after
echo "results: $dir"
exit "$rc"                         # the stdlib harnesses' verdict (1: below their pass line)
