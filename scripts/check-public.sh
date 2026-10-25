#!/usr/bin/env bash
# Scan this repo for private infrastructure details, credentials and off-limits wording before publishing.
# Adapted from scripts/check-public.sh of the DeepSeek-V4.1 and GLM-5.3 TensorFold recipes.
# Credentials (API keys, tokens, private keys, credential assignments) are scripts/check_secrets.py's patterns, run
# from here over the same files; its value-masked findings fail this check too.
#
#   scripts/check-public.sh           precise patterns; exit 1 on any finding
#   scripts/check-public.sh --broad   also list every secret / password / api-key / bearer / authorization mention
#                                     for a manual read (informational, does not fail)
#   scripts/check-public.sh --history the same patterns over every blob of every commit (git grep), every commit
#                                     message, every line any commit added (credentials), and every commit's author /
#                                     committer address (only ALLOWED_AUTHOR)
#
# Scans every tracked and untracked (not ignored) file, the engine patches included, except this script and the local
# list.
#
# Your own private names (hostnames, user names, tailnet name, home paths, internal repos) belong in
# .check-public.local (git-ignored, one extended regex per line, # comments allowed) or in CHECK_PUBLIC_EXTRA='a|b'.
# They are matched case-insensitively. Lines of the form  regex<TAB>path-glob  skip that path for that regex.
#
# Wording rules enforced here: no history of past hardware events (see the "event history wording" pattern), and
# exactly one permitted GPU-clock sentence (ALLOWED_CLOCK_LINE); any other line that mentions clocks or MHz fails.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
SELF="scripts/check-public.sh"
LOCAL_LIST=".check-public.local"
SECRETS="scripts/check_secrets.py"
ALLOWED_AUTHOR="${ALLOWED_AUTHOR:-58495321+jayleaton@users.noreply.github.com}"
ALLOWED_CLOCK_LINE='We suggest a 2100-2200 MHz GPU clock cap to help with temperature control.'

# name|regex (grep -E, case-insensitive)|path globs excluded for this pattern (space-separated, may be empty)
PATTERNS=(
    "private IPv4 192.168/16|\\b192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|"
    "private IPv4 10/8|\\b10\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|"
    "private IPv4 172.16/12|\\b172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|"
    "CGNAT / Tailscale IPv4|\\b100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|"
    'Tailscale IPv6|fd7a:|'
    'Tailscale names|\.ts\.net\b|tailnet|tailscale|'
    'gmail address|[a-z0-9._%+-]+@gmail\.com|'
    'MAC address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b|'
    'serial number|serial[ _-]?(number|no\.?)[ :=]|'
    'SSH public key|ssh-(ed25519|rsa|dss) AAAA|'
    'home directories|/home/[a-z0-9_-]+/|'
    'root home paths|(^|[^a-z0-9_.-])/root/|'
    'machine hostnames (dgx-01 style)|\bdgx-?[0-9]{1,2}\b|'
    'per-user work paths|\.cache/tf[0-9]+-port|~/ai/|\.prod-stack|test-window-lease|\.t3/|'
    'agent tooling / chatter|\bt3[-_ ]?(code|gateway|agent)|agent-storage|claude[ -]code|\bcaptain\b|\bcoordinator\b|\bsub-?agents?\b|thread:mcp|\b(gpu|perf|dense|exchange|research|window|coding) agent\b|'
    'AI attribution|co-authored-by|generated with \[?claude|noreply@anthropic|'
    'event history wording|overheat|over-heat|heat[- ]?(event|incident|damage|death)|thermal[- ](incident|event|shutdown|trip|runaway|throttl)|\bincidents?\b|post-?mortem|node (loss|losses|lost|died|went down)|lost (a|the|one) node|(crash|crashed|crashes|died|shut ?down)[^.]*(heat|thermal|temperature)|(heat|thermal|temperature)[^.]*(crash|crashed|crashes|died|shut ?down)|'
    'GPU / board identifiers|\bGPU-[0-9a-f]{8}-[0-9a-f]{4}|\bMIG-[0-9a-f]{8}|'
)
if [[ -f "$LOCAL_LIST" ]]; then
    while IFS= read -r line; do
        [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
        re="${line%%$'\t'*}"; skip=""
        [[ "$line" == *$'\t'* ]] && skip="${line#*$'\t'}"
        PATTERNS+=("local list ($LOCAL_LIST)|$re|$skip")
    done < "$LOCAL_LIST"
fi
[[ -n "${CHECK_PUBLIC_EXTRA:-}" ]] && PATTERNS+=("CHECK_PUBLIC_EXTRA|$CHECK_PUBLIC_EXTRA|")

# Known-good lines a pattern may pass: pattern name -> extended regex (case-insensitive) over "path:line:text".
# patches/ is imported byte-identical from MiaAI-Lab's recipe and is never edited here, so its public attribution
# passes line by line rather than by skipping the directory.
declare -A LINE_OK=(
    # /home/<name>/ for these generic placeholders
    ["home directories"]='/home/(user|you|<you>|<user>|ubuntu)/'
    # the build container's own root cache, removed in the image's last layer
    ["root home paths"]='^zig/Dockerfile:[0-9]+:.* /root/\.cache/zig( |$)'
    # the patch author line of MiaAI-Lab's public commit (git format-patch header)
    ["gmail address"]='^patches/tasks/[^:]+\.patch:[0-9]+:From: MiaAI-Lab <mia\.ailab1@gmail\.com>$'
    # the trailers / author line of MiaAI-Lab's public commits that the task patches were exported from
    ["AI attribution"]='^patches/tasks/[^:]+\.patch:[0-9]+:(Co-Authored-By: Claude [A-Za-z]+ [0-9.]+|From: Claude) <noreply@anthropic\.com>$'
    # MiaAI-Lab's own description of how its patches were written, kept as its credit
    ["agent tooling / chatter"]='^(docs/CREDITS-MIAAI-LAB\.md|docs/speed-borrow-glm\.md):[0-9]+:.*(Mia.s AI Lab|developed) with \[?Claude Code\b'
)
# Clock mentions that are not GPU clocks: the engine's Zig clock types in the patches, and a wall-clock time.
# docs/speed-borrow-deepseek.md (MiaAI-Lab's speed survey) discusses a same-clock benchmark rule; it passes as a whole
# until its wording is settled.
# device/ holds the operator's device artifacts copied verbatim (tests/test_repo.py pins their sha256): they pass the clock
# wording and the private names list, and every other pattern still applies to them. A line naming the operator's
# GPU unit (dgx-gpu-clock-cap) or the device file's SPARK_CLOCK_* variables passes the clock rule too.
CLOCK_OK='^device/[^:]+:[0-9]+:|dgx-gpu-clock-cap|SPARK_CLOCK_[A-Z_*]+|^docs/OPERATIONS\.md:[0-9]+:.*300-2100 MHz by `dgx-gpu-clock-cap\.service`|^patches/[^:]+\.patch:[0-9]+:\+.*(FixedClock|clock\.clock\(\)|Io\.Clock)|^docs/speed-borrow-glm\.md:[0-9]+:.*wall-clock|^docs/speed-borrow-deepseek\.md:[0-9]+:'
# e-mail addresses of the public patch authors (format-patch headers) and the one credited contributor's address
EMAIL_OK='^patches/tasks/[^:]+\.patch:[0-9]+:(miaai-lab@users\.noreply\.github\.com|mia\.ailab1@gmail\.com|noreply@anthropic\.com)$'
EMAIL_OK+='|^docs/speed-borrow-deepseek\.md:[0-9]+:coinupbtc@users\.noreply\.github\.com$'

line_ok() { # name; stdin: path:line:text hits; drops that pattern's known-good lines
    if [[ -n "${LINE_OK[$1]:-}" ]]; then grep -viE -- "${LINE_OK[$1]}"; else cat; fi
}

mapfile -t FILES < <(git ls-files -co --exclude-standard 2>/dev/null |
                     grep -vxF "$SELF" | grep -vxF "$LOCAL_LIST" |
                     while IFS= read -r f; do [[ -f "$f" && ! -L "$f" ]] && printf '%s\n' "$f"; done)
if [[ ${#FILES[@]} -eq 0 ]]; then echo "no files to scan (not a git checkout?)" >&2; exit 2; fi

# a clock / MHz mention is allowed only as the exact permitted sentence on its own line (list or quote marker ok)
CLOCK_RE='clock|[0-9] ?mhz\b|\bmhz\b|-lgc\b|lock-gpu-clocks'
# In the engine series (engine/patches, the engine's source) "clock" is mostly code (clock_gettime, a clock cache); a line
# there fails only when it is about GPU clocks.
ENGINE_GPU_CLOCK_RE='[0-9] ?mhz|mhz|-lgc|lock-gpu-clocks|(gpu|sm|graphics) clock'
clock_ok() { # stdin: grep -Hn lines; drops the permitted sentence and CLOCK_OK
    local allowed=${ALLOWED_CLOCK_LINE//./\\.}
    grep -vE "^[^:]+:[0-9]+:[[:space:]]*([-*>#]+[[:space:]]*)?${allowed}[[:space:]]*$" | grep -viE -- "$CLOCK_OK" |
        awk -v gpu="$ENGINE_GPU_CLOCK_RE" 'tolower($0) !~ /^engine\/patches\/[^:]+:[0-9]+:/ || tolower($0) ~ gpu'
}

fail=0
for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"; rest="${entry#*|}"
    skip="${rest##*|}"; re="${rest%|*}"
    hits=$(for f in "${FILES[@]}"; do
               skipped=0
               read -ra globs <<< "$skip"
               # shellcheck disable=SC2053
               for g in "${globs[@]}"; do [[ "$f" == $g ]] && { skipped=1; break; }; done
               [[ "$name" == "local list"* && "$f" == device/* ]] && skipped=1
               [[ $skipped -eq 1 ]] && continue
               grep -HnIiE -- "$re" "$f" 2>/dev/null
           done | line_ok "$name")
    if [[ -n "$hits" ]]; then
        fail=1
        echo "== $name"
        cut -c1-220 <<< "$hits"
    fi
done

clocks=$(for f in "${FILES[@]}"; do grep -HnIiE -- "$CLOCK_RE" "$f" 2>/dev/null; done | clock_ok)
if [[ -n "$clocks" ]]; then
    fail=1; echo "== clock wording (only this exact line is allowed: $ALLOWED_CLOCK_LINE)"; cut -c1-220 <<< "$clocks"
fi

# credentials: one pattern set, shared with scripts/redact.py; allowlist in scripts/check-secrets-allow.txt
if [[ ! -f "$SECRETS" ]]; then
    fail=1; echo "== credentials: $SECRETS is missing (it holds the credential patterns)"
elif ! secrets=$(python3 -I "$SECRETS" "${FILES[@]}" 2>&1); then
    fail=1; echo "== credentials ($SECRETS)"; echo "$secrets"
fi

# e-mail addresses other than placeholder / example domains
emails=$(for f in "${FILES[@]}"; do grep -HnoIiE '[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}' "$f" 2>/dev/null; done |
         grep -viE '@(([a-z0-9-]+\.)*example(\.(com|org|net))?)$' | grep -vF ":$ALLOWED_AUTHOR" | grep -viE -- "$EMAIL_OK" |
         cut -c1-200)
if [[ -n "$emails" ]]; then fail=1; echo "== e-mail addresses (not on the example-domain allowlist)"; echo "$emails"; fi

# files that should never be published
bad=$(printf '%s\n' "${FILES[@]}" | grep -E '\.(log|out|pem|key)$|(^|/)\.env$|^config/[^/]+\.env$|id_(rsa|ed25519)' || true)
if [[ -n "$bad" ]]; then fail=1; echo "== files that should not be published"; echo "$bad"; fi

if [[ "${1:-}" == --history ]]; then
    commits=$(git rev-list --all 2>/dev/null)
    echo "== history: $(wc -w <<< "$commits") commit(s)"
    bad_auth=$(git log --all --format='%h %ae %ce' | awk -v ok="$ALLOWED_AUTHOR" '$2 != ok || $3 != ok')
    if [[ -n "$bad_auth" ]]; then fail=1; echo "== commits not authored / committed as $ALLOWED_AUTHOR"; echo "$bad_auth"; fi
    if [[ -f "$SECRETS" ]] && ! hsecrets=$(python3 -I "$SECRETS" --history 2>&1); then
        fail=1; echo "== history: credentials ($SECRETS)"; echo "$hsecrets"
    fi
    msgs=$(git log --all --format='%h %B')
    for entry in "${PATTERNS[@]}"; do
        name="${entry%%|*}"; rest="${entry#*|}"; skip="${rest##*|}"; re="${rest%|*}"
        pathspec=(-- . ":!$SELF")
        for g in $skip; do pathspec+=(":!$g"); done
        [[ "$name" != "local list"* ]] || pathspec+=(":!device/")   # the operator's verbatim files, as in the tree scan
        hits=""
        for c in $commits; do
            h=$(git grep -I -n -i -E -e "$re" "$c" "${pathspec[@]}" 2>/dev/null | sed "s/^$c://" | line_ok "$name")
            [[ -n "$h" ]] && hits+="$c:${h//$'\n'/$'\n'$c:}"$'\n'
        done
        m=$(grep -niE -e "$re" <<< "$msgs" || true)
        [[ -n "$m" ]] && hits+="commit message: $m"$'\n'
        if [[ -n "${hits//[[:space:]]/}" ]]; then fail=1; echo "== history: $name"; cut -c1-220 <<< "$hits" | sed '/^$/d'; fi
    done
    hclocks=""
    for c in $commits; do
        h=$(git grep -I -n -i -E -e "$CLOCK_RE" "$c" -- . ":!$SELF" 2>/dev/null | sed "s/^$c://" | clock_ok)
        [[ -n "$h" ]] && hclocks+="$c: $h"$'\n'
    done
    m=$(grep -niE -e "$CLOCK_RE" <<< "$msgs" || true)
    [[ -n "$m" ]] && hclocks+="commit message: $m"$'\n'
    if [[ -n "${hclocks//[[:space:]]/}" ]]; then fail=1; echo "== history: clock wording"; cut -c1-220 <<< "$hclocks"; fi
fi

if [[ "${1:-}" == --broad ]]; then
    echo "== broad keyword list (manual review; informational)"
    for f in "${FILES[@]}"; do
        grep -HnIiE 'secret|password|api[_-]?key|bearer|authorization' "$f" 2>/dev/null
    done | cut -c1-200
fi

if [[ $fail -ne 0 ]]; then
    echo; echo "check-public: FINDINGS above (${#FILES[@]} files scanned)"; exit 1
fi
echo "check-public: clean (${#FILES[@]} files scanned$([[ -f "$LOCAL_LIST" ]] && echo ", plus $LOCAL_LIST"))"
