#!/usr/bin/env bash
# Export the engine this recipe serves as a patch series against upstream TensorFold, for ENGINE_PATCHES=engine/patches.
#
#   scripts/export-engine.sh ENGINE_CHECKOUT [COMMIT]     (COMMIT: default the checkout's HEAD; committed state only)
#
# The engine is TensorFold at TF_REF (v1.1.2) plus MiaAI-Lab's minimax_m3_vl family (patches/0001-minimax-m3-family.patch,
# Mia's AI Lab's commit, unchanged) plus the work on top of it. The series is:
#   engine/patches/0001-minimax-m3-family.patch   a link to patches/0001 (MiaAI-Lab's, byte for byte)
#   engine/patches/0002-minimax-m3-engine.patch   COMMIT's tree against the tree that 0001 gives, as one binary-safe diff
# Files come over as they are in COMMIT, with their licence and attribution headers. engine/export-deny.txt lists path
# globs that must never be exported: a match fails the export. Two lists change what is exported:
#   engine/export-exclude.txt   (in the repo) path globs left out: a file the branch added is dropped, a file it changed
#                               keeps the base version
#   .engine-redact.local        (git-ignored) lines "regex<TAB>replacement" (sed -E) applied to the exported text files,
#                               for private names that must not leave the machine; each hit is counted in SOURCE-EXPORT.md
# It then checks that TF_REF + 0001 + 0002 gives exactly the exported tree, and writes engine/SOURCE-EXPORT.md.
# Run scripts/check-public.sh afterwards (it scans the series). The engine checkout is never modified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
die() { echo "export-engine: $*" >&2; exit 1; }
[[ $# -ge 1 ]] || die "usage: scripts/export-engine.sh ENGINE_CHECKOUT [COMMIT]"
SRC=$(cd "$1" && pwd); REV=${2:-HEAD}
TF_REF=$(sed -nE 's/^TF_REF="\$\{TF_REF:-([0-9a-f]{40})\}"$/\1/p' scripts/config.sh)
[[ $TF_REF =~ ^[0-9a-f]{40}$ ]] || die "could not read TF_REF from scripts/config.sh"
MIA=patches/0001-minimax-m3-family.patch
OUT=engine/patches
EXCLUDE=engine/export-exclude.txt
DENY=engine/export-deny.txt
REDACT=.engine-redact.local

g() { git -C "$SRC" "$@"; }
head=$(g rev-parse --verify "$REV^{commit}") || die "$REV is not a commit in $SRC"
g merge-base --is-ancestor "$TF_REF" "$head" || die "$head does not contain TF_REF $TF_REF"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
export GIT_INDEX_FILE=$tmp/index

# the tree MiaAI-Lab's patch gives on TF_REF
g read-tree "$TF_REF"
g apply --cached "$ROOT/$MIA" || die "$MIA does not apply to TF_REF"
base_tree=$(g write-tree)

# COMMIT's tree, with the exclusions and redactions
g read-tree "$head"
excluded=0
if [[ -f $EXCLUDE ]]; then
    while IFS= read -r glob; do
        [[ -z ${glob//[[:space:]]/} || $glob == \#* ]] && continue
        while IFS= read -r path; do
            [[ -n $path ]] || continue
            if entry=$(g ls-tree "$base_tree" -- "$path") && [[ -n $entry ]]; then
                g update-index --cacheinfo "$(awk '{print $1","$3}' <<< "$entry"),$path"
            else
                g update-index --force-remove -- "$path"
            fi
            excluded=$((excluded + 1)); echo "$path" >> "$tmp/excluded"
        done < <(g diff --name-only "$base_tree" "$head" -- ":(glob)$glob")
    done < "$EXCLUDE"
fi
redacted=0
if [[ -f $REDACT ]]; then
    sedargs=()
    while IFS= read -r line; do
        [[ -z ${line//[[:space:]]/} || $line == \#* ]] && continue
        [[ $line == *$'\t'* ]] || die "$REDACT: a line is regex<TAB>replacement"
        re=${line%%$'\t'*}; rep=${line#*$'\t'}
        sedargs+=(-e "s#${re//#/\\#}#${rep//#/\\#}#g")
    done < "$REDACT"
    while IFS= read -r path; do
        mode_sha=$(g ls-files -s -- "$path" | awk '{print $1" "$2}')
        [[ -n $mode_sha ]] || continue
        g cat-file blob "${mode_sha#* }" > "$tmp/blob"
        grep -qI . "$tmp/blob" || continue                      # binary: left as it is
        sed -E "${sedargs[@]}" "$tmp/blob" > "$tmp/new"
        cmp -s "$tmp/blob" "$tmp/new" && continue
        sha=$(g hash-object -w "$tmp/new")
        g update-index --cacheinfo "${mode_sha% *},$sha,$path"
        redacted=$((redacted + 1))
    done < <(g diff --cached --name-only --diff-filter=AM "$base_tree")
fi
tree=$(g write-tree)

# paths that must never be exported (engine/export-deny.txt): a match fails the export
if [[ -f $DENY ]]; then
    while IFS= read -r glob; do
        [[ -z ${glob//[[:space:]]/} || $glob == \#* ]] && continue
        hits=$(g diff --name-only "$base_tree" "$tree" -- ":(glob,icase)$glob")
        [[ -z $hits ]] || die "$DENY forbids these files in the export (fix them on the engine branch): $(tr '\n' ' ' <<< "$hits")"
    done < "$DENY"
fi

mkdir -p "$OUT"
ln -sfn "../../$MIA" "$OUT/0001-minimax-m3-family.patch"
{
    printf 'From: JJ Eaton <58495321+jayleaton@users.noreply.github.com>\n'
    printf 'Subject: [PATCH 2/2] minimax_m3_vl: the engine this recipe serves, on MiaAI-Lab'"'"'s family\n\n'
    printf 'TensorFold %s plus %s, then this diff: exported from engine commit %s.\n' "${TF_REF:0:12}" "$MIA" "${head:0:12}"
    printf 'Authors and credits: docs/CREDITS.md; how it was made: engine/SOURCE-EXPORT.md.\n---\n'
    g diff --binary --full-index "$base_tree" "$tree"
} > "$OUT/0002-minimax-m3-engine.patch"

# TF_REF + the series must give the exported tree
g read-tree "$TF_REF"
for p in "$OUT"/*.patch; do g apply --cached "$ROOT/$p" || die "$p does not apply in order"; done
[[ $(g write-tree) == "$tree" ]] || die "TF_REF + $OUT/*.patch does not reproduce the exported tree"

stat=$(g diff --shortstat "$base_tree" "$tree")
cat > engine/SOURCE-EXPORT.md <<EOF
# Engine source export

\`ENGINE_PATCHES=engine/patches\`: \`zig/Dockerfile\` applies these to upstream
[TensorFold](https://github.com/ashhart/TensorFold) at \`TF_REF\` (\`$TF_REF\`, the v1.1.2 tag), in filename order.

| Patch | Author | What |
| --- | --- | --- |
| \`0001-minimax-m3-family.patch\` | Mia's AI Lab (MiaAI-Lab) | a link to \`$MIA\`, unchanged: the \`minimax_m3_vl\` family |
| \`0002-minimax-m3-engine.patch\` | JJ Eaton (jayleaton) | the engine this recipe serves, on top of 0001: $stat |

Exported by \`scripts/export-engine.sh\` from engine commit \`$head\` (\`ENGINE_COMMIT\`). Every file comes over as it is in that
commit, with its licence and attribution headers; TensorFold's \`LICENSE\`, \`NOTICE\`, \`THIRD_PARTY_NOTICES.md\` and
\`LICENSES/\` apply to the patched tree ([LICENSING](../docs/LICENSING.md)). Changes made by the export: $excluded file(s) left out
by \`engine/export-exclude.txt\`, $redacted file(s) with private names replaced by placeholders. The script checks that
\`TF_REF\` plus the two patches reproduces the exported tree exactly.
EOF
if [[ -s $tmp/excluded ]]; then
    { echo; echo "Left out:"; echo; sed 's/^/- `/; s/$/`/' "$tmp/excluded"; } >> engine/SOURCE-EXPORT.md
fi
echo "engine/patches: 0001 (MiaAI-Lab) + 0002 from ${head:0:12} ($stat); excluded $excluded, redacted $redacted"
echo "ENGINE_COMMIT=$head"
