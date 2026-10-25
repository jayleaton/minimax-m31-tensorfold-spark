#!/usr/bin/env bash
# Every CPU-only check of this repo, the set to run before a commit: bash -n and shellcheck on every script, pytest over
# tests/ (the recipe, the guard, the credential tools and the bench harnesses), and the public check. No GPU, Docker, SSH
# or network (127.0.0.1 inside the bench tests). Run it on a Spark or another Linux box, not on a laptop with a live
# server's state.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
mapfile -t scripts < <(ls start.sh stop.sh scripts/*.sh tests/*.sh; find bench -name '*.sh' 2>/dev/null)
for script in "${scripts[@]}"; do bash -n "$script"; done
echo "bash -n: ok (${#scripts[@]} scripts)"
if command -v shellcheck > /dev/null; then shellcheck -x -S error "${scripts[@]}" && echo "shellcheck: ok (errors; MiaAI-Lab's imported scripts carry style and warning notes, kept as imported)"
else echo "shellcheck: not installed, skipped"; fi
python3 -m pytest -q -p no:cacheprovider tests "$@"
bash scripts/check-public.sh
