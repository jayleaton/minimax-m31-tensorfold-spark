#!/usr/bin/env bash
# Push the image scripts/prepare.sh built to GitHub Container Registry as $GHCR_IMAGE:<TAG_BASE>-<image hash> and :latest,
# labelled with this repository (so the package takes the repository's visibility and access).
# The image hash is config.sh's image_hash (patches/, zig/Dockerfile and the pins), the same value prepare.sh writes
# into the image's tf.patches label and looks for when it pulls: prepare.sh pulls $GHCR_IMAGE:<TAG_BASE>-<hash>.
# From the GLM-5.3-Flash recipe's scripts/publish-image.sh (Mia's AI Lab). It pushes to a registry: run it only on purpose.
# No registry name or repository is chosen for this recipe yet, and none is guessed: set GHCR_IMAGE (e.g. ghcr.io/<owner>/<name>)
# and REPO_URL (the repository the package belongs to). This recipe is private, and an image of it built WITHOUT the
# MiniMax family (patches/ empty) cannot serve the checkpoint: publish only an image that passed ./start.sh's family check.
# Run it on the head Spark (the one that runs ./start.sh); the worker gets the image from the head or the registry.
# Needs a GitHub token with write:packages: `gh auth login -s write:packages`, or GHCR_TOKEN / GITHUB_TOKEN set.
# Usage: scripts/publish-image.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

[[ -n "$GHCR_IMAGE" ]] || die "GHCR_IMAGE is not set (no registry name is chosen for this recipe): GHCR_IMAGE=ghcr.io/<owner>/<name> scripts/publish-image.sh"
[[ -n "${REPO_URL:-}" ]] || die "REPO_URL is not set: REPO_URL=https://github.com/<owner>/<repo> scripts/publish-image.sh"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
family_gate >/dev/null    # refuses (ALLOW_UNSUPPORTED=1 overrides) an image whose build lacks the family
hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ "$hash" == "$(image_hash)" ]] ||
  die "$IMAGE was built from other patches or pins ($hash, not $(image_hash)); run scripts/prepare.sh first"
tag="${TAG_BASE}-${hash}"
latest=latest

token="${GHCR_TOKEN:-${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
[[ -n "$token" ]] || die "no GitHub token: run gh auth login -s write:packages, or set GHCR_TOKEN"
user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null || echo "${GHCR_IMAGE#ghcr.io/}" | cut -d/ -f1)}"

log "Labelling $IMAGE as $GHCR_IMAGE:$tag"
docker build -q -t "$GHCR_IMAGE:$tag" -t "$GHCR_IMAGE:$latest" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.licenses=Apache-2.0 \
  --label org.opencontainers.image.description="MiniMax-M3.1-Flash EXL3 on two DGX Sparks: TensorFold $TF_TAG native engine plus patches/, pins $hash" \
  - <<<"FROM $IMAGE" >/dev/null
# the labels above sit on top of the built image: its tf.patches label must come through unchanged
[[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$GHCR_IMAGE:$tag")" == "$hash" ]] ||
  die "$GHCR_IMAGE:$tag lost its tf.patches label"
log "Pushing $GHCR_IMAGE:$tag and :$latest (a few GB; only changed layers upload)"
# log in with a throwaway Docker config, so no registry credential stays in ~/.docker after the push
login_dir=$(mktemp -d)
trap 'rm -rf -- "$login_dir"' EXIT
echo "$token" | DOCKER_CONFIG="$login_dir" docker login ghcr.io -u "$user" --password-stdin >/dev/null
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$tag"
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$latest"
digest=$(docker image inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$GHCR_IMAGE:$tag" | grep -m1 "^$GHCR_IMAGE@" |
         cut -d@ -f2)
log "Done: docker pull $GHCR_IMAGE:$tag"
log "Pin it in scripts/config.sh: IMAGE_TAG=$tag IMAGE_DIGEST=$digest"
