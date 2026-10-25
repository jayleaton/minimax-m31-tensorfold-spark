#!/usr/bin/env bash
# Prepare two Sparks to serve MODEL_ID (jayleaton's EXL3 3.55 bpw pack of MiniMax-M3.1-Flash by default) with TensorFold v1.1.2's
# native engine plus the minimax_m3_vl family:
#   1. preflight: docker and the GPU on both Sparks, key-based ssh, the link between them
#   2. the engine image on this Spark (zig/Dockerfile: TensorFold v1.1.2 at TF_REF plus ENGINE_PATCHES/*.patch, on BASE_IMAGE),
#      then on the worker (docker save | docker load: only what it lacks is stored)
#   3. THE FAMILY GATE: the image's `tensorfold-native capabilities --json` must list minimax_m3_vl, BEFORE the download
#      (ALLOW_UNSUPPORTED=1 goes on). --image-only (scripts/serve.sh prebuild) stops here: no weights are read
#   4. the checkpoint into HF_CACHE (resumable), at its pinned revision: every shard the index names, the index's total_size,
#      the license file (MODEL_LICENSE_FILE), and the pack's SHA256SUMS once per revision (SHA=1 checks it again). The
#      repository is PRIVATE: it needs your Hugging Face token (HF_TOKEN, or `hf auth login`). The token is read by `hf` on
#      this Spark only: it is not printed, written anywhere by these scripts, or passed to the worker or a container
#   5. rank 1's copy. WEIGHTS=local (default): the snapshot copied into the worker's HF cache over ssh (rsync, resumable) and
#      checked file by file, by size. WEIGHTS=nfs: a read-only NFS volume of this Spark's HF_CACHE on the worker, checked the same
#   WEIGHTS=dir: the pack is already a plain folder on each Spark (MODEL_DIR, WORKER_MODEL_DIR): steps 4 and 5 only check it
#      (the index's shards and total_size, the license and SHA256SUMS here, the worker's folder by size); nothing is downloaded
#      or copied and no token is needed
#   6. with DRAFTS=dflash2 (or DRAFTS=1 DRAFTER=dflash2): the DFlash2 draft, incoai/MiniMax-M3.1-Flash-DFlash2 at its pinned
#      revision (Apache-2.0, public: fetched with no token, even when this Spark has one), into DFLASH2_DIR here (hf download
#      --local-dir, resumable), then copied to WORKER_DFLASH2_DIR on the worker (rsync over ssh, resumable); every file is checked
#      against its sha256 pin (config.sh DFLASH2_PINS) on both Sparks, once per folder and revision (SHA=1 checks again)
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch (use it after changing ENGINE_PATCHES); PULL=0 never pulls a published image.
# --dflash2 runs step 6 only (no image, no pack): the draft on both Sparks, whatever DRAFTS says.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0; IMAGE_ONLY=0; DFLASH2_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --image-only) IMAGE_ONLY=1 ;;
    --dflash2) DFLASH2_ONLY=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# fetch_dflash2 (step 6): the DFlash2 draft in DFLASH2_DIR here and WORKER_DFLASH2_DIR on the worker, every file matching its pin.
# The repository is public: hf runs without HF_TOKEN and with HF_HUB_DISABLE_IMPLICIT_TOKEN=1, so no token of this Spark is sent.
fetch_dflash2() {
  local d wd ok cls avail
  d=$(dflash2_dir_here)
  ok="$STATE_DIR/dflash2-ok-${DFLASH2_REVISION:0:12}-$(sha256sum <<<"$d" | cut -c1-12)"
  if [[ "${SHA:-0}" != 1 && -f "$ok" ]] && bash -c "$(dflash2_verify_cmd "$d" quick)"; then
    log "DFlash2 draft here: $d ($DFLASH2_REPO @ ${DFLASH2_REVISION:0:8}, checked)"
  else
    if ! bash -c "$(dflash2_verify_cmd "$d")" >/dev/null 2>&1; then
      command -v hf >/dev/null || die "the Hugging Face CLI ('hf') is not installed on this Spark: pip install -U huggingface_hub"
      mkdir -p "$d" || die "cannot create DFLASH2_DIR $d"
      avail=$(df -B1 --output=avail "$d" | tail -1)
      (( avail > 5 * 1024 ** 3 )) || die "only $(( avail / 1024 ** 3 )) GiB free for $d: the DFlash2 draft is 4.2 GB (5 GiB free are asked for)"
      log "Fetching $DFLASH2_REPO @ ${DFLASH2_REVISION:0:8} into $d (4.2 GB the first time; resumes). Public repository: no token is used"
      env -u HF_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=0 \
        hf download "$DFLASH2_REPO" config.json README.md model.safetensors --revision "$DFLASH2_REVISION" --local-dir "$d" >/dev/null ||
        die "$DFLASH2_REPO: the download failed (it resumes: run scripts/prepare.sh --dflash2 again)"
    fi
    bash -c "$(dflash2_verify_cmd "$d")" ||
      die "$DFLASH2_REPO @ ${DFLASH2_REVISION:0:8} in $d: a file does not match its sha256 pin (above): delete it and run scripts/prepare.sh --dflash2 again"
    touch "$ok"; log "DFlash2 draft here: $d, every file matches its sha256 pin"
  fi
  cls=$(dflash2_class head) || die "$cls"
  wd=$(worker_dflash2_dir) || die "cannot resolve WORKER_DFLASH2_DIR on $(worker_host 1)"
  ok="$STATE_DIR/dflash2-ok-worker-${DFLASH2_REVISION:0:12}-$(sha256sum <<<"$(worker_host 1):$wd" | cut -c1-12)"
  if [[ "${SHA:-0}" != 1 && -f "$ok" ]] && worker 1 "$(dflash2_verify_cmd "$wd" quick)"; then
    log "DFlash2 draft on $(worker_host 1): $wd (checked)"
    return 0
  fi
  if ! worker 1 "$(dflash2_verify_cmd "$wd")" >/dev/null 2>&1; then
    command -v rsync >/dev/null && worker 1 'command -v rsync' >/dev/null 2>&1 || die "rsync is needed on both Sparks to copy the DFlash2 draft (apt install rsync)"
    worker 1 "mkdir -p $(printf '%q' "$wd")" || die "cannot create $wd on $(worker_host 1)"
    log "Copying the DFlash2 draft to $(worker_host 1):$wd (4.2 GB the first time; resumes)"
    rsync -a --partial --info=progress2 -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
      "$d/config.json" "$d/README.md" "$d/model.safetensors" "$(worker_host 1):$wd/" ||
      die "copying the DFlash2 draft to $(worker_host 1) failed (it resumes: run scripts/prepare.sh --dflash2 again)"
  fi
  worker 1 "$(dflash2_verify_cmd "$wd")" ||
    die "the DFlash2 draft in $wd on $(worker_host 1) does not match its sha256 pins (above): delete the file there and run scripts/prepare.sh --dflash2 again"
  touch "$ok"; log "DFlash2 draft on $(worker_host 1): $wd, every file matches its sha256 pin"
}

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$STATE_DIR"; [[ "$WEIGHTS" == dir ]] || mkdir -p "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it: pgrep -af prepare.sh"
if (( DFLASH2_ONLY )); then
  check_workers
  need_worker 1
  fetch_dflash2
  log "Done (--dflash2): the DFlash2 draft is on both Sparks; start it with DRAFTS=dflash2 (docs/DFLASH2.md)"
  exit 0
fi
log "Preflight checks on both Sparks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed on this node: is the NVIDIA driver working?"
check_workers
need_worker 1
worker 1 'docker info >/dev/null 2>&1' || die "$(worker_host 1) cannot talk to its docker daemon (docker group?)"
worker 1 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on $(worker_host 1)"
detect_links
check_link_pins
log "Link: this node $HEAD_ADDR ($HEAD_DEV, RoCE $HEAD_HCAS) <-> $(worker_host 1) $WORKER_ADDR ($WORKER_DEV, RoCE $WORKER_HCAS); weights: $WEIGHTS$([[ "$WEIGHTS" == nfs ]] && echo " from $(nfs_server 1)")"

# ---------------------------------------------------------------- 2. the image
hash=$(image_hash)
built=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && -n "$GHCR_IMAGE" && "$built" != "$hash" ]]; then
  prebuilt=$(prebuilt_image)            # the pinned digest (config.sh's IMAGE_TAG / IMAGE_DIGEST), else the hash's tag
  log "Pulling the prebuilt image $prebuilt if there is one (PULL=0 builds instead)"
  if docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$hash" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built=$hash
    log "Using $prebuilt as $IMAGE"
  else
    log "No prebuilt image for these pins ($prebuilt): building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built" != "$hash" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "Pulling $BASE_IMAGE"; docker pull "$BASE_IMAGE" >/dev/null; }
  pdir=$(engine_patches_dir)
  [[ -d "$pdir" ]] || die "ENGINE_PATCHES=$ENGINE_PATCHES is not a folder"
  log "Building $IMAGE (TensorFold $TF_TAG ${TF_REF:0:12}, $(compgen -G "$pdir/*.patch" | wc -l) patches from $ENGINE_PATCHES${ENGINE_COMMIT:+ @ ${ENGINE_COMMIT:0:12}}, Zig $ZIG_VERSION, sm_$SM)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches"; cp "$pdir"/*.patch "$ctx/patches/" 2>/dev/null || true; cp zig/Dockerfile "$ctx/Dockerfile"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" -f "$ctx/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" --build-arg TF_TAG="$TF_TAG" --build-arg SM="$SM" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" --build-arg PATCHES_HASH="$hash" "$ctx"
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built (hash $hash; --rebuild forces it)"
fi

# ---------------------------------------------------------------- 3. the family gate
# Before the worker gets the image and before 157.1 GiB are downloaded: a build without the family cannot serve this checkpoint.
family_gate || die "image $IMAGE missing: the build above did not produce it"

ident=$(image_ident "$IMAGE")
if [[ "$(worker_image_ident 1 "$IMAGE")" != "$ident" ]]; then
  if [[ "${PULL:-1}" == 1 && -n "$GHCR_IMAGE" ]] && docker image inspect "$(prebuilt_image)" >/dev/null 2>&1 && worker 1 docker pull "$(prebuilt_image)" >/dev/null 2>&1 &&
     [[ "$(worker_image_ident 1 "$(prebuilt_image)")" == "$ident" ]]; then
    worker 1 docker tag "$(prebuilt_image)" "$IMAGE"
    log "Using $(prebuilt_image) as $IMAGE on $(worker_host 1)"
  else
    log "Copying $IMAGE to $(worker_host 1) (docker save | docker load; only missing layers are stored)"
    docker save "$IMAGE" | worker 1 docker load >/dev/null
  fi
  [[ "$(worker_image_ident 1 "$IMAGE")" == "$ident" ]] || die "$(worker_host 1)'s $IMAGE differs from this Spark's"
fi
log "Image $IMAGE identical on both Sparks"
if (( IMAGE_ONLY )); then log "Done (--image-only): the image is built, on both Sparks, and lists $REQUIRED_FAMILY; no weights were read"; exit 0; fi

# ---------------------------------------------------------------- 4. the checkpoint
# check_snapshot <dir>: the snapshot is complete: the index, every shard it names (MODEL_SHARDS of them), none short of the
# index's total_size in all. Prints the figures; fails with the reason. python3 -I: no code is loaded from the snapshot's folder.
check_snapshot() {
  python3 -I -c '
import json, os, sys
d, want_shards, want_total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.isfile(idx):
    sys.exit("no model.safetensors.index.json")
meta = json.load(open(idx))
shards = sorted(set(meta["weight_map"].values()))
total = int(meta.get("metadata", {}).get("total_size", 0))
size = 0
for s in shards:
    p = os.path.join(d, s)
    if not os.path.isfile(p):
        sys.exit("missing shard %s" % s)
    size += os.stat(p).st_size
if want_shards and len(shards) != want_shards:
    sys.exit("%d shards, expected %d" % (len(shards), want_shards))
if want_total and total != want_total:
    sys.exit("the index says total_size %d, expected %d" % (total, want_total))
if size < total:
    sys.exit("the shards hold %d bytes, fewer than the index total_size %d: a download is incomplete" % (size, total))
print("%d shards, %d bytes (index total_size %d)" % (len(shards), size, total))
' "$1" "$2" "$3"
}
want_shards=$MODEL_SHARDS; want_total=$MODEL_TOTAL_BYTES
SNAP=$(snapshot_dir)
if [[ "$WEIGHTS" == dir ]]; then
  fig=$(check_snapshot "$SNAP" "$want_shards" "$want_total" 2>&1) ||
    die "WEIGHTS=dir: the pack in $SNAP (MODEL_DIR) is not complete: $fig. Nothing is downloaded with WEIGHTS=dir: put the whole pack there"
  log "Checkpoint in $SNAP (WEIGHTS=dir): $fig"
elif [[ -n "$(snapshot_rev)" ]] && fig=$(check_snapshot "$SNAP" "$want_shards" "$want_total" 2>&1); then
  log "Checkpoint already complete: $fig"
else
  command -v hf >/dev/null || die "the Hugging Face CLI ('hf') is not installed on this Spark: pip install -U huggingface_hub"
  need_hf_token
  avail=$(df -B1 --output=avail "$HF_CACHE" | tail -1)
  # the cache is keyed by the repo's name and the revision: a copy under another name or revision does not count, so refuse here
  # when the disk cannot hold the whole thing instead of filling it half way
  need_gib=$(awk -v g="$MODEL_GIB" 'BEGIN {printf "%d", (g == "?" ? 200 : g) * 1.1 + 1}')
  (( avail > need_gib * 1024 ** 3 )) || die "$MODEL_ID @ $(snapshot_rev | cut -c1-8) is not in $HF_CACHE/hub yet and only $(( avail / 1024 ** 3 )) GiB are free there; it is $MODEL_GIB GiB and $need_gib GiB free are asked for. If you already have this checkpoint under another revision, set MODEL_REVISION; otherwise free the disk"
  log "Downloading $MODEL_ID${MODEL_REVISION:+ @ ${MODEL_REVISION:0:8}} into $HF_CACHE/hub ($MODEL_GIB GiB the first time; resumes). Private repository: using your Hugging Face login"
  if ! hf download "$MODEL_ID" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --cache-dir "$HF_CACHE/hub" >/dev/null; then
    die "$MODEL_ID: the download failed (is the token a read token that can see this private repository? hf auth whoami)"
  fi
  SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
  fig=$(check_snapshot "$SNAP" "$want_shards" "$want_total") || die "$MODEL_ID: the snapshot at $SNAP is not complete: $fig"
  log "Checkpoint: $SNAP ($fig)"
fi

# The license the snapshot ships: never serve or share weights without it (read it: the MiniMax Community License for this pack)
[[ -s "$SNAP/$MODEL_LICENSE_FILE" ]] || die "$MODEL_ID @ $(snapshot_rev | cut -c1-8) ships no $MODEL_LICENSE_FILE: do not serve or share it"
# SHA256SUMS, when the pack ships one (jayleaton's packs do: every shard), once per revision; SHA=1 checks it again
sums_ok="$STATE_DIR/sha256-ok-${MODEL_ID//\//--}-$(snapshot_rev)"
[[ "$WEIGHTS" != dir ]] || sums_ok+="-dir-$(sha256sum <<<"$SNAP" | cut -c1-12)"   # a folder of its own: checked once itself
if [[ -f "$SNAP/SHA256SUMS" ]] && [[ "${SHA:-0}" == 1 || ! -f "$sums_ok" ]]; then
  log "Checking $SNAP/SHA256SUMS ($(wc -l < "$SNAP/SHA256SUMS") files; a few minutes)"
  (cd "$SNAP" && sha256sum --quiet -c SHA256SUMS) || die "$MODEL_ID: a file does not match SHA256SUMS (above): delete it from the cache and run scripts/prepare.sh again"
  touch "$sums_ok"; log "SHA256SUMS: every file matches"
fi

# ---------------------------------------------------------------- 6. the DFlash2 draft (DRAFTS=dflash2), before the copy steps exit
if dflash2_wanted; then fetch_dflash2; fi

# ---------------------------------------------------------------- 5. rank 1's copy of the checkpoint
if [[ "$WEIGHTS" == dir ]]; then
  want=$(model_dir_manifest head) || die "cannot list $SNAP"
  have=$(model_dir_manifest worker 2>/dev/null || true)
  [[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
    die "$(worker_host 1)'s folder $(worker_model_dir) (WORKER_MODEL_DIR) does not hold the same pack as $SNAP (above): WEIGHTS=dir copies nothing"; }
  log "Worker ($(worker_host 1)) holds the same pack in $(worker_model_dir) (every file a rank reads, by size)"
  prepared_state > "$PREPARED_MARKER"
  log "Done: both Sparks are ready. Start the server with scripts/serve.sh start (port $PORT)."
  exit 0
fi
manifest() {  # the files a rank reads, with their sizes, as paths under HF_CACHE
  (cd "$HF_CACHE" && find -L "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" -type f -printf '%p %s\n' | LC_ALL=C sort)
}
want=$(manifest)
if [[ "$WEIGHTS" == local ]]; then
  wcache=$(worker_hf_cache 1)
  rel="hub/models--${MODEL_ID//\//--}"
  have=$(worker 1 "cd $(printf '%q' "$wcache") 2>/dev/null && find -L '$rel/snapshots/$(snapshot_rev)' -type f -printf '%p %s\n' | LC_ALL=C sort" 2>/dev/null || true)
  if [[ "$have" != "$want" ]]; then
    command -v rsync >/dev/null && worker 1 'command -v rsync' >/dev/null 2>&1 || die "rsync is needed on both Sparks for WEIGHTS=local (apt install rsync), or set WEIGHTS=nfs"
    log "Copying $MODEL_ID @ $(snapshot_rev | cut -c1-8) to $(worker_host 1):$wcache ($MODEL_GIB GiB the first time; resumes). Set WORKER to the worker's link address for the fastest copy"
    worker 1 "mkdir -p $(printf '%q' "$wcache/$rel")" || die "cannot create $wcache/$rel on $(worker_host 1)"
    # the repository's cache folder as it is (refs, blobs, the snapshot's symlinks), so the worker's cache works like this one's
    rsync -a --partial --info=progress2 -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "$HF_CACHE/$rel/" "$(worker_host 1):$wcache/$rel/" ||
      die "copying the checkpoint to $(worker_host 1) failed (it resumes: run scripts/prepare.sh again)"
    have=$(worker 1 "cd $(printf '%q' "$wcache") && find -L '$rel/snapshots/$(snapshot_rev)' -type f -printf '%p %s\n' | LC_ALL=C sort" 2>/dev/null || true)
  fi
  [[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
    die "$(worker_host 1)'s copy in $wcache differs from this Spark's (above)"; }
  log "Worker ($(worker_host 1)) holds the same snapshot in $wcache (every file, by size; SHA=1 scripts/prepare.sh checks this Spark's against SHA256SUMS)"
  prepared_state > "$PREPARED_MARKER"
  log "Done: both Sparks are ready. Start the server with scripts/serve.sh start (port $PORT)."
  exit 0
fi
ensure_nfs_volume 1
if ! out=$(worker_nfs 1 true 2>&1); then
  die "$(worker_host 1) cannot mount this Spark's :$NFS_PATH from $(nfs_server 1) over NFS ($NFS_VOLUME):
$(tail -4 <<<"$out" | sed 's/^/    /')
    This Spark must export HF_CACHE (as $NFS_PATH) read-only to that worker's address on its link (README: NFS)"
fi
have=$(worker 1 "docker run --rm --network none -v '$NFS_VOLUME:/hf:ro' -w /hf --entrypoint find '$IMAGE' -L 'hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)' -type f -printf '%p %s\n'" 2>/dev/null | LC_ALL=C sort || true)
[[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
  die "$(worker_host 1) does not see every file over NFS ($NFS_VOLUME)"; }
log "Worker ($(worker_host 1)) reads the checkpoint over NFS"

prepared_state > "$PREPARED_MARKER"
log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
