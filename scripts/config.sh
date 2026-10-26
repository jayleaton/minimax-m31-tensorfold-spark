# shellcheck shell=bash
# Shared settings for start.sh, stop.sh, scripts/serve.sh and scripts/*.sh. A setting's value comes from the first of these
# that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. scripts/local.sh (this setup's own values; sourced as bash), then ./.env (KEY=value lines, read, never run): both are
#      yours, not the repository's; where both set a key, local.sh wins
#   3. the serving config, $CONFIG (default config/prod.env: cp config/prod.env.example config/prod.env; KEY=value lines,
#      read, never run). config/prod.env.example documents every setting below; it is the production profile
#   4. the defaults below
# The loading of local.sh and .env is the GLM-5.3-Flash recipe's (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold);
# the config file and its placeholder refusal are the GLM-5.3 and DeepSeek-V4.1 recipes' (jayleaton).
# STATUS: the minimax_m3_vl family comes from ENGINE_PATCHES (engine/patches in the profile); the profile is measured in
# results/window-all-3. Defaults below that are not verified pins stay empty (the engine's own default applies); the production values
# are in config/prod.env.example, each with where it comes from (calculator, sibling recipe, or NOT MEASURED).
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# _cfg_read <file>: KEY=value lines into the environment for every key still unset (quotes stripped, # comments dropped)
_cfg_read() {
  local _line _key _value
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$1"
}
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
[[ ! -f "$_cfg_root/.env" ]] || _cfg_read "$_cfg_root/.env"
CONFIG="${CONFIG:-config/prod.env}"
[[ "$CONFIG" == /* ]] && _cfg_file=$CONFIG || _cfg_file="$_cfg_root/$CONFIG"
if [[ -f "$_cfg_file" ]]; then
  # the heat limits are the device's (/etc/spark/thermal.env, section "temperature" below), never the recipe config's
  if grep -qE '^[[:space:]]*(export[[:space:]]+)?GUARD_(PAUSE_C|RESUME_C|KILL_C|KILL_HOLD_S)=' "$_cfg_file"; then
    printf '[%s] ERROR: %s defines GUARD_* limits: they live on the device (/etc/spark/thermal.env); remove them\n' "$(basename "$0")" "$CONFIG" >&2
    exit 2
  fi
  _cfg_read "$_cfg_file"
  # refuse a config that still holds the example's <placeholders> (GLM's rule): every such key must be filled in or removed
  while IFS= read -r _n; do
    [[ "${!_n}" == *"<"*">"* ]] || continue
    printf '[%s] ERROR: %s: set %s (still %s; its config/*.env.example says how)\n' "$(basename "$0")" "$CONFIG" "$_n" "${!_n}" >&2
    exit 2
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$_cfg_file" | LC_ALL=C sort -u)
elif [[ "$CONFIG" != config/prod.env ]]; then
  printf '[%s] ERROR: no config file %s\n' "$(basename "$0")" "$CONFIG" >&2; exit 2
fi
unset _n _cfg_file

# ---------------------------------------------------------------- the two Sparks
# This machine serves rank 0 and the API; WORKER (an ssh target, key-based) is rank 1. The two are cabled directly
# through their ConnectX-7 ports (scripts/nodes.sh finds the link).
TP=2
WORKER="${WORKER:-}"; FABRIC_PEER="${FABRIC_PEER:-}"
MASTER_PORT="${MASTER_PORT:-29620}"  # the NCCL_COMM_ID port (keep it on the private link); MASTER_PORT+1 must be free too (tp_watch)
# The rendezvous address: this node's address on the link to the worker (scripts/nodes.sh sets it from the link when it
# is empty).
MASTER_ADDR="${MASTER_ADDR:-}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"     # NCCL's bootstrap socket's netdev on both nodes (default: the link's)

# ---------------------------------------------------------------- the checkpoint, the engine
# The weights: jayleaton's EXL3 pack of MiniMaxAI/MiniMax-M3.1-Flash (revision 54e4e69d), routed experts at 3.55 bits a weight on
# average (attention q/o 6, k/v 8, MSA index projections FP16, shared expert and dense MLP 6, head 8, embedding BF16; the
# vision tower, projector and dspark/ draft copied unchanged), model_type minimax_m3_vl: 199.9 GB in all, 26 shards
# (index total_size 197,542,720,916 bytes) plus dspark/ (2.30 GB) and the files it ships, among them SHA256SUMS and the
# MiniMax Community License (LICENSE). The repository is PRIVATE: scripts/prepare.sh needs your Hugging Face token
# (HF_TOKEN, or `hf auth login`, which saves it at $HF_HOME/token). The token is read by `hf` on this Spark only; it is
# never written to a file here, printed, or sent to the worker or a container. Mia-AiLab's 3.06 bpw EAQ pack (the base
# recipe's) is still known: MODEL_ID=Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw selects its pins.
MODEL_ID="${MODEL_ID:-jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31}"
case "$MODEL_ID" in
  jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31)
    _rev=5bff3d732e9dfa236a41b6f3054dd8550bef595f; _shards=26; _total=197542720916; _gib=186.2 ;;
  Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
    _rev=1f404fc3c665883d0d001cfb32fb430108160589; _shards=21; _total=168662578580; _gib=157.1 ;;
  *) _rev=""; _shards=0; _total=0; _gib="?" ;;
esac
MODEL_REVISION="${MODEL_REVISION-$_rev}"
# What the pinned revision's index says (checked by scripts/prepare.sh; 0 skips the check for another MODEL_ID)
MODEL_SHARDS="${MODEL_SHARDS:-$_shards}"
MODEL_TOTAL_BYTES="${MODEL_TOTAL_BYTES:-$_total}"
MODEL_GIB="${MODEL_GIB:-$_gib}"         # for messages only: the whole snapshot, GiB
MODEL_LICENSE_FILE="${MODEL_LICENSE_FILE:-LICENSE}"   # prepare.sh refuses a snapshot without it (read it before serving)
unset _rev _shards _total _gib
# The engine: TensorFold v1.1.2 (ashhart/TensorFold, the native Zig binary), at the commit the tag points to. The tag
# resolves to 4ff784410f70b95767684d337e59dfd6d94fec8b (git ls-remote, peeled); the Dockerfile checks the checkout against it.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_TAG="${TF_TAG:-v1.1.2}"
TF_REF="${TF_REF:-4ff784410f70b95767684d337e59dfd6d94fec8b}"
# TensorFold's .zig-version is 0.17.0; the tarball's sha256 is ziglang.org's download index for aarch64-linux
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
# The image's base: the CUDA toolchain (nvcc builds the sm_121 fatbins), NCCL and libibverbs the engine loads, as the
# sibling DeepSeek recipe builds on. Not yet built for this recipe.
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
SM="${SM:-121}"                         # GB10
# The family the checkpoint needs. start.sh and prepare.sh refuse to go on until the image's `tensorfold-native
# capabilities --json` lists it. ALLOW_UNSUPPORTED=1 goes on anyway (engine development only: the server will fail to load).
REQUIRED_FAMILY="${REQUIRED_FAMILY:-minimax_m3_vl}"
ALLOW_UNSUPPORTED="${ALLOW_UNSUPPORTED:-0}"

# The engine patches: ENGINE_PATCHES is the folder whose *.patch files zig/Dockerfile applies to TF_REF in filename order.
# This recipe does not edit them: the engine branch owns the minimax_m3_vl family and its loader fixes. The default is
# patches/ (MiaAI-Lab's series at fd1d379, unchanged); point ENGINE_PATCHES at the engine branch's series at its pinned commit
# (and record the commit in ENGINE_COMMIT) when it lands. A relative path is under this checkout.
ENGINE_PATCHES="${ENGINE_PATCHES:-patches}"
ENGINE_COMMIT="${ENGINE_COMMIT:-}"      # the pinned commit ENGINE_PATCHES was cut from (recorded in the image; empty: patches/)
engine_patches_dir() {   # an absolute path, ~/... (this Spark's home), or a path under this checkout
  case "$ENGINE_PATCHES" in /*) echo "$ENGINE_PATCHES" ;; "~/"*) echo "$HOME/${ENGINE_PATCHES#"~/"}" ;; *) echo "$_cfg_root/$ENGINE_PATCHES" ;; esac
}
# The image is built from ENGINE_PATCHES, zig/Dockerfile and these pins. Its tag is IMAGE; the hash goes into the image's
# tf.patches label, which is how prepare.sh knows a rebuild is due.
image_hash() {
  ( cd "$_cfg_root" || exit 1
    cat "$(engine_patches_dir)"/*.patch 2>/dev/null || true
    cat zig/Dockerfile
    echo "$TF_REPO $TF_TAG $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE sm$SM $ENGINE_COMMIT"
  ) | sha256sum | cut -c1-12
}
IMAGE="${IMAGE:-tensorfold-minimax-m3:v1.1.2}"
# A published image of these pins, if any: none is published, and no registry name is chosen yet. Set GHCR_IMAGE (and
# REPO_URL for scripts/publish-image.sh) to use one; with it empty prepare.sh always builds locally.
GHCR_IMAGE="${GHCR_IMAGE:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
TAG_BASE="${TAG_BASE:-v1.1.2}"          # a published tag is <TAG_BASE>-<image hash>
prebuilt_image() {
  local tag="${TAG_BASE}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-minimax-m3-flash-tf}"   # the same name on both Sparks
# A docker volume of the same name on each Spark, mounted at /cache in its rank: what the engine keeps across starts (the
# RoCE failure marker, TF_ROCE_MARK). Created by docker on first use; scripts/serve.sh roce-reset removes the marker.
CACHE_VOL="${CACHE_VOL:-minimax-m3-flash-cache}"

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-MiniMax-M3.1-Flash-EXL3}"
# The API listens on this Spark's loopback by default (the GLM and DeepSeek recipes' rule: no auth unless API_KEY=1). Put a
# reverse proxy with authentication in front, or set HOST=0.0.0.0 together with API_KEY=1, to serve other machines.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# Everything below is EMPTY by default: nothing was measured for this checkpoint, so start.sh passes no flag and the engine's
# own default applies. The base model declares a 1,048,576-token window; the quantized pack was scored at 2,048 tokens at
# most, and no window, stream count or KV size is qualified. Set a value to pass it (the build's capabilities are checked).
CONTEXT="${CONTEXT:-}"                 # --context: a request's positions at most (prompt and reply)
PARALLEL="${PARALLEL:-}"               # --parallel: requests at once
MAX_TOKENS="${MAX_TOKENS:-}"           # --max-tokens: a reply's budget when a request sets none
THINKING="${THINKING:-}"               # 1: --thinking, 0: --no-thinking, empty: the engine's/template's default
KV="${KV:-}"                           # --kv-cache bf16 | fp8 (empty: the engine's default, bf16). fp8: K/V as e4m3 rows, 47,040 B a token a rank instead of
                                       # 76,800, index keys stay bf16; needs patches/tasks/0016, quality NOT MEASURED (docs/fp8-kv.md); both ranks alike
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"   # --prompt-cache-gib (0 turns kept prompt states off)
# Speculative decoding. 0 passes --no-drafts on both ranks. 1 lets the engine draft: copies of text already in the context,
# which need no memory and no weights (a follower must agree with rank 0, so both ranks get the same flag). The pack's
# `dspark/` folder (a 2.3 GB mxfp8 draft, MiniMax's own; both known packs carry it) is a second drafter: DRAFTER=dspark loads it
# on both ranks (about 2 GiB a rank before the KV pool, so the pool is smaller; scripts/quant_budget.py --serve --drafter
# prices it) and it drafts blocks of 7 that the target verifies, so the reply is the one plain decode gives. NOT MEASURED:
# whether it pays on this MoE (a verify window reads more experts); docs/dspark-plan.md section 8 gives the break-even.
# DRAFTS=dflash2: drafts on (copies) with the DFlash2 draft of DFLASH2_DIR (files, below) on both ranks; DRAFTER then defaults to
# dflash2 (that folder, mounted read-only at /dflash2 in both ranks, so both pass the same --drafter /dflash2). The engine
# (mm3/dflash2) tells the kind from the folder's config.json; docs/DFLASH2.md is the draft, its memory and its measurement.
DRAFTS="${DRAFTS:-0}"
DRAFTER="${DRAFTER:-}"                 # empty: copy drafts only. dspark: the checkpoint's dspark/ folder; dflash2: DFLASH2_DIR; or a folder (seen by both ranks)
# TF_MM3_DSPARK_STATS=1 (an engine with mm3/dspark, DRAFTER=dspark): rank 0 logs the draft's acceptance by position for
# every finished stream, just before the server's `done <id> ...` line: `dspark acceptance stream: rounds=R reached=a1/../a8
# kept=k1/../k8 kept_per_round=X conf1_rounds=.. conf1_kept=..` (rounds = head rounds; reached[i] / kept[i] = rounds that
# verified / kept a draft at depth i + 1), and `dspark acceptance total: ...` at exit either way. bench/drafts.py
# --server-log-cmd reads the stream lines (docs/DRAFTS.md); config/drafts.env.example sets it. Like every TF_MM3_* it reaches
# both ranks (scripts/nodes.sh rank_env); empty or 0: totals at exit only. It changes no reply.
TF_MM3_DSPARK_STATS="${TF_MM3_DSPARK_STATS:-}"
# TF_MM3_DRAFT_DEPTH (an engine with mm3/dspark): the most drafts a round verifies, 1..7 (empty: the draft's block of 7);
# the depth rule may still pick fewer or none. Like every TF_MM3_* it reaches both ranks. It changes no reply.
TF_MM3_DRAFT_DEPTH="${TF_MM3_DRAFT_DEPTH:-}"
# TF_MM3_DSPARK_CHANCES=1 (an engine with mm3/dspark, DRAFTER=dspark): each round's depth is sized by the draft's own
# confidences against the verify costs, at most TF_MM3_DRAFT_DEPTH (or the block); TF_MM3_CALIB gives those costs: auto (a
# matching cached table, else timed at boot, ~28 s), measure (timed every boot) or off/empty (the engine's static table).
# The profile measured in window-all-3 sets both (config/prod.env.example). Both reach both ranks; neither changes a reply.
TF_MM3_DSPARK_CHANCES="${TF_MM3_DSPARK_CHANCES:-}"
TF_MM3_CALIB="${TF_MM3_CALIB:-}"
# RoCE for the ranks' small gathers (the family's cuda/roce.zig, as GLM's): 1 leaves it on (the engine's default: RoCE when
# its setup succeeds, NCCL otherwise), 0 passes TF_ROCE=0 (every gather on NCCL). Both ranks get the same value (the ranks'
# fingerprint does not carry it yet: docs/rank-contract.md). ROCE_MAX_KB caps a RoCE gather (TF_ROCE_MAX_KB; empty: the
# engine's 4096). TF_COMM_BACKEND, the DeepSeek recipe's switch, is not read by this family.
ROCE="${ROCE:-1}"
ROCE_MAX_KB="${ROCE_MAX_KB:-}"
# Image input (the engine branch mm3/vision). VISION=1 passes --vision, --vision-max-images and --vision-image-tokens to both
# ranks (they agree on a request's image rows); 0 passes none, and an image part is then refused with HTTP 400. The
# MiniMax-M3.1 vision tower loads and runs on rank 0 only (this Spark, the head): 2.27 GiB with its scratch and its first call's plans, plus every
# rank's image rows (12 KiB an image token); the engine takes both out of the KV pool, which both ranks size to rank 0's
# (scripts/quant_budget.py --serve --vision-image-tokens; MEM_GATE_GIB does not move). The ranks' fingerprint refuses
# unequal --vision flags at boot. An
# image is at most 576 tokens (672 x 672 pixels at most, 28 x 28 pixels a token); JPEG, PNG and WebP are read (the image
# carries libjpeg-turbo 3.1.4.1 and libwebp 1.6.0, zig/Dockerfile), other formats and video parts are refused. An engine
# without mm3/vision refuses --vision at start. OFF by default: with images the all-resident pool no longer holds a
# 131,072-token context on a node at 110-113 GiB available, and with the KV tier the slice cache is short at the 112 GiB
# gate (docs/KNOWN-ISSUES.md). With VISION=1, lower CONTEXT to what the calculator gives (docs/OPERATIONS.md section 3a).
# bench/vision.py smoke checks a running server.
export VISION="${VISION:-0}"
export VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-4}"         # --vision-max-images: images a request at most (one more: HTTP 400)
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-2304}"         # --vision-image-tokens: a request's images together; 2304 = 4 x 576,
                                                           # so every image keeps its full size
# 1: --vision-urls on rank 0: an image part may name a public https URL, which the head downloads (10 MB an image, 20 MB
# a request, private and loopback addresses refused). 0: data: URLs only, and an https part is refused.
export VISION_URLS="${VISION_URLS:-0}"
# More environment for BOTH ranks, as NAME=value words (a family patch's switches), e.g. EXTRA_ENV="TF_FOO=1 TF_BAR=2".
# Every exported TENSORFOLD_*, TF_MM3_*, TF_ROCE_*, TF_CUDA_*, TF_SERVE_* and TF_EXL3_* variable goes to both ranks too
# (config/prod.env's lines are exported), e.g. TENSORFOLD_MEMORY_RESERVE_GIB.
EXTRA_ENV="${EXTRA_ENV:-}"

# ---------------------------------------------------------------- files
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME:-$HOME/.cache/huggingface}/token}"   # where `hf auth login` saves it (only tested for existence)
# Where rank 1 reads the checkpoint. WEIGHTS=local (the default, as the GLM and DeepSeek recipes run): a byte-identical
# copy in the worker's own HF cache (WORKER_HF_CACHE, default its $HF_HOME or ~/.cache/huggingface), which
# scripts/prepare.sh copies from this Spark over the link (rsync over ssh: the token never reaches the worker) and checks
# file by file. WEIGHTS=nfs (MiaAI-Lab's base recipe): the worker reads this Spark's HF_CACHE over NFS through a read-only
# docker volume NFS_VOLUME that prepare.sh creates; this Spark must export NFS_PATH (default: HF_CACHE; "/" for an NFSv4
# export with fsid=0) to the worker's link address, and NFS_SERVER is the head's address the worker mounts from (default:
# the head's link address). One copy of the checkpoint, but rank 1's load goes over NFS (its speed and page-cache cost on
# the worker are NOT MEASURED: docs/loader-speed.md).
# WEIGHTS=dir: each Spark already holds the pack as a plain folder (what `hf download --local-dir` leaves, or a copy made
# by hand), MODEL_DIR here and WORKER_MODEL_DIR on the worker (default: MODEL_DIR). A leading ~/ is each node's own home.
# Each rank mounts its node's folder read-only at /model. Nothing is downloaded or copied: prepare.sh checks the index's
# shards and total_size here and SHA256SUMS once, and preflight and prepare.sh compare the worker's folder with this one's
# file by file, by size (the shards, the top-level json / jinja / txt files, the license, SHA256SUMS; dspark/ with DRAFTER=dspark).
WEIGHTS="${WEIGHTS:-local}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-}"
MODEL_DIR="${MODEL_DIR:-}"
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-}"
# The DFlash2 draft (DRAFTS=dflash2): incoai/MiniMax-M3.1-Flash-DFlash2 at the pinned revision (Apache-2.0, public: no token),
# a plain folder on each Spark: DFLASH2_DIR here, WORKER_DFLASH2_DIR on the worker (a leading ~/ is each node's own home).
# Default: beside the pack, the way the weights lie on each node: with WEIGHTS=dir in MODEL_DIR's parent folder (WORKER_MODEL_DIR's
# on the worker), else in each node's own HF cache under dflash2/. scripts/prepare.sh fetches it here and copies it to the worker
# (resumable), checking every file against the sha256 pins below; start.sh mounts each node's folder read-only at /dflash2.
DFLASH2_DIR="${DFLASH2_DIR:-}"
WORKER_DFLASH2_DIR="${WORKER_DFLASH2_DIR:-}"
DFLASH2_FOLDER=MiniMax-M3.1-Flash-DFlash2
_dflash2_default=0
if [[ -z "$DFLASH2_DIR" ]]; then
  _dflash2_default=1
  if [[ "$WEIGHTS" == dir && -n "$MODEL_DIR" ]]; then _d=${MODEL_DIR%/}; DFLASH2_DIR="${_d%/*}/$DFLASH2_FOLDER"; unset _d
  else DFLASH2_DIR="$HF_CACHE/dflash2/$DFLASH2_FOLDER"; fi
fi
DFLASH2_REPO=incoai/MiniMax-M3.1-Flash-DFlash2
DFLASH2_REVISION=ac07a7c1ecaae26a7ace595756124dca974b432d
DFLASH2_IN_CONTAINER=/dflash2
# "<sha256> <bytes> <file>": every file of the revision (model.safetensors 4,247,776,488 bytes)
DFLASH2_PINS="64c32db3c3bc5de64a048d949c5acd91352ae9b0f02ef636e3382c838f1d0815 - config.json
8ec2de932fdd0e2d72a36ba17a0c593cfcba053609661a22fcb841047ae93282 - README.md
ad861f4ddd8f7204cda1f407837403e9095e1fb1f36d4c9f4188185261cc4fd1 4247776488 model.safetensors"
# The config.json classes of a DFlash2 draft (the engine's dflash2_config.architectures): incoai's, and RadixArk's, which the
# engine refuses for now (its block attention is causal by the reference's default rule, not built; its card has no license)
DFLASH2_CLASSES="DFlash2DraftModel MiniMaxDFlash2DraftModel"
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"
NFS_VOLUME="${NFS_VOLUME:-minimax-m3-flash-hf}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/tensorfold-minimax-m3}"      # on every Spark: saved server logs, the memory guard
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tensorfold-minimax-m3}"   # locks and the prepared marker
LOG_DIR="${LOG_DIR:-$CACHE_DIR/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------- the KV tier
# The engine's KV tier (mm3/kv-tier): every page of every stream on NVMe, one O_DIRECT file a rank under TF_MM3_KV_TIER
# in the container, which start.sh mounts from KV_TIER_DIR here and WORKER_KV_TIER_DIR on the worker (default: the same; a
# leading ~/ is each node's own home). The device holds the pages' summaries, an exact index horizon and a slice cache.
# Empty KV_TIER_DIR: no tier (TF_MM3_KV_TIER must then be unset). KV_TIER_KEEP_GIB: disk each Spark keeps free beyond it.
KV_TIER_DIR="${KV_TIER_DIR:-}"
WORKER_KV_TIER_DIR="${WORKER_KV_TIER_DIR:-}"
KV_TIER_KEEP_GIB="${KV_TIER_KEEP_GIB:-300}"
KV_TIER_IN_CONTAINER=/kvtier

# ---------------------------------------------------------------- prepared per-rank weights
# PREPARED=1 (an engine that reads TF_MM3_PREPARED; one without it ignores both variables): each rank keeps its share of the
# weights, as the engine lays them out on the GPU, in one file of its node's folder, HEAD_PREPARED here and WORKER_PREPARED on
# the worker (default: each node's ~/.cache/tensorfold-minimax-m3/prepared; a leading ~/ is each node's own home), mounted at
# /prepared. A boot that finds no file for its key writes mm3-r<rank>of<world>-<key16>.tfprep (~94 GiB a rank,
# TF_MM3_PREPARED_WRITE=1); later boots read it. The key covers the pack revision, the engine build and the layout settings:
# a change makes a new key and leaves the old file stale (scripts/serve.sh prepared clean). Both ranks get the same two
# variables (rank_env), so the switch is on for both or for neither; each node's folder holds only its own rank's file.
# preflight wants PREPARED_FREE_GIB free on a node whose folder has no file yet.
PREPARED="${PREPARED:-0}"
HEAD_PREPARED="${HEAD_PREPARED:-$CACHE_DIR/prepared}"
WORKER_PREPARED="${WORKER_PREPARED:-}"; [[ -n "$WORKER_PREPARED" ]] || WORKER_PREPARED=\~/.cache/tensorfold-minimax-m3/prepared
PREPARED_FREE_GIB="${PREPARED_FREE_GIB:-100}"
PREPARED_IN_CONTAINER=/prepared

# ---------------------------------------------------------------- memory
# A Spark's GPU shares its memory with the host: a rank that takes it all does not fail cleanly, the machine freezes and has to
# be power-cycled. Both Sparks have the same MemTotal (127.6 GB) since the worker's BIOS and firmware match the head's
# (2026-10-24), so either rank can bind: every floor below is one value checked on BOTH ranks (the DeepSeek recipe, PR #79).
# The levels, highest first (docs/OPERATIONS.md, Memory):
#  - MEM_GATE_GIB: before a load, after the page cache is dropped (echo 3), MemAvailable (MemFree plus what the kernel can
#    still reclaim) must reach this on both Sparks, and MemFree itself at least MEM_GATE_GIB - MEM_GATE_SLACK_GIB (on GB10
#    the page cache is the GPU's memory too, so the slack bounds what may still be cache), within MEM_GATE_TIMEOUT s, or the
#    start stops. With MEM_GATE_DROP_CACHES=1 the gate drops the page cache (`sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`;
#    the scoped sudoers line is in AGENTS.md) while it waits, and DROP_CACHES=1 drops it (echo 3) before the load, after the
#    server answers and after the canary. The value is scripts/quant_budget.py --serve for the configured CONTEXT x PARALLEL
#    and KV (weights a rank + the engine's reserve, runtime and workspaces + that KV), not a measurement. 0 turns the gate off.
#  - TENSORFOLD_MEMORY_RESERVE_GIB (config/prod.env, both ranks): what the engine leaves free when it sizes its KV pool. The
#    engine's own default is a tenth of MemTotal (12.2 GiB here), which would leave this pack about 6 GiB of KV a rank.
#  - the guard's hard STOP (scripts/guard, 4 GiB of MemAvailable on either node), when the guard runs.
#  - MEMGUARD_GIB: scripts/memguard.sh runs on both Sparks while the server is up and kills the rank's container when
#    MemAvailable stays under this floor for MEMGUARD_HOLD_S seconds without coming back over it, or at once (two samples)
#    under MEMGUARD_HARD_GIB. The pair watch (below) then stops the other rank. 0 turns it off.
#  - earlyoom, where the nodes run it (`-m 2,1` on ours: SIGTERM under 2% of MemTotal, 2.43 GiB): the last backstop. Its
#    `--avoid ^(...|tensorfold-|...)` lowers a rank's score but does not exempt it. MEMGUARD_HARD_GIB must stay above
#    earlyoom's level so the recipe, not earlyoom, picks what dies; `scripts/serve.sh preflight` reads earlyoom's arguments on
#    both nodes and warns when it does not.
MEM_GATE_GIB="${MEM_GATE_GIB:-0}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
MEM_GATE_SLACK_GIB="${MEM_GATE_SLACK_GIB:-4}"
MEM_GATE_DROP_CACHES="${MEM_GATE_DROP_CACHES:-1}"
DROP_CACHES="${DROP_CACHES:-1}"
MEMGUARD_GIB="${MEMGUARD_GIB:-3}"
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-4}"
MEMGUARD_HARD_GIB="${MEMGUARD_HARD_GIB:-2.75}"

# ---------------------------------------------------------------- temperature: the device's limits
# Heat and GPU limits are settings of each machine, not of this recipe. Every Spark carries them in SPARK_THERMAL_ENV
# (/etc/spark/thermal.env: GUARD_PAUSE_C, GUARD_RESUME_C, GUARD_KILL_C, GUARD_KILL_HOLD_S and the SPARK_CLOCK_* values), and
# spark-thermal-guard.service publishes the node's state in SPARK_THERMAL_RUN/state: normal, throttled (at or above the pause
# limit, until the resume limit) or emergency (at or above the kill limit for the hold). This recipe reads both: the file's
# values win over the environment and the recipe config (which may not define them), start.sh refuses while the file is
# missing on either node or either node is not `normal`, memguard stops its rank on `emergency`, and the guard pauses on
# `throttled`. scripts/install-device-limits.sh installs the file and the services on a fresh pair (it asks for sudo).
SPARK_THERMAL_ENV="${SPARK_THERMAL_ENV:-/etc/spark/thermal.env}"
SPARK_THERMAL_RUN="${SPARK_THERMAL_RUN:-/run/spark-thermal}"
if [[ -r "$SPARK_THERMAL_ENV" ]]; then
  # read, never run; and these keys are the device's, whatever the environment says
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(GUARD_(PAUSE_C|RESUME_C|KILL_C|KILL_HOLD_S)|SPARK_[A-Z_]+)=([^#[:space:]]*)[[:space:]]*$ ]] || continue
    export "${BASH_REMATCH[1]}=${BASH_REMATCH[3]}"
  done < "$SPARK_THERMAL_ENV"
  unset _line
else
  unset GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S
fi
# need_device_limits: refuse (start) while this Spark has no device limits file
need_device_limits() {
  [[ -r "$SPARK_THERMAL_ENV" ]] ||
    die "no device thermal limits on this Spark ($SPARK_THERMAL_ENV): install them on both Sparks first with scripts/install-device-limits.sh (it asks for sudo; AGENTS.md section 1)"
}

# ---------------------------------------------------------------- the pair
# A lone rank waits for its peer forever (an NCCL collective that never completes). PAIR_WATCH=1 runs scripts/pairwatch.sh on
# this Spark while the server is up: every PAIR_WATCH_S seconds it checks both containers, and when one has exited it stops
# the other and writes STATE_DIR/failfast (the exit codes), so the watchdog heals the pair on its next tick instead of after
# WATCH_FAILS ticks (the DeepSeek recipe's fail-fast, done from outside the engine: the family's own RankLost path is
# NOT MEASURED). A worker out of reach is not taken for an exit.
PAIR_WATCH="${PAIR_WATCH:-1}"
PAIR_WATCH_S="${PAIR_WATCH_S:-5}"
# The link the ranks must use, when you want it pinned (GLM's check): LINK_NETDEV (e.g. enp1s0f1np1) and LINK_HCAS (the RoCE
# devices, comma-separated) are compared with what scripts/nodes.sh detects on this Spark, and a start refuses on a mismatch
# (a renamed netdev after a firmware or kernel update, a cable on another port). Empty: whatever the route to the worker uses.
LINK_NETDEV="${LINK_NETDEV:-}"
LINK_HCAS="${LINK_HCAS:-}"
# After the server answers: scripts/canary.py (a greedy chat, a reply with separated reasoning, a parsed tool call, a JSON
# reply). strict stops both ranks on a failure (their last logs are saved in STATE_DIR), warn only says so, off skips it.
CANARY="${CANARY:-strict}"
# The engine's request features from the mm3/api-tools engine branch: tool_choice "required" and a named function, response_format,
# thinking off (chat_template_kwargs.enable_thinking false), thinking_budget and the five reasoning_effort levels. 0 (until
# ENGINE_COMMIT includes that branch): the canary skips them and the bench harnesses report a refusal as SKIP. 1: a refusal fails
# the canary and the harnesses (both read it from the environment). Either way an accepted feature must behave.
export ENGINE_API_TOOLS="${ENGINE_API_TOOLS:-0}"
# The guard (scripts/guard): set GUARD_CONFIG to its JSON and start.sh and the watchdog obey its lifecycle decision.
GUARD_CONFIG="${GUARD_CONFIG:-}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }
[[ "$ALLOW_UNSUPPORTED" =~ ^[01]$ ]] || die "ALLOW_UNSUPPORTED is 0 or 1, not $ALLOW_UNSUPPORTED"
[[ "$MODEL_SHARDS" =~ ^[0-9]+$ && "$MODEL_TOTAL_BYTES" =~ ^[0-9]+$ ]] || die "MODEL_SHARDS and MODEL_TOTAL_BYTES are whole numbers (0 skips the check)"
[[ "$WEIGHTS" =~ ^(local|nfs|dir)$ ]] || die "WEIGHTS is local, nfs or dir, not $WEIGHTS"
for _n in KV_TIER_DIR WORKER_KV_TIER_DIR DFLASH2_DIR WORKER_DFLASH2_DIR; do
  [[ -z "${!_n}" || "${!_n}" == /* || "${!_n}" == "~/"* ]] || die "$_n is an absolute path or one under ~/, not ${!_n}"
done
[[ "$KV_TIER_KEEP_GIB" =~ ^[0-9]+$ ]] || die "KV_TIER_KEEP_GIB is a whole number of GiB, not $KV_TIER_KEEP_GIB"
if [[ -n "$KV_TIER_DIR" ]]; then
  [[ "${TF_MM3_KV_TIER:-}" == "$KV_TIER_IN_CONTAINER" ]] || die "KV_TIER_DIR is set: TF_MM3_KV_TIER is $KV_TIER_IN_CONTAINER (where start.sh mounts it), not ${TF_MM3_KV_TIER:-unset}"
  [[ "${TF_MM3_KV_TIER_GIB:-}" =~ ^[0-9]+$ ]] || die "the KV tier needs TF_MM3_KV_TIER_GIB, whole GiB a rank (scripts/quant_budget.py --serve --kv-tier)"
elif [[ -n "${TF_MM3_KV_TIER:-}" ]]; then
  die "TF_MM3_KV_TIER is set without KV_TIER_DIR: nothing would be mounted at $TF_MM3_KV_TIER"
fi
# the prepared weights: PREPARED=1 mounts each node's folder at /prepared and gives both ranks the same two variables
[[ "$PREPARED" =~ ^[01]$ ]] || die "PREPARED is 0 or 1, not $PREPARED"
[[ "$PREPARED_FREE_GIB" =~ ^[0-9]+$ ]] || die "PREPARED_FREE_GIB is a whole number of GiB, not $PREPARED_FREE_GIB"
for _n in HEAD_PREPARED WORKER_PREPARED; do
  [[ "${!_n}" == /* || "${!_n}" == \~/* ]] || die "$_n is an absolute path or one under ~/, not ${!_n}"
done
if [[ "$PREPARED" == 1 ]]; then
  [[ -z "${TF_MM3_PREPARED:-}" || "$TF_MM3_PREPARED" == "$PREPARED_IN_CONTAINER" ]] ||
    die "PREPARED=1 mounts the folder at $PREPARED_IN_CONTAINER: leave TF_MM3_PREPARED unset (it is $TF_MM3_PREPARED)"
  export TF_MM3_PREPARED=$PREPARED_IN_CONTAINER TF_MM3_PREPARED_WRITE="${TF_MM3_PREPARED_WRITE:-1}"
  [[ "$TF_MM3_PREPARED_WRITE" =~ ^[01]$ ]] || die "TF_MM3_PREPARED_WRITE is 1 (write a missing file) or 0 (read only), not $TF_MM3_PREPARED_WRITE"
elif [[ -n "${TF_MM3_PREPARED:-}${TF_MM3_PREPARED_WRITE:-}" ]]; then
  die "TF_MM3_PREPARED / TF_MM3_PREPARED_WRITE are set without PREPARED=1: nothing would be mounted at $PREPARED_IN_CONTAINER (set PREPARED=1 instead)"
fi
if [[ "$WEIGHTS" == dir ]]; then
  for _n in MODEL_DIR WORKER_MODEL_DIR; do
    [[ -z "${!_n}" || "${!_n}" == /* || "${!_n}" == "~/"* ]] || die "$_n is an absolute path or one under ~/, not ${!_n}"
  done
  [[ -n "$MODEL_DIR" ]] || die "WEIGHTS=dir needs MODEL_DIR (this Spark's folder of the pack; WORKER_MODEL_DIR the worker's, default the same)"
  unset _n
fi
[[ "$ENGINE_API_TOOLS" =~ ^[01]$ ]] || die "ENGINE_API_TOOLS is 0 or 1, not $ENGINE_API_TOOLS"
[[ -z "$TF_MM3_DRAFT_DEPTH" || "$TF_MM3_DRAFT_DEPTH" =~ ^[1-7]$ ]] || die "TF_MM3_DRAFT_DEPTH is 1..7 or empty, not $TF_MM3_DRAFT_DEPTH"
[[ -z "$TF_MM3_DSPARK_CHANCES" || "$TF_MM3_DSPARK_CHANCES" =~ ^[01]$ ]] || die "TF_MM3_DSPARK_CHANCES is 1 (depth by confidence), 0 or empty, not $TF_MM3_DSPARK_CHANCES"
[[ -z "$TF_MM3_CALIB" || "$TF_MM3_CALIB" =~ ^(auto|measure|off)$ ]] || die "TF_MM3_CALIB is auto, measure, off or empty, not $TF_MM3_CALIB"
[[ -z "$TF_MM3_DSPARK_STATS" || "$TF_MM3_DSPARK_STATS" =~ ^[01]$ ]] || die "TF_MM3_DSPARK_STATS is 1 (a line a finished stream), 0 or empty, not $TF_MM3_DSPARK_STATS"
[[ "$VISION" =~ ^[01]$ && "$VISION_URLS" =~ ^[01]$ ]] || die "VISION and VISION_URLS are 0 or 1, not $VISION / $VISION_URLS"
[[ "$VISION_MAX_IMAGES" =~ ^[1-9][0-9]*$ && "$VISION_IMAGE_TOKENS" =~ ^[1-9][0-9]*$ ]] ||
  die "VISION_MAX_IMAGES and VISION_IMAGE_TOKENS are positive counts, not $VISION_MAX_IMAGES / $VISION_IMAGE_TOKENS"
[[ "$VISION_URLS" == 0 || "$VISION" == 1 ]] || die "VISION_URLS=1 needs VISION=1 (--vision-urls without --vision is refused)"
# the commit ENGINE_PATCHES was cut from: a commit hash (7 to 40 hex digits), not a branch name, or empty
[[ -z "$ENGINE_COMMIT" || "$ENGINE_COMMIT" =~ ^[0-9a-f]{7,40}$ ]] || die "ENGINE_COMMIT is the commit hash ENGINE_PATCHES was cut from (7 to 40 hex digits) or empty, not $ENGINE_COMMIT"
if [[ -r "$SPARK_THERMAL_ENV" ]]; then
  for _n in GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S; do
    [[ "${!_n:-}" =~ ^[1-9][0-9]*$ ]] || die "$SPARK_THERMAL_ENV: $_n is not a whole number (${!_n:-missing})"
  done
  (( GUARD_RESUME_C < GUARD_PAUSE_C && GUARD_PAUSE_C <= GUARD_KILL_C && GUARD_KILL_C < 94 )) ||
    die "$SPARK_THERMAL_ENV: the limits must order GUARD_RESUME_C < GUARD_PAUSE_C <= GUARD_KILL_C < 94 (the Sparks degrade at 94 C), not $GUARD_RESUME_C / $GUARD_PAUSE_C / $GUARD_KILL_C"
  unset _n
fi
[[ -z "$MODEL_REVISION" || "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "MODEL_REVISION is a full 40-character commit hash (or empty for the cache's refs/main), not $MODEL_REVISION"
[[ "$TF_REF" =~ ^[0-9a-f]{40}$ ]] || die "TF_REF is a full 40-character commit hash, not $TF_REF"

hub_dir()         { echo "$HF_CACHE/hub/models--${1//\//--}"; }
model_cache_dir() { hub_dir "$MODEL_ID"; }
# The snapshot this setup serves: the pin, else what refs/main names here.
snapshot_rev() { local rev=$MODEL_REVISION; [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir)/refs/main" 2>/dev/null); echo "$rev"; }
# model_dir_here: the WEIGHTS=dir folder on this Spark (a leading ~/ is this node's home)
model_dir_here() { local d=$MODEL_DIR; [[ "$d" != "~/"* ]] || d="$HOME/${d#"~/"}"; echo "$d"; }
# kv_tier_here: the KV tier's folder on this Spark (a leading ~/ is this node's home)
kv_tier_here() { local d=$KV_TIER_DIR; [[ "$d" != "~/"* ]] || d="$HOME/${d#"~/"}"; echo "$d"; }
# prepared_here: the prepared weights' folder on this Spark (HEAD_PREPARED; a leading ~/ is this node's home)
prepared_here() { local d=$HEAD_PREPARED; [[ "$d" != \~/* ]] || d="$HOME/${d#\~/}"; echo "$d"; }
# dflash2_dir_here: the DFlash2 draft's folder on this Spark (a leading ~/ is this node's home)
dflash2_dir_here() { local d=$DFLASH2_DIR; [[ "$d" != "~/"* ]] || d="$HOME/${d#"~/"}"; echo "$d"; }
# dflash2_wanted: this start loads the DFlash2 draft (DRAFTS=dflash2, or DRAFTS=1 with DRAFTER=dflash2)
dflash2_wanted() { [[ "$DRAFTS" == dflash2 || ( "$DRAFTS" == 1 && "$DRAFTER" == dflash2 ) ]]; }
# snapshot_dir: the checkpoint folder on this Spark: the HF cache's snapshot, or MODEL_DIR (WEIGHTS=dir)
snapshot_dir() { if [[ "$WEIGHTS" == dir ]]; then model_dir_here; else echo "$(model_cache_dir)/snapshots/$(snapshot_rev)"; fi; }
# The checkpoint as the containers see it (HF_CACHE mounted at /hf on both Sparks; the WEIGHTS=dir folder at /model).
model_in_container() { if [[ "$WEIGHTS" == dir ]]; then echo /model; else echo "/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi; }
# head_model_mount: rank 0's (and the info check's) read-only docker mount of the checkpoint
head_model_mount() { if [[ "$WEIGHTS" == dir ]]; then echo "$(model_dir_here):/model:ro"; else echo "$HF_CACHE:/hf:ro"; fi; }

# need_hf_token: the private repository needs a token. Tests that one exists; never reads it into a variable or prints it.
need_hf_token() {
  [[ -n "${HF_TOKEN:-}" || -s "$HF_TOKEN_FILE" ]] ||
    die "$MODEL_ID is a PRIVATE repository and no Hugging Face token was found (looked for HF_TOKEN and $HF_TOKEN_FILE): run 'hf auth login' with a read token that can see it, or export HF_TOKEN"
}
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN

# What scripts/prepare.sh last left ready (start.sh runs it again when this line changes).
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  echo "model=$MODEL_ID@$(snapshot_rev) image=$(image_ident "$IMAGE") patches=$(image_hash) w=$(worker_host 1) weights=$WEIGHTS$([[ "$WEIGHTS" != dir ]] || echo ":$MODEL_DIR:$WORKER_MODEL_DIR")$(! dflash2_wanted || echo " dflash2=$DFLASH2_REVISION:$DFLASH2_DIR:$WORKER_DFLASH2_DIR")"
}

# ---------------------------------------------------------------- API key
# The API is open on the network by default: anyone who can reach PORT can use the GPUs. API_KEY=1 makes rank 0 require a key
# (Authorization: Bearer <key>, or x-api-key) on every request but /health. The key lives in API_KEY_FILE (default:
# $STATE_DIR/api-key, mode 600; start.sh generates one when it is missing) and reaches the container as a read-only file for
# tensorfold-native's --api-key-file, never in the command line or the environment. One key (or "label: key") a line.
API_KEY="${API_KEY:-0}"
API_KEY_FILE="${API_KEY_FILE:-$STATE_DIR/api-key}"
API_KEY_IN_CONTAINER=/run/secrets/tensorfold-api-key
# api_key_ensure: create API_KEY_FILE (mode 600, one random key) when it does not exist; never overwrites a file
api_key_ensure() {
  [[ -e "$API_KEY_FILE" ]] && { [[ "$(stat -c %a "$API_KEY_FILE")" =~ ^[0-7]00$ ]] || chmod 600 "$API_KEY_FILE"; return 0; }
  mkdir -p "$(dirname "$API_KEY_FILE")"
  ( umask 077; python3 -c 'import secrets; print("sk-tf-" + secrets.token_urlsafe(32))' > "$API_KEY_FILE" ) || die "could not write $API_KEY_FILE"
  log "Generated an API key in $API_KEY_FILE (mode 600); clients send it as Authorization: Bearer \$(cat $API_KEY_FILE)"
}
# api_header_file: a mode-600 file holding the Authorization header, for curl -H @file (the key stays out of ps)
api_header_file() {
  local h="$STATE_DIR/curl-auth-header"
  ( umask 077; printf 'Authorization: Bearer %s\n' "$(head -n 1 "$API_KEY_FILE" | sed 's/^[^:]*: *//')" > "$h" ) && echo "$h"
}
