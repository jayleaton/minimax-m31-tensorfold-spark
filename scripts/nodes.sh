# shellcheck shell=bash
# The two Sparks: this machine (rank 0) and WORKER (rank 1, over ssh). The link between them is found from the routes and
# subnets between them (detect_link): one direct CX7 cable, its PCIe twin as a second rail when it reaches the peer.
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.
# From the node helpers of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (Mia's AI Lab), cut to two Sparks.

# wvar <NAME> <i>: the name of worker i's own setting: NAME for worker 1 (WORKER, FABRIC_PEER, NFS_SERVER). The helpers
# keep the GLM scripts' per-worker form; this recipe has the one worker.
wvar() { if (( $2 == 1 )); then echo "$1"; else echo "$1$2"; fi; }
wval() { local v; v=$(wvar "$1" "$2"); echo "${!v:-}"; }
worker_host() { wval WORKER "$1"; }
# The workers this start uses (1 .. TP-1), and every worker configured at all (stop.sh stops them all).
worker_ids() { seq 1 $((TP - 1)); }
configured_workers() { local i; for i in 1 2 3; do [[ -z "$(worker_host "$i")" ]] || echo "$i"; done; }

# check_workers: this recipe serves two Sparks (TP=2), and WORKER is set.
check_workers() {
  [[ "$TP" == 2 ]] || die "TP is 2 (two Sparks): this recipe runs tensor-parallel over two nodes, not $TP"
  [[ -n "$(worker_host 1)" ]] || die "WORKER is not set: put WORKER=user@<address of the second Spark> in scripts/local.sh (see scripts/local.sh.example)"
}

# DRY_RUN=1 goes on without workers it cannot reach (a fictitious WORKER): its values print as <WORKER:...>.
declare -A WORKER_DOWN=()
# worker <i> <command...>: run a command on worker i (rank i). Key-based ssh only, no prompts.
worker() {
  local i=$1; shift
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$(worker_host "$i")" "$@"
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log <i>: the same, for rank i on worker i (into ~/.cache/tensorfold-minimax-m3/logs there)
worker_save_log() {
  worker "$1" "$(declare -f save_log); save_log \"\$HOME/.cache/tensorfold-minimax-m3/logs\" $1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  local i=$1 v h
  v=$(wvar WORKER "$i"); h=$(worker_host "$i")
  [[ -n "$h" ]] || die "$v is not set: put $v=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker "$i" true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $h ($v): its values print as <$v:...>"
    WORKER_DOWN[$i]=1
    return 0
  fi
  die "cannot ssh to $h without a password: set up key-based ssh (ssh-copy-id $h)"
}
need_workers() { local i; for i in $(worker_ids); do need_worker "$i"; done; }

# Worker i's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank i. WORKER_HF_CACHE (worker 1) / WORKER_HF_CACHE<i> wins: a worker whose
# cache is not its HF_HOME (e.g. a shared models disk).
worker_hf_cache() {
  local c; c=$(wval WORKER_HF_CACHE "$1")
  if [[ -n "$c" ]]; then echo "$c"; return; fi
  [[ -z "${WORKER_DOWN[$1]:-}" ]] || { echo "<$(wvar WORKER "$1"):hf-cache>"; return 0; }
  worker "$1" 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# WEIGHTS=dir (config.sh): worker_model_dir is the pack's folder on the worker, WORKER_MODEL_DIR (default MODEL_DIR) with a
# leading ~/ resolved in the worker's home; model_dir_manifest <head|worker> lists what a rank reads there, "<path> <size>"
# a line, sorted: the shards and every top-level json / jinja / txt file (not hidden ones: a tool's own state), the license,
# SHA256SUMS, and dspark/ with DRAFTER=dspark. Fails when the folder cannot be read.
worker_model_dir() {
  local d=${WORKER_MODEL_DIR:-$MODEL_DIR} h
  [[ "$d" == "~/"* ]] || { echo "$d"; return 0; }
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { echo "<WORKER:$d>"; return 0; }
  h=$(worker 1 'printf %s "$HOME"') && [[ "$h" == /* ]] || return 1
  echo "$h/${d#"~/"}"
}
# worker_kv_tier_dir: the KV tier's folder on worker 1 (WORKER_KV_TIER_DIR, default KV_TIER_DIR; ~/ is its home)
worker_kv_tier_dir() {
  local d=${WORKER_KV_TIER_DIR:-$KV_TIER_DIR} h
  [[ "$d" == "~/"* ]] || { echo "$d"; return 0; }
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { echo "<WORKER:$d>"; return 0; }
  h=$(worker 1 'printf %s "$HOME"') && [[ "$h" == /* ]] || return 1
  echo "$h/${d#"~/"}"
}
# kv_tier_room_cmd <dir>: a shell command printing the GiB the tier may hold there: free space plus its own files
kv_tier_room_cmd() {
  local q; q=$(printf '%q' "$1")
  echo "mkdir -p $q && echo \$(( ( \$(df -B1 --output=avail $q | tail -1) + \$(du -sb $q | cut -f1) ) / 1073741824 ))"
}
# kv_tier_gate: with KV_TIER_DIR, both folders exist and each disk holds TF_MM3_KV_TIER_GIB + KV_TIER_KEEP_GIB; sets
# TIER_HERE_GIB / TIER_WORKER_GIB. 1 when either is short (the caller refuses the start).
kv_tier_gate() {
  [[ -n "$KV_TIER_DIR" ]] || return 0
  local need=$(( TF_MM3_KV_TIER_GIB + KV_TIER_KEEP_GIB )) w
  TIER_HERE_GIB=$(bash -c "$(kv_tier_room_cmd "$(kv_tier_here)")") || TIER_HERE_GIB=0
  w=$(worker_kv_tier_dir) || return 1
  TIER_WORKER_GIB=$(worker 1 "$(kv_tier_room_cmd "$w")") || TIER_WORKER_GIB=0
  (( ${TIER_HERE_GIB:-0} >= need && ${TIER_WORKER_GIB:-0} >= need ))
}
model_dir_manifest() {
  local d q
  if [[ "$1" == head ]]; then d=$(model_dir_here); else d=$(worker_model_dir) || return 1; fi
  q="cd $(printf '%q' "$d") && { find -L . -maxdepth 1 -type f ! -name '.*' \\( -name '*.safetensors' -o -name '*.json' -o -name '*.jinja' -o -name '*.txt' -o -name $(printf '%q' "$MODEL_LICENSE_FILE") -o -name SHA256SUMS \\) -printf '%P %s\\n'"
  [[ "$DRAFTER" != dspark ]] || q+="; find -L dspark -type f -printf '%p %s\\n'"
  q+="; } | LC_ALL=C sort"
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q"; fi
}

# The DFlash2 draft (DRAFTS=dflash2, config.sh "files"). worker_dflash2_dir: its folder on worker 1: WORKER_DFLASH2_DIR, else
# DFLASH2_DIR when that was given, else beside the worker's pack (WEIGHTS=dir: WORKER_MODEL_DIR's parent folder; otherwise the
# worker's own HF cache under dflash2/); a leading ~/ is resolved in the worker's home.
worker_dflash2_dir() {
  local d=$WORKER_DFLASH2_DIR h
  if [[ -z "$d" && "$_dflash2_default" == 0 ]]; then d=$DFLASH2_DIR
  elif [[ -z "$d" && "$WEIGHTS" == dir ]]; then d=$(worker_model_dir) || return 1; d="${d%/}"; d="${d%/*}/$DFLASH2_FOLDER"
  elif [[ -z "$d" ]]; then d="$(worker_hf_cache 1)/dflash2/$DFLASH2_FOLDER"
  fi
  [[ "$d" == "~/"* ]] || { echo "$d"; return 0; }
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { echo "<WORKER:$d>"; return 0; }
  h=$(worker 1 'printf %s "$HOME"') && [[ "$h" == /* ]] || return 1
  echo "$h/${d#"~/"}"
}
# dflash2_dir <head|worker>: the draft's folder on that node
dflash2_dir() { if [[ "$1" == head ]]; then dflash2_dir_here; else worker_dflash2_dir; fi; }
# dflash2_class_cmd <dir>: a shell command printing the class config.json's "architectures" names (no python needed on the node)
dflash2_class_cmd() {
  local q; q=$(printf '%q' "$1/config.json")
  echo "tr -d ' \\n\\t\\r' < $q | grep -o '\"architectures\":\\[\"[A-Za-z0-9_]*\"' | sed 's/.*\\[\"//; s/\"\$//'"
}
# dflash2_class <head|worker>: that node's draft class; fails (and says why) when it is not a DFlash2 draft folder
dflash2_class() {
  local d c
  d=$(dflash2_dir "$1") || { echo "cannot resolve the folder on the $1"; return 1; }
  if [[ "$1" == head ]]; then [[ -f "$d/config.json" ]] || { echo "no $d/config.json on the $1"; return 1; }; c=$(bash -c "$(dflash2_class_cmd "$d")" 2>/dev/null || true)
  else worker 1 "test -f $(printf '%q' "$d/config.json")" || { echo "no $d/config.json on the $1"; return 1; }; c=$(worker 1 "$(dflash2_class_cmd "$d")" 2>/dev/null || true)
  fi
  [[ " $DFLASH2_CLASSES " == *" ${c:-none} "* ]] || { echo "$d/config.json on the $1 names ${c:-no class} in architectures, not a DFlash2 draft ($DFLASH2_CLASSES)"; return 1; }
  echo "$c"
}
# dflash2_verify_cmd <dir> [quick]: a shell command that fails unless every pinned file there has its sha256 (quick: only
# config.json's sha256 and model.safetensors' size, for a folder already checked once)
dflash2_verify_cmd() {
  local q sha size f sums=""; q=$(printf '%q' "$1")
  if [[ "${2:-}" == quick ]]; then
    read -r sha _ f < <(grep ' config.json$' <<<"$DFLASH2_PINS"); read -r _ size _ < <(grep ' model.safetensors$' <<<"$DFLASH2_PINS")
    echo "cd $q && [ \"\$(stat -c %s model.safetensors 2>/dev/null)\" = $size ] && printf '%s  %s\\n' $sha config.json | sha256sum --quiet --status -c -"
    return
  fi
  while read -r sha _ f; do sums+="$sha  $f\\n"; done <<<"$DFLASH2_PINS"
  echo "cd $q && printf '$sums' | sha256sum --quiet -c -"
}
# dflash2_on <head|worker> <command>: run a shell command on that node
dflash2_on() { if [[ "$1" == head ]]; then bash -c "$2"; else worker 1 "$2"; fi; }

# Weights over NFS: rank i mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on worker i, from
# nfs_server i: NFS_SERVER (worker 1) / NFS_SERVER<i>, else the head's address on that worker's link (detect_links).
# ensure_nfs_volume i creates it (or checks the one there names the same export); worker_nfs i <cmd...> runs a command
# in a throwaway container on worker i with the volume at /hf.
nfs_server() { local s; s=$(wval NFS_SERVER "$1"); echo "${s:-${LINK_HEAD_ADDR[$1]:-}}"; }
ensure_nfs_volume() {
  local i=$1 server have h
  server=$(nfs_server "$i"); h=$(worker_host "$i")
  [[ -n "$server" ]] || die "no NFS server address for $h: set $(wvar NFS_SERVER "$i") to the head's address on that worker's link"
  have=$(worker "$i" "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "$i" "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $h"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$h's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / $(wvar NFS_SERVER "$i")"
  fi
}
worker_nfs() { local i=$1; shift; worker "$i" "docker run --rm --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }
# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and NCCL gets one x4: ~112 Gb/s of the port's 200.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d $n ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: one rail's traffic reaches the peer's twin - a full-size, unfragmentable
# ping bound to that rail's own port (-I <netdev>: an address alone would be routed out of whichever port of the same
# subnet the kernel picks, usually the cabled one), so an unaddressed, differently-subnetted or smaller-MTU twin fails
# here instead of as vendor_err 0x81 (transport retry exceeded) in the first all-gather (issue #66). Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# The same function on worker i (its definition is sent over ssh).
worker_link_info() { worker "$1" "$(declare -f link_info); link_info $2"; }
# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing". worker_image_ident <i> <image>: on worker i.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}
# cx7_peer <i>: worker i's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker "$1" 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# probe_rails: before a second rail (the PCIe twin of the cabled port, or a second cabled port) is used, check that it
# reaches the peer. The RDMA ring pairs the two nodes' devices by list position, so a twin that is up and has a GID
# but no path to the peer's twin (no address, another subnet, another MTU, a cable on a different port) ends in vendor_err
# 0x81. Rail k of the head is probed against rail k of the worker, in both directions; a rail that fails is dropped from both
# lists with a warning (the first device, the link itself, is never probed: the launcher reached the worker over it).
probe_rails() {
  local -a h w keep_h=() keep_w=()
  local k hdev hip hmtu wdev wip wmtu mtu=0 ok
  IFS=, read -ra h <<<"$HEAD_HCAS"; IFS=, read -ra w <<<"$WORKER_HCAS"
  (( ${#h[@]} > 1 && ${#h[@]} == ${#w[@]} )) || return 0
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { warn "DRY_RUN: the second rail ($HEAD_HCAS / $WORKER_HCAS) is not probed: the worker cannot be reached"; return 0; }
  keep_h=("${h[0]}"); keep_w=("${w[0]}")
  for (( k = 1; k < ${#h[@]}; k++ )); do
    read -r hdev hip hmtu <<<"$(rail_info "${h[k]}")"
    read -r wdev wip wmtu <<<"$(worker 1 "$(declare -f rail_info); rail_info ${w[k]}")"
    ok=0
    if [[ "$hip" =~ ^[0-9.]+$ && "$wip" =~ ^[0-9.]+$ && "$hmtu" =~ ^[0-9]+$ && "$wmtu" =~ ^[0-9]+$ ]]; then
      mtu=$(( hmtu < wmtu ? hmtu : wmtu ))
      if rail_ping "$hdev" "$wip" "$mtu" && worker 1 "$(declare -f rail_ping); rail_ping $wdev $hip $mtu"; then ok=1; fi
    fi
    if (( ok )); then
      keep_h+=("${h[k]}"); keep_w+=("${w[k]}")
      log "Rail ${h[k]} <-> ${w[k]}: $hip <-> $wip reaches the peer (MTU $mtu): used"
    else
      warn "second rail ${h[k]} (${hip:--}, MTU ${hmtu:--}) <-> ${w[k]} (${wip:--}, MTU ${wmtu:--}) does not reach the peer: NOT used, running on fewer rails. Give both twins an IPv4 address on one subnet per rail with the same MTU on both Sparks (README), or set NCCL_RAILS=1 to silence this"
    fi
  done
  HEAD_HCAS=$(IFS=,; echo "${keep_h[*]}"); WORKER_HCAS=$(IFS=,; echo "${keep_w[*]}")
}

# TP=2: HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the
# two ranks talk over (NCCL and the rendezvous). FABRIC_PEER overrides the worker's link address when WORKER is reached
# over another network.
detect_link() {
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer 1); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "${NCCL_RAILS:-2}" == 1 ]] || probe_rails
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# The NCCL settings of rank r's container (docker -e arguments): the RoCE link's netdev, RoCE devices (both rails) and GID
# index, as TensorFold's two-node RoCE notes (zig/src/fabric/tp/README.md) run them (NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_GID_INDEX).
# Every other exported NCCL_* variable goes to both ranks unchanged (an A/B of NCCL settings: NCCL_NET_PLUGIN=none,
# NCCL_MAX_NCHANNELS=4, NCCL_IB_QPS_PER_CONNECTION=2, NCCL_DEBUG_SUBSYS=INIT,NET ...); none set: no change. The names and
# values are checked here, in the main shell, before anything is stopped: a value is [A-Za-z0-9_.,:=/-] only (no spaces,
# quotes, $, ;, newlines), because it ends up in a command that the worker's shell runs. NCCL_SOCKET_IFNAME, NCCL_IB_HCA
# and NCCL_IB_GID_INDEX are this script's own (from the link it finds); NCCL_RAILS is its setting, not NCCL's.
NCCL_EXTRA_ENV=()
NCCL_ARGS=()
nccl_extra_collect() {
  local name val
  NCCL_EXTRA_ENV=()
  while IFS= read -r name; do
    [[ "$name" =~ ^NCCL_[A-Za-z0-9_]+$ ]] || continue
    val=${!name}
    case "$name" in
      NCCL_RAILS) continue ;;
      NCCL_SOCKET_IFNAME|NCCL_IB_HCA|NCCL_IB_GID_INDEX) warn "$name is set by this script (from the link it finds): your value is ignored"; continue ;;
    esac
    if [[ -z "$val" ]]; then warn "$name is empty: not passed on"; continue; fi
    [[ "$val" =~ ^[A-Za-z0-9_.,:=/-]+$ ]] || die "$name has a value with characters outside A-Z a-z 0-9 _ . , : = / - (it goes into a shell command on the worker): not passed on, nothing started"
    [[ "$name" == NCCL_DEBUG ]] || NCCL_EXTRA_ENV+=("$name=$val")
  done < <(compgen -e | LC_ALL=C sort | grep -E '^NCCL_' || true)
}

nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  NCCL_ARGS=(-e "NCCL_SOCKET_IFNAME=$dev" -e "NCCL_IB_HCA=$hcas" -e "NCCL_IB_GID_INDEX=$gid")
  [[ -z "${NCCL_DEBUG:-}" ]] || NCCL_ARGS+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
  # cuMem off, as the DeepSeek recipe runs its ranks (credit: BobClawblaw); an exported NCCL_CUMEM_* below is passed after these and wins
  NCCL_ARGS+=(-e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0)
  local x; for x in "${NCCL_EXTRA_ENV[@]}"; do NCCL_ARGS+=(-e "$x"); done
  # the same arguments as one line for a remote shell: any word with a character outside [A-Za-z0-9_.,:=/@%+-] is
  # shell-quoted (printf %q), so a value can never be taken for shell syntax there (the values nccl_extra_collect lets
  # through need no quoting, which keeps the line as it always was)
  local out="" w
  for w in "${NCCL_ARGS[@]}"; do
    if [[ "$w" =~ ^[A-Za-z0-9_.,:=/@%+-]+$ ]]; then out+="$w "; else out+="$(printf '%q' "$w") "; fi
  done
  echo "${out% }"
}
rank_nccl_env() { nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}"; }
# rank_nccl_args <r>: the same as the array NCCL_ARGS (docker arguments), for a command run here without a shell between
rank_nccl_args() { nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}" >/dev/null; }

# detect_links: the one link of two Sparks (start.sh and prepare.sh call it; the GLM scripts' name for it at any TP).
detect_links() { detect_link; }

# mem_available_gib: MemAvailable here, in GiB (one decimal)
mem_available_gib() { awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
# memfree <head|worker>: MemFree in whole GiB (empty when the worker cannot be read). MemFree, not MemAvailable: on GB10 the
# page cache is the GPU's memory too, and a load beside a full page cache gets a smaller KV pool (GLM, DeepSeek).
memfree() {
  local q="awk '/^MemFree:/ {printf \"%d\", \$2 / 1048576}' /proc/meminfo"
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q" 2>/dev/null || true; fi
}
# drop_caches <1|3> [force]: sync and drop the page cache on both Sparks (1: page cache only; 3: also dentries and inodes),
# never swap. sudo -n only: the scoped sudoers line in AGENTS.md allows exactly these commands. DROP_CACHES=0 skips it
# unless forced (the memory gate forces level 1 while it waits, under MEM_GATE_DROP_CACHES).
drop_caches() {
  local level=$1 force=${2:-0} cmd
  [[ "$force" == 1 || "$DROP_CACHES" == 1 ]] || return 0
  cmd="sync; sudo -n sh -c 'echo $level > /proc/sys/vm/drop_caches'"
  if bash -c "$cmd" 2>/dev/null; then log "Dropped the page cache here (MemFree now $(memfree head) GiB)"
  else warn "drop_caches refused here (no scoped sudo -n: AGENTS.md)"; fi
  if worker 1 "$cmd" 2>/dev/null; then log "Dropped the page cache on $(worker_host 1) (MemFree now $(memfree worker) GiB)"
  else warn "drop_caches refused on $(worker_host 1) (no scoped sudo -n: AGENTS.md)"; fi
}
# mem_gate: MemFree >= MEM_GATE_GIB on both Sparks, waiting up to MEM_GATE_TIMEOUT s (dropping the page cache every 10 s with
# MEM_GATE_DROP_CACHES=1). Sets HERE_GB / WORKER_GB. 1 when the time ran out (the caller refuses the start).
mem_gate() {
  local t0=$SECONDS
  (( MEM_GATE_GIB > 0 )) || return 0
  while :; do
    HERE_GB=$(memfree head); WORKER_GB=$(memfree worker)
    if (( ${HERE_GB:-0} >= MEM_GATE_GIB && ${WORKER_GB:-0} >= MEM_GATE_GIB )); then
      log "MemFree $HERE_GB / $WORKER_GB GiB (here / $(worker_host 1)) >= MEM_GATE_GIB $MEM_GATE_GIB"; return 0
    fi
    (( SECONDS - t0 < MEM_GATE_TIMEOUT )) || return 1
    log "Waiting for memory: MemFree ${HERE_GB:-?} / ${WORKER_GB:-?} GiB (here / $(worker_host 1)), want $MEM_GATE_GIB"
    [[ "$MEM_GATE_DROP_CACHES" != 1 ]] || drop_caches 1 1 >/dev/null 2>&1
    sleep 10
  done
}

# earlyoom_level <head|worker>: the GiB of MemAvailable under which earlyoom sends SIGTERM on that node ("-" when earlyoom is
# not active or its -m cannot be read). earlyoom's -m PERCENT[,KILL] is a share of MemTotal.
earlyoom_level() {
  local q='systemctl is-active -q earlyoom 2>/dev/null || { echo -; exit 0; }
a=$(ps -o args= -C earlyoom 2>/dev/null | head -1)
m=$(printf "%s\n" "$a" | sed -nE "s/.*-m ?([0-9.]+).*/\1/p")
[ -n "$m" ] || m=10
awk -v m="$m" "/^MemTotal:/ {printf \"%.2f\", \$2 * m / 100 / 1048576}" /proc/meminfo'
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q" 2>/dev/null || echo -; fi
}

# The memory guard (scripts/memguard.sh) on this Spark and on the worker, which gets a copy of the script.
MEMGUARD_REMOTE_DIR='$HOME/.cache/tensorfold-minimax-m3'
memguard_start() {
  awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}' || return 0
  "$_cfg_root/scripts/memguard.sh" start "$CONTAINER_NAME" "$MEMGUARD_GIB" "$LOG_DIR/memguard.log" "$STATE_DIR/memguard.pid" "$MEMGUARD_HOLD_S" "$MEMGUARD_HARD_GIB" ||
    die "the memory guard did not start here"
  local i
  for i in $(worker_ids); do
    worker "$i" "mkdir -p $MEMGUARD_REMOTE_DIR/logs && cat > $MEMGUARD_REMOTE_DIR/memguard.sh && chmod +x $MEMGUARD_REMOTE_DIR/memguard.sh" \
      < "$_cfg_root/scripts/memguard.sh" || die "could not copy the memory guard to $(worker_host "$i")"
    worker "$i" "$MEMGUARD_REMOTE_DIR/memguard.sh start '$CONTAINER_NAME' $MEMGUARD_GIB $MEMGUARD_REMOTE_DIR/logs/memguard.log $MEMGUARD_REMOTE_DIR/memguard.pid $MEMGUARD_HOLD_S $MEMGUARD_HARD_GIB" ||
      die "the memory guard did not start on $(worker_host "$i")"
  done
  log "Memory guard on: each rank's container is killed when the memory available stays under $MEMGUARD_GIB GiB for $MEMGUARD_HOLD_S s, or falls under $MEMGUARD_HARD_GIB GiB, or its node's device state turns emergency (the device's limits, $SPARK_THERMAL_ENV on each node) (log: memguard.log)"
}
memguard_stop() {
  local i
  "$_cfg_root/scripts/memguard.sh" stop "$STATE_DIR/memguard.pid" 2>/dev/null || true
  for i in $(configured_workers); do
    worker "$i" "[ -x $MEMGUARD_REMOTE_DIR/memguard.sh ] && $MEMGUARD_REMOTE_DIR/memguard.sh stop $MEMGUARD_REMOTE_DIR/memguard.pid; true" 2>/dev/null || true
  done
}

# The pair watch (scripts/pairwatch.sh) on this Spark: stops the surviving rank when the other exits (config.sh, PAIR_WATCH).
pairwatch_start() {
  [[ "$PAIR_WATCH" == 1 ]] || return 0
  rm -f "$STATE_DIR/failfast"
  "$_cfg_root/scripts/pairwatch.sh" start "$CONTAINER_NAME" "$(worker_host 1)" "$PAIR_WATCH_S" "$STATE_DIR" "$LOG_DIR/pairwatch.log" ||
    die "the pair watch did not start"
  log "Pair watch on: when one rank exits, the other is stopped within ~$PAIR_WATCH_S s (log: $LOG_DIR/pairwatch.log)"
}
pairwatch_stop() { "$_cfg_root/scripts/pairwatch.sh" stop "$STATE_DIR" 2>/dev/null || true; }

# thermal_stops: the thermal-stop records memguard left (one line a node, "<node>: <epoch> <reading> C"); empty when none
thermal_stops() {
  { [[ ! -f "$STATE_DIR/thermal-stop" ]] || sed 's/^/head: /' "$STATE_DIR/thermal-stop"
    worker 1 "cat $MEMGUARD_REMOTE_DIR/thermal-stop 2>/dev/null" 2>/dev/null | sed 's/^/worker: /'
  } | grep -E '^(head|worker): [0-9]+ [0-9]+ C$' || true
}
# device_state <head|worker>: that node's spark-thermal-guard state (normal | throttled | emergency), "stale" when its state
# file is older than 30 s, "missing" without one
device_state() {
  local q="f=$SPARK_THERMAL_RUN/state; [ -r \$f ] || { echo missing; exit 0; }; [ \$(( \$(date +%s) - \$(stat -c %Y \$f) )) -le 30 ] && cat \$f || echo stale"
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q" 2>/dev/null || echo unreachable; fi
}
# device_limits_sha <head|worker>: sha256 of that node's device limits file ("missing" without one)
device_limits_sha() {
  local q="sha256sum $SPARK_THERMAL_ENV 2>/dev/null | cut -c1-64 || echo missing"
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q" 2>/dev/null || echo unreachable; fi
}
# hottest_now <head|worker>: memguard's reading on that node now (whole C), empty when nothing could be read
hottest_now() {
  local q; q="$(sed -n '/^hottest_c() {/,/^}/p' "$_cfg_root/scripts/memguard.sh"); hottest_c"
  if [[ "$1" == head ]]; then bash -c "$q"; else worker 1 "$q" 2>/dev/null; fi
}

# roce_marks: the nodes whose cache volume holds the RoCE failure marker (TF_ROCE_MARK under /cache), one a line; empty: none
roce_marks() {
  [[ "${TF_ROCE_MARK:-}" == /cache/* ]] || return 0
  local q="docker run --rm --network none -v '$CACHE_VOL:/cache' --entrypoint test '$IMAGE' -e '$TF_ROCE_MARK'"
  bash -c "$q" >/dev/null 2>&1 && echo head
  worker 1 "$q" >/dev/null 2>&1 && echo worker
  return 0
}

# check_link_pins: LINK_NETDEV / LINK_HCAS, when set, must be what detect_link found on this Spark (GLM's link check)
check_link_pins() {
  [[ -z "$LINK_NETDEV" || "$LINK_NETDEV" == "${NODE_DEV[0]}" ]] ||
    die "the link to the worker runs over ${NODE_DEV[0]}, not LINK_NETDEV=$LINK_NETDEV: a renamed netdev (firmware or kernel update) or a cable on another port; check ibdev2netdev and ip -br addr on both Sparks (AGENTS.md section 2)"
  [[ -z "$LINK_HCAS" || "$LINK_HCAS" == "${NODE_HCAS[0]}" ]] ||
    die "the link's RoCE devices here are ${NODE_HCAS[0]}, not LINK_HCAS=$LINK_HCAS (ibdev2netdev; AGENTS.md section 2)"
  return 0
}

# ---------------------------------------------------------------- the engine image: capabilities and the family gate
# GPU memory a container's processes hold so far (GiB); on the worker through ssh, with this definition.
# (From the GLM-5.3-Flash recipe's start.sh, Mia's AI Lab.)
gpu_gib() {
  local pids
  pids=$(docker top "$1" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}

# engine_caps: the image's `tensorfold-native capabilities --json` (no GPU, driver or model needed). Fails when there is no image.
engine_caps() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 || return 1
  docker run --rm --network none --entrypoint tensorfold-native "$IMAGE" capabilities --json
}

# family_gate: refuses to go on while this TensorFold build does not list REQUIRED_FAMILY (minimax_m3_vl) among the families
# its capabilities report. TensorFold v1.1.2 lists none for this model, and the checkpoint has never been loaded in any
# engine, so until patches/ adds the family this is the answer, and nothing is stopped, downloaded or started.
# ALLOW_UNSUPPORTED=1 goes on anyway (engine development only). Returns 0 when it is fine to go on, 2 when the image does
# not exist yet (nothing to ask; the caller decides), and exits through die otherwise.
family_gate() {
  local caps why families errf
  errf=$(mktemp) || die "mktemp failed"
  # stdout alone is parsed: without a visible GPU the binary writes 'cuInit: CUDA_ERROR_NO_DEVICE' to stderr, which must not reach the JSON
  if ! caps=$(engine_caps 2>"$errf"); then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      why="$IMAGE did not answer 'tensorfold-native capabilities --json':"$'\n'"$(tail -5 "$errf" | sed 's/^/    /')"
      rm -f "$errf"
      (( ALLOW_UNSUPPORTED )) && { warn "ALLOW_UNSUPPORTED=1: $why"; return 0; }
      die "$why"
    fi
    rm -f "$errf"
    return 2
  fi
  rm -f "$errf"
  if families=$(python3 -I -c '
import json, sys
want = sys.argv[1]
c = json.load(sys.stdin)
fam = c.get("families", {})
print("TensorFold %s, families: %s" % (c.get("version", "?"), ", ".join(sorted(fam)) or "none"))
sys.exit(0 if want in fam else 1)' "$REQUIRED_FAMILY" <<<"$caps"); then
    log "Engine: $families (has $REQUIRED_FAMILY)"
    return 0
  fi
  why="MiniMax-M3.1-Flash family not in this TensorFold build; apply patches/ ($IMAGE: $families; it needs $REQUIRED_FAMILY).
    TensorFold v1.1.2 has no minimax_m3_vl family, and $MODEL_ID has never been loaded in any engine: this server cannot start yet.
    patches/ holds the engine patches that add the family (see patches/README.md); rebuild the image with scripts/prepare.sh --rebuild after adding them."
  if (( ALLOW_UNSUPPORTED )); then
    warn "ALLOW_UNSUPPORTED=1: going on although $why"
    return 0
  fi
  die "$why
    ALLOW_UNSUPPORTED=1 overrides this for engine development; expect the load to fail."
}

# checkpoint_info: `tensorfold-native info <checkpoint>` in the image (what it would serve; no GPU), printed indented. Its exit
# code is the answer: 1 when no registered family serves the checkpoint's model_type.
checkpoint_info() {
  local out rc=0
  out=$(docker run --rm --network none -v "$(head_model_mount)" --entrypoint tensorfold-native "$IMAGE" info "$(model_in_container)" 2>&1) || rc=$?
  sed 's/^/    info: /' <<<"$out"
  return "$rc"
}

# check_serve_flags <serve arguments after the model>: every --flag must be one the build's capabilities list, and a flag with a
# fixed set of values must get one of them: the binary refuses the rest only at start, after the old server is stopped.
# (After scripts/zig_flags.py of the sibling DeepSeek recipe, from the GLM-5.3-Flash recipe, Mia's AI Lab.)
check_serve_flags() {
  engine_caps | python3 -I -c '
import json, sys
flags = json.load(sys.stdin)["serve"]["flags"]
args = sys.argv[1:]
valued = {"--host", "--port", "--name", "--context", "--parallel", "--max-tokens", "--drafter", "--tp", "--rank", "--master",
          "--master-port", "--prompt-cache-gib", "--backend", "--temperature", "--top-p", "--top-k", "--min-p",
          "--reasoning-effort", "--alias", "--api-key", "--api-key-file", "--chat-template", "--device", "--kv-cache",
          "--vision-max-images", "--vision-image-tokens"}
errors, i = [], 0
while i < len(args):
    a = args[i]; i += 1
    if not a.startswith("--"):
        continue
    name, eq, value = a.partition("=")
    if name not in flags:
        errors.append("%s is not served by this tensorfold-native" % name)
        continue
    values = flags[name].get("values")
    if not eq and (values is not None or name in valued):
        value = args[i] if i < len(args) else ""
        i += 1
    if values is not None and value not in values:
        errors.append("%s takes one of %s, not %s" % (name, ", ".join(values), value))
if errors:
    if any(e.startswith("--vision") for e in errors):
        errors.append("this engine reads no images (it lacks mm3/vision): set VISION=0, or build one that has it")
    sys.exit("tensorfold serve: error: " + "; ".join(errors))
' "$@"
}

# serves_flag <flag>: true when the image's capabilities list the serve flag (e.g. --tp)
serves_flag() { engine_caps 2>/dev/null | python3 -I -c 'import json,sys; sys.exit(0 if sys.argv[1] in json.load(sys.stdin)["serve"]["flags"] else 1)' "$1"; }

# ---------------------------------------------------------------- the ranks' environment
# rank_env <rank>: the NAME=value words rank <rank> gets, one per line, as docs/rank-contract.md section 2 names them:
# TF_MM3_WORLD / TF_MM3_RANK and NCCL_COMM_ID (<rank 0 link address>:MASTER_PORT; the liveness line listens on MASTER_PORT + 1),
# TF_ROCE_GID_INDEX from the link scan (TF_ROCE_HCA is not set: roce.zig falls back to NCCL_IB_HCA, which NCCL_ARGS sets from
# the same scan), TF_ROCE=0 / TF_ROCE_MAX_KB from ROCE / ROCE_MAX_KB, MM31_SERVE_PROFILE (the serve flags both ranks run, as
# one word, so the guard's identity check sees a CONTEXT or KV change: the engine does not read it), then every exported
# TENSORFOLD_*, TF_MM3_*, TF_ROCE_* and TF_CUDA_* variable, and EXTRA_ENV. TF_TP_* and TF_COMM_BACKEND are not passed: this
# family reads neither. Values are limited to A-Z a-z 0-9 _ . , : = / - because they end up in a command the worker's shell runs.
rank_env() {
  local r=$1 x name val master=${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}
  printf '%s\n' "TF_MM3_WORLD=$TP" "TF_MM3_RANK=$r" "NCCL_COMM_ID=$master:$MASTER_PORT" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  [[ -z "${NODE_GID[$r]:-}" ]] || printf '%s\n' "TF_ROCE_GID_INDEX=${NODE_GID[$r]}"
  [[ "$ROCE" != 0 ]] || printf '%s\n' "TF_ROCE=0"
  [[ -z "$ROCE_MAX_KB" ]] || printf '%s\n' "TF_ROCE_MAX_KB=$ROCE_MAX_KB"
  [[ -z "${SERVE_PROFILE:-}" ]] || printf '%s\n' "MM31_SERVE_PROFILE=$SERVE_PROFILE"
  while IFS= read -r name; do
    [[ "$name" =~ ^(TENSORFOLD|TF_MM3|TF_ROCE|TF_CUDA)_[A-Za-z0-9_]+$ ]] || continue
    case "$name" in TF_MM3_WORLD|TF_MM3_RANK|TF_ROCE_GID_INDEX|TF_ROCE_MAX_KB) continue ;; esac
    # a credential never rides in a container's environment (docker inspect shows it): the API key goes as a file
    [[ ! "$name" =~ (^|_)(API_?KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?|TOKEN)(_|$) ]] ||
      die "$name looks like a credential: it would be visible in docker inspect on both Sparks; use API_KEY=1 (a key file) instead"
    val=${!name}
    [[ -n "$val" ]] || continue   # an empty setting (e.g. the KV tier's, off) is the engine's default: not passed on
    [[ "$val" =~ ^[A-Za-z0-9_.,:=/-]+$ ]] || die "$name has a value with characters outside A-Z a-z 0-9 _ . , : = / - (it goes into a shell command on the worker)"
    printf '%s\n' "$name=$val"
  done < <(compgen -e | LC_ALL=C sort)
  for x in $EXTRA_ENV; do
    [[ "$x" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.,:=/-]*$ ]] || die "EXTRA_ENV word '$x' is not NAME=value with a value of A-Z a-z 0-9 _ . , : = / -"
    printf '%s\n' "$x"
  done
}
