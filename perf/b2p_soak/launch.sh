#!/bin/bash
# One long BindCraft 2 campaign, detached, with the drift sampler beside it.
#
# A soak is longer than any agent turn, so the campaign must outlive the shell that started it
# (setsid, its own log) and everything an interruption would have to be explained from has to be
# on disk while it runs: the campaign log, a timestamp per gradient round, and the drift series.
#
# The same script runs on both boards. Everything host-specific is an environment variable:
#
#   SOAK_VENV    interpreter with ttnn, jax and BindCraft 2's dependencies
#   SOAK_BC2     BindCraft 2 checkout (BCX_BC2 for the harness)
#   SOAK_PARAMS  AlphaFold 2 parameter directory
#   SOAK_REPO    tt-bio checkout to run from
#   SOAK_CARD    the chip, as TT_VISIBLE_DEVICES numbers them, leased for the whole campaign
#   SOAK_OUT     output root; the campaign gets $SOAK_OUT/$tag
#   SOAK_HOURS   wall-clock cap, default 12
#
# launch.sh <tag> [campaign_run.py arguments...]
set -uo pipefail
tag=${1:?tag}; shift
: "${SOAK_VENV:?}" "${SOAK_BC2:?}" "${SOAK_PARAMS:?}" "${SOAK_REPO:?}" "${SOAK_CARD:?}"
out=${SOAK_OUT:-$HOME/b2p_soak/out}/$tag
hours=${SOAK_HOURS:-12}
mkdir -p "$out"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }

export BCX_BC2=$SOAK_BC2
export PYTHONPATH=$SOAK_REPO:$SOAK_BC2
export TT_VISIBLE_DEVICES=$SOAK_CARD
export TT_BIO_LEASE_CARDS=$SOAK_CARD
export TT_BIO_LEASE_HOLDER=worker:b2p-soak
# The lease has to outlast the campaign, or it expires mid-soak and another row is told the chip
# is free while this one is still folding on it.
export TT_BIO_LEASE_TIMEOUT=$(( hours * 3600 + 3600 ))
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/b2p_soak/cache/xla}
export TT_METAL_CACHE=${TT_METAL_CACHE:-$HOME/b2p_soak/cache/tt-metal}
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$TT_METAL_CACHE"

say "soak $tag card=$SOAK_CARD cap=${hours}h repo=$SOAK_REPO commit=$(git -C "$SOAK_REPO" rev-parse --short HEAD 2>/dev/null)"
say "args: $*"
cd "$SOAK_REPO" || exit 1
setsid nohup timeout -s INT $(( hours * 3600 )) \
  "$SOAK_VENV" -u perf/bcx_p10_campaign/campaign_run.py --params "$SOAK_PARAMS" \
  --out "$out/project" "$@" < /dev/null > "$out/campaign.log" 2>&1 &
pid=$!
echo "$pid" > "$out/pid"
say "campaign pid $pid log $out/campaign.log"
setsid nohup "$SOAK_VENV" -u perf/b2p_soak/drift.py "$pid" "$out/drift.jsonl" \
  --every "${SOAK_DRIFT_EVERY:-30}" --project "$out/project" \
  --cache "$JAX_COMPILATION_CACHE_DIR" --cache "$TT_METAL_CACHE" \
  < /dev/null > "$out/drift.log" 2>&1 &
say "drift sampler pid $! -> $out/drift.jsonl"
