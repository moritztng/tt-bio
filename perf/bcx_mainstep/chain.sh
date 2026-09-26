#!/bin/bash
# Two arms, one after the other, on card 1. Each writes its own fresh project dir: a directory
# holding a .campaign_state.json resumes and runs zero rounds.
cd "$(dirname "$0")/../.."
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for spec in "confirm_s100 8 0" "shipped_s100 12 0 --shipped"; do
  set -- $spec
  tag=$1; rounds=$2; exact=$3; shift 3
  rm -rf "perf/bcx_mainstep/out/$tag"
  log "START $tag rounds=$rounds exact=$exact extra=$*"
  bash perf/bcx_mainstep/arm.sh "$tag" "$rounds" "$exact" "$@" \
      > "perf/bcx_mainstep/out/$tag.log" 2>&1
  log "END $tag rc=$?"
done
log "CHAIN FINISHED"
