#!/bin/bash
cd "$(dirname "$0")/../.."
log() { echo "[$(date -u +%FT%TZ)] $*"; }
rm -rf perf/bcx_mainstep/out/n288_s100
log "START n288_s100 binder=146"
bash perf/bcx_mainstep/arm.sh n288_s100 12 0 --binder 146 > perf/bcx_mainstep/out/n288_s100.log 2>&1
log "END n288_s100 rc=$?"
log "START prof"
bash perf/bcx_mainstep/prof.sh > perf/bcx_mainstep/out/prof_s100.log 2>&1
log "END prof rc=$?"
log "CHAIN2 FINISHED"
