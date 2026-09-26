#!/bin/bash
# The public configuration on main: the five multimer_v3 design models pdl1.json ships, at the
# 288-token device axis the acceptance tree measured 64.6 s/step at. Waits for chain2's card.
cd "$(dirname "$0")/../.."
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for i in $(seq 1 240); do grep -q 'CHAIN2 FINISHED' perf/bcx_mainstep/out/chain2.log && break; sleep 10; done
rm -rf perf/bcx_mainstep/out/public_s100
log "START public_s100 shipped pool, binder 146"
bash perf/bcx_mainstep/arm.sh public_s100 12 0 --shipped --binder 146 > perf/bcx_mainstep/out/public_s100.log 2>&1
log "END public_s100 rc=$?"
log "CHAIN3 FINISHED"
