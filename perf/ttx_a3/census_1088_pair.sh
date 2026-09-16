#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
OUT=perf/ttx_a3/gate6
PROG=$OUT/progress
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

# One arm per PROCESS. Never flip the flag inside a live device context: that is the harness
# trap this campaign already fell into, and the hang it manufactured was mis-attributed for a
# whole pass.
for arm in on off; do
  name="cen1088-$arm"
  log "$name START card=3 loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  if [ "$arm" = off ]; then EXTRA="TT_BIO_SDPA_FUSED_LARGE_S=0"; else EXTRA="TT_BIO_SDPA_FUSED_LARGE_S=1"; fi
  env $EXTRA TT_VISIBLE_DEVICES=3 timeout -k 30 900 \
    $P scripts/lever_census.py --tt-bio $P --label "rf3-1088-$arm" \
      --out "$WT/$OUT/census_1088_$arm.json" \
      -- -m tt_bio.main predict "$WT/perf/size512/fixtures/cdk2x2_1088.yaml" \
         --model rf3 --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0 \
         --out_dir "$WT/$OUT/out_1088_$arm" > "$OUT/$name.log" 2>&1
  log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
log "CEN1088_DONE"
