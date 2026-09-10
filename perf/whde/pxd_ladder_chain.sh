#!/bin/bash
# Wait for the pxdesign checkpoint (served slowly from a CN bucket by a sibling downloader,
# so this must not fetch it itself -- fetch_file stages on a stable .part and two writers
# would corrupt it), then walk the ladder_* rungs on one chip.
# usage: pxd_ladder_chain.sh <chip> <results-dir>
set -u
chip=$1; RES=$2; WT=/home/mthuening/scratch/wt-wh-perf-design-embed
CKPT=/home/tt-admin/.boltz/pxdesign/pxdesign_v0.1.0.pt
export TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH=$WT OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip
export TT_BIO_LEASE_HOLDER=worker:wh-perf-design-embed TT_BIO_LEASE_TIMEOUT=1800
cd "$WT" || exit 2
mkdir -p "$RES"

for i in $(seq 1 480); do            # up to 2 h of waiting, checked every 15 s
  [ -f "$CKPT" ] && break
  sleep 15
done
if [ ! -f "$CKPT" ]; then echo "PXD CHAIN: checkpoint never arrived"; exit 3; fi
echo "PXD CHAIN: checkpoint present $(date -u +%H:%M:%SZ), size $(stat -c%s "$CKPT")"

for rung in 128 256 512 768; do
  y=perf/pxdesign/targets/ladder_${rung}.yaml
  o=$RES/pxdesign_ladder_${rung}_wh${chip}.json
  timeout 3600 /home/mthuening/work/tt-bio/env/bin/python -u perf/whde/design_step_probe.py \
    --model pxdesign --inputs "$y" --n-step 30 --num-designs 1 --out "$o" \
    > "$RES/pxdesign_ladder_${rung}_wh${chip}.log" 2>&1
  echo "PXD rung $rung rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  tail -2 "$RES/pxdesign_ladder_${rung}_wh${chip}.log"
done
echo "PXD CHAIN DONE $(date -u +%H:%M:%SZ)"
