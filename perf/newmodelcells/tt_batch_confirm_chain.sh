#!/bin/bash
# Three TT runs owed by state doc section 3.5, chained under one benchlock hold.
set -x
WT=/home/moritz/.coworker/wt/pxdesign-realistic-campaign-batch
cd "$WT" || exit 1
PY=/home/moritz/tt-bio/env/bin/python3
E="env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-realistic-campaign-batch PYTHONPATH=$WT"

# 1. b=16 at 400 real steps: decides whether the fitted non-monotonic turn is real.
$E $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
  --yaml perf/newmodelcells/laczc_512_tt.yaml --n_step 400 --n_sample 16 --rounds 4 \
  --label c400_n16 --out perf/newmodelcells/batchcurve400/c400_n16.json
echo "STEP1_RC=$?"

# 2. digest reproduction: rounds 5 -> seeds [0,1,2,3,0], round 4 repeats round 0's seed.
$E $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
  --yaml perf/newmodelcells/laczc_512_tt.yaml --n_step 400 --n_sample 8 --rounds 5 \
  --label d400_n8r5 --out perf/newmodelcells/batchcurve400/d400_n8r5.json
echo "STEP2_RC=$?"

# 3. chunk-vs-batch: b=32 with the per-forward chunk capped at 8, in the fit regime the
#    ladder was built in, so it compares directly against s8_n32 / s24_n32.
for S in 8 24; do
  $E $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
    --yaml perf/newmodelcells/laczc_512_tt.yaml --n_step $S --n_sample 32 \
    --max_parallel_samples 8 --rounds 3 \
    --label mps8_s${S}_n32 --out perf/newmodelcells/batchchunk/mps8_s${S}_n32.json
  echo "STEP3_${S}_RC=$?"
done
echo "TT_CHAIN_DONE"
