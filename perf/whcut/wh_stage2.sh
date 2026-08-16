#!/bin/bash
# Stage 2 on Wormhole, after the §5.1 fold chain: the four ESMFold2 rows re-run on the
# harness that actually produced their reference numbers, then the §6.2 parity gate.
#
# WHY THE RE-RUN. §5.1 asks esmfold2 rows to reproduce 83.843 s at 512 and a 7.6 % win at
# 1024, and labels both "esmfold2-fast --fast". Three things are wrong with that:
#   * `esmfold2-fast` is a DIFFERENT CHECKPOINT (24 trunk blocks, no MSA encoder), not the
#     --fast flag. `xmodel_ab.py --model esmfold2 --fast` folds the other checkpoint.
#   * 83.843 s is the `esmfold2` checkpoint at 512 (wh-perf-esmfold2 §16.4 line 1152).
#     `esmfold2-fast` at 512 is 49.576 s.
#   * every one of those numbers came from perf/wh-esmfold2/fold_ab512.py, not from
#     xmodel_ab.py, so a wall from the other harness is not comparable to them.
#
# Arms here are overrides against the SHIPPED default, so running `base` on the assembled
# tree and matching the recorded `A` column is the real confirmation: it measures what a
# user gets now that the lever ships on.
#
#   model          size  must reproduce (was)
#   esmfold2        512   83.843  (93.512)
#   esmfold2       1024  367.032 (395.224)
#   esmfold2-fast   512   49.576  (54.630)
#   esmfold2-fast  1024  203.679 (220.374)
set -u
TREE=/home/cust-team/mthuening/whbase/wt-whcut
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=$TREE/perf/whcut/out/wh
CHAIN_PID=${CHAIN_PID:-773336}
cd "$TREE" || exit 1
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "STAGE2 START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"

run_esm() {  # model size card
  local m=$1 s=$2 c=$3
  echo "=== $m $s card $c start $(date -u +%H:%M:%S)"
  env TT_VISIBLE_DEVICES=$c TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" -u perf/wh-esmfold2/fold_ab512.py --model "$m" --size "$s" --fast \
      --arms base --rounds 1 --out "$OUT/esm_${m}_${s}.json" > "$OUT/esm_${m}_${s}.log" 2>&1
  echo "EXIT $m $s = $?"
}
( run_esm esmfold2 1024 28; run_esm esmfold2 512 28 ) > "$OUT/stage2_c28.log" 2>&1 &
P1=$!
( run_esm esmfold2-fast 1024 29; run_esm esmfold2-fast 512 29 ) > "$OUT/stage2_c29.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "ESMFOLD2 ROWS DONE $(date -u +%FT%TZ)"

echo "WH PARITY START $(date -u +%FT%TZ)"
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$TREE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py \
    --workers UF-EV-A13-GWH02:28,UF-EV-A13-GWH02:29 \
    --fold-timeout 4800 --fresh --workdir "$TREE/perf/whcut/out/parity-wh" \
    --leg esmc-300m --leg esmc-600m --leg saprot-35m --leg saprot-650m \
    --leg esmfold2-trpcage --leg boltz2-trpcage-nomsa --leg boltz2-prot-nomsa \
    --leg boltz2-prot-msa --leg boltz2-ubiquitin-msa --leg boltz2-hsa-nomsa \
    --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa \
    --leg opendde-trpcage-nomsa --leg boltzgen --leg rfd3-featurizer
echo "STAGE2 EXIT $? $(date -u +%FT%TZ)"
