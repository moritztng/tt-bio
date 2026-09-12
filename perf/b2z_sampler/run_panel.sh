#!/usr/bin/env bash
# Fan the panel across one chip per target. Each process holds exactly one card (TT_BIO_LEASE_CARDS
# pins it: an unpinned open brings up every visible chip), loads the model once and folds its whole
# grid. Results land under results/ after every fold, so a run that is cut short still lands what
# it measured.
set -u
WT=/home/tt-admin/wt/b2z-sampler-steps
PY=/home/mthuening/work/tt-bio/env/bin/python
OUTD=$WT/perf/b2z_sampler/results
LOGD=/home/tt-admin/wt/b2z-sampler-steps/perf/b2z_sampler/logs
mkdir -p "$OUTD" "$LOGD" "$OUTD/cif"
i=0
for T in cdk2x2_512 cdk2x2_298 cdk2x2_128 affinity_fkg affinity_dhfr affinity_tryp 9bk6 8hel 7xi5 hsa; do
  CARD=$((7 + i))
  i=$((i + 1))
  ( cd "$WT" && TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:b2z-sampler-steps \
      "$PY" perf/b2z_sampler/sweep.py --target "$T" \
        --out "$OUTD/$T.json" --cifdir "$OUTD/cif" ) \
    > "$LOGD/$T.log" 2>&1 &
  echo "launched $T on card $CARD pid $!"
done
wait
echo "PANEL DONE"
