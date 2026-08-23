#!/usr/bin/env bash
# od512 re-run, 8 pairs. The 5-pair run at 20:10 was noise-dominated: per-fold loadavg reached
# 13.88 and the within-arm spread was 1.43 s against this model 0.05-0.13 s quiet-box floor.
# od298 at 20:34 showed the instrument is tight (0.070 s within-arm) when loadavg stays under ~6,
# so this waits for that and then folds. Same tt_baseline.py --ab-env, no new harness.
set -u
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=0
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=90

# Wait for the previous chain to be gone so two processes never open card 0 under one holder.
while pgrep -f run_measure.sh >/dev/null; do echo "$(date -Is) waiting for run_measure.sh"; sleep 15; done

# Wait for loadavg <= 6, the band od298 was measured in, up to 20 min.
t0=$SECONDS
while [ $((SECONDS-t0)) -lt 1200 ]; do
  l=$(cut -d" " -f1 /proc/loadavg)
  awk -v a="$l" "BEGIN{exit !(a+0<=6.0)}" && { echo "$(date -Is) load $l, folding"; break; }
  echo "$(date -Is) load $l, waiting"; sleep 20
done

bash /home/ttuser/.coworker/scripts/benchlock.sh protenix-opendde-token-bucket-flip-measure -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure \
      PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm \
  "$PY" -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 8 \
    --ab-env TT_BIO_PROTENIX_TOKEN_BUCKET --ab-values 1,0 \
    --target perf/size512/fixtures/cdk2x2_512.yaml --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "od512 rerun 8 pairs, page protocol, qb1 card 0, paired token-bucket A/B" \
    --msa-dir $WT/.msa_tokenbucket --keep-cif perf/tokenbucket/cif_od512r \
    --out perf/tokenbucket/od512_paired.json
echo "=== $(date -Is) od512 rerun END rc=$? ==="
