#!/bin/bash
# D43(2) follow-up: only 5 levers serve at crop 256 (dit_sigf32 pair_bf16z tok_recycle_mm tok_w8 transition_b8; the
# other 28 leave P digests bit-identical, pair_mm/pair_mm_tr off crash). transition_b8 off passes b00/b07 but fails
# b08. Grade every serving lever off, and transition_b8 paired with each other one, on all 9 batches.
#   run9x.sh CARD
C=${1:?card}; B=$HOME/r4t/perf/bis; SRC=$B/src; OUT=$B/t9x; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
ALL=$(tr " " "\n" < $B/levers.txt | grep -v "^$")
mkdir -p $OUT; cd $SRC
while pgrep -f gate_default.sh >/dev/null; do sleep 15; done
echo "START $(date -u +%FT%TZ) card $C src $(cat .sha)" >> $OUT/run.log
for arm in dit_sigf32 pair_bf16z tok_recycle_mm tok_w8 transition_b8+tok_recycle_mm transition_b8+pair_bf16z transition_b8+tok_w8 transition_b8+dit_sigf32 five; do
  drop=$(echo $arm | tr + "\n"); [ $arm = five ] && drop="dit_sigf32 pair_bf16z tok_recycle_mm tok_w8 transition_b8"
  lev=$(echo $ALL | tr " " "\n" | grep -vxF "$(echo $drop | tr " " "\n")" | paste -sd,)
  for b in 00 01 02 03 04 05 06 07 08; do
    for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
    env PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
      TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_LEVERS=$lev \
      timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --batch $D/batch$b.npz \
      --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck_$arm$b > $OUT/${arm}_batch$b.log 2>&1
    echo "RC_${arm}_$b $? $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck_$arm$b
  done
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
