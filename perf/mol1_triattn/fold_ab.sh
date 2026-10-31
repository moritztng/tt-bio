#!/bin/sh
# Mol-1 fold A/B for triangle attention: base (protenix.NORMAL_LEVERS, wide default off = the tree before this row)
# vs + triatt_ln + triatt_k1 with the any-width fixes. Random-weight Mol-1 checkpoint (timing only), single sequence,
# upstream defaults (20 cycles, 200 steps, 1 sample), seeds 101,102: the second fold is the warm one.
# Runs after chain3 on qb1 card 3. Output: fold_ab/<arm>_<S>.{pt,log}
W=/home/ttuser/.coworker/wt/mol1-perf-triattn
T=$W/.foldtree
while kill -0 1893781 2>/dev/null; do sleep 20; done
mkdir -p $W/perf/mol1_triattn/fold_ab
cd $T; export PYTHONPATH=$T
/home/ttuser/tt-bio-dev/env/bin/python3 -c "import tt_bio,sys; assert tt_bio.__file__.startswith(\"$T\"), tt_bio.__file__" || exit 3
BASE=normal+transition_ws+ws_l1acc+trimul_t1_c256+trimul_t1_c384
for S in 256 512 1024; do
  SEQ=$(python3 -c "import random;random.seed($S);print(''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range($S)))")
  for ARM in base triatt; do
    if [ $ARM = base ]; then LV=$BASE; WIDE=0; else LV=$BASE+triatt_ln+triatt_k1; WIDE=1; fi
    O=$W/perf/mol1_triattn/fold_ab/${ARM}_$S
    PYTHONPATH=$T TT_BIO_LEVERS=$LV TT_BIO_MM_DEFAULT_WIDE=$WIDE TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:mol1-perf-triattn \
      timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 perf/mol1_smoke/e2e_tt.py /home/ttuser/mol1trimul/mol1_random.pt "$SEQ" $O.pt 101,102 > $O.log 2>&1
    echo "rc=$?" >> $O.log
  done
done
echo ALLDONE > $W/perf/mol1_triattn/fold_ab/DONE
