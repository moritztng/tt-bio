#!/bin/bash
# bc4-bet-tape8 sweep1 on qb2 card 3: main vs precision's stack vs bfp8 cotangents vs both.
# float64 VJP of the cotangent arm (+ forced grid), census per arm, timing x3 interleaved, reps 6.
WT=/home/ttuser/.coworker/wt/bc4-bet-tape8; cd $WT
D=$WT/_bc4out/sweep1_$(git rev-parse --short HEAD); mkdir -p $D; echo $$ > $D/pid
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bc4-bet-tape8 TT_BIO_LEASE_TIMEOUT=3600
( while :; do printf "%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!3/tt_aiclk)"; sleep 1; done ) >> $D/aiclk.tsv 9>&- & P=$!
ST="TT_BIO_TRIMUL_INPROJ_B8=1 TT_BIO_TRIMUL_INPROJ_B8_BIAS=1 TT_BIO_TRIATT_QKV_B8=1"
CT="TT_BIO_COTAN_B8=1"
PY=$HOME/bcx_e2e_venv/bin/python
vjp() { tag=$1; arm=$2; shift 2; echo "vjp_$arm$tag start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
  env "$@" timeout -s INT -k 120 2400 $PY perf/bc4_precision/vjp_grade.py --arms $arm --tag "$tag" > $D/vjp_$arm$tag.log 2>&1
  echo "vjp_$arm$tag rc=$? end $(date +%s)" >> $D/card; cp perf/bcx_afgrad/vjp_n224_bc4p_${arm}_s0$tag.json $D/ 2>/dev/null; }
run() { arm=$1; shift; echo "$arm start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
  env "$@" timeout -s INT -k 120 6000 $PY perf/spd/bc2_evo_devprof.py \
     --params $HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz --out $D/$arm --warm 2 --reps 6 $EXTRA > $D/$arm.log 2>&1
  echo "$arm rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card; }
EXTRA=--census
run m_cen X=0; run cot_cen $CT; run stack_cen $ST; run all_cen $ST $CT
EXTRA=
for r in 0 1 2; do run m$r X=0; run stack$r $ST; run cot$r $CT; run all$r $ST $CT; done
vjp _cot off $CT
vjp _cot_g11 off $CT TT_BIO_FORCE_GRID=11,10
vjp "" off X=0
kill $P; touch $D/done
