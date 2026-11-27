#!/bin/bash
# G1 gate_bw default-on grade: PairUpdate n64/r48 seed 7 bf16 arm vs float64 (r4t-tape block_gradcheck + D39 floor),
# lnbw on (training default), mmvjp off; arms differ only by TT_BIO_GATE_BW_FUSED. off, on, off2 (A/A),
# brk (on + a 1.10x gradient on tri_mult_out.linear_g.weight, must FAIL).   gateg.sh CARD TREE OUT
C=${1:?card}; T=${2:?tree}; OUT=${3:?out}; PY=/home/ttuser/tt-bio-dev/env/bin/python3; mkdir -p $OUT
cd $T; echo "START $(date -u +%FT%TZ) tree $(git rev-parse --short HEAD) load $(cut -d" " -f1 /proc/loadavg)" >> $OUT/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $OUT/aiclk.log 2>&1 &
CL=$!
for arm in off on off2 brk; do
  case $arm in off|off2) G=0;; on|brk) G=1;; esac; F=""; [ $arm = brk ] && F="--scale-grad pu.tri_mult_out.linear_g.weight"
  TT_BIO_GATE_BW_FUSED=$G TT_BIO_LNBW_WEIGHTS=1 TT_BIO_MMVJP_MIN=0 PYTHONPATH=$T TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    timeout 2400 $PY perf/r4t_tape/block_gradcheck.py --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 $F \
    --floor perf/r4t_tape/floor/pu_n64r48_s7.json --out $OUT/$arm.json > $OUT/$arm.log 2>&1
  echo "RC_$arm $? $(date -u +%T) gate=$G" >> $OUT/run.log
done
kill $CL; echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
