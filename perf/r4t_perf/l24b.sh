#!/bin/bash
# L2/L4 default-on grade, second form: PairUpdate n64/r48 seed 7 bf16 arm vs float64 (r4t-tape block_gradcheck +
# D39 floor), ONE merged tree (wk/r4t-perf + wk/r4t-tape) so off/on differ only by the flags. Arms: off, on, off2 (A/A),
# l4 (MMVJP only), l2 (LNBW only), brk (on + --break, must FAIL).   l24b.sh CARD TREE OUT
C=${1:?card}; T=${2:?tree}; OUT=${3:?out}; PY=/home/ttuser/tt-bio-dev/env/bin/python3; mkdir -p $OUT
cd $T; echo "START $(date -u +%FT%TZ) tree $(git rev-parse --short HEAD) = $(git rev-parse --short HEAD^1) + tape $(git rev-parse --short HEAD^2)" >> $OUT/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $OUT/aiclk.log 2>&1 &
CL=$!
for arm in off on off2 l4 l2 brk; do
  case $arm in off|off2) E="0 0";; on|brk) E="1 1";; l4) E="0 1";; l2) E="1 0";; esac; set -- $E; F=""; [ $arm = brk ] && F="--break"
  TT_BIO_LNBW_WEIGHTS=$1 TT_BIO_MMVJP_MIN=$2 PYTHONPATH=$T TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    timeout 2400 $PY perf/r4t_tape/block_gradcheck.py --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 $F \
    --floor perf/r4t_tape/floor/pu_n64r48_s7.json --out $OUT/$arm.json > $OUT/$arm.log 2>&1
  echo "RC_$arm $? $(date -u +%T) lnbw=$1 mmvjp=$2" >> $OUT/run.log
done
kill $CL; echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
