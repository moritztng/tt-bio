#!/bin/bash
# R1's kill measurement: the 512 aa Boltz-2 cell on `main` and on wk/train-a1-defork, grad off.
# Arms interleaved ABBA so host drift does not land on one of them. The digest is the answer;
# the median is the perf half of the same question. AICLK sampled during every fold.
set -u
BR=/home/ttuser/.coworker/wt/train-a1-defork
MN=/home/ttuser/train-a1-ab-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$BR/perf/train_a1_defork/out
mkdir -p "$OUT"
run() {  # run <tree> <tag>
  cd "$1" || exit 1
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      TT_BIO_LEASE_HOLDER=worker:train-a1-defork PYTHONPATH="$1" \
      "$PY" perf/b2z2_cell_recheck/cell_now.py \
        --out "$OUT/cell_$2.json" --reps 3 --warm 1 --clock "tenstorrent!0" \
        --allow-cotenant 2>&1 | grep -viE "^Config\{|DEBUG *\| *ttnn"
  echo "RC[$2]=${PIPESTATUS[0]}"
}
for r in 1 2; do
  echo "=== round $r : MAIN ==="; run "$MN" "main_r$r"
  echo "=== round $r : DEFORK ==="; run "$BR" "defork_r$r"
done
echo ABCELLDONE
