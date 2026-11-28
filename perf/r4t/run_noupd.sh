#!/bin/bash
# D38: the G3 no-update loss-parity arm. One process per dumped batch, `--steps 1`, so every line is
# read at the released weights (parity_from_step.py keeps step 1 only). Two arms in one window:
#   P  r4t-step's own config (run_real.sh env, TT_BIO_SDPA_OWN_FORWARD=1, the default lever set)
#   C  the same with TT_BIO_LEVERS=none: every precision lever the CLI reaches switched off
# P is graded under D37; C only attributes a P failure (lever cost vs defect).
#
#   run_noupd.sh CARD SRC OUT     SRC = a checkout of the code under test, OUT = log dir
# Waits for booking $OWNER (default r4t-accuracy-noupd) to hold CARD with the node free, re-running
# `book.py add` while it waits so the window only moves earlier.
C=${1:?card}; SRC=${2:?src}; OUT=${3:?out}; OWNER=${OWNER:-r4t-accuracy-noupd}
D=${DUMPS:-$HOME/r4t/step}; PY=${PY:-$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python}
BOOK=$HOME/spd/feed/book.py
mkdir -p "$OUT"; cd "$SRC" || exit 1
ok=
for i in $(seq 1 480); do
  who=$(python3 $BOOK active $C 2>/dev/null)
  if [ "$who" = "$OWNER" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ]; then ok=1; break; fi
  [ $((i % 10)) = 1 ] && python3 $BOOK add $OWNER $C ${SECS:-5400} --note "r4t-accuracy D38 no-update arm P+C" >/dev/null 2>&1
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> "$OUT/wait.log"; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" >> "$OUT/run.log"; exit 4; }
echo "START $(date -u +%FT%TZ) card $C src $(cat "$SRC/.sha" 2>/dev/null)" >> "$OUT/run.log"
for arm in P C; do
  for b in 00 01 02 03 04 05 06 07 08; do
    lev=; [ $arm = C ] && lev=none
    env PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-accuracy \
        TT_BIO_SDPA_OWN_FORWARD=1 ${lev:+TT_BIO_LEVERS=$lev} \
        timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --batch $D/batch$b.npz \
        --exempt perf/r4t_step/g2_exempt.tsv --out "$OUT/ck_$arm$b" > "$OUT/${arm}_batch$b.log" 2>&1
    echo "RC_${arm}_$b $? $(date -u +%T)" >> "$OUT/run.log"
    rm -rf "$OUT/ck_$arm$b"
  done
done
echo "END $(date -u +%FT%TZ)" >> "$OUT/run.log"
python3 $BOOK rm $OWNER >/dev/null 2>&1
