#!/bin/bash
# The G3 curve leg on one booked BH chip: trunk_step --curve-dump over the committed 2,000-step plan's
# dumped inputs, checkpoint every 10 steps (ring of 3, 1.44 GB each; the one a loss spike > 5x the
# trailing median started from is pinned so it replays), and on a crash resume from the newest checkpoint written
# since the last attempt (the CurveLog truncates to the resume step, so the log stays one gap-free run).
#
#   run_g3_curve.sh CARD SRC DUMP OUT     SRC = a checkout of the code under test (.sha beside it)
# Waits for booking $OWNER to hold CARD with the node free, re-running `book.py add` while it waits so
# the window only moves earlier. Done flag: an END or GAVE-UP line in OUT/run.log.
C=${1:?card}; SRC=${2:?src}; DUMP=${3:?dump}; OUT=${4:?out}; OWNER=${OWNER:-r4t-accuracy-g3}
PY=${PY:-$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python}
BOOK=$HOME/spd/feed/book.py
mkdir -p "$OUT"; cd "$SRC" || exit 1
ok=
for i in $(seq 1 2880); do
  who=$(python3 $BOOK active $C 2>/dev/null)
  if [ "$who" = "$OWNER" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ]; then ok=1; break; fi
  [ $((i % 10)) = 1 ] && python3 $BOOK add $OWNER $C ${SECS:-34200} --note "r4t-accuracy G3 curve leg, 2000 steps" >/dev/null 2>&1
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> "$OUT/wait.log"; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" >> "$OUT/run.log"; exit 4; }
sha=$(cat "$SRC/.sha" 2>/dev/null)
echo "START $(date -u +%FT%TZ) card $C src $sha" >> "$OUT/run.log"
for try in 1 2 3 4; do
  last=$(ls -1 "$OUT"/ck/adapter-*.safetensors 2>/dev/null | tail -1)
  env PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-accuracy \
      TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_SHA=$sha \
      $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --curve-dump "$DUMP" --curve-out "$OUT/ours.jsonl" \
      --save-every 10 --keep 3 --pin-spike 5 --exempt perf/r4t_step/g2_exempt.tsv --out "$OUT/ck" ${last:+--resume "$last"} \
      >> "$OUT/leg.log" 2>&1
  rc=$?
  echo "RC try$try $rc $(date -u +%T) resumed_from=${last:-none}" >> "$OUT/run.log"
  [ $rc = 0 ] && { echo "END $(date -u +%FT%TZ)" >> "$OUT/run.log"; break; }
  # Resume only past a NEW checkpoint: a failure before step 10 (a G2 guard refusal included, which a
  # resumed run would no longer check) or one that made no progress since the last resume stops here.
  now=$(ls -1 "$OUT"/ck/adapter-*.safetensors 2>/dev/null | tail -1)
  if [ $try = 4 ] || [ -z "$now" ] || [ "$now" = "$last" ]; then echo "GAVE-UP $(date -u +%FT%TZ)" >> "$OUT/run.log"; break; fi
done
python3 $BOOK rm $OWNER >/dev/null 2>&1
