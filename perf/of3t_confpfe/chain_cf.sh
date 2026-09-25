#!/usr/bin/env bash
# of3t-confpfe on qb2 card 0: go384's chain (perf/of3t_go384/chain.sh) with the row, card and
# scratch changed, scored against ref384c, the 384 reference rebuilt with the pae and pde heads
# our step scores (run_ref.sh). Waits for both reference modes before scoring.
W=/home/ttuser/.coworker/wt/of3t-confpfe; S=/home/ttuser/of3t_confpfe/cf; P=$W/perf/of3t_confpfe; L=$S/chain.log
R=/home/ttuser/of3t_confpfe/ref384c; CK=/home/ttuser/of3-weights/of3-p2-155k.pt
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][${CARD:-0}]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain.txt 2>&1 &
M=$!
bash $P/devarm_cf.sh CF384 $W --weights-out $S/weights_walked_CF384.pt >> $L 2>&1
rc=$?
kill $M
cp $S/aiclk_chain.txt $P/AICLK_CF384.txt
[ "$rc" = 0 ] || { echo "=== arm failed rc $rc, not scoring $(date -u +%FT%TZ)" >> $L; exit "$rc"; }
set -e
source /home/ttuser/tt-bio-dev/env/bin/activate
python3 perf/of3t_fullstep64/bijmap.py --weights $S/weights_walked_CF384.pt --checkpoint $CK \
  --out $P/BIJECTION_CF384.json >> $L 2>&1
python3 - "$S/weights_walked_CF384.pt" "$P/DEVICE_SHAPES_CF384.json" <<'PY'
import hashlib, json, sys, torch
w, out = sys.argv[1:]
d = torch.load(w, map_location="cpu")
json.dump({"weights_walked_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
           "shapes": {k: list(v.shape) for k, v in d.items()}, "source": w}, open(out, "w"))
PY
until [ "$(grep -c '=== .* exit 0' $R/chain.log 2>/dev/null)" = 2 ]; do
  grep -q '=== .* exit [1-9]' $R/chain.log 2>/dev/null && { echo "=== reference failed" >> $L; exit 4; }
  sleep 60
done
echo "=== score start $(date -u +%FT%TZ)" >> $L
python3 perf/of3t_fullstep64/score.py --f64 $R/f64/grads_f64.pt --bf16 $R/bf16/grads_bf16.pt \
  --bijection $P/BIJECTION_CF384.json --shapes $P/DEVICE_SHAPES_CF384.json \
  --arm CF384=$S/grad_CF384.pt --out $P/SCORE_CF384.json >> $L 2>&1
python3 perf/of3t_orchestrator/sections/section_ratio.py $P/SCORE_CF384.json CF384 \
  $P/SECTIONS_CF384.json >> $L 2>&1
echo "=== chain done $(date -u +%FT%TZ)" >> $L
