#!/usr/bin/env bash
# of3t-go384 on qb2 card 3: arm GO384 on this worktree's tree (walked weights dumped), the
# bijection and device shapes built from them as of3t-ieatom's score_pf64f.sh does, then
# of3t-fullstep64 score.py against the 384 float64 reference with upstream's bf16 as the bar.
W=/home/ttuser/.coworker/wt/of3t-go384; S=/home/ttuser/of3t_go384; P=$W/perf/of3t_go384; L=$S/chain.log
R=/home/ttuser/of3t_denoise/ref384; CK=/home/ttuser/of3-weights/of3-p2-155k.pt
cd $W
(while true; do echo "$(date -u +%T) $(/home/ttuser/.local/bin/tt-smi -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][3]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $S/aiclk_chain.txt 2>&1 &
M=$!
bash $P/devarm.sh GO384 $W --weights-out $S/weights_walked_GO384.pt >> $L 2>&1
rc=$?
kill $M
cp $S/aiclk_chain.txt $P/AICLK_GO384.txt
[ "$rc" = 0 ] || { echo "=== arm failed rc $rc, not scoring $(date -u +%FT%TZ)" >> $L; exit "$rc"; }
set -e
source /home/ttuser/tt-bio-dev/env/bin/activate
echo "=== score start $(date -u +%FT%TZ)" >> $L
python3 perf/of3t_fullstep64/bijmap.py --weights $S/weights_walked_GO384.pt --checkpoint $CK \
  --out $P/BIJECTION_GO384.json >> $L 2>&1
python3 - "$S/weights_walked_GO384.pt" "$P/DEVICE_SHAPES_GO384.json" <<'PY'
import hashlib, json, sys, torch
w, out = sys.argv[1:]
d = torch.load(w, map_location="cpu")
json.dump({"weights_walked_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
           "shapes": {k: list(v.shape) for k, v in d.items()}, "source": w}, open(out, "w"))
PY
python3 perf/of3t_fullstep64/score.py --f64 $R/grads_f64.pt --bf16 $R/grads_bf16.pt \
  --bijection $P/BIJECTION_GO384.json --shapes $P/DEVICE_SHAPES_GO384.json \
  --arm GO384=$S/grad_GO384.pt --out $P/SCORE_GO384.json >> $L 2>&1
echo "=== chain done $(date -u +%FT%TZ)" >> $L
