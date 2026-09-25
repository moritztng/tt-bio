#!/usr/bin/env bash
# of3t-confpfe: RF384 (pre-fix tree, the arm launched as CF384 at 01:05Z) scored against ref384c,
# as chain.sh would have scored it; the chain was stopped before its score step (PREREGISTERED.md).
set -e
W=/home/ttuser/.coworker/wt/of3t-confpfe; S=/home/ttuser/of3t_confpfe; P=$W/perf/of3t_confpfe; R=$S/ref384c
CK=/home/ttuser/of3-weights/of3-p2-155k.pt; L=$S/score_rf.log
cd $W
source /home/ttuser/tt-bio-dev/env/bin/activate
python3 perf/of3t_fullstep64/bijmap.py --weights $S/weights_walked_RF384.pt --checkpoint $CK \
  --out $P/BIJECTION_RF384.json >> $L 2>&1
python3 - "$S/weights_walked_RF384.pt" "$P/DEVICE_SHAPES_RF384.json" <<'PY'
import hashlib, json, sys, torch
w, out = sys.argv[1:]
d = torch.load(w, map_location="cpu")
json.dump({"weights_walked_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
           "shapes": {k: list(v.shape) for k, v in d.items()}, "source": w}, open(out, "w"))
PY
python3 perf/of3t_fullstep64/score.py --f64 $R/f64/grads_f64.pt --bf16 $R/bf16/grads_bf16.pt \
  --bijection $P/BIJECTION_RF384.json --shapes $P/DEVICE_SHAPES_RF384.json \
  --arm RF384=$S/grad_RF384.pt --out $P/SCORE_RF384.json >> $L 2>&1
python3 perf/of3t_orchestrator/sections/section_ratio.py $P/SCORE_RF384.json RF384 \
  $P/SECTIONS_RF384.json >> $L 2>&1
echo "=== score_rf done $(date -u +%FT%TZ)" >> $L
