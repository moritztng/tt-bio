#!/usr/bin/env bash
# of3t-ieatom: score PF64F (59646c2bb + D264 + D263, card 3) exactly as SCORE_PW64.json and
# SCORE_PF64C.json: bijmap on its walked weights, of3t-fullstep64 score.py against the 384
# float64/bf16 references, score93.py for the 93, and the gradient compare against PW64F.
set -euo pipefail
W=/home/ttuser/.coworker/wt/of3t-ieatom; S=/home/ttuser/of3t_ieatom; P=$W/perf/of3t_ieatom
R=/home/ttuser/of3t_denoise/ref384
CK=/home/ttuser/of3-weights/of3-p2-155k.pt
source /home/ttuser/tt-bio-dev/env/bin/activate
cd $W
python3 perf/of3t_fullstep64/bijmap.py --weights $S/weights_walked_PFF.pt --checkpoint $CK \
  --out $P/BIJECTION_PFF.json
python3 - "$S/weights_walked_PFF.pt" "$P/DEVICE_SHAPES_PFF.json" <<'PY'
import hashlib, json, sys, torch
w, out = sys.argv[1:]
d = torch.load(w, map_location="cpu")
json.dump({"weights_walked_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
           "shapes": {k: list(v.shape) for k, v in d.items()}, "source": w}, open(out, "w"))
PY
python3 perf/of3t_fullstep64/score.py --f64 $R/grads_f64.pt --bf16 $R/grads_bf16.pt \
  --bijection $P/BIJECTION_PFF.json --shapes $P/DEVICE_SHAPES_PFF.json \
  --arm PF64F=$S/grad_PF64F.pt --out $P/SCORE_PF64F.json
python3 perf/of3t_ieatom/score93.py --f64 $R/grads_f64.pt --bf16 $R/grads_bf16.pt \
  --bijection $P/BIJECTION_PFF.json --shapes $P/DEVICE_SHAPES_PFF.json \
  --arm PF64F=$S/grad_PF64F.pt --out $P/SCORE93_PF64F.json
