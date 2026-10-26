"""bg_rngcheck.py DESIGN_ARGS... : featurize the BoltzGen page designs twice on CPU (no device opens), perturbing torch's
global RNG between designs the second time. The features must match: the featurizer may draw only from numpy's stream.

    TT_VISIBLE_DEVICES=0 python perf/spd/bg_rngcheck.py design perf/dsfix/fixtures/bg_R3.yaml --model boltzgen \
        --steps design --num_designs 3 --out_dir OUT --seed 0      # RNGCHECK lines land in OUT/run.log
"""
import hashlib, runpy, sys, numpy as np, torch
import ttnn
def _no_device(*a, **k): raise RuntimeError("rngcheck must not open a device")
for _n in ("open_device", "open_mesh_device", "CreateDevice", "CreateDevices"):
    if hasattr(ttnn, _n): setattr(ttnn, _n, _no_device)
from tt_bio.boltzgen.task.predict import predict as P

def run(self, config=None, run_prediction=True):
    out = []
    for perturb in (False, True):
        np.random.seed(0); torch.manual_seed(0)
        self.data.num_workers = 0
        for i, b in enumerate(self.data.predict_dataloader()):
            if perturb: torch.randn(1000)
            h = hashlib.sha256()
            for k in sorted(b):
                v = b[k]
                if torch.is_tensor(v): h.update(k.encode()); h.update(v.cpu().contiguous().numpy().tobytes())
            out.append((perturb, i, h.hexdigest()[:16]))
    print("RNGCHECK", out, flush=True)
    a = [x[2] for x in out if not x[0]]; b = [x[2] for x in out if x[0]]
    print("RNGCHECK same_features_under_perturb", a == b, flush=True)
    raise SystemExit(0)
P.Predict.run = run
sys.argv = ["tt-bio"] + sys.argv[1:]
runpy.run_module("tt_bio.main", run_name="__main__")
