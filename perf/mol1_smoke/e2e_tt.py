"""Mol-1 end to end on TT at several seeds, upstream defaults (20 cycles, 200 steps, 1 sample).
Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/e2e_tt.py CKPT SEQ OUT_PT SEED[,SEED...]
MOL1_A3M=<path> folds with that MSA. Saves {seed: all-atom coords} plus per-fold seconds and the AICLK sampled during each fold."""
import os
import sys
import time

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from perf.clocksample import during  # noqa: E402
from tt_bio.protenix import Protenix  # noqa: E402
from tt_bio.protenix_data import build_protein_features  # noqa: E402

ckpt, seq, out, seeds = sys.argv[1:5]
model = Protenix.load_from_checkpoint(ckpt)
a3m = os.environ.get("MOL1_A3M")
feats = build_protein_features(seq, open(a3m).read() if a3m else None)
res = {"coords": {}, "s": {}, "clock": {}, "plddt": {}}
for sd in [int(x) for x in seeds.split(",")]:
    with during() as clk, torch.no_grad():
        t = time.time()
        x, conf = model.fold(feats, n_step=200, n_sample=1, seed=sd, n_cycles=20, return_confidence=True)
        res["s"][sd] = round(time.time() - t, 2)
    res["coords"][sd] = torch.as_tensor(x).reshape(-1, 3).float()
    res["clock"][sd] = clk.summary().get(0)
    c = conf[0] if isinstance(conf, list) else conf
    res["plddt"][sd] = float(torch.as_tensor(c["plddt"]).float().mean())
    print(sd, res["s"][sd], res["plddt"][sd], clk.line(), flush=True)
torch.save(res, out)
