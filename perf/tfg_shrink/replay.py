"""Replay recorded TT steps (x_noisy, x0) through Guidance on CPU, dense (core off) vs fast (auto) vs recorded x.
Each step is fed its RECORDED inputs, so the comparison is per step, not closed loop."""
import sys, torch
from tt_bio.tfg.guidance import Guidance
from tt_bio.tfg.rigid import RigidSchedule
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
p = sys.argv[1]
d = torch.load(p + "_traj.pt", weights_only=False); f = torch.load(p + "_feats.pt", weights_only=False)
steps = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(len(d["x"]))
g = {c: Guidance(f, schedule=RigidSchedule(core=c)) for c in ("off", "auto")}
n = len(d["x"])
for k in steps:
    kw = dict(t_hat=d["t_hat"][k], sigma_t=d["sigma_t"][k], eta=d["eta"][k], step=k, n_step=n)
    out = {c: g[c]._step(d["x_noisy"][k], d["x0"][k], **kw) for c in g}
    rec = d["x"][k]
    print(k, "dense-auto %.2e" % (out["off"] - out["auto"]).abs().max(), "auto-rec %.2e" % (out["auto"] - rec).abs().max(),
          "dense-rec %.2e" % (out["off"] - rec).abs().max(), flush=True)
