"""Movable-group geometry along a traced guided fold (trace.py output).
usage: python traj_an.py DIR/<n>   -> per step, per sample: Rg of moving group in x_noisy, x0, x; min singular value
of the best linear map from the step-0-free reference (final x0 of the least-shrunk sample)."""
import sys, torch
from tt_bio.tfg import rigid
p = sys.argv[1]
d = torch.load(p + "_traj.pt", weights_only=False); f = torch.load(p + "_feats.pt", weights_only=False)
fixed, moving, _, _ = rigid.contact_groups(d["x0"][0], f)
def rg(x):  # [.., N, 3]
    c = x - x.mean(-2, keepdim=True); return c.square().sum(-1).mean(-1).sqrt()
xn, x0, x = d["x_noisy"], d["x0"], d["x"]
S = x.shape[1]
print("steps", x.shape[0], "samples", S, "moving atoms", len(moving), "fixed", len(fixed))
fin = rg(x[-1][:, moving]); print("final moving Rg per sample", [round(v, 2) for v in fin.tolist()],
                                  "fixed Rg", [round(v, 2) for v in rg(x[-1][:, fixed]).tolist()])
early = set(range(100, 190, 3))
for k in list(range(0, 100, 10)) + list(range(100, 200)):
    a = rg(xn[k][:, moving]); b = rg(x0[k][:, moving]); c = rg(x[k][:, moving])
    print(f"{k:3d}{'*' if k in early else ' '} t={d['t_hat'][k]:8.3f}", " | ".join(
        f"{a[s]:5.1f} {b[s]:5.1f} {c[s]:5.1f}" for s in range(S)))
