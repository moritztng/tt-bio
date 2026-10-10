"""What the rigid x0 pass does to the movable group along a traced guided fold (trace.py output).

Per early-pass step and sample: severe movable-fixed overlaps in the denoiser's x0 (the search scores such an entry pose
inf and always re-docks it), the rotation from x0's movable pose to the guided one (Kabsch angle), and the movable-group
Rg in x0, in the guided step output x and in the next step's x0. The guided x0 is recomputed on CPU from the recorded x0.
usage: python redock.py DIR/<n> [threads]
"""
import math
import sys

import torch

from tt_bio.tfg import epitope, rigid
from tt_bio.tfg.rigid import RigidSchedule

p = sys.argv[1]
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
d = torch.load(p + "_traj.pt", weights_only=False)
f = torch.load(p + "_feats.pt", weights_only=False)
x0s, xs = d["x0"], d["x"]
fixed, moving, _, _ = rigid.contact_groups(x0s[0], f)
rad = torch.as_tensor(rigid.rdkit_vdws, dtype=torch.float32)[f["ref_element"].argmax(-1)]
rsum = rad[moving][:, None] + rad[fixed][None, :]


def rg(x):
    c = x - x.mean(-2, keepdim=True)
    return c.square().sum(-1).mean(-1).sqrt()


def angle(a, b):
    """Rotation angle (deg) of the best rigid fit of a onto b, both [n, 3]."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    s = torch.ones(3)
    s[2] = torch.sign(torch.det(u @ vh))
    R = u @ torch.diag(s) @ vh
    return math.degrees(math.acos(max(-1.0, min(1.0, (torch.trace(R).item() - 1) / 2))))


sch = RigidSchedule()
S = x0s.shape[1]
print("step | per sample: severe pairs in x0, rotation x0->guided (deg), Rg moving x0 -> x -> next x0")
for k in range(x0s.shape[0]):
    if not rigid.x0_step_active(k, sch):
        continue
    x0 = x0s[k]
    g = epitope.guide_x0(x0.clone(), f, k, sch)
    dist = torch.cdist(x0[:, moving], x0[:, fixed])
    sev = (dist < 0.75 * rsum).flatten(1).sum(-1)
    nxt = x0s[k + 1] if k + 1 < x0s.shape[0] else x0
    cells = []
    for s in range(S):
        moved = (g[s, moving] - x0[s, moving]).abs().max().item() > 0
        ang = angle(x0[s, moving], g[s, moving]) if moved else 0.0
        cells.append(f"{int(sev[s]):3d} {ang:5.1f} {rg(x0[s, moving]):5.1f}>{rg(xs[k][s, moving]):5.1f}>{rg(nxt[s, moving]):5.1f}")
    print(f"{k:3d} t={d['t_hat'][k]:7.3f} | " + " | ".join(cells), flush=True)
fin = rg(xs[-1][:, moving])
print("final moving Rg", [round(v, 2) for v in fin.tolist()])
