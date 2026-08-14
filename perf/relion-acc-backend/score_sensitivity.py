"""How much reference error does it take to change which orientation the coarse score picks?

The mechanism by which interpolant error would compound across a refinement is score reordering: the
coarse diff2 selects which orientations reach the fine pass, that sets the assignment, the assignment
builds the map, and the map is the next iteration's reference. If a reference perturbation of size eps
never moves the argmin, it cannot compound through the orientation path.

This is the transfer function, measured on the real shape with the same diff2 code RELION calls. It is
NOT a measurement of the fslice shear. Read it alongside the shear's measured error, do not substitute.

Perturbation: ref <- ref * (1 + eps * g), g standard normal, independently on real and imaginary parts.
Relative and flat in frequency, which is the property the shear's error was measured to have.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-acc-backend")
import tt_bio.cryoem.relion as R

torch.set_grad_enabled(False)
mdlX, mdlY, mdlZ = 100, 199, 199
mdlInitY, mdlInitZ = -99, -99
maxR, pf = 98, 2.0
maxR2p = int(maxR * maxR * pf * pf)
imgX, imgY = 99, 196
P = imgX * imgY
No, Nt = 186, 9
NPART = 12
OUT = "/home/ttuser/relion-scratch/sens.txt"

rng = np.random.default_rng(7)
vol = rng.standard_normal(mdlX * mdlY * mdlZ * 2).astype(np.float32).reshape(-1, 2)
rad = np.linspace(0.02, 1.0, mdlX * mdlY * mdlZ).astype(np.float32)
vol *= (rad ** -1.5)[:, None] * 0.01
mdl = torch.from_numpy(vol)

pix = torch.arange(P, dtype=torch.int64)
x = (pix % imgX).float()
yi = pix // imgX
y = torch.where(yi > maxR, yi - imgY, yi).float()

# Hoisted: the euler sets and the particle data are built once and reused for every eps, so the only
# thing that changes across rows is the perturbation size.
parts = []
for ip in range(NPART):
    g = torch.Generator().manual_seed(1000 + ip)
    eul = torch.empty(No, 9)
    for i in range(No):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        eul[i] = torch.from_numpy(q.reshape(-1).astype(np.float32))
    tx = torch.randn(Nt, generator=g) * 0.05
    ty = torch.randn(Nt, generator=g) * 0.05
    img_r = torch.randn(P, generator=g)
    img_i = torch.randn(P, generator=g)
    w = torch.rand(P, generator=g) * 0.5
    ph = x.unsqueeze(0) * tx.unsqueeze(1) + y.unsqueeze(0) * ty.unsqueeze(1)
    s, c = torch.sin(ph), torch.cos(ph)
    parts.append((eul, c * img_r - s * img_i, c * img_i + s * img_r, w))

# Exact scores once.
refs, exact = [], []
for eul, sh_r, sh_i, w in parts:
    rr, ri = R._project(torch, mdl, x, y, eul, mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2p, pf)
    refs.append((rr, ri))
    dr = rr.unsqueeze(1) - sh_r.unsqueeze(0)
    di = ri.unsqueeze(1) - sh_i.unsqueeze(0)
    exact.append(((dr * dr + di * di) * w).sum(-1))

lines = ["shape No=%d Nt=%d P=%d, %d particles per eps" % (No, Nt, P, NPART),
         "bf16 round-off is ~4e-3 relative and is settled free (+0.000016 A), so 4e-3 is the control row",
         "%-9s %-13s %-15s %-14s" % ("eps", "argmin moved", "orientation chg", "median rel d(diff2)")]
for eps in [float(e) for e in (sys.argv[1:] or ["4e-3", "1e-2", "3e-2", "1e-1", "3e-1"])]:
    moved = orient = 0
    rels = []
    for k, (eul, sh_r, sh_i, w) in enumerate(parts):
        gg = torch.Generator().manual_seed(50000 + k)
        rr, ri = refs[k]
        ar = rr * (1 + eps * torch.randn(rr.shape, generator=gg))
        ai = ri * (1 + eps * torch.randn(ri.shape, generator=gg))
        dr = ar.unsqueeze(1) - sh_r.unsqueeze(0)
        di = ai.unsqueeze(1) - sh_i.unsqueeze(0)
        d = ((dr * dr + di * di) * w).sum(-1)
        rels.append(float(((d - exact[k]).abs() / exact[k].abs().clamp_min(1e-30)).median()))
        ie, ip_ = int(exact[k].argmin()), int(d.argmin())
        moved += ie != ip_
        orient += (ie // Nt) != (ip_ // Nt)
    lines.append("%-9.0e %-13s %-15s %-14.3e"
                 % (eps, "%d/%d" % (moved, NPART), "%d/%d" % (orient, NPART), float(np.median(rels))))
    open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
