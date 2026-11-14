"""Cycle-aligned trunk single-track comparison: upstream (40 snapshots/cycle) vs tt-bio (48/cycle).
python -I cyc_cmp.py UP.pt TT.pt [N_TOKEN]; compares the three s-writing blocks of every cycle + cycle end."""
import sys, torch
up = torch.load(sys.argv[1], weights_only=False); tt = torch.load(sys.argv[2], weights_only=False)
n = int(sys.argv[3]) if len(sys.argv) > 3 else up[0].shape[0]
pu, pt = len(up) // 20, len(tt) // 20
small = up[-1].abs().amax(0) < 512
def tok(x): return x - x.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm().clamp_min(1e-30))
for c in range(20):
    for name, ou, ot in (("init", 0, 0), ("w1", pu - 8, pt - 8), ("w2", pu - 5, pt - 5), ("end", pu - 1, pt - 1)):
        u = up[c * pu + ou].double(); t = tt[c * pt + ot][:n].double()
        print(f"cyc {c:2d} {name:4s} absmax up {u.abs().max():9.4g} tt {t.abs().max():9.4g} rel {rel(t,u):.3g} "
              f"tok small {rel(tok(t[:,small]),tok(u[:,small])):.3g} big {rel(tok(t[:,~small]),tok(u[:,~small])):.3g} "
              f"tokstd small up {tok(u[:,small]).std():.3g} tt {tok(t[:,small]).std():.3g}")
