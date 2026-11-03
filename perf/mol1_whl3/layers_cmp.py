"""Per-PairformerBlock trunk single track, tt-bio vs upstream fp32: where WH's small channels lose their token signal.

    python -I layers_cmp.py UP_LAYERS.pt TT_LAYERS.pt [N_TOKEN]
For each block: overall rel err, and the token part (s minus its token mean) rel err on the channels whose upstream
final |s| stays under 512 (the only ones the confidence head's clamp lets through) and on the rest.
"""
import sys, torch
up = torch.load(sys.argv[1], weights_only=False)
tt = torch.load(sys.argv[2], weights_only=False)
n = int(sys.argv[3]) if len(sys.argv) > 3 else up[0].shape[0]
print(f"blocks: upstream {len(up)}, tt {len(tt)}")
small = up[-1].abs().amax(0) < 512
def tok(x): return x - x.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm().clamp_min(1e-30))
for i, (u, t) in enumerate(zip(up, tt)):
    t = t[:n]
    print(f"{i:4d} absmax up {u.abs().max():9.4g} tt {t.abs().max():9.4g}  rel {rel(t, u):.3g}  "
          f"token-part rel small {rel(tok(t[:, small]), tok(u[:, small])):.3g} big {rel(tok(t[:, ~small]), tok(u[:, ~small])):.3g}")
