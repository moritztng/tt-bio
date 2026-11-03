"""Per-cycle trunk single track: every WH tree (and upstream, if given) on the 8 small channels' token part.
python -I tri_cmp.py REF_LAYERS.pt NAME=TT_LAYERS.pt ... ; REF is upstream (40/cycle) or a tt-bio dump (48/cycle)."""
import sys, torch
ref = torch.load(sys.argv[1], weights_only=False)
tts = [(a.split("=")[0], torch.load(a.split("=")[1], weights_only=False)) for a in sys.argv[2:]]
n = ref[0].shape[0]
pr = len(ref) // 20
small = ref[-1].abs().amax(0) < 512
print("small channels:", int(small.sum()), small.nonzero().flatten().tolist())
def tok(x): return x - x.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm().clamp_min(1e-30))
for c in range(20):
    line = f"cyc {c:2d}"
    for name, off in (("w1", pr - 8), ("w2", pr - 5), ("end", pr - 1)):
        u = ref[c * pr + off].double()
        line += f" | {name} ref std {tok(u[:, small]).std():.3g}"
        for nm, tt in tts:
            pt = len(tt) // 20
            t = tt[c * pt + off - pr + pt][:n].double()
            line += f" {nm} {rel(tok(t[:, small]), tok(u[:, small])):.3g}"
    print(line)
