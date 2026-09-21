"""Where does the residual sign disagreement live after the layer_norm_z fix? If one module
dominates it is the next defect; if it tracks element count it is the bf16 forward floor."""
import os, math, numpy as np, torch
from collections import defaultdict
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1o, w1t = load_w(L+"/shipped", 1), load_w(L+"/theirs", 1)
names = sorted(n for n in w1t if n in w1o and PREFIX+n in sd
               and tuple(sd[PREFIX+n].shape) == tuple(w1t[n].shape))
w2o, w2t = load_w(L+"/shipped", 2), load_w(L+"/theirs", 2)
agg = defaultdict(lambda: [0, 0])          # module -> [disagree, compared]
per_tensor = []
for n in names:
    do = (w2o[n].astype(np.float64) - w1o[n].astype(np.float64)).ravel()
    dt = (w2t[n].astype(np.float64) - w1t[n].astype(np.float64)).ravel()
    so, st = np.sign(do), np.sign(dt); m = (so != 0) & (st != 0)
    d = int((so[m] != st[m]).sum()); c = int(m.sum())
    top = n.split(".")[0]
    if top == "diffusion_transformer":
        top = "diffusion_transformer." + n.split(".")[-2]
    agg[top][0] += d; agg[top][1] += c
    if c: per_tensor.append((d / c, c, n))
print("%-52s %12s %10s %8s" % ("group", "compared", "disagree", "rate"))
for k in sorted(agg, key=lambda k: -agg[k][0]):
    d, c = agg[k]
    print("%-52s %12d %10d %7.4f" % (k[:52], c, d, d / c if c else 0))
tot_d = sum(v[0] for v in agg.values()); tot_c = sum(v[1] for v in agg.values())
print("%-52s %12d %10d %7.4f" % ("TOTAL", tot_c, tot_d, tot_d / tot_c))
print()
per_tensor.sort(reverse=True)
print("worst 10 tensors by sign-disagreement rate (compared >= 1000):")
for r, c, n in [p for p in per_tensor if p[1] >= 1000][:10]:
    print("   %.4f  n=%-9d %s" % (r, c, n[-70:]))
print()
print("best 5:")
for r, c, n in [p for p in per_tensor if p[1] >= 1000][-5:]:
    print("   %.4f  n=%-9d %s" % (r, c, n[-70:]))
