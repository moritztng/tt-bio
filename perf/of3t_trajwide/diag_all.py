"""Prove the k=1 offset is the load-time bf16 quantisation and nothing else, over ALL
scored tensors, and ask whether the narrow rungs scope could ever have seen it."""
import os, math, numpy as np, torch, ml_dtypes
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1o, w1t = load_w(L+"/shipped", 1), load_w(L+"/theirs", 1)
names = sorted(n for n in w1t if n in w1o and PREFIX+n in sd
               and tuple(sd[PREFIX+n].shape) == tuple(w1t[n].shape))
exact_bf16 = exact_f32 = neither = 0
bf16_names, other = [], []
for n in names:
    b = sd[PREFIX+n].to(torch.float64).numpy()
    o = w1o[n].astype(np.float64)
    if np.array_equal(o, b):
        exact_f32 += 1
    elif np.array_equal(o, b.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)):
        exact_bf16 += 1; bf16_names.append(n)
    else:
        neither += 1; other.append(n)
print("scored tensors           :", len(names))
print("our k=1 == ckpt exactly  :", exact_f32)
print("our k=1 == bf16(ckpt)    :", exact_bf16)
print("neither                  :", neither, other[:6])
# so: did OUR optimizer move anything at k=1? it moved nothing iff neither==0.
print("=> zero-motion at k=1 on our side:", neither == 0)
print()
# which module owns the bf16 set
from collections import Counter
c = Counter(n.split(".")[0] for n in bf16_names)
print("bf16-resident tensors by top-level module:", dict(c))
c2 = Counter(n.split(".")[0] for n in names)
print("all scored tensors by top-level module   :", dict(c2))
cond = [n for n in bf16_names if n.startswith("diffusion_conditioning")]
print()
print("bf16-resident tensors under diffusion_conditioning (the narrow rungs scope):", len(cond))
print("  => the narrow rung could see this defect:", len(cond) > 0)
