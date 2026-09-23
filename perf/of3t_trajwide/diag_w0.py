"""Decompose the k=1 offset the scorer reads. Both sides are differenced against the
float64 checkpoint, so a side whose w_0 is not the checkpoint carries a STATIC offset in
every d_k it reports. Ask which side that is, how big it is, and where it lives."""
import os, sys, glob, math, json
import numpy as np, torch
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"

def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z:
        return {n: z[n] for n in z.files}

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

w1o, w1t = load_w(L + "/shipped", 1), load_w(L + "/theirs", 1)
names = sorted(n for n in w1t if n in w1o and PREFIX + n in sd
               and tuple(sd[PREFIX + n].shape) == tuple(w1t[n].shape))
print("scored names:", len(names))

# dtype the checkpoint actually stores, and what the dumps are
print("ckpt dtypes:", {str(sd[PREFIX+n].dtype) for n in names})
print("ours dump dtypes:", {str(w1o[n].dtype) for n in names})
print("theirs dump dtypes:", {str(w1t[n].dtype) for n in names})

def decomp(w, tag):
    tot = 0.0; per = []
    for n in names:
        b = sd[PREFIX+n].to(torch.float64).numpy()
        d = w[n].astype(np.float64) - b
        s = float(d.ravel() @ d.ravel())
        bn = float(b.ravel() @ b.ravel())
        tot += s
        per.append((math.sqrt(s), n, math.sqrt(bn), w[n].size))
    per.sort(reverse=True)
    nz = [p for p in per if p[0] > 0]
    print("\n[%s] ||w1 - ckpt|| = %.6e over %d tensors; %d tensors nonzero" %
          (tag, math.sqrt(tot), len(names), len(nz)))
    for a, n, bn, sz in per[:12]:
        rel = a / bn if bn else float("nan")
        # per-element rms relative, the bf16 tell: 2**-9 = 1.95e-03
        print("   %.6e  rel=%.3e  n=%-7d  %s" % (a, rel, sz, n))
    return per

po = decomp(w1o, "ours/shipped k=1")
pt = decomp(w1t, "theirs k=1")

# Is the offset STATIC? compare our k=1 vs our k=20 residual against the checkpoint,
# restricted to the tensors that carry it.
w20o = load_w(L + "/shipped", 20)
top = [p[1] for p in po[:5]]
print("\nstatic-offset check on the 5 largest carriers, ours k=1 vs k=20 vs ckpt:")
for n in top:
    b = sd[PREFIX+n].to(torch.float64).numpy()
    d1 = w1o[n].astype(np.float64) - b
    d20 = w20o[n].astype(np.float64) - b
    moved = d20 - d1
    print("   %-64s ||d1||=%.4e ||d20||=%.4e ||d20-d1||=%.4e" %
          (n[-64:], np.linalg.norm(d1), np.linalg.norm(d20), np.linalg.norm(moved)))

# bf16 round-trip test: is our k=1 value exactly the bf16 rounding of the checkpoint?
import ml_dtypes
print("\nbf16 round-trip test on the 5 largest carriers (ours k=1 vs bf16(ckpt)):")
for n in top:
    b = sd[PREFIX+n].to(torch.float64).numpy()
    bf = b.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    o = w1o[n].astype(np.float64)
    print("   %-52s ||ours-bf16(ckpt)||=%.4e  ||ours-ckpt||=%.4e  exact=%s" %
          (n[-52:], np.linalg.norm(o - bf), np.linalg.norm(o - b),
           bool(np.array_equal(o, bf))))
