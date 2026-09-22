"""Split the k=20 trajectory error by on-device storage dtype. 264 of the 549 scored
tensors are bf16-resident and 285 are fp32. If the divergence is a bf16 effect it lives
in the 264; if it is broad it does not."""
import os, math, numpy as np, torch, ml_dtypes
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1o, w1t = load_w(L+"/shipped", 1), load_w(L+"/theirs", 1)
names = sorted(n for n in w1t if n in w1o and PREFIX+n in sd
               and tuple(sd[PREFIX+n].shape) == tuple(w1t[n].shape))
bf16 = set()
for n in names:
    b = sd[PREFIX+n].to(torch.float64).numpy()
    if not np.array_equal(w1o[n].astype(np.float64), b):
        bf16.add(n)
K = 20
w20o, w20t = load_w(L+"/shipped", K), load_w(L+"/theirs", K)
def agg(sel, tag):
    num = den = onorm = 0.0; cos_n = 0.0
    for n in sel:
        do = w20o[n].astype(np.float64) - w1o[n].astype(np.float64)
        dt = w20t[n].astype(np.float64) - w1t[n].astype(np.float64)
        d = do - dt
        num += float(d.ravel() @ d.ravel()); den += float(dt.ravel() @ dt.ravel())
        onorm += float(do.ravel() @ do.ravel()); cos_n += float(do.ravel() @ dt.ravel())
    num, den, onorm = math.sqrt(num), math.sqrt(den), math.sqrt(onorm)
    cos = cos_n / (onorm * den) if onorm and den else float("nan")
    print("%-18s n=%3d  rel_d=%.6e  ||d_ours||=%.4e ||d_theirs||=%.4e  ratio=%.4f  cos=%.4f  angle=%.1f deg"
          % (tag, len(sel), num/(den+1e-30), onorm, den, onorm/den if den else float("nan"),
             cos, math.degrees(math.acos(max(-1.0, min(1.0, cos))))))
agg(names, "ALL 549")
agg(sorted(bf16), "bf16-resident")
agg(sorted(set(names) - bf16), "fp32-resident")
cond = [n for n in names if n.startswith("diffusion_conditioning")]
agg(cond, "cond (26, fp32)")
