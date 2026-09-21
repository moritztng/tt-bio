"""At k=2 the Adam moments are zero, so the update is lr * m_hat/(sqrt(v_hat)+eps) =
lr*sign(g) elementwise. d_2 therefore encodes the SIGN of the gradient and nothing else,
which makes the disagreement rate directly measurable rather than inferred from rel_d."""
import os, math, numpy as np, torch
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
tot = dis = both_nz = 0
mag_o = mag_t = 0.0
for n in names:
    do = (w2o[n].astype(np.float64) - w1o[n].astype(np.float64)).ravel()
    dt = (w2t[n].astype(np.float64) - w1t[n].astype(np.float64)).ravel()
    so, st = np.sign(do), np.sign(dt)
    m = (so != 0) & (st != 0)
    both_nz += int(m.sum()); tot += do.size
    dis += int((so[m] != st[m]).sum())
    mag_o += float(do @ do); mag_t += float(dt @ dt)
f = dis / both_nz
print("elements total            : %d" % tot)
print("both sides nonzero        : %d" % both_nz)
print("sign disagreements        : %d" % dis)
print("disagreement fraction f   : %.6f" % f)
print("predicted rel_d = 2*sqrt(f): %.6f" % (2 * math.sqrt(f)))
print("measured  rel_d at k=2     : 0.941270   (traj_shipped_w0own.json)")
print("||d_ours||=%.6e ||d_theirs||=%.6e ratio=%.6f"
      % (math.sqrt(mag_o), math.sqrt(mag_t), math.sqrt(mag_o/mag_t)))
# is the step really sign-like? every |element| should equal lr=1.8e-6
do_all = np.concatenate([(w2t[n].astype(np.float64)-w1t[n].astype(np.float64)).ravel()
                         for n in names[:40]])
a = np.abs(do_all[do_all != 0])
print("their |d_2| elementwise: min=%.4e median=%.4e max=%.4e  (lr=1.8e-06)"
      % (a.min(), np.median(a), a.max()))
