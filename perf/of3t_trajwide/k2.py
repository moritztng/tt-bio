"""rel_d at k=2 with each side's own w_0, against the FIXED reference, plus the gradient
sign-disagreement rate that drives it. Pre-fix this read rel_d 0.941270 and f 0.231766."""
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
print("scored tensors: %d  (pre-fix 549)" % len(names))
g = torch.load("/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt",
               map_location="cpu", weights_only=False)["grad_f64"]
tot = sum(float(v.double().pow(2).sum()) for v in g.values() if torch.is_tensor(v))
got = sum(float(g[n].double().pow(2).sum()) for n in names if n in g)
print("scored sq-grad-norm %.6f of boundary %.6f = %.4f%% of boundary, %.4f%% of model"
      % (got, tot, 100*got/tot, 100*got/10.279642678524981))
w2o, w2t = load_w(L+"/shipped", 2), load_w(L+"/theirs", 2)
num = den = onorm = cosn = 0.0; dis = nz = 0
for n in names:
    do = (w2o[n].astype(np.float64) - w1o[n].astype(np.float64)).ravel()
    dt = (w2t[n].astype(np.float64) - w1t[n].astype(np.float64)).ravel()
    d = do - dt
    num += float(d@d); den += float(dt@dt); onorm += float(do@do); cosn += float(do@dt)
    so, st = np.sign(do), np.sign(dt); m = (so != 0) & (st != 0)
    nz += int(m.sum()); dis += int((so[m] != st[m]).sum())
num, den, onorm = math.sqrt(num), math.sqrt(den), math.sqrt(onorm)
cos = cosn/(onorm*den); f = dis/nz
print()
print("k=2 rel_d              %.6e      (pre-fix 9.412700e-01)" % (num/(den+1e-30)))
print("   ||d_ours||          %.6e" % onorm)
print("   ||d_theirs||        %.6e   ratio %.6f" % (den, onorm/den))
print("   cos                 %.6f  angle %.1f deg   (pre-fix 49.6 at k=20)" % (cos, math.degrees(math.acos(max(-1,min(1,cos))))))
print("   sign disagree f     %.6f      (pre-fix 0.231766)  over %d elements" % (f, nz))
print("   2*sqrt(f)           %.6f" % (2*math.sqrt(f)))
