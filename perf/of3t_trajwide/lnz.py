import os, numpy as np, torch
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1t, w1o = load_w(L+"/theirs", 1), load_w(L+"/shipped", 1)
ck = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
print("checkpoint keys containing 'layer_norm_z':")
for k, v in ck.items():
    if "layer_norm_z" in k: print("   ", k, tuple(v.shape))
print("\ncheckpoint keys the module did not want (unexpected candidates, top-level scan):")
unexp = sorted(set(ck) - set(w1t))
print("   count:", len(unexp))
for k in unexp[:30]: print("   ", k, tuple(ck[k].shape))
print("\ntheir layer_norm_z.weight at k=1:", w1t["diffusion_transformer.layer_norm_z.weight"][:8])
print("   all ones?", bool(np.all(w1t["diffusion_transformer.layer_norm_z.weight"] == 1.0)))
key = "diffusion_transformer.layer_norm_z.weight"
print("\nis it in OUR dump?", key in w1o)
if key in w1o:
    print("   ours:", w1o[key][:8], " all ones?", bool(np.all(w1o[key] == 1.0)))
