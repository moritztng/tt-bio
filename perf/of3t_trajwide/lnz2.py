import os, numpy as np, torch
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1o = load_w(L+"/shipped", 1)
hits = sorted(n for n in w1o if "layer_norm_z" in n)
print("OUR dump keys containing layer_norm_z:", len(hits))
for n in hits[:6]: print("   ", n, w1o[n].shape)
print()
k0 = "diffusion_transformer.blocks.0.attention_pair_bias.layer_norm_z.weight"
if k0 in w1o:
    c = sd[PREFIX+k0].to(torch.float64).numpy()
    o = w1o[k0].astype(np.float64)
    print("block0 layer_norm_z.weight")
    print("   checkpoint[:8]:", np.round(c[:8], 5))
    print("   ours      [:8]:", np.round(o[:8], 5))
    print("   ours == ckpt (or bf16 of it):", bool(np.array_equal(o, c)))
    print("   checkpoint all ones?", bool(np.all(c == 1.0)))
    print("   ||ckpt - 1||/||1|| = %.6f" % (np.linalg.norm(c - 1.0) / np.linalg.norm(np.ones_like(c))))
# how far from ones are the 24 trained per-block values?
devs = []
for i in range(24):
    k = "diffusion_transformer.blocks.%d.attention_pair_bias.layer_norm_z.weight" % i
    if PREFIX+k in sd:
        c = sd[PREFIX+k].to(torch.float64).numpy()
        devs.append(np.linalg.norm(c - 1.0) / np.linalg.norm(np.ones_like(c)))
print()
print("24 per-block layer_norm_z: relative deviation from all-ones")
print("   min=%.4f median=%.4f max=%.4f" % (min(devs), float(np.median(devs)), max(devs)))
