import os, numpy as np, torch
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
w1t, w1o = load_w(L+"/theirs", 1), load_w(L+"/shipped", 1)
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
ck = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
def blk(d, tag):
    ks = sorted(n for n in d if n.startswith("diffusion_transformer.blocks.0."))
    print("== %s : %d keys under diffusion_transformer.blocks.0." % (tag, len(ks)))
    for n in ks: print("    ", n.replace("diffusion_transformer.blocks.0.", ""), tuple(np.shape(d[n])))
blk(w1t, "THEIR dump")
blk(ck, "CHECKPOINT")
blk(w1o, "OUR dump")
