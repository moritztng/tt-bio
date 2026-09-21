"""Locate the 1 missing / 24 unexpected load_state_dict keys on the reference side, using only
the dumps and the checkpoint. A reference parameter left at its random init would perturb THEIR
forward everywhere and make the trajectory error partly their artifact rather than ours."""
import os, numpy as np, torch
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt"); PREFIX = "diffusion_module."
L = "/tmp/of3t/trajwide/w"
def load_w(d, k):
    with np.load(os.path.join(d, "k%02d.npz" % k)) as z: return {n: z[n] for n in z.files}
sd = torch.load(CKPT, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
w1t = load_w(L+"/theirs", 1); w1o = load_w(L+"/shipped", 1)
ck = {k[len(PREFIX):] for k in sd if k.startswith(PREFIX)}
their = set(w1t)
print("their dumped params        :", len(their))
print("checkpoint diffusion_module:", len(ck))
no_ck = sorted(their - ck)
print("their params with NO checkpoint entry (=> left at init):", len(no_ck))
for n in no_ck: print("   ", n, w1t[n].shape)
print()
# shape mismatches would also silently skip
mism = [n for n in sorted(their & ck) if tuple(sd[PREFIX+n].shape) != tuple(w1t[n].shape)]
print("shape mismatches:", len(mism), mism[:5])
print()
# does any such param still sit at a value that differs from the checkpoint at k=1?
scored = sorted(n for n in w1t if n in w1o and PREFIX+n in sd
                and tuple(sd[PREFIX+n].shape) == tuple(w1t[n].shape))
print("scored:", len(scored))
print("unscored of theirs:", len(their) - len(scored))
unsc = sorted(their - set(scored))
in_ck = [n for n in unsc if n in ck]
print("  of which HAVE a checkpoint entry (so they loaded, just not in our dump):", len(in_ck))
print("  of which have NO checkpoint entry (the real risk):", len(unsc) - len(in_ck))
