"""Offline: device s (P, C) vs float64 reference s on batch08; per row group and the insertion term itself."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/ttuser/r4t/tape/b08/src")
from tt_bio.train.rfd4_objective import insertion_loss
from tt_bio.rfd4 import network as N, engine as E
B = "/home/ttuser/r4t/tape/b08/"
ref = torch.load(B + "out_P/ref.pt")
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
glue = N.FlowGlue(state, config)
d0 = torch.load(B + "out_P/dev.pt")
m, ins = d0["mask"].bool(), d0["ins"]
batch = {"insertion_source_idx": ins.numpy(), "padding_mask": m.numpy()}
rec = np.load("/home/ttuser/r4t/step/batch08.npz")
def lg(s):
    with torch.no_grad():
        q, k = glue.insertion_q_proj(s.float()), glue.insertion_k_proj(s.float())
        return (torch.bmm(q, k.transpose(1, 2)) * glue._insertion_head_scale).double().numpy()
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
Lref = insertion_loss(lg(ref["sref"]), batch)[0]
print("loss upstream rec", rec["loss/insertion_prediction"], "ref64", Lref)
for arm in ("P", "C"):
    try: d = torch.load(B + f"out_{arm}/dev.pt")
    except Exception as e: print(arm, e); continue
    s, sr = d["sdev"].double(), ref["sref"].double()
    L = insertion_loss(lg(d["sdev"]), batch)[0]
    for b in range(4):
        ir = ((ins[b] >= 0) & m[b]).numpy(); rr = ((ins[b] < 0) & m[b]).numpy()
        e = (s[b] - sr[b]).numpy()
        print(f"{arm} cand{b}: s rel ins {np.linalg.norm(e[ir])/np.linalg.norm(sr[b].numpy()[ir]):.3e} rest {np.linalg.norm(e[rr])/np.linalg.norm(sr[b].numpy()[rr]):.3e} | loss {L[b]:.4e} vs ref {Lref[b]:.4e} rel {abs(L[b]-Lref[b])/Lref[b]:.3e}")
