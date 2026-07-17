"""Validate on-device IPA linear projections vs the reference internals (PCC 1.0).

The IPA attention (scalar q.k + point) is the documented ceiling: it needs subtile
head/point-dim reshapes (head=12, head_dim=16, P_q/P_v=4/8, point coords=3) that ttnn stock ops
cannot express on device. A full on-device IPA needs a custom tt-metal
point-attention kernel (separate domain, deferred). What IS on device here: the
IPA linear projections (q, kv, qp, kvp, pair bias b) -- validated PCC 1.0.
"""
import pickle, torch
from tt_bio.tenstorrent import get_device, WeightScope
from tt_bio.abodybuilder3 import abb3_compute_kernel_config, IPALayer, _from_torch, _to_torch

def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))

I = pickle.load(open("/tmp/abb3_cache/abb3_ipa_internals.pkl", "rb"))
sd = torch.load("/tmp/abb3_cache/abodybuilder3_plddt.pt", map_location="cpu", weights_only=True)
scope = "ipa_layers.0"
w = WeightScope({k[len(scope) + 1:]: v for k, v in sd.items() if k.startswith(scope + ".")})
ck = abb3_compute_kernel_config()
ipa = IPALayer(w, ck)

s = _from_torch(I["s"]); z = _from_torch(I["z"])
rot = _from_torch(I["rot_mats"]); trans = _from_torch(I["trans"]); mask = _from_torch(I["mask"])
out = ipa(s, z, rot, trans, mask)


def chk(name, tt, ref):
    o = _to_torch(tt).reshape(ref.shape)
    print(f"{name:12s} PCC={pcc(o.float(), ref.float()):.5f}  shape {tuple(o.shape)}")


N = I["s"].shape[1]
chk("q", out["q"], I["q"].reshape(1, N, -1))
chk("kv", out["kv"], torch.cat([I["k"], I["v"]], dim=-1).reshape(1, N, -1))
chk("qp", out["qp"], I["q_pts"].permute(0, 1, 4, 2, 3).reshape(1, N, -1))
chk("kvp", out["kvp"], torch.cat([I["k_pts"], I["v_pts"]], dim=-2).permute(0, 1, 4, 2, 3).reshape(1, N, -1))
chk("b", out["b"], I["b"])
