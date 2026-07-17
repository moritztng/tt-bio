"""Capture ABodyBuilder3 IPA (block 0) internal intermediates on the reference,
feeding the golden block-0 (s, z, rot_mats, trans, mask). A bisect oracle for the
on-device IPA port: each ttnn sub-step PCC-gates against the matching intermediate.

Dumps /tmp/abb3_cache/abb3_ipa_internals.pkl with:
  q, k, v                 [1, N, 12, 16]
  q_pts, k_pts, v_pts     [1, N, 12, 4|8, 3]   (after rigid apply)
  b                       [1, N, N, 12]
  a_scalar                [1, 12, N, N]         (q.k + pair bias, pre point-att)
  pt_att                  [1, 12, N, N]         (point-attention logits, pre mask)
  a                       [1, 12, N, N]         (full logits, post mask)
  attn                    [1, 12, N, N]         (softmax(a))
  o                       [1, N, 12, 16]
  o_pt                    [1, N, 12, 8, 3]       (after invert_apply)
  o_pt_norm               [1, N, 96]
  o_pair                  [1, N, 1536]
  cat                     [1, N, 2112]
  delta                   [1, N, 128]           (== golden ipa_delta)
"""
import math
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio._vendor.abodybuilder3.openfold.model.structure_module import StructureModule
from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid, Rotation
from tt_bio._vendor.abodybuilder3.openfold.utils.tensor_utils import permute_final_dims


def main():
    cache = os.environ.get("TT_BIO_CACHE", "/tmp/abb3_cache")
    os.makedirs(cache, exist_ok=True)
    sd = torch.load(ensure_abb3_weights(cache), map_location="cpu", weights_only=True)
    m = StructureModule(**ABB3_CONFIG)
    m.load_state_dict(sd, strict=True)
    m.eval()
    ipa = m.ipa_layers[0]

    gold = pickle.load(open(os.environ.get("ABB3_GOLDEN", os.path.join(cache, "abb3_golden.pkl")), "rb"))
    b0 = gold["blocks"][0]
    s = b0["ipa_s_in"]
    z = b0["ipa_z_in"]
    r = Rigid(Rotation(rot_mats=b0["ipa_rot_mats"], quats=None), b0["ipa_trans"])
    mask = b0["ipa_mask"]

    H, C = ipa.no_heads, ipa.c_hidden
    Pq, Pv = ipa.no_qk_points, ipa.no_v_points

    with torch.no_grad():
        q = ipa.linear_q(s).view(*s.shape[:-1], H, C)
        kv = ipa.linear_kv(s).view(*s.shape[:-1], H, 2 * C)
        k, v = torch.split(kv, C, dim=-1)

        q_pts = ipa.linear_q_points(s)
        q_pts = torch.stack(torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1), dim=-1)
        q_pts = r[..., None].apply(q_pts).view(*s.shape[:-1], H, Pq, 3)

        kv_pts = ipa.linear_kv_points(s)
        kv_pts = torch.stack(torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1), dim=-1)
        kv_pts = r[..., None].apply(kv_pts).view(*s.shape[:-1], H, -1, 3)
        k_pts, v_pts = torch.split(kv_pts, [Pq, Pv], dim=-2)

        b = ipa.linear_b(z[0] if isinstance(z, list) else z)

        a = torch.matmul(permute_final_dims(q, (1, 0, 2)), permute_final_dims(k, (1, 2, 0)))
        a = a * math.sqrt(1.0 / (3 * C))
        a = a + math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        hw = ipa.softplus(ipa.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        hw = hw * math.sqrt(1.0 / (3 * (Pq * 9.0 / 2)))
        pt_att = pt_att * hw
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = ipa.inf * (square_mask - 1)
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a_full = a + pt_att + square_mask.unsqueeze(-3)
        attn = ipa.softmax(a_full)

        o = torch.matmul(attn, v.transpose(-2, -3).to(attn.dtype)).transpose(-2, -3)
        o_flat = o.reshape(*o.shape[:-2], -1)

        o_pt = torch.sum(
            (attn[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + ipa.eps).reshape(*o_pt.shape[:-3], -1)
        o_pt_flat = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        o_pair = torch.matmul(attn.transpose(-2, -3), z[0] if isinstance(z, list) else z).to(attn.dtype)
        o_pair = o_pair.reshape(*o_pair.shape[:-2], -1)

        cat = torch.cat((o_flat, *torch.unbind(o_pt_flat, dim=-1), o_pt_norm, o_pair), dim=-1)
        delta = ipa.linear_out(cat.to(z[0].dtype if isinstance(z, list) else z.dtype))

    out = dict(q=q, k=k, v=v, q_pts=q_pts, k_pts=k_pts, v_pts=v_pts, b=b,
               a_scalar=a, pt_att=pt_att, a=a_full, attn=attn, o=o_flat, o_pt=o_pt,
               o_pt_norm=o_pt_norm, o_pair=o_pair, cat=cat, delta=delta,
               head_weights=ipa.head_weights, mask=mask, s=s, z=z,
               rot_mats=b0["ipa_rot_mats"], trans=b0["ipa_trans"])
    path = os.path.join(cache, "abb3_ipa_internals.pkl")
    pickle.dump(out, open(path, "wb"))
    # Self-consistency: delta must match the golden ipa_delta.
    ref = b0["ipa_delta"].float().flatten()
    d = delta.float().flatten()
    pcc = float(((ref - ref.mean()) * (d - d.mean())).sum() / ((ref - ref.mean()).norm() * (d - d.mean()).norm()))
    print(f"IPA internals -> {path}")
    print(f"  self-consistency delta PCC vs golden ipa_delta = {pcc:.6f} (expect ~1.0)")
    for k_ in ("q", "k", "v", "q_pts", "k_pts", "v_pts", "b", "a_scalar", "pt_att",
               "attn", "o", "o_pt", "o_pt_norm", "o_pair", "cat", "delta"):
        print(f"  {k_:10s} {tuple(out[k_].shape)}")


if __name__ == "__main__":
    main()
