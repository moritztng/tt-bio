"""Grade each `tt_bio.diffdock.device_loop` block on a card against its float64 torch twin.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/diffdock_port/device_loop_grade.py --out g.json

Edge vectors span 0-40 A in every direction, plus exact zeros (pad edges). Inputs are drawn once
on the host, so card and reference see the same fp32 numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.detach().double().flatten(), ref.detach().double().flatten()
    return {"max_abs": float((g - r).abs().max()),
            "rel_frobenius": float((g - r).norm() / r.norm().clamp_min(1e-30)),
            "pcc": float(torch.corrcoef(torch.stack([g, r]))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import device_loop as dl
    from tt_bio.diffdock.features import gaussian_smearing_reference
    from tt_bio.diffdock.harmonics import spherical_harmonics
    from tt_bio.diffdock.score_model import load_released
    from tt_bio.tenstorrent import get_device

    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg).double()
    g = torch.Generator().manual_seed(0)
    vec = (torch.rand(args.edges, 3, generator=g) * 2 - 1) * 40
    vec[:32] = 0
    vec = vec.float()
    dev = get_device()
    xyz = [dl.upload(dev, vec[:, k]) for k in range(3)]
    back = lambda t, w: ttnn.to_torch(t)[:args.edges, :w]
    out = {"edges": args.edges}

    def record(name, got, ref):
        """Measure one leg and print it at once: a later failure keeps the earlier numbers."""
        out[name] = stats(got, ref)
        print(name, out[name], flush=True)
        return out[name]

    dist = vec.double().norm(dim=-1)
    record("norm3", back(dl.norm3(*xyz), 1)[:, 0], dist)
    for ls in ((0, 1, 2), (2,)):
        ref = spherical_harmonics(vec.double(), ls)
        record(f"harmonics_{''.join(map(str, ls))}", back(dl.harmonics(*xyz, ls), ref.shape[1]), ref)
    d_dev = dl.upload(dev, dist.float())
    for name in ("lig", "rec", "cross"):
        mod = getattr(model, f"{name}_distance_expansion")
        ref = gaussian_smearing_reference(dist.float(), mod)
        record(f"smear_{name}", back(dl.Smearing(dev, mod)(d_dev), ref.shape[1]), ref)
    for name in ("rec_edge_embedding", "cross_edge_embedding", "lig_edge_embedding",
                 "rec_sigma_embedding"):
        seq = getattr(model, name)
        x = torch.randn(args.edges, seq[0].in_features, generator=g).float()
        ref = seq(x.double())
        record(name, back(dl.MLP(dev, seq.float())(dl.upload(dev, x)), ref.shape[1]), ref)
        seq.double()
    # --- the pose update ------------------------------------------------------------
    from tt_bio.diffdock.sampling import axis_angle_to_matrix, guard_scores

    B = 320
    aa = torch.randn(B, 3, generator=g).float() * 1.5
    aa[:32] = 0                       # the small-angle series, and the exact-zero axis
    aa[32:64] *= 1e-7
    cols = [dl.upload(dev, aa[:, k]) for k in range(3)]
    ref_m = axis_angle_to_matrix(aa.double())
    got_m = dl.axis_angle_matrix(*cols)
    record("axis_angle_matrix", 
        torch.cat([ttnn.to_torch(c)[:B, :1] for c in got_m], 1), ref_m.reshape(B, 9))

    pts = torch.randn(B, 3, generator=g).float() * 12
    pcols = [dl.upload(dev, pts[:, k]) for k in range(3)]
    ref_r = torch.bmm(pts.double()[:, None, :], ref_m.transpose(1, 2))[:, 0]
    record("rotate", 
        torch.cat([ttnn.to_torch(c)[:B, :1] for c in dl.rotate(pcols, got_m)], 1), ref_r)

    score = torch.randn(B, 3, generator=g).float()
    score[3, 1], score[7, 0], score[11, 2] = float("nan"), float("inf"), float("-inf")
    ref_g = guard_scores(score.double())[0]
    got_g = dl.guard([dl.upload(dev, score[:, k]) for k in range(3)])
    record("guard", torch.cat([ttnn.to_torch(c)[:B, :1] for c in got_g], 1), ref_g)

    # --- gathered edge geometry -----------------------------------------------------
    n_nodes, E = 512, args.edges
    pos = (torch.rand(n_nodes, 3, generator=g) * 2 - 1).float() * 30
    src = torch.randint(0, n_nodes, (E,), generator=g)
    dst = torch.randint(0, n_nodes, (E,), generator=g)
    dst[:32] = src[:32]                       # pad edges: a zero vector, zero harmonics
    geo = dl.EdgeGeometry(dev, src, dst, E)
    pos_dev = dl.upload(dev, pos)
    vcols = geo.vectors(pos_dev)
    ref_v = pos.double()[dst] - pos.double()[src]
    record("gathered_edge_vectors", 
        torch.cat([ttnn.to_torch(c)[:E, :1] for c in vcols], 1), ref_v)
    ref_sh = spherical_harmonics(ref_v, (0, 1, 2))
    record("gathered_harmonics", back(dl.harmonics(*vcols), ref_sh.shape[1]), ref_sh)
    d_gath = dl.norm3(*vcols)
    ref_m = (ref_v.norm(dim=-1) <= 5.0).double()
    got_m = ttnn.to_torch(geo.within(d_gath, 5.0))[:E, 0]
    out["within"] = {"mismatches": int((got_m.double() - ref_m).abs().sum()),
                     "kept": int(ref_m.sum())}
    print("within", out["within"], flush=True)

    print(json.dumps({k: v for k, v in out.items() if isinstance(v, (int, float))}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
