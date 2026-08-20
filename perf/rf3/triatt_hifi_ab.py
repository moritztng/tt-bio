#!/usr/bin/env python3
"""The fused-SDPA triangle attention against the materialised fp32 softmax, over the whole trunk.

Both legs in ONE process off one checkpoint load, flag flipped per call, arm A run twice so the
A/A floor is measured rather than assumed. Reports per-recycle time and the z / s divergence
between the arms -- which is NOT the accuracy bar (neither arm is the reference; see
`perf/rf3/triatt_ckc_sweep.py` for the arm-vs-fp64 numbers) but does say how far the two paths pull
apart over 48 blocks.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def rel_rms(a, b):
    return float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = T.get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

    def leg(on: bool):
        T._TRIATT_FUSED_HIFI = on
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(args.n_recycles):
            s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
        ttnn.synchronize_device(device)
        per = (time.perf_counter() - t0) / args.n_recycles
        return per, torch.Tensor(ttnn.to_torch(z)).float(), torch.Tensor(ttnn.to_torch(s)).float()

    leg(False), leg(True)                       # warm both programs
    legs = {"materialised": [], "fused_hifi": [], "materialised_aa": []}
    keep = {}
    for rep in range(args.reps):
        for name, on in (("materialised", False), ("fused_hifi", True),
                         ("materialised_aa", False)):
            per, z, s = leg(on)
            legs[name].append(per)
            keep.setdefault(name, (z, s))
            print(f"  rep {rep} {name:16s} {per * 1e3:9.1f} ms/recycle", flush=True)
    T._TRIATT_FUSED_HIFI = False

    med = {k: statistics.median(v) for k, v in legs.items()}
    zA, sA = keep["materialised"]
    zB, sB = keep["fused_hifi"]
    zA2, sA2 = keep["materialised_aa"]
    rep = {
        "aa": args.aa, "n_token": int(host.n_token), "n_recycles": args.n_recycles,
        "per_recycle_s": {k: round(v, 4) for k, v in med.items()},
        "all_legs_s": legs,
        "trunk_speedup": round(med["materialised"] / med["fused_hifi"], 4),
        "aa_floor": round(med["materialised"] / med["materialised_aa"], 4),
        "z_rel_rms_vs_materialised": rel_rms(zB, zA),
        "s_rel_rms_vs_materialised": rel_rms(sB, sA),
        "z_rel_rms_aa": rel_rms(zA2, zA),
        "s_rel_rms_aa": rel_rms(sA2, sA),
        "served": dict(T.TRIATT_FUSED_HIFI_STATS),
        "kernel_rejects": {str(k): v for k, v in T._triatt_sdpa.REJECTS.items()},
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
