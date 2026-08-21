#!/usr/bin/env python3
"""The head-major qkv projection on RF3's BIASED triangle attention, over the whole trunk.

RF3 biases `linear_g` and `linear_o`; nothing biases the qkv projection, in any model. The gate
that admitted K1a read `self.biased` anyway, so RF3 -- the one model in the tree that trips it --
ran `minimal_matmul` + `nlp_create_qkv_heads` at all 96 triangle-attention calls per recycle and
the census read the lever as 0 served / 0 declined at every size on the ladder.

Both legs in ONE process off one checkpoint load, flag flipped per call, arm A run twice so the
A/A floor is measured rather than assumed. The kernel is bit-exact by construction (same block
config, same transaction schedule, only the destination tile id moves), so the claim here is
`torch.equal` on the trunk's z and s -- not a tolerance.
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
    import tt_bio.triatt_qkv as QKV
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    # Everything this campaign landed, on: the fold this lever has to move is the pass-4 fold.
    T._TRIATT_FUSED_HIFI = True
    T._OPM_SMALL_DEPTH = True

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
        QKV._ENABLED = on
        before = tuple(QKV.STATS)
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(args.n_recycles):
            s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
        ttnn.synchronize_device(device)
        per = (time.perf_counter() - t0) / args.n_recycles
        served = QKV.STATS[0] - before[0]
        declined = QKV.STATS[1] - before[1]
        return (per, torch.Tensor(ttnn.to_torch(z)).float(),
                torch.Tensor(ttnn.to_torch(s)).float(), served, declined)

    leg(False), leg(True)                       # warm both programs
    legs = {"stock": [], "head_major": [], "stock_aa": []}
    counts, keep = {}, {}
    for rep in range(args.reps):
        for name, on in (("stock", False), ("head_major", True), ("stock_aa", False)):
            per, z, s, served, declined = leg(on)
            legs[name].append(per)
            counts[name] = {"served": served, "declined": declined}
            keep.setdefault(name, (z, s))
            print(f"  rep {rep} {name:12s} {per * 1e3:9.1f} ms/recycle  "
                  f"served={served} declined={declined}", flush=True)
    QKV._ENABLED = True

    med = {k: statistics.median(v) for k, v in legs.items()}
    zA, sA = keep["stock"]
    zB, sB = keep["head_major"]
    zA2, sA2 = keep["stock_aa"]
    rep = {
        "aa": args.aa, "n_token": int(host.n_token), "n_recycles": args.n_recycles,
        "per_recycle_s": {k: round(v, 4) for k, v in med.items()},
        "all_legs_s": legs,
        "calls": counts,
        "trunk_speedup": round(med["stock"] / med["head_major"], 4),
        "aa_floor": round(med["stock"] / med["stock_aa"], 4),
        "z_bit_exact": bool(torch.equal(zA, zB)),
        "s_bit_exact": bool(torch.equal(sA, sB)),
        "z_max_abs_diff": float((zA - zB).abs().max()),
        "s_max_abs_diff": float((sA - sB).abs().max()),
        "z_aa_bit_exact": bool(torch.equal(zA, zA2)),
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
