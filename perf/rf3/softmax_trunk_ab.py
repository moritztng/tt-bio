#!/usr/bin/env python3
"""What the accurate softmax costs the RF3 trunk, measured rather than extrapolated.

`_accurate_softmax` costs 4.22x the fused kernel on the softmax op alone at
[1,16,1024,1024] (`perf/rf3/results/sm_mech.json`). The op is one of many in a pairformer
block, so the trunk number is the one that matters for the perf cells. Both arms off one
featurization, arm A run twice so the A/A floor is measured first: a difference inside the
A/A spread is not a difference.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \\
        python3 perf/rf3/softmax_trunk_ab.py --aa 512
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--feat_cache", default="/home/ttuser/rf3_perf_work/featcache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3 import remap as rf3_remap
    from tt_bio.rf3.host import HostInputs
    from tt_bio.tenstorrent import get_device
    from perf.rf3.featcache import featurized
    from perf.rf3.tt_rf3_bench import net_config

    fo = featurized(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                    n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                    seed=42, cache_dir=args.feat_cache or None)
    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    host = HostInputs.build(fo["feats"], device)

    def build(accurate: bool):
        rf3_remap.PAIRFORMER_FLAGS["accurate_softmax"] = accurate
        try:
            return rf3_model.load(
                args.ckpt, kcfg,
                n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
                n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
                n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
                num_timesteps=20, with_confidence=False)
        finally:
            rf3_remap.PAIRFORMER_FLAGS["accurate_softmax"] = True

    def times(tt):
        tt.trunk(host, args.n_recycles)                 # warm
        ts = []
        for _ in range(args.reps):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            tt.trunk(host, args.n_recycles)
            ttnn.synchronize_device(device)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts

    off = build(False)
    a1 = times(off)
    a2 = times(off)
    del off
    on = build(True)
    b1 = times(on)

    def med(x):
        return x[len(x) // 2]
    aa = abs(med(a2) - med(a1)) / med(a1)
    rep = {"aa": args.aa, "n_token": host.n_token, "n_atom": host.n_atom,
           "n_recycles": args.n_recycles, "reps": args.reps,
           "arm_a_fused_s": [round(t, 4) for t in a1],
           "arm_a_repeat_s": [round(t, 4) for t in a2],
           "arm_b_accurate_s": [round(t, 4) for t in b1],
           "a_a_spread_frac": round(aa, 5),
           "b_over_a": round(med(b1) / med(a1), 4),
           "cost_frac": round(med(b1) / med(a1) - 1, 5)}
    rep["inside_a_a"] = bool(abs(rep["cost_frac"]) <= aa)
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
