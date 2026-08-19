#!/usr/bin/env python3
"""Is TriMul E6 bit-exact in the CONFIDENCE head's Pairformer, not just the trunk's?

PAIRFORMER_FLAGS is shared by the trunk stack and the confidence head's 4-block stack, so
turning gated_move on turned it on in both. trimul_ab.py scored the trunk and found z
bit-exact there; this scores the head, whose output is what plddt is read off. The flag is
read per call, so both arms run off one checkpoint load against one set of coordinates.
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
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
    ap.add_argument("--aa", type=int, default=128)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=args.n_recycles, diffusion_batch_size=1,
                   seed=args.seed)[0]
    f, rep_atom_idxs = fo["feats"], fo["ground_truth"]["rep_atom_idxs"]

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.rf3.sampler import Draws
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config
    from perf.rf3.trimul_ab import trimuls

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps, with_confidence=True)

    head_tms = trimuls(tt.confidence_head)
    host = HostInputs.build(f, device)
    s_inputs, s, z = tt.trunk(host, args.n_recycles)
    x_pred, _ = tt.sampler.sample(
        lambda x, t: tt.diffusion_module(host, x, t, s_inputs, s, z),
        torch.zeros(1, host.n_atom, 3), 1, draws=Draws())

    arms = {}
    for name, gated in (("on", True), ("off", False), ("on2", True)):
        for tm in head_tms:
            tm.gated_move = gated
        arms[name] = tt.confidence(s_inputs, s, z, x_pred, rep_atom_idxs)
        print(f"  ran head with gated_move={gated}", flush=True)
    for tm in head_tms:
        tm.gated_move = True

    keys = sorted(arms["on"])

    def cmp(a, b):
        d = (a - b).abs()
        return {"bit_exact": bool(torch.eq(a, b).all()),
                "max_abs_diff": float(d.max()),
                "rel_rms": float(((a - b) ** 2).mean().sqrt() / (a ** 2).mean().sqrt())}

    rep = {"aa": args.aa, "n_token": host.n_token, "n_trimul_in_head": len(head_tms),
           "on_vs_on2": {k: cmp(arms["on"][k], arms["on2"][k]) for k in keys},
           "on_vs_off": {k: cmp(arms["on"][k], arms["off"][k]) for k in keys},
           "plddt_mean": {n: float(arms[n]["plddt_logits"].mean()) for n in arms}}
    rep["deterministic"] = all(v["bit_exact"] for v in rep["on_vs_on2"].values())
    rep["e6_bit_exact_in_head"] = all(v["bit_exact"] for v in rep["on_vs_off"].values())
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
