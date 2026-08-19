#!/usr/bin/env python3
"""Does the TriMul E6 fused chunk+gate move pay on RF3's 48-block Pairformer?

`gated_move` is a per-instance opt-in, not a global default -- the same kernel wins on
opendde's channel widths and loses on boltz2's call mix -- and RF3's `PAIRFORMER_FLAGS`
never passed it, so RF3's trunk has been running the four-way split all along. That
trunk is half of the 512 aa wall before the atom-pair-track lever and 86% of it after,
so this is worth an A/B rather than an assumption.

The flag is read per call, not baked in at construction, so both arms run against ONE
checkpoint load with the legs interleaved -- all-A-then-all-B has read +13.3% where
interleaved gave +5.2% on a hot card. The trunk alone is timed, not a whole fold: it is
the only phase the flag can touch, and timing it alone keeps a rung inside a couple of
minutes.

E6 is meant to be a dataflow change and nothing else, so the arms should agree
bit-exactly on `z`. That is checked, not assumed.
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


def trimuls(root) -> list:
    """Every TriangleMultiplication reachable from `root`, found by walk rather than by
    path, so a stack that gets restructured does not silently leave half of them out."""
    from tt_bio.tenstorrent import TriangleMultiplication
    seen, found, stack = set(), [], [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, TriangleMultiplication):
            found.append(o)
        if isinstance(o, (list, tuple)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(v for v in vars(o).values()
                         if isinstance(v, (list, tuple)) or hasattr(v, "__dict__"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                   seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

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
        num_timesteps=50, with_confidence=False)

    tms = trimuls(tt.recycler)
    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

    def leg(gated: bool):
        for tm in tms:
            tm.gated_move = gated
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(args.n_recycles):
            s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
        ttnn.synchronize_device(device)
        return (time.perf_counter() - t0) / args.n_recycles, \
            torch.Tensor(ttnn.to_torch(z)).float()

    print(f"{len(tms)} TriangleMultiplication instances in the trunk", flush=True)
    leg(False), leg(True)                       # warm both program variants
    legs = {"off": [], "on": []}
    zs = {}
    for rep in range(args.reps):
        for name, gated in (("off", False), ("on", True)):
            per, z = leg(gated)
            legs[name].append(per)
            zs.setdefault(name, z)
            print(f"  rep {rep} {name:3s}  {per * 1e3:8.1f} ms/recycle", flush=True)
    for tm in tms:
        tm.gated_move = False

    med = {k: statistics.median(v) for k, v in legs.items()}
    d = (zs["off"] - zs["on"]).abs()
    rep = {"aa": args.aa, "n_token": host.n_token, "n_trimul": len(tms),
           "n_recycles_per_leg": args.n_recycles, "reps": args.reps,
           "per_recycle_s": {k: round(v, 4) for k, v in med.items()},
           "all_legs_s": legs,
           "speedup_on_trunk": round(med["off"] / med["on"], 4),
           "z_bit_exact": bool(torch.eq(zs["off"], zs["on"]).all()),
           "z_max_abs_diff": float(d.max()),
           "z_rel_max": float(d.max() / (zs["off"].abs().max().item() or 1.0))}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
