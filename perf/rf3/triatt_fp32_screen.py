#!/usr/bin/env python3
"""How much of the trunk is the fp32 softmax, and what would a fused one be worth?

Triangle attention is 67% of the RF3 trunk at 512 aa -- 85.1 ms of a 126 ms block -- and
it runs at 2.39 TF/s against the 17.41 TF/s this card was measured at on the K-thin shape
the attention actually has. The reason is structural, not a tuning gap: RF3's
PAIRFORMER_FLAGS sets fp32_softmax=True, which routes past _tri_att_sdpa (the fused SDPA
that never materialises a score tensor) into _fp32_softmax_attention, which materialises
n_heads x I^3 scores in bf16, again in fp32, softmaxes them and casts back. At 512 aa that
is ~10 GB of DRAM traffic per call against a 440.4 GB/s roof, i.e. ~23 ms of pure
bandwidth in a 43 ms call.

This is a SCREEN, not a lever. fp32_softmax is a correctness finding the port paid four
passes for, so turning it off is not shippable and the rel_rms below is why. What the arm
buys is a measured bound on what an fp32 softmax fused INTO the SDPA kernel would be
worth, instead of a projection.
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


def triatts(root) -> list:
    from tt_bio.tenstorrent import TriangleAttention
    seen, found, stack = set(), [], [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, TriangleAttention):
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
    ap.add_argument("--n_recycles", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
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

    tas = [t for t in triatts(tt.recycler.pairformer)]
    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

    def leg(fp32: bool):
        for t in tas:
            t.fp32_softmax = fp32
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

    print(f"{len(tas)} TriangleAttention instances in the 48-block stack", flush=True)
    leg(True), leg(False)
    legs, zs = {"fp32": [], "bf16": []}, {}
    for rep in range(args.reps):
        for name, fp32 in (("fp32", True), ("bf16", False)):
            per, z = leg(fp32)
            legs[name].append(per)
            zs.setdefault(name, z)
            print(f"  rep {rep} {name}  {per * 1e3:9.1f} ms/recycle", flush=True)
    for t in tas:
        t.fp32_softmax = True

    med = {k: statistics.median(v) for k, v in legs.items()}
    a, b = zs["fp32"], zs["bf16"]
    rel_rms = float(((a - b) ** 2).mean().sqrt() / (a ** 2).mean().sqrt())
    rep = {"aa": args.aa, "n_token": host.n_token, "n_triatt": len(tas),
           "per_recycle_s": {k: round(v, 4) for k, v in med.items()},
           "all_legs_s": legs,
           "trunk_speedup_if_fused": round(med["fp32"] / med["bf16"], 4),
           "z_rel_rms_fp32_vs_bf16": rel_rms,
           "z_bit_exact": bool(torch.eq(a, b).all())}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
