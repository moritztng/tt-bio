#!/usr/bin/env python3
"""Inside the MSA module: which of its five parts carries the 12.7%.

Pass 2 priced the MSA module at "11x a Pairformer block" from pre-lever numbers. Pass 3's
third decomposition re-priced it at 0.2948 s per recycle over 4 blocks against 1.836 s over
48 Pairformer blocks, which is 1.93x a Pairformer block, not 11x -- the label expired when
the fused triangle attention removed the traffic that set it. 1.93x is not obviously wrong
for a block that also carries an outer product and a pair-weighted average, so the question
this instrument answers is whether any ONE of the five parts is out of line with what it
computes, at the shipped MSA depth.

Same convention as trunk_decompose.py: syncs on, so the totals are an attribution and not
a headline, and the inner parts are wrapped before the module attribute that reaches them.
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

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ROOF_SQUARE_TFS = 103.31
ROOF_KTHIN_TFS = 17.41
ROOF_DRAM_GBS = 440.4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
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
    from perf.rf3.trunk_decompose import Acc

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

    acc = Acc(device)
    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

    msa = tt.recycler.msa_module
    pf = msa.pairformer_layer
    # innermost first: wrapping an attribute replaces the object with a function.
    for attr, name in (("triangle_multiplication_start", "msa.pf.tri_mul_start"),
                       ("triangle_multiplication_end", "msa.pf.tri_mul_end"),
                       ("triangle_attention_start", "msa.pf.tri_att_start"),
                       ("triangle_attention_end", "msa.pf.tri_att_end"),
                       ("transition_z", "msa.pf.transition_z")):
        if hasattr(pf, attr):
            acc.wrap(pf, attr, name)
    acc.wrap(msa, "subsampler", "msa.subsampler")
    acc.wrap(msa, "outer_product", "msa.outer_product")
    acc.wrap(msa, "pair_weighted_averaging", "msa.pwa")
    acc.wrap(msa, "msa_transition", "msa.transition_m")
    acc.wrap(msa, "pairformer_layer", "msa.pairformer_layer")
    acc.wrap(tt.recycler, "msa_module", "msa.TOTAL")

    n_block = msa.n_block
    depth = tuple(host.msa_stack[0].shape)

    def run():
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(args.n_recycles):
            s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0

    run()
    acc.on = True
    acc.reset()
    wall = run()
    acc.on = False

    per = {k: v / args.n_recycles for k, v in acc.t.items()}
    total = per.get("msa.TOTAL", 0.0)
    inner = sum(v for k, v in per.items()
                if k not in ("msa.TOTAL", "msa.pairformer_layer")
                and not k.startswith("msa.pf."))
    pf_named = sum(v for k, v in per.items() if k.startswith("msa.pf."))
    rep = {"aa": args.aa, "n_token": host.n_token, "n_msa_block": n_block,
           "msa_feat_shape": depth, "n_recycles": args.n_recycles,
           "synced_per_recycle_s": wall / args.n_recycles,
           "per_recycle_s": {k: round(v, 5) for k, v in per.items()},
           "calls_per_recycle": {k: v // args.n_recycles for k, v in acc.n.items()},
           "per_msa_block_ms": {k: round(v / n_block * 1e3, 3) for k, v in per.items()
                                if k not in ("msa.TOTAL", "msa.subsampler")},
           "unattributed_in_total_s": total - inner - per.get("msa.pairformer_layer", 0.0),
           "pairformer_unattributed_s": per.get("msa.pairformer_layer", 0.0) - pf_named}
    print(f"{args.aa} aa, {host.n_token} tokens, msa feats {depth}, "
          f"{n_block} MSA blocks, module {total:.4f} s/recycle (syncs on)")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        share = v / total * 100 if total else 0.0
        print(f"  {k:26s} {v:8.4f} s  {share:5.1f}%   "
              f"{v / n_block * 1e3:8.3f} ms/block")
    print(f"  {'pairformer unattributed':26s} "
          f"{rep['pairformer_unattributed_s']:8.4f} s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
