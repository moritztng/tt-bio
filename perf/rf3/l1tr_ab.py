#!/usr/bin/env python3
"""Lever 8, the L1-resident pair transpose: is it bit-exact, and what is it worth per block?

Both arms in one process off one checkpoint load, the flag flipped on the live modules between
them, so the only difference is where the ending variant's two `_pair_transpose` calls put their
result. A memory config cannot change a value; this measures that rather than asserting it, with
`torch.equal` on the full 48-block trunk's z AND s after two recycles.
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
    ap.add_argument("--aa", type=int, default=768)
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

    blocks = tt.recycler.pairformer.blocks
    print(f"{len(blocks)} blocks, reserve on the modules: "
          f"{blocks[0].triangle_attention_end.transpose_l1_reserve} B/core", flush=True)

    def set_reserve(v):
        for b in blocks:
            b.triangle_attention_end.transpose_l1_reserve = v
            b.triangle_attention_start.transpose_l1_reserve = v

    def run():
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(args.n_recycles):
            s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0, s, z

    # Which route each arm's transpose actually took, read off the destination rather than
    # inferred from the constant: a gate that does not fire has to say so.
    dest = {}
    orig_tmc = T._transpose_memory_config

    def spy(t, reserve_per_core=0):
        mc = orig_tmc(t, reserve_per_core)
        if len(t.shape) == 3 and int(t.shape[0]) > 256:
            dest[str(mc.buffer_type).split(".")[-1]] = dest.get(
                str(mc.buffer_type).split(".")[-1], 0) + 1
        return mc

    T._transpose_memory_config = spy

    out = {"aa": args.aa, "tokens": int(z_init.shape[1]), "arms": {}}
    ref = None
    for name, reserve in (("headroom", 0), ("reserve", T._TRANSPOSE_L1_RESERVE_PER_CORE),
                          ("headroom_aa", 0)):
        set_reserve(reserve)
        dest.clear()
        _dt, s, z = run()                                   # warm
        ttnn.deallocate(s); ttnn.deallocate(z)
        routes = dict(dest)
        ts = []
        for _ in range(args.reps):
            dt, s, z = run()
            ts.append(dt)
            if name == "headroom" and ref is None:
                ref = (ttnn.to_torch(s), ttnn.to_torch(z))
                got = None
            elif name == "reserve":
                got = (ttnn.to_torch(s), ttnn.to_torch(z))
            ttnn.deallocate(s); ttnn.deallocate(z)
        med = statistics.median(ts)
        out["arms"][name] = {"per_recycle_s": med / args.n_recycles, "total_s": med,
                             "reps_s": [round(t, 4) for t in ts],
                             "transpose_dest": routes}
        print(f"{name:12s} {med / args.n_recycles:.4f} s/recycle  routes {routes}", flush=True)
    T._transpose_memory_config = orig_tmc

    eq_s = bool(torch.equal(ref[0], got[0]))
    eq_z = bool(torch.equal(ref[1], got[1]))
    out["bit_exact"] = {"s": eq_s, "z": eq_z,
                        "max_abs_z": float((ref[1].float() - got[1].float()).abs().max()),
                        "max_abs_s": float((ref[0].float() - got[0].float()).abs().max())}
    a, b, c = (out["arms"][k]["per_recycle_s"] for k in ("headroom", "reserve", "headroom_aa"))
    out["speedup"] = a / b
    out["aa_floor"] = a / c
    print(f"\n{args.aa} aa: {a:.4f} -> {b:.4f} s/recycle, {a / b:.4f}x, A/A floor {a / c:.4f}, "
          f"bit-exact z={eq_z} s={eq_s} (max abs {out['bit_exact']['max_abs_z']})")
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
