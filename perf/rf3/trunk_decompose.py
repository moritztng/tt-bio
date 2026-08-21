#!/usr/bin/env python3
"""Where the RF3 trunk's time goes, per component, and what roof each one is against.

After the atom-pair-track lever the trunk is 77% of a 512 aa fold and 90% of a 1024 aa
one, and it grows N^2.5. Clearing the 4x bar needs a further 2.6x on it at 512 aa, so the
first question is not what to try but which of the six things a Pairformer block does
carries the cost.

Every component of every block is wrapped, and the totals accumulated by name, so this is
the whole 48-block stack rather than one block assumed to stand for the rest. The syncs
are ON: this over-syncs and inflates the total, which is what makes the numbers usable as
an attribution and useless as a headline
(`tt-bio-isolated-op-timing-oversync-inflates-cost`). The recycles figure to quote is the
one from tt_rf3_bench.py without --breakdown.

Each component is priced against the roofs measured on this card in perf/rf3/roofs.py,
not against a spec sheet: 103.31 TF/s at HiFi4 on a square matmul, 17.41 TF/s on a K-thin
one, 440.4 GB/s DRAM.
"""
from __future__ import annotations

import argparse
import contextlib
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


class Acc:
    """Accumulating timer: many calls per name, synced at both ends of each."""

    def __init__(self, device):
        import ttnn
        self._ttnn, self._device = ttnn, device
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.on = False

    @contextlib.contextmanager
    def span(self, name):
        if not self.on:
            yield
            return
        self._ttnn.synchronize_device(self._device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._ttnn.synchronize_device(self._device)
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
            self.n[name] = self.n.get(name, 0) + 1

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **kw):
            with self.span(name):
                return fn(*a, **kw)
        setattr(obj, attr, timed)

    def reset(self):
        self.t, self.n = {}, {}


def instrument(tt, acc: Acc):
    r = tt.recycler
    acc.wrap(r, "process_zh", "trunk.process_zh")
    acc.wrap(r, "process_sh", "trunk.process_sh")
    acc.wrap(r, "template_embedder", "trunk.template")
    acc.wrap(r, "msa_module", "trunk.msa")
    for b in r.pairformer.blocks:
        acc.wrap(b, "triangle_multiplication_start", "block.tri_mul_start")
        acc.wrap(b, "triangle_multiplication_end", "block.tri_mul_end")
        acc.wrap(b, "triangle_attention_start", "block.tri_att_start")
        acc.wrap(b, "triangle_attention_end", "block.tri_att_end")
        acc.wrap(b, "transition_z", "block.transition_z")
        acc.wrap(b, "attention_pair_bias", "block.attn_pair_bias")
        acc.wrap(b, "transition_s", "block.transition_s")
    return len(r.pairformer.blocks)


def trimul_cost(i: int, c_hidden: int = 128, c_z: int = 128) -> dict:
    """FLOPs and DRAM bytes for one TriangleMultiplication at I tokens.

    Two [I, I, c_z] -> [I, I, 2*c_hidden] projections in, the [I, I] x [I, I] contraction
    over the third token index per channel, one [I, I, c_hidden] -> [I, I, c_z] out.
    """
    proj = 2 * i * i * c_z * 2 * c_hidden + 2 * i * i * c_hidden * c_z
    contract = 2 * c_hidden * i * i * i
    bytes_ = 2 * (3 * i * i * c_z + 4 * i * i * c_hidden)
    return {"gflop": (proj + contract) / 1e9, "contract_share": contract / (proj + contract),
            "gbyte": bytes_ / 1e9}


def triatt_cost(i: int, n_heads: int = 4, head_dim: int = 32, c_z: int = 128) -> dict:
    """qkv projections plus the per-row [I, I] x [I, I] attention, per token row."""
    d = n_heads * head_dim
    qkv = 2 * i * i * c_z * 3 * d
    attn = 2 * i * (2 * n_heads * i * i * head_dim)
    out = 2 * i * i * d * c_z
    bytes_ = 2 * (i * i * c_z * 2 + i * i * d * 4)
    return {"gflop": (qkv + attn + out) / 1e9, "attn_share": attn / (qkv + attn + out),
            "gbyte": bytes_ / 1e9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    # Triangle attention's fp32-softmax tail keeps its score block L1-resident when the block fits
    # a per-core budget. This overrides that budget for an A/B; 0 keeps the shipped value.
    ap.add_argument("--fp32_l1_bytes_per_core", type=int, default=0)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                   seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    from tt_bio import tenstorrent as tt_mod
    if args.fp32_l1_bytes_per_core:
        tt_mod._FP32_SOFTMAX_L1_BYTES_PER_CORE = args.fp32_l1_bytes_per_core
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

    acc = Acc(device)
    tt_mod.FP32_SOFTMAX_STATS.update({k: 0 for k in tt_mod.FP32_SOFTMAX_STATS})
    host = HostInputs.build(f, device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)
    # After the template projection, not before: wrapping an attribute replaces the
    # module object with a function, and `embed_template_feats` is reached off it.
    n_blocks = instrument(tt, acc)

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

    run()                                   # warm
    acc.on = True
    acc.reset()
    wall = run()
    acc.on = False

    per = {k: v / args.n_recycles for k, v in acc.t.items()}
    synced_recycle = wall / args.n_recycles
    named = sum(per.values())
    rows = sorted(per.items(), key=lambda kv: -kv[1])

    i_tok = host.n_token
    tm, ta = trimul_cost(i_tok), triatt_cost(i_tok)
    rep = {"aa": args.aa, "n_token": i_tok, "n_blocks": n_blocks,
           "fp32_softmax_stats": dict(tt_mod.FP32_SOFTMAX_STATS),
           "fp32_l1_bytes_per_core": (args.fp32_l1_bytes_per_core
                                      or tt_mod._FP32_SOFTMAX_L1_BYTES_PER_CORE),
           "n_recycles": args.n_recycles,
           "synced_per_recycle_s": synced_recycle,
           "per_recycle_s": {k: round(v, 5) for k, v in per.items()},
           "calls_per_recycle": {k: v // args.n_recycles for k, v in acc.n.items()},
           "unattributed_s": synced_recycle - named,
           "per_block_ms": {k: round(v / n_blocks * 1e3, 3)
                            for k, v in per.items() if k.startswith("block.")},
           "roofline": {
               "tri_mul": {**tm,
                           "floor_s_square": tm["gflop"] / ROOF_SQUARE_TFS / 1e3,
                           "floor_s_kthin": tm["gflop"] / ROOF_KTHIN_TFS / 1e3,
                           "floor_s_dram": tm["gbyte"] / ROOF_DRAM_GBS},
               "tri_att": {**ta,
                           "floor_s_square": ta["gflop"] / ROOF_SQUARE_TFS / 1e3,
                           "floor_s_kthin": ta["gflop"] / ROOF_KTHIN_TFS / 1e3,
                           "floor_s_dram": ta["gbyte"] / ROOF_DRAM_GBS}}}
    print(f"{args.aa} aa, {i_tok} tokens, {n_blocks} blocks, "
          f"{synced_recycle:.3f} s/recycle WITH the attribution syncs on")
    for k, v in rows:
        share = v / synced_recycle * 100
        extra = (f"   {v / n_blocks * 1e3:7.3f} ms/block" if k.startswith("block.")
                 else "")
        print(f"  {k:24s} {v:8.3f} s  {share:5.1f}%{extra}")
    print(f"  {'unattributed':24s} {rep['unattributed_s']:8.3f} s")
    print(f"  fp32_softmax {rep['fp32_softmax_stats']} "
          f"l1_bytes_per_core={rep['fp32_l1_bytes_per_core']}")
    for name, c in (("tri_mul", tm), ("tri_att", ta)):
        r = rep["roofline"][name]
        print(f"  roof {name}: {c['gflop']:8.2f} GFLOP, {c['gbyte']:6.3f} GB -> "
              f"floor {r['floor_s_square'] * 1e3:6.2f} ms square / "
              f"{r['floor_s_kthin'] * 1e3:7.2f} ms K-thin / "
              f"{r['floor_s_dram'] * 1e3:6.2f} ms DRAM")
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
