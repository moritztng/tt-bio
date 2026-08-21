#!/usr/bin/env python3
"""Does the fused-vs-materialised accuracy ordering flip with sequence length?

The op-level sweep (`triatt_ckc_sweep.py`) says the fused SDPA at HiFi4 is 1.5-1.9x MORE accurate
than `_fp32_softmax_attention` at 128 and 512 tokens. The port's four parity fixtures, which are 8
to 53 tokens, say the Pairformer block is 3.4-5.3x LESS accurate with it. Length is the only
uncontrolled variable between the two, so this walks it directly: one real captured
triangle-attention call, sliced to n tokens and zero-padded back to a tile multiple exactly the way
a short fixture reaches the device, scored against fp64 on the same bf16 operands.

Real operands throughout, so this is not a synthetic-input measurement -- only the length is
synthetic.
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


def rel_rms(got, want):
    return float(((got - want) ** 2).mean().sqrt() / (want ** 2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--lens", default="8,12,41,53,64,96,128,192,256,384,512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]

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

    grab = {}
    orig = T._fp32_softmax_attention

    def spy(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype=ttnn.bfloat16,
            bias_scale_inv=None):
        shp = tuple(int(d) for d in q.shape)
        if (not grab and len(shp) == 4 and shp[0] == shp[2] and shp[0] > 1
                and abs(scale_inv - (bias_scale_inv or scale_inv)) < 1e-12):
            grab.update(q=ttnn.to_torch(q), k=ttnn.to_torch(k), v=ttnn.to_torch(v),
                        bias=ttnn.to_torch(bias), scale=scale_inv)
        return orig(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype, bias_scale_inv)

    T._fp32_softmax_attention = spy
    host = HostInputs.build(fo["feats"], device)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)
    tt.recycler(host, tmpl, host.msa_stack[0], s_inputs, s_init, z_init,
                ttnn.mul(s_init, 0.0), ttnn.mul(z_init, 0.0))
    T._fp32_softmax_attention = orig
    assert grab, "no Pairformer triangle-attention call captured"
    qh, kh, vh, bh, scale = (grab[x] for x in ("q", "k", "v", "bias", "scale"))
    print(f"captured q{list(qh.shape)} scale={scale:.6f}", flush=True)

    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rows = []
    for n in [int(x) for x in args.lens.split(",")]:
        if n > qh.shape[0]:
            continue
        pad = -n % 32
        cut = lambda t, d2: torch.nn.functional.pad(t, (0, 0, 0, pad) if d2 else (0, pad, 0, pad))
        # Batch is untiled, so a short fixture reaches the device as n rows of a tile-padded
        # sequence: q/k/v [n, H, n->n+pad, 32] and bias [1, H, n->n+pad, n->n+pad], zeros in the pad.
        q = cut(qh[:n, :, :n, :], True)
        k = cut(kh[:n, :, :n, :], True)
        v = cut(vh[:n, :, :n, :], True)
        b = cut(bh[:, :, :n, :n], False)
        S = n + pad
        ref = torch.empty((n, q.shape[1], S, q.shape[3]), dtype=torch.float64)
        q64, k64, v64, b64 = q.double(), k.double(), v.double(), b.double()
        for i in range(0, n, 8):
            sc = q64[i:i + 8] @ k64[i:i + 8].transpose(-1, -2)
            ref[i:i + 8] = torch.softmax((sc + b64) * scale, dim=-1) @ v64[i:i + 8]
        qd, kd, vd, bd = up(q), up(k), up(v), up(b)
        row = {"n": n, "padded": S, "pad_frac": round(pad / S, 4)}
        o = T._fp32_softmax_attention(qd, kd, vd, bd, scale_inv=scale,
                                      compute_kernel_config=kcfg, out_dtype=ttnn.bfloat16,
                                      bias_scale_inv=scale)
        row["materialised"] = rel_rms(ttnn.to_torch(o).double(), ref)
        ttnn.deallocate(o)
        o = T._tri_att_sdpa_hifi(qd, kd, vd, bd, scale)
        row["fused_hifi"] = rel_rms(ttnn.to_torch(o).double(), ref) if o is not None else None
        if o is not None:
            ttnn.deallocate(o)
        # the same maths in torch bf16 storage: the floor neither arm can beat
        cl = torch.empty((n, q.shape[1], S, q.shape[3]), dtype=torch.bfloat16)
        for i in range(0, n, 8):
            sc = q[i:i + 8].float() @ k[i:i + 8].float().transpose(-1, -2)
            cl[i:i + 8] = (torch.softmax((sc + b.float()) * scale, dim=-1)
                           @ v[i:i + 8].float()).bfloat16()
        row["bf16_ceiling"] = rel_rms(cl.double(), ref)
        row["ratio_fused_over_materialised"] = (
            round(row["fused_hifi"] / row["materialised"], 4) if row["fused_hifi"] else None)
        rows.append(row)
        for t in (qd, kd, vd, bd):
            ttnn.deallocate(t)
        fu = "declined" if row["fused_hifi"] is None else f"{row['fused_hifi']:.6e}"
        print(f"  n={n:4d} pad {row['pad_frac']:.2f}  materialised {row['materialised']:.6e}  "
              f"fused {fu}  ceiling {row['bf16_ceiling']:.6e}  "
              f"ratio {row['ratio_fused_over_materialised']}", flush=True)

    rep = {"aa": args.aa, "shape": [int(d) for d in qh.shape], "rows": rows}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
