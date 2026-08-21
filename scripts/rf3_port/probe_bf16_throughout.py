#!/usr/bin/env python3
"""Is the Pairformer s-track's 11x a defect, or the wrong denominator?

The trunk block scores s rel_rms 0.021 against a torch-autocast golden whose own
bf16-vs-fp32 spread is 0.0019, which reads as 11x. But autocast is not what the device
does: it keeps activations in fp32 between ops and casts only matmul inputs, while ttnn
carries bf16 activations throughout. This runs the same block three ways on the same
captured real input and asks whether the port sits at the ceiling of the precision model
it actually implements.

    fp32          no autocast, fp32 weights and activations           -- the truth
    autocast      what upstream runs (Lightning bf16 AMP)             -- today's golden
    bf16_through  bf16 weights AND bf16 activations, no autocast      -- what ttnn does

    python3 scripts/rf3_port/probe_bf16_throughout.py \
        --ckpt ~/rf3_ref_work/rf3_latest.ckpt --golden ~/rf3_ref_work/trunk_io_ligands.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

C_S, C_Z = 384, 128


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def block(block_sd: dict):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import PairformerBlock
    blk = PairformerBlock(
        c_s=C_S, c_z=C_Z, p_drop=0.0,
        triangle_multiplication={"d_hidden": 128},
        triangle_attention={"n_head": 4, "d_hidden": 32},
        attention_pair_bias={"n_head": 16},
    )
    missing, unexpected = blk.load_state_dict(block_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"block weights mismatch: {len(missing)} / {len(unexpected)}")
    return blk.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.",
                    help="checkpoint prefix of the block to score")
    ap.add_argument("--out")
    args = ap.parse_args()

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    if not sd:
        raise KeyError(f"no weights under {args.stack!r}")

    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()

    blk = block(sd)
    with torch.no_grad():
        # The attention force-casts its input to bf16, so an fp32 control has to switch
        # that off or it dies on bf16 activations against fp32 weights. Same control the
        # parity harness uses.
        blk.attention_pair_bias.force_bfloat16 = False
        s_f32, z_f32 = blk(s.clone(), z.clone())
        blk.attention_pair_bias.force_bfloat16 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            s_amp, z_amp = blk(s.clone(), z.clone())
        # Third arm: bf16 STORAGE throughout. Every leaf module's output is rounded to
        # bf16 and carried on in fp32 containers, and the weights are bf16-rounded too.
        # Running the vendored block in true torch bf16 is not possible -- it builds an
        # fp32 Beta_II internally and dies on the dtype mismatch -- and rounding at every
        # op boundary is the closer model of ttnn anyway: bf16 tensors in DRAM/L1 with
        # fp32 accumulation inside each kernel (fp32_dest_acc_en, HiFi4), not bf16
        # accumulation. Autocast is the arm that does NOT match: it keeps fp32
        # activations between ops and only casts matmul inputs.
        blk_b = block({k: v.bfloat16().float() for k, v in sd.items()})
        blk_b.attention_pair_bias.force_bfloat16 = False

        def round_bf16(x):
            if torch.is_tensor(x) and x.is_floating_point():
                return x.bfloat16().float()
            if isinstance(x, (tuple, list)):
                return type(x)(round_bf16(v) for v in x)
            return x

        for m in blk_b.modules():
            if not list(m.children()):
                m.register_forward_hook(lambda _m, _i, out: round_bf16(out))
        s_bf, z_bf = blk_b(round_bf16(s.clone()), round_bf16(z.clone()))

    rep = {
        "tokens": int(z.shape[-2]),
        "stack": args.stack,
        "s_ref_std": round(float(s_f32.std()), 4),
        "z_ref_std": round(float(z_f32.std()), 4),
        # The two candidate ceilings, both against the fp32 truth.
        "s_autocast_vs_fp32": round(rel_rms(s_amp, s_f32), 6),
        "z_autocast_vs_fp32": round(rel_rms(z_amp, z_f32), 6),
        "s_bf16_storage_vs_fp32": round(rel_rms(s_bf, s_f32), 6),
        "z_bf16_storage_vs_fp32": round(rel_rms(z_bf, z_f32), 6),
        # And against each other, which is the number a device score against the
        # autocast golden is really being compared with.
        "s_bf16_storage_vs_autocast": round(rel_rms(s_bf, s_amp), 6),
        "z_bf16_storage_vs_autocast": round(rel_rms(z_bf, z_amp), 6),
    }
    rep["s_bf16_penalty"] = round(rep["s_bf16_storage_vs_fp32"]
                                 / rep["s_autocast_vs_fp32"], 2)
    rep["z_bf16_penalty"] = round(rep["z_bf16_storage_vs_fp32"]
                                 / rep["z_autocast_vs_fp32"], 2)
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
