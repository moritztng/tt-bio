#!/usr/bin/env python3
"""Mechanism for the AttentionPairBias 13x: bf16 attention logits.

`probe_apb_precision.py` leaves one candidate standing -- the gap survives with the pair
bias neutralised, with fp32 storage around the op, and with the bias-scale convention
flipped. What is left is the score path. tt-bio's fp32-softmax attention takes the q@k
matmul's bf16 output and upcasts THAT (`tenstorrent.py:1447`), so the logits are rounded
to bf16 before the reduction. This measures what that rounding is worth on the real
operating point, on the host, and reports what dtype the reference's own einsum runs in.

    python3 scripts/rf3_port/probe_apb_logits.py \
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


def rel_rms(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--out")
    args = ap.parse_args()

    from tt_bio._vendor.rf3.model.layers.pairformer_layers import PairformerBlock

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    blk = PairformerBlock(c_s=C_S, c_z=C_Z, p_drop=0.0,
                          triangle_multiplication={"d_hidden": 128},
                          triangle_attention={"n_head": 4, "d_hidden": 32},
                          attention_pair_bias={"n_head": 16})
    blk.load_state_dict(sd, strict=False)
    blk.eval()
    apb = blk.attention_pair_bias

    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()

    with torch.no_grad():
        a = apb.ln_1(s)
        q = apb.to_q(a) / (24.0 ** 0.5)
        k = apb.to_k(a)
        v = apb.to_v(a)
        g = apb.to_g(a)
        bias = apb.to_b(apb.ln_0(z))
        logits = torch.einsum("...ihd,...jhd->...ijh", q, k) + bias

        def head(x):
            return x

        def attend(lg, vv):
            p = torch.softmax(lg, dim=-2)
            o = torch.einsum("...ijh,...jhc->...ihc", p, vv)
            return apb.to_a((g * o).flatten(start_dim=-2))

        out_f32 = attend(logits, v)
        out_bf_logits = attend(logits.bfloat16().float(), v)
        out_bf_v = attend(logits, v.bfloat16().float())
        out_bf_qk = attend(torch.einsum("...ihd,...jhd->...ijh",
                                        q.bfloat16().float(), k.bfloat16().float())
                           + bias.bfloat16().float(), v.bfloat16().float())

        # What dtype does the reference's own score einsum run in under autocast? If
        # autocast does not put einsum on its bf16 list, the golden's logits are fp32 and
        # the port's are not, which is the whole gap.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            amp_logits = torch.einsum("...ihd,...jhd->...ijh", q, k)
            amp_linear = apb.to_q(a)

    rep = {
        "tokens": int(z.shape[-2]),
        "logit_std": round(float(logits.std()), 4),
        "logit_absmax": round(float(logits.abs().max()), 4),
        "logit_p99_abs": round(float(logits.abs().flatten().kthvalue(
            int(0.99 * logits.numel()))[0]), 4),
        "autocast_einsum_dtype": str(amp_logits.dtype),
        "autocast_linear_dtype": str(amp_linear.dtype),
        # each rounding's own cost on the op's output, measured against the fp32 path
        "cost_bf16_logits": round(rel_rms(out_bf_logits, out_f32), 6),
        "cost_bf16_v": round(rel_rms(out_bf_v, out_f32), 6),
        "cost_bf16_qk_and_v": round(rel_rms(out_bf_qk, out_f32), 6),
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
