#!/usr/bin/env python3
"""Trace the port's OWN AttentionPairBias chain and score every intermediate against fp32.

`probe_apb_stagewise.py` scored every stage teacher-forced on the reference's fp32
intermediate and found all nine at or near their ceiling (0.36x-2.22x), while the composed
op is 13.43x. So no op is individually wrong; the error is made by propagation. This runs
the port's real chain, keeps every intermediate, and scores it against the reference's fp32
stage with the reference's OWN autocast-bf16 stage as the ceiling. The stage where the
port's error jumps clear of the reference's is the one that propagates.

Also reports the softmax input's conditioning: a per-row (per query, per head) constant
offset is invisible to softmax but not to bf16 rounding, so `within_row_std` against
`global_std` says whether the logits' signal survives a bf16 round at all.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \
        python3 scripts/rf3_port/probe_apb_chain.py \
            --ckpt ~/rf3_ref_work/rf3_latest.ckpt \
            --golden ~/rf3_ref_work/trunk_io_ligands.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from scripts.rf3_port.probe_apb_stagewise import (  # noqa: E402
    HEAD_DIM, N_HEAD, PADDED, apb_inputs, ref_stages, rel_rms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--stack", default="shadow.recycler.pairformer_stack.0.")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--fp32-scores", action="store_true",
                    help="keep the q@k^T scores in fp32 instead of bf16")
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS, remap_pairformer_block
    from tt_bio.tenstorrent import (CORE_GRID_MAIN, PairformerLayer, batched_matmul,
                                    get_device)

    sd = {k[len(args.stack):]: v.float()
          for k, v in torch.load(args.ckpt, map_location="cpu",
                                 weights_only=False)["model"].items()
          if k.startswith(args.stack)}
    gold = torch.load(args.golden, weights_only=False)
    s, z = gold["in"]
    s = s.float().unsqueeze(0) if s.dim() == 2 else s.float()
    z = z.float().unsqueeze(0) if z.dim() == 3 else z.float()
    if args.crop:
        s, z = s[:, :args.crop], z[:, :args.crop, :args.crop]

    got, apb = apb_inputs(sd, s, z)
    a0, zz = got["in"][0].float(), got["in"][2].float()
    hi = ref_stages(apb, a0, zz, bf16=False)
    lo = ref_stages(apb, a0, zz, bf16=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer = PairformerLayer(32, 4, HEAD_DIM, N_HEAD, True, remap_pairformer_block(sd), cfg,
                            **PAIRFORMER_FLAGS)
    op = layer.attention_pair_bias
    scale = HEAD_DIM ** -0.5
    bsi = 1.0 / op._bias_scale

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    def host(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    # ---- the port's chain, op for op as AttentionPairBias.__call__ runs it ----
    trace = {}
    a = ttnn.layer_norm(tt(a0), weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                        epsilon=1e-5, compute_kernel_config=cfg)
    trace["a"] = host(a).reshape(hi["a"].shape)
    qkv = ttnn.linear(a, op.qkv_weight, bias=op.qkv_bias, compute_kernel_config=cfg,
                      core_grid=CORE_GRID_MAIN)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.unsqueeze(qkv, 1), num_heads=N_HEAD, num_kv_heads=N_HEAD,
        transpose_k_heads=False)
    for nm, dv in (("q", q), ("k", k), ("v", v)):
        trace[nm] = host(dv)[..., :HEAD_DIM].permute(0, 2, 1, 3)
    zn = ttnn.layer_norm(tt(zz), weight=op.z_norm_weight, bias=op.z_norm_bias, epsilon=1e-5,
                         compute_kernel_config=cfg)
    zb = ttnn.linear(zn, op.z_weight, compute_kernel_config=cfg, core_grid=CORE_GRID_MAIN)
    bias = ttnn.permute(zb, (0, 3, 1, 2))
    trace["b"] = (host(bias) * bsi).permute(0, 2, 3, 1)

    kt = ttnn.permute(k, (0, 1, 3, 2))
    sc = batched_matmul(q, kt, compute_kernel_config=cfg,
                        dtype=ttnn.float32 if args.fp32_scores else None)
    trace["logits"] = host(sc).permute(0, 2, 3, 1)
    sc_f = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
    bias_f = ttnn.multiply(ttnn.typecast(bias, ttnn.float32,
                                         memory_config=bias.memory_config()), bsi)
    att = ttnn.add_(sc_f, bias_f, input_tensor_a_activations=[
        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)])
    att = ttnn.softmax_in_place(att)
    trace["probs"] = host(att).permute(0, 2, 3, 1)
    attn_bf = ttnn.typecast(att, ttnn.bfloat16, memory_config=att.memory_config())
    o = batched_matmul(attn_bf, v, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
    trace["o"] = host(o)[..., :HEAD_DIM].permute(0, 2, 1, 3)
    o = o[:, :, :, :HEAD_DIM]
    o = ttnn.permute(o, (0, 1, 3, 2))
    o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
    o = ttnn.permute(o, (0, 2, 1))
    g = ttnn.linear(a, op.g_weight, compute_kernel_config=cfg, core_grid=CORE_GRID_MAIN)
    gated = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                          dtype=op.dtype)
    trace["gated"] = host(gated).reshape(hi["gated"].shape)
    out = ttnn.linear(gated, op.o_weight, compute_kernel_config=cfg,
                      core_grid=CORE_GRID_MAIN)
    trace["out"] = host(out).reshape(hi["out"].shape)

    rows = []
    for nm in ("a", "q", "k", "v", "b", "logits", "probs", "o", "gated", "out"):
        h, l = hi[nm], lo[nm]
        e_dev, e_ref = rel_rms(trace[nm], h), rel_rms(l, h)
        rows.append({"stage": nm, "err_vs_fp32": round(e_dev, 6),
                     "ref_bf16_err": round(e_ref, 6),
                     "x_ref": round(e_dev / e_ref, 2) if e_ref else None,
                     "fp32_std": round(float(h.std()), 4)})

    # Where does the o error come from -- the probs error, or the device matmul? Recompute
    # o on the host in fp64 from each arm's probs against the SAME fp32 v.
    def host_pv(pr):
        return torch.einsum("...ijh,...jhc->...ihc", pr.double(),
                            hi["v"].double().permute(0, 1, 2, 3)).float()
    attrib = {
        "o_from_port_probs_host": round(rel_rms(host_pv(trace["probs"]), hi["o"]), 6),
        "o_from_ref_bf16_probs_host": round(rel_rms(host_pv(lo["probs"]), hi["o"]), 6),
        "o_device_from_port_probs": round(rel_rms(trace["o"], hi["o"]), 6),
    }
    # structure of the probs error: is mass conserved, and do argmaxes move?
    def pstats(pr):
        rs = pr.sum(dim=2)                                    # sum over keys j
        am = pr.argmax(dim=2)
        return rs, am
    rs_hi, am_hi = pstats(hi["probs"])
    out_p = {}
    for nm, pr in (("port", trace["probs"]), ("ref_bf16", lo["probs"]), ("fp32", hi["probs"])):
        rs, am = pstats(pr)
        out_p[nm] = {"rowsum_mean": round(float(rs.mean()), 6),
                     "rowsum_min": round(float(rs.min()), 6),
                     "rowsum_max": round(float(rs.max()), 6),
                     "argmax_flip_frac": round(float((am != am_hi).float().mean()), 6),
                     "pmax_mean": round(float(pr.max(dim=2).values.mean()), 6)}
    attrib["probs_structure"] = out_p

    # conditioning of the softmax input: what a bf16 round of the logits costs relative to
    # the within-row spread the softmax actually sees
    lgs = hi["logits"] * scale                       # [B,I,J,H], softmax over J
    within = float((lgs - lgs.mean(dim=2, keepdim=True)).std())
    rnd = float((lgs.to(torch.bfloat16).float() - lgs).abs().mean())
    cond = {"scaled_logits_global_std": round(float(lgs.std()), 4),
            "scaled_logits_within_row_std": round(within, 4),
            "scaled_logits_absmax": round(float(lgs.abs().max()), 2),
            "bf16_round_mean_abs_err": round(rnd, 4),
            "round_err_over_within_row_std": round(rnd / within, 4)}

    # does the composed op agree with this replication?
    whole = op(ttnn.layer_norm(tt(a0), weight=layer.pre_norm_s_weight,
                               bias=layer.pre_norm_s_bias, epsilon=1e-5,
                               compute_kernel_config=cfg), tt(zz))
    rep_check = rel_rms(host(whole).reshape(hi["out"].shape), trace["out"])

    rep = {"stack": args.stack, "tokens": int(zz.shape[-2]), "crop": args.crop,
           "fp32_scores": args.fp32_scores,
           "replication_vs_op_rel_rms": float(f"{rep_check:.3e}"),
           "op_vs_golden_x_ceiling": round(
               rel_rms(host(whole).reshape(hi["out"].shape), lo["out"])
               / rel_rms(hi["out"], lo["out"]), 2),
           "chain": rows, "attribution": attrib, "softmax_conditioning": cond}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
