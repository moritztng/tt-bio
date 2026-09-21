#!/usr/bin/env python3
"""D116: is the trunk's backward defect an fp32 island, a reference version, an accumulation
order, or the tape's own softmax rule?

One experiment answers all four, because the discriminator is cheap: take the y the card
actually produced, and evaluate the tape's backward RULE on it in exact float64 on the host.
If the rule is fine and the silicon is the problem, that arm lands on the float64 reference.
It does not.

Every number is a relative L2 against a float64 softmax evaluated on the SAME bf16 input
values the card saw, so what is scored is the operation and never the input rounding.
"""
import argparse, json, os, time

import torch
import ttnn

from tt_bio.tenstorrent import get_device


def rel_l2(a, b):
    d = (a - b).double()
    n = b.double()
    return float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(n))


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def stats(t):
    t = t.double()
    return dict(mean=float(t.mean()), rms=float(t.pow(2).mean().sqrt()),
                min=float(t.min()), max=float(t.max()))


def to_dev(dev, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def shipped_bw(y, g):
    """EXACTLY tt_bio/taped_ttnn.py::_v_softmax's backward on origin/main (lines 173-175)."""
    inner = ttnn.sum(ttnn.multiply(g, y), dim=-1, keepdim=True)
    return ttnn.multiply(y, ttnn.subtract(g, inner))


def renorm_bw(y, g):
    """of3t-apbgrad's repair: inner = sum(g*y) / sum(y)."""
    inner = ttnn.sum(ttnn.multiply(g, y), dim=-1, keepdim=True)
    rows = ttnn.sum(y, dim=-1, keepdim=True)
    return ttnn.multiply(y, ttnn.subtract(g, ttnn.divide(inner, rows)))


def one_case(dev, name, shape, std, mask_frac, seed, fp32_store=False):
    torch.manual_seed(seed)
    x = (torch.randn(*shape) * std)
    if mask_frac:
        n = shape[-1]
        k = int(n * mask_frac)
        x[..., n - k:] = -1e4                       # padding columns, upstream's mask value
    x = x.to(torch.bfloat16)                        # the values the card will see
    g = (torch.randn(*shape)).to(torch.bfloat16)

    dt = ttnn.float32 if fp32_store else ttnn.bfloat16
    x_tt = to_dev(dev, x, dt)
    g_tt = to_dev(dev, g, dt)
    y_tt = ttnn.softmax(x_tt, dim=-1)

    dx_ship_tt = shipped_bw(y_tt, g_tt)
    dx_rn_tt = renorm_bw(y_tt, g_tt)
    # A/A: the same device call twice must be bit-identical, or no arm below is attributable
    aa = torch.equal(ttnn.to_torch(shipped_bw(y_tt, g_tt)), ttnn.to_torch(dx_ship_tt))

    y = ttnn.to_torch(y_tt).double()
    dx_ship_dev = ttnn.to_torch(dx_ship_tt).double()
    dx_rn_dev = ttnn.to_torch(dx_rn_tt).double()

    # --- the float64 reference, on the card's own input values -----------------------------
    x64, g64 = x.double(), g.double()
    p = torch.softmax(x64, dim=-1)                       # rows sum to 1 to 1e-16
    inner_ref = (g64 * p).sum(-1, keepdim=True)
    dx_ref = p * (g64 - inner_ref)

    # --- the SAME TWO RULES, in exact float64, on the card's own y -------------------------
    # This is the discriminator. Infinite precision, device operand.
    ship_inner = (g64 * y).sum(-1, keepdim=True)
    dx_ship_f64 = y * (g64 - ship_inner)
    rows = y.sum(-1, keepdim=True)
    dx_rn_f64 = y * (g64 - ship_inner / rows)

    # --- what the row sum actually does ----------------------------------------------------
    dev_rows = rows.squeeze(-1)
    # decompose y - p into a per-row uniform SCALE part and a SHAPE part
    err = y - p
    scale_part = (rows - 1.0) * p
    shape_part = err - scale_part
    en = float(torch.linalg.vector_norm(err))

    # break control: score the reference against a column-permuted reference. If the metric
    # cannot tell these apart it is measuring nothing.
    perm = torch.randperm(shape[-1])
    dx_break = dx_ref[..., perm]

    return dict(
        case=name, shape=list(shape), std=std, mask_frac=mask_frac, seed=seed,
        storage="float32" if fp32_store else "bfloat16",
        aa_bit_identical=bool(aa),
        rowsum=stats(dev_rows), rowsum_dev_from_1=stats(dev_rows - 1.0),
        y_vs_f64=dict(rel_l2=rel_l2(y, p), cos=cos(y, p)),
        err_decomp=dict(scale_share=float(torch.linalg.vector_norm(scale_part)) / en,
                        shape_share=float(torch.linalg.vector_norm(shape_part)) / en,
                        err_norm=en),
        vjp=dict(
            shipped_rule_f64=dict(rel_l2=rel_l2(dx_ship_f64, dx_ref), cos=cos(dx_ship_f64, dx_ref)),
            renorm_rule_f64=dict(rel_l2=rel_l2(dx_rn_f64, dx_ref), cos=cos(dx_rn_f64, dx_ref)),
            shipped_device=dict(rel_l2=rel_l2(dx_ship_dev, dx_ref), cos=cos(dx_ship_dev, dx_ref)),
            renorm_device=dict(rel_l2=rel_l2(dx_rn_dev, dx_ref), cos=cos(dx_rn_dev, dx_ref)),
            break_control=dict(rel_l2=rel_l2(dx_break, dx_ref), cos=cos(dx_break, dx_ref)),
        ),
        # the invariant the attention backward below softmax depends on: rows of dx vanish
        dx_rowsum_rms=dict(
            reference=float(dx_ref.sum(-1).pow(2).mean().sqrt()),
            shipped_rule_f64=float(dx_ship_f64.sum(-1).pow(2).mean().sqrt()),
            renorm_rule_f64=float(dx_rn_f64.sum(-1).pow(2).mean().sqrt()),
            shipped_device=float(dx_ship_dev.sum(-1).pow(2).mean().sqrt()),
            renorm_device=float(dx_rn_dev.sum(-1).pow(2).mean().sqrt()),
        ),
        # relative to the cotangent's own scale, so it is readable next to a rel_l2
        dx_rowsum_rel=dict(
            shipped_device=float(dx_ship_dev.sum(-1).pow(2).mean().sqrt()
                                 / torch.linalg.vector_norm(dx_ref) * (dx_ref[..., 0].numel() ** 0.5)),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_d116/rowsum.json")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()

    dev = get_device()
    rows = []
    t0 = time.time()

    # 1. the trunk AttentionPairBias shape D116 was filed on, and an n-ladder on the row length
    for n in (64, 128, 256, 384, 512, 1024):
        rows.append(one_case(dev, f"apb_n{n}", (1, 4, 384, n), 3.0, 0.0, a.seed))
    # 2. the real trunk shape, 16 heads
    rows.append(one_case(dev, "apb_trunk_16h", (1, 16, 384, 384), 3.0, 0.0, a.seed))
    # 3. peakedness sweep: how flat/peaked the row is
    for std in (0.5, 1.0, 3.0, 6.0, 12.0):
        rows.append(one_case(dev, f"std{std}", (1, 4, 384, 384), std, 0.0, a.seed))
    # 4. masked rows, the attention case that actually runs
    for mf in (0.25, 0.5, 0.75):
        rows.append(one_case(dev, f"mask{mf}", (1, 4, 384, 384), 3.0, mf, a.seed))
    # 5. does fp32 STORAGE fix the row sum, i.e. is a forward-side repair available?
    rows.append(one_case(dev, "apb_trunk_fp32store", (1, 16, 384, 384), 3.0, 0.0, a.seed,
                         fp32_store=True))
    # 6. seed replicate, so a single draw is not the finding
    rows.append(one_case(dev, "apb_trunk_16h_seed2", (1, 16, 384, 384), 3.0, 0.0, a.seed + 1))

    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=os.uname().nodename, seconds=round(time.time() - t0, 1), cases=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
