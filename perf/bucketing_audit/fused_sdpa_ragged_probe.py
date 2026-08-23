"""P1.4: does the FUSED triangle-attention SDPA read its ragged key tail, and by how much?

The audit's 71-76x number was measured on `ttnn.transformer.scaled_dot_product_attention`.
`tt_bio.triatt_sdpa.sdpa` demonstrably SERVES ragged calls (1208 of 1208 on a 98-aa protenix-v2
fold) because `sdpa_generic.plan` derives `use_padded_mask` from `padded_shape` and so cannot see
raggedness at all -- but its own error at a ragged axis was unmeasured. This measures it.

No poisoned tail is needed, and that matters: a logical-ragged VIEW over a poisoned padded buffer is
not constructible from Python (`ttnn.reshape` refuses to shrink a logical volume), which is what
stopped the first pass. The defect is self-poisoning instead. `from_torch` zeroes the physical tail,
so a padded key column has k = 0 and therefore a raw score of 0, and the caller's additive bias
covers only the logical extent so the padded column's bias is 0 too. Its score is exactly 0. Push
the REAL scores down with a constant bias of -20 and the arithmetic is:

    correct     sum_i exp(s_i) v_i / sum_i exp(s_i)                    over the 98 real columns
    with tail   sum_i exp(s_i) v_i / (sum_i exp(s_i) + n_pad * 1)      v of a padded key is 0

At bias -20 the real denominator is ~98*exp(-20) = 2e-7 and the 30 padded columns add 30, so an
unmasked kernel returns ~1e-8 of the right answer. There is no threshold to tune.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=... python3 <this>
"""
import json
import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tt_bio import triatt_sdpa
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device

torch.manual_seed(893)

B, H, DH = 1, 4, 32
BIAS = -20.0          # pushes every real score far below the padded columns' 0
SCALE = DH ** -0.5


def _tt(t, dev):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def reference(q, k, v, bias):
    """fp32 softmax attention over the LOGICAL extent only."""
    s = (q.float() @ k.float().transpose(-1, -2)) * SCALE + bias.float()
    return torch.softmax(s, dim=-1) @ v.float()


def run(dev, S):
    q = torch.randn(B, H, S, DH) * 0.1
    k = torch.randn(B, H, S, DH) * 0.1
    v = torch.randn(B, H, S, DH)
    bias = torch.full((1, 1, S, S), BIAS)
    # bf16 round-trip the inputs so the reference sees exactly what the device sees.
    q, k, v, bias = (t.to(torch.bfloat16).float() for t in (q, k, v, bias))
    want = reference(q, k, v, bias)[0, 0]

    qt, kt, vt, bt = (_tt(t, dev) for t in (q, k, v, bias))
    padded = int(kt.padded_shape[-2])
    out = {"S": S, "padded": padded, "ragged": padded != S, "n_pad": padded - S}

    # Chunk pairs the fused gate can accept: both must divide the PADDED extent, which is what
    # `use_padded_mask` is computed from.
    served = None
    for qc in (padded, padded // 2, 64, 32):
        for kc in (padded, padded // 2, 64, 32):
            if padded % qc or padded % kc or qc < 32 or kc < 32:
                continue
            o = triatt_sdpa.sdpa(qt, kt, vt, bt, SCALE, qc, kc)
            if o is not None:
                served = (qc, kc)
                got = ttnn.to_torch(o).float()[0, 0]
                break
        if served:
            break
    if served is None:
        out["fused"] = "declined at every chunk pair"
    else:
        out["fused_chunks"] = list(served)
        out["fused_ratio"] = float((got.abs().sum() / want.abs().sum()).item())
        out["fused_relerr"] = float(((got - want).abs().max() / want.abs().max()).item())

    # The stock op at the same shapes, as the already-known-bad control.
    o = ttnn.transformer.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=bt, is_causal=False, scale=SCALE)
    got = ttnn.to_torch(o).float()[0, 0]
    out["stock_ratio"] = float((got.abs().sum() / want.abs().sum()).item())
    out["stock_relerr"] = float(((got - want).abs().max() / want.abs().max()).item())

    # What an unmasked tail predicts, from the arithmetic above.
    s = (q.float() @ k.float().transpose(-1, -2)) * SCALE + bias.float()
    den = torch.exp(s).sum(-1)
    out["predicted_ratio_if_unmasked"] = float((den / (den + out["n_pad"])).mean().item())
    return out


if __name__ == "__main__":
    dev = get_device()
    print("grid", COMPUTE_GRID_MAIN)
    rows = [run(dev, S) for S in (128, 98, 160, 130)]
    print()
    print("%5s %7s %7s %14s %14s %14s %14s" % (
        "S", "padded", "n_pad", "fused ratio", "fused relerr", "stock ratio", "stock relerr"))
    for r in rows:
        print("%5d %7d %7d %14.6g %14.6g %14.6g %14.6g" % (
            r["S"], r["padded"], r["n_pad"], r.get("fused_ratio", float("nan")),
            r.get("fused_relerr", float("nan")), r["stock_ratio"], r["stock_relerr"]))
    print()
    for r in rows:
        print("S=%d predicted ratio if the tail is unmasked: %.6g  (chunks %s)"
              % (r["S"], r["predicted_ratio_if_unmasked"], r.get("fused_chunks", r.get("fused"))))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fused_sdpa_ragged.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
