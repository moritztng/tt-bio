#!/usr/bin/env python3
"""Which function does each triangle-attention path actually compute?

`khole.py`'s float64 grade has never agreed with either device path on either board class. The
cheapest way to find out why is to stop reading the docstrings and cross every candidate call
convention against every candidate reference, at one length both paths serve.

Two axes:
  * the scalar. `_tri_att_sdpa_hifi(q, k, v, bias, scale)` names it `scale`;
    `_fp32_softmax_attention(..., scale_inv=...)` names the reciprocal. khole passes
    `scale ** -1` to BOTH, which cannot be right for both.
  * the bias. `_fp32_softmax_attention`'s docstring says the additive bias arrives PRE-BAKED by
    sqrt(h), and `bias_scale_inv` undoes that. khole passes a raw randn with bias_scale_inv=1.0.

A device arm matching a reference to ~1e-3 rel names the convention; everything else is noise.
"""
import itertools
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch                                                           # noqa: E402
import ttnn                                                            # noqa: E402
from tt_bio import tenstorrent as T                                    # noqa: E402

N, BATCH, HEADS, HEAD_DIM = 256, 8, 4, 32
S = HEAD_DIM ** -0.5          # 0.176777, the true softmax multiplier
SQ = HEAD_DIM ** 0.5          # 5.65685, its reciprocal


def refs(q, k, v, b):
    q, k, v, b = (t.double() for t in (q, k, v, b))
    qk = torch.einsum("shqd,shkd->shqk", q, k)
    out = {}
    for name, logits in {
        "mul_s_after_bias":  (qk + b) * S,      # bias treated as pre-baked by sqrt(h)
        "mul_s_before_bias": qk * S + b,        # bias treated as raw z
        "mul_sq_after_bias": (qk + b) * SQ,     # what khole currently asks the fused path for
        "mul_sq_before_bias": qk * SQ + b,
    }.items():
        out[name] = torch.einsum("shqk,shkd->shqd", logits.softmax(-1), v)
    return out


def rel(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)) / torch.sqrt(torch.mean(b ** 2)))


def main():
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    g = torch.Generator().manual_seed(100)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)   # noqa: E731
    hq, hk, hv = (mk(BATCH, HEADS, N, HEAD_DIM) for _ in range(3))
    hb = mk(1, HEADS, N, N)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    q, k, v, bias = (up(t) for t in (hq, hk, hv, hb))
    R = refs(hq, hk, hv, hb)

    arms = {}
    for scal, label in ((S, "scale=s"), (SQ, "scale=sqrt_h")):
        o = T._tri_att_sdpa_hifi(q, k, v, bias, scal)
        arms[f"fused({label})"] = None if o is None else ttnn.to_torch(o).double()
        if o is not None:
            ttnn.deallocate(o)
    for si, bsi in itertools.product((S, SQ), (1.0, S, SQ)):
        o = T._fp32_softmax_attention(q, k, v, bias, scale_inv=si,
                                      compute_kernel_config=T._SOFTMAX_PRECISE_CKC,
                                      out_dtype=ttnn.bfloat16, bias_scale_inv=bsi,
                                      accurate_softmax=False)
        tag = f"fp32(scale_inv={'s' if si == S else 'sqrt_h'},bias_scale_inv=" \
              f"{ {1.0: '1', S: 's', SQ: 'sqrt_h'}[bsi] })"
        arms[tag] = ttnn.to_torch(o).double()
        ttnn.deallocate(o)

    table = {}
    for a, t in arms.items():
        if t is None:
            table[a] = "declined"
            continue
        table[a] = {r: round(rel(t, R[r]), 6) for r in R}
        best = min(table[a], key=table[a].get)
        print(f"{a:38s} best={best:20s} rel={table[a][best]:.6f}  all={table[a]}", flush=True)
    ttnn.close_device(dev)

    out = ROOT / "perf/land_standing/out/convention/convention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": N, "batch": BATCH, "heads": HEADS, "head_dim": HEAD_DIM,
                               "table": table}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
