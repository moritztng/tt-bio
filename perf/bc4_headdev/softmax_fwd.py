#!/usr/bin/env python3
"""Where the card softmax forward loses precision, and a composition that keeps it. On layer L's
real IPA logits, against float64: ttnn.softmax; the composition exp(x - max) / sum with a plain
ttnn.sum; and the same with the row sum split into a bf16-exact high part plus the remainder, so
the reduce's TF32 operand rounding sees exact values. Also exp and the plain sum on their own.

usage: softmax_fwd.py --params <npz> --inputs <sm_inputs.npz> [--layer 1]
"""
import argparse, json

import numpy as np


def logits(a):
    from tt_bio import af2_structure as S
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if k.startswith(S.ROOT)}
    k = S.pack(p)
    d = np.load(a.inputs)
    single, pair = (np.asarray(d[f"{nm}_0"], np.float64) for nm in ("single", "pair"))
    n = single.shape[0]
    mu = single.mean(-1, keepdims=True); var = ((single - mu) ** 2).mean(-1, keepdims=True)
    act0 = ((single - mu) / np.sqrt(var + 1e-5) * p[S.ROOT + "single_layer_norm//scale"]
            + p[S.ROOT + "single_layer_norm//offset"]) @ p[S.ROOT + "initial_projection//weights"] \
        + p[S.ROOT + "initial_projection//bias"]
    o = S.NP(np)
    cap, sm = [], o.softmax
    o.softmax = lambda x: (cap.append(x), sm(x))[1]
    S.stack(o, k, act0, pair, S.key_bias(np.ones(n)), layers=a.layer)
    return cap[a.layer - 1], sm


def main(a):
    x, sm = logits(a)
    y64 = sm(x)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import autograd as A, tenstorrent as T
    dev = T.get_device(trace="diffusion")
    put = lambda v: ttnn.from_torch(torch.tensor(v, dtype=torch.float32), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
    get = lambda t: ttnn.to_torch(t).double().numpy()
    cfg = A.precise_config()
    xt = put(x)
    m = ttnn.max(xt, dim=-1, keepdim=True)
    e = ttnn.exp(ttnn.subtract(xt, m))
    e64 = np.exp(x - get(m))
    s_plain = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=cfg)
    hi = ttnn.typecast(ttnn.typecast(e, ttnn.bfloat16), ttnn.float32)
    s_split = ttnn.add(ttnn.sum(hi, dim=-1, keepdim=True, compute_kernel_config=cfg),
                       ttnn.sum(ttnn.subtract(e, hi), dim=-1, keepdim=True, compute_kernel_config=cfg))
    J = put(np.ones((x.shape[-1], x.shape[-1])))           # e @ J: every column is the row sum
    mmc = lambda v: ttnn.matmul(v, J, compute_kernel_config=cfg)
    s_mm = ttnn.add(mmc(hi), mmc(ttnn.subtract(e, hi)))
    rel = lambda e_, r: float(np.linalg.norm(e_ - r) / np.linalg.norm(r))
    s64 = e64.sum(-1, keepdims=True)
    yb = get(ttnn.softmax(xt, dim=-1, compute_kernel_config=cfg))
    yp, ys = get(ttnn.divide(e, s_plain)), get(ttnn.divide(e, s_split))
    ratio = np.where(y64 > 1e-3, yb / np.maximum(y64, 1e-300), np.nan)
    out = {"layer": a.layer, "exp_rel": rel(get(e), e64),
           "sum_plain_maxrel": float(np.abs(get(s_plain) / s64 - 1).max()),
           "sum_split_maxrel": float(np.abs(get(s_split) / s64 - 1).max()),
           "sum_mm_split_maxrel": float(np.abs(get(s_mm) / s64 - 1).max()),
           "composed_mm_split": {"rel": rel(get(ttnn.divide(e, s_mm)), y64),
                                 "maxabs": float(np.abs(get(ttnn.divide(e, s_mm)) - y64).max())},
           "ttnn_softmax": {"rel": rel(yb, y64), "maxabs": float(np.abs(yb - y64).max()),
                            "row_scale_share": float(np.nanstd(np.nanmean(ratio, -1)) / np.nanstd(ratio))},
           "composed_plain": {"rel": rel(yp, y64), "maxabs": float(np.abs(yp - y64).max())},
           "composed_split": {"rel": rel(ys, y64), "maxabs": float(np.abs(ys - y64).max())}}
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layer", type=int, default=1)
    main(ap.parse_args())
