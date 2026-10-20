#!/usr/bin/env python3
"""Which half of the card softmax is wrong on the IPA huge-logit rows: the forward, or the backward
fed an exact y. Captures layer 1's real IPA logits [H, N, N] in float64 (af2_structure.NP), runs
ttnn.softmax and autograd.softmax_bw on card, and grades each against float64 per head.

usage: softmax_probe.py --params <npz> --inputs <sm_inputs.npz>
"""
import argparse, json

import numpy as np


def main(a):
    from tt_bio import af2_structure as S
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if k.startswith(S.ROOT)}
    k = S.pack(p)
    d = np.load(a.inputs)
    single, pair = (np.asarray(d[f"{nm}_0"], np.float64) for nm in ("single", "pair"))
    real = single.shape[0]
    n = -(-real // 32) * 32
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    single = np.pad(single, ((0, n - real), (0, 0)))
    pair = np.pad(pair, ((0, n - real), (0, n - real), (0, 0)))
    mu = single.mean(-1, keepdims=True); var = ((single - mu) ** 2).mean(-1, keepdims=True)
    act0 = ((single - mu) / np.sqrt(var + 1e-5) * p[S.ROOT + "single_layer_norm//scale"]
            + p[S.ROOT + "single_layer_norm//offset"]) @ p[S.ROOT + "initial_projection//weights"] \
        + p[S.ROOT + "initial_projection//bias"]
    o = S.NP(np)
    cap, sm = [], o.softmax
    o.softmax = lambda x: (cap.append(x), sm(x))[1]
    S.stack(o, k, act0, pair, S.key_bias(mask), layers=a.layer)
    x = cap[a.layer - 1]                                               # [H, N, N], padded keys masked
    y64 = sm(x)
    g = np.random.default_rng(3).standard_normal(x.shape) * mask[None, :, None]
    dx64 = y64 * (g - (g * y64).sum(-1, keepdims=True))

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import autograd as A, tenstorrent as T
    dev = T.get_device(trace="diffusion")
    put = lambda v: ttnn.from_torch(torch.tensor(v, dtype=torch.float32), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
    get = lambda t: ttnn.to_torch(t).double().numpy()[:, :real]
    cfg = A.precise_config()
    yc_t = ttnn.softmax(put(x), dim=-1, compute_kernel_config=cfg)
    yc = ttnn.to_torch(yc_t).double().numpy()
    dx_exact_y = get(A.softmax_bw(put(y64), put(g), config=cfg))
    dx_card_y = get(A.softmax_bw(yc_t, put(g), config=cfg))
    dx_refined = get(A.softmax_bw(yc_t, put(g), config=cfg, refine=True))
    yc, y64r, dx64, xr = yc[:, :real], y64[:, :real], dx64[:, :real], x[:, :real]
    rel = lambda e, r: float(np.linalg.norm(e - r) / np.linalg.norm(r))
    xs = np.sort(xr, -1)
    out = {"n": n, "real": real,
           "row_max_median_per_head": np.median(xs[..., -1], -1).round(1).tolist(),
           "top2_gap_median_per_head": np.median(xs[..., -1] - xs[..., -2], -1).round(1).tolist(),
           "fwd_rel": rel(yc, y64r), "fwd_rel_per_head": [round(rel(yc[h], y64r[h]), 6) for h in range(len(x))],
           "fwd_maxabs_per_head": [float(f"{np.abs(yc[h] - y64r[h]).max():.3g}") for h in range(len(x))],
           "fwd_rowsum_maxdev": float(np.abs(yc.sum(-1) - 1).max()),
           "bw_exact_y_rel": rel(dx_exact_y, dx64), "bw_card_y_rel": rel(dx_card_y, dx64),
           "bw_card_y_err_over_g_per_head": [float(f"{np.linalg.norm(dx_card_y[h] - dx64[h]) / np.linalg.norm(g[h]):.3g}") for h in range(len(x))],
           "bw_refined_rel": rel(dx_refined, dx64),
           "bw_refined_err_over_g_per_head": [float(f"{np.linalg.norm(dx_refined[h] - dx64[h]) / np.linalg.norm(g[h]):.3g}") for h in range(len(x))]}
    bad = np.unravel_index(np.argmax(np.abs(yc - y64r)), yc.shape)
    out["worst_fwd"] = {"at": [int(i) for i in bad], "card": float(yc[bad]), "ref": float(y64r[bad]),
                        "row_max": float(xr[bad[:2]].max()), "x": float(xr[bad])}
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layer", type=int, default=1, help="whose logits: 1..8")
    main(ap.parse_args())
