#!/usr/bin/env python3
"""Does `_tri_att_sdpa_hifi` use its `scale` argument? Compare the two outputs to each other."""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                     # noqa: E402
from tt_bio import tenstorrent as T                                    # noqa: E402

N, BATCH, HEADS, HEAD_DIM = 256, 64, 4, 32
S = HEAD_DIM ** -0.5
SQ = HEAD_DIM ** 0.5


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
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev)                          # noqa: E731
    q, k, v, bias = (up(t) for t in (hq, hk, hv, hb))
    qk = torch.einsum("shqd,shkd->shqk", hq.double(), hk.double())
    ref_s = torch.einsum("shqk,shkd->shqd", ((qk + hb.double()) * S).softmax(-1), hv.double())
    ref_sq = torch.einsum("shqk,shkd->shqd", ((qk + hb.double()) * SQ).softmax(-1), hv.double())

    outs = {}
    for scal, lab in ((S, "s"), (SQ, "sqrt_h"), (1.0, "one")):
        o = T._tri_att_sdpa_hifi(q, k, v, bias, scal)
        outs[lab] = ttnn.to_torch(o).double().clone()
        ttnn.deallocate(o)

    rms = lambda t: float(torch.sqrt(torch.mean(t ** 2)))               # noqa: E731
    res = {"ref_s_rms": rms(ref_s), "ref_sq_rms": rms(ref_sq)}
    for lab, t in outs.items():
        res[f"out_{lab}_rms"] = rms(t)
        res[f"out_{lab}_abs_vs_ref_s"] = rms(t - ref_s)
        res[f"out_{lab}_abs_vs_ref_sq"] = rms(t - ref_sq)
    res["s_vs_sqrt_h_maxabs"] = float((outs["s"] - outs["sqrt_h"]).abs().max())
    res["s_vs_one_maxabs"] = float((outs["s"] - outs["one"]).abs().max())
    for kk, vv in res.items():
        print(f"{kk:28s} {vv:.8f}", flush=True)
    ttnn.close_device(dev)
    out = ROOT / "perf/land_standing/out/convention/scale_used.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": N, "batch": BATCH, **res}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
