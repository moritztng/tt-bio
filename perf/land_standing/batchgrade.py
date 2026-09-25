#!/usr/bin/env python3
"""Does the leading (batch) dim change what the fused path COMPUTES, not just which rung it picks?

The firing census established that batch does not change the rung. The grade is a different
question: khole.py runs batch == n and reports the fused arm 9.3x its reference's rms, while the
same call at batch 8 sits at 0.021 relative. One of those is wrong and the difference is the
batch. Same n, same seed order, same call, batch swept.
"""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                     # noqa: E402
from tt_bio import tenstorrent as T                                    # noqa: E402

N, HEADS, HEAD_DIM = 256, 4, 32
S = HEAD_DIM ** -0.5


def main():
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    rows = []
    for batch in (8, 32, 64, N):
        g = torch.Generator().manual_seed(100)
        mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
        hq, hk, hv = (mk(batch, HEADS, N, HEAD_DIM) for _ in range(3))
        hb = mk(1, HEADS, N, N)
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=dev)                        # noqa: E731
        q, k, v, bias = (up(t) for t in (hq, hk, hv, hb))
        qk = torch.einsum("shqd,shkd->shqk", hq.double(), hk.double())
        ref = torch.einsum("shqk,shkd->shqd", ((qk + hb.double()) * S).softmax(-1), hv.double())
        o = T._tri_att_sdpa_hifi(q, k, v, bias, S)
        got = ttnn.to_torch(o).double()
        r = {"batch": batch, "out_shape": list(got.shape), "ref_shape": list(ref.shape),
             "ref_rms": float(torch.sqrt(torch.mean(ref ** 2))),
             "out_rms": float(torch.sqrt(torch.mean(got ** 2))),
             "abs_rmsd": float(torch.sqrt(torch.mean((got - ref) ** 2)))}
        r["rel"] = r["abs_rmsd"] / r["ref_rms"]
        rows.append(r)
        print(f"batch={batch:4d} out{r['out_shape']} ref{r['ref_shape']} "
              f"ref_rms={r['ref_rms']:.5f} out_rms={r['out_rms']:.5f} "
              f"abs={r['abs_rmsd']:.5f} rel={r['rel']:.5f}", flush=True)
        for t in (q, k, v, bias, o):
            ttnn.deallocate(t)
    ttnn.close_device(dev)
    out = ROOT / "perf/land_standing/out/convention/batch_grade.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": N, "heads": HEADS, "head_dim": HEAD_DIM, "rows": rows},
                              indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
