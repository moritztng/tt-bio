#!/usr/bin/env python3
"""Does khole.py's own Rungs wrapper change the result it then grades?

Same operands, same call, the only variable is whether `triatt_sdpa.sdpa` is wrapped by the
counter khole installs to record which rungs were offered.
"""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_tapedfwd"))
import torch, ttnn                                                     # noqa: E402
from tt_bio import tenstorrent as T, triatt_sdpa as TS                 # noqa: E402

N, BATCH, HEADS, HEAD_DIM = 256, 64, 4, 32
S = HEAD_DIM ** -0.5


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
    ref = torch.einsum("shqk,shkd->shqd", ((qk + hb.double()) * S).softmax(-1), hv.double())
    rms = lambda t: float(torch.sqrt(torch.mean(t ** 2)))               # noqa: E731

    res = {"ref_rms": rms(ref)}
    o = T._tri_att_sdpa_hifi(q, k, v, bias, S ** -1)
    plain = ttnn.to_torch(o).double().clone()
    ttnn.deallocate(o)
    res["plain_abs"] = rms(plain - ref)

    import khole                                                        # noqa: E402
    rungs = khole.Rungs(TS.sdpa)
    TS.sdpa = rungs
    rungs.seen, rungs.on = [], True
    T._TRIATT_HIFI_OVER_L1.clear()
    o = T._tri_att_sdpa_hifi(q, k, v, bias, S ** -1)
    wrapped = ttnn.to_torch(o).double().clone()
    ttnn.deallocate(o)
    TS.sdpa = rungs.real
    res["wrapped_abs"] = rms(wrapped - ref)
    res["plain_vs_wrapped_maxabs"] = float((plain - wrapped).abs().max())
    res["rungs_seen"] = [(r["q_chunk"], r["k_chunk"], r["served"]) for r in rungs.seen]

    for kk, vv in res.items():
        print(f"{kk:28s} {vv}", flush=True)
    ttnn.close_device(dev)
    out = ROOT / "perf/land_standing/out/convention/rungs_wrapper.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": N, "batch": BATCH, **res}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
