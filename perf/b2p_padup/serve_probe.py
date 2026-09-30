#!/usr/bin/env python3
"""Where does the pad-up change the fused-HiFi triangle-attention route, at a given model's geometry?

For each axis, calls `_tri_att_sdpa_hifi` twice on the same operands, pad-up off then on (the
module constant is read at call time), and records served/declined for both, the length the pad-up
served at, and rel L2 of the pad-up output against a float64 softmax reference on the same
bf16 operands where the pad-up served. q/k/v [N, H, N, D] bf16, bias [1, H, N, N], batch = N as
triangle attention runs it.

OpenFold3's trunk: H 4, D 32, one_k_chunk=True (`openfold3.trunk` is the only site defaulting
`tri_att_sdpa_hifi` on, and it passes `tri_att_one_k_chunk` with it).

Usage: serve_probe.py --sizes 64,128,... [--one-k-chunk] [--out f.json]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--one-k-chunk", action="store_true")
    ap.add_argument("--ref-rows", type=int, default=8, help="batch rows graded against float64")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    dev = T.get_device()
    H, D = a.heads, a.head_dim
    res = {"grid": list(T.COMPUTE_GRID_MAIN), "heads": H, "head_dim": D,
           "one_k_chunk": a.one_k_chunk, "pad_up_tiles": T._TRIATT_HIFI_PAD_UP_TILES, "sizes": {}}
    tiles = T._TRIATT_HIFI_PAD_UP_TILES
    for S in [int(x) for x in a.sizes.split(",")]:
        torch.manual_seed(S)
        host = [torch.randn(S, H, S, D).to(torch.bfloat16) for _ in range(3)]
        hb = (torch.randn(1, H, S, S) * 2).to(torch.bfloat16)
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v, b = up(host[0]), up(host[1]), up(host[2]), up(hb)
        rec = {}
        for arm, t in (("off", 0), ("on", tiles)):
            T._TRIATT_HIFI_PAD_UP_TILES = t
            T.TRIATT_FUSED_HIFI_PADDED.pop(S, None)
            t0 = time.perf_counter()
            o = T._tri_att_sdpa_hifi(q, k, v, b, D ** -0.5, one_k_chunk=a.one_k_chunk)
            dt = time.perf_counter() - t0
            rec[arm] = {"served": o is not None, "padded_to": T.TRIATT_FUSED_HIFI_PADDED.get(S),
                        "pick": T.TRIATT_FUSED_HIFI_PICKS.get((T.TRIATT_FUSED_HIFI_PADDED.get(S, S),) * 2),
                        "wall_s_first_call": round(dt, 3)}
            if o is not None and arm == "on" and rec["off"]["served"] is False:
                r = a.ref_rows
                got = ttnn.to_torch(o)[:r].double()
                qq, kk, vv = (x[:r].double() for x in host)
                # kernel convention: the bias is added before the scale, exp((qk + bias - max) * scale)
                ref = torch.softmax((qq @ kk.transpose(-1, -2) + hb.double()) * D ** -0.5, -1) @ vv
                rec["on"]["rel_l2_vs_f64"] = float((got - ref).norm() / ref.norm())
            if o is not None:
                ttnn.deallocate(o)
        T._TRIATT_HIFI_PAD_UP_TILES = tiles
        rec["route_changes"] = rec["off"]["served"] != rec["on"]["served"]
        res["sizes"][S] = rec
        print(S, json.dumps(rec), flush=True)
        for x in (q, k, v, b):
            ttnn.deallocate(x)
        a.out.write_text(json.dumps(res, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
