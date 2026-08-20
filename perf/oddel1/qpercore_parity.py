#!/usr/bin/env python3
"""Bit-exactness and wall of the q-per-core work split, driven through the fold's own picker.

Arm A forces `q_per_core = 1` (main's split). Arm B lets `_q_split` search. Both go through
`tenstorrent._tri_att_sdpa_at`, so the q_chunk under test is the one the fold actually picks --
screening `_sdpa_chunks_shipped`'s return value instead measures a config the fold never executes
(that is the K4 lesson recorded in `_sdpa_chunks_shipped`'s own comment).

A/A runs arm A twice. On a card with a known bit-flip defect a non-empty A/A diff is the card, not
the change; on a clean card it is the noise floor for the A/B verdict.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WARM, REPS = 1, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512,640,768,1024")
    ap.add_argument("--heads", default="12,2")
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import os
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as PM
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    # qb2 is two dual-chip p300 boards; a bare single-chip open fails without the mesh descriptor.
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "grid": [g.x, g.y], "cores": g.x * g.y, "rows": []}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)

    picked = []
    real_sdpa = PM.sdpa

    def spy(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default=None):
        o = real_sdpa(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default)
        picked.append((q_chunk, k_chunk, o is not None))
        return o
    PM.sdpa = spy
    T._triatt_sdpa.sdpa = spy

    for H in [int(h) for h in a.heads.split(",")]:
        for S in [int(s) for s in a.sizes.split(",")]:
            torch.manual_seed(S * 1000 + H)
            tt = lambda x: ttnn.from_torch(  # noqa: E731
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q = tt(torch.randn(S, H, S, a.dh))
            k = tt(torch.randn(S, H, S, a.dh))
            v = tt(torch.randn(S, H, S, a.dh))
            bias = tt(torch.randn(1, H, S, S))
            scale = a.dh ** -0.5

            def leg(on):
                PM._Q_PER_CORE = on
                picked.clear()
                PM.STATS[0] = PM.STATS[1] = 0
                o = T._tri_att_sdpa_at(q, k, v, bias, scale)
                ttnn.synchronize_device(dev)
                host = ttnn.to_torch(o)
                cfg = {"picked": [list(p) for p in picked], "served": PM.STATS[0]}
                ts = []
                for _ in range(WARM):
                    ttnn.deallocate(T._tri_att_sdpa_at(q, k, v, bias, scale))
                ttnn.synchronize_device(dev)
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    o2 = T._tri_att_sdpa_at(q, k, v, bias, scale)
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o2)
                ttnn.deallocate(o)
                cfg["ms"] = round(st.median(ts), 4)
                cfg["ms_all"] = [round(t, 4) for t in ts]
                return host, cfg

            a0, ca0 = leg(False)
            a1, ca1 = leg(False)
            b0, cb = leg(True)
            row = {"S": S, "H": H, "A": ca0, "A2": ca1, "B": cb,
                   "aa_equal": bool(torch.equal(a0, a1)),
                   "ab_equal": bool(torch.equal(a0, b0)),
                   "ab_maxdiff": float((a0.float() - b0.float()).abs().max()),
                   "split_changed": ca0["picked"] != cb["picked"] or ca0["served"] != cb["served"],
                   "speedup": round(ca0["ms"] / cb["ms"], 4) if cb["ms"] else None}
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
            for t in (q, k, v, bias):
                ttnn.deallocate(t)
    res["pm_over_l1"] = sorted(str(x) for x in PM._PM_OVER_L1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    bad = [r for r in res["rows"] if not r["ab_equal"]]
    print(f"AB mismatches: {len(bad)}; AA mismatches: "
          f"{len([r for r in res['rows'] if not r['aa_equal']])}", flush=True)


if __name__ == "__main__":
    main()
