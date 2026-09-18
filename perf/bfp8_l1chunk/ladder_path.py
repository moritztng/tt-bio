#!/usr/bin/env python3
"""KERNEL-PATH: which (q_chunk, k_chunk) the shipped LADDER lands on, per dtype, wide-k off and on.

The op A/B named the configs by hand. This asks the production entry point, `_tri_att_sdpa_at`,
what it actually picks, and reads the answer off SDPA_CHUNK_PICKS rather than inferring it.
"""
from __future__ import annotations

import argparse, json, os, socket, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_sdpa as TS
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    S, H, D, B = a.seq, a.heads, a.head_dim, a.rows
    scale = D ** -0.5
    g = torch.Generator().manual_seed(0)

    def mk(dt):
        t = [ttnn.from_torch(torch.randn(B, H, S, D, generator=g), dtype=dt,
                             layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
        t.append(ttnn.from_torch(torch.randn(1, H, S, S, generator=g), dtype=dt,
                                 layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return t

    res = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(TT.COMPUTE_GRID_MAIN), "shape": {"B": B, "H": H, "S": S, "D": D},
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), "cases": {}}

    for widek, up in ((False, False), (True, False), (False, True), (True, True)):
        for dname, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
            TT.SDPA_CHUNK_PICKS.clear()
            for k in TT.SDPA_ROUTE_COUNTS:
                TT.SDPA_ROUTE_COUNTS[k] = 0
            TT._SDPA_QK_OVER_L1.clear()
            TT._SDPA_Q_CHUNK_OVER_L1.clear()
            TS._PM_OVER_L1.clear()
            TS.REJECTS.clear()
            old = TT._SDPA_WIDE_K if hasattr(TT, "_SDPA_WIDE_K") else None
            os.environ["TT_BIO_SDPA_WIDE_K"] = "1" if widek else "0"
            TT._SDPA_WIDE_K_UP = up
            q, k_, v, b = mk(dt)
            o = TT._tri_att_sdpa_at(q, k_, v, b, scale)
            assert o is not None
            key = f"wide_k={int(widek)}/up={int(up)}/{dname}"
            res["cases"][key] = {
                "picks": {str(kk): vv for kk, vv in TT.SDPA_CHUNK_PICKS.items()},
                "route": dict(TT.SDPA_ROUTE_COUNTS),
                "out_dtype": str(o.dtype),
                "rejects": {str(kk): vv for kk, vv in TS.REJECTS.items()},
                "QK_OVER_L1": [str(x) for x in TT._SDPA_QK_OVER_L1],
            }
            print(key, "->", res["cases"][key]["picks"], res["cases"][key]["route"],
                  "rejects", res["cases"][key]["rejects"])
            ttnn.deallocate(o)
            if old is not None:
                TT._SDPA_WIDE_K = old

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
