#!/usr/bin/env python3
"""Op-level screen of the wide-k SDPA ladder, driven through the production entry point.

The pass-3 lesson this obeys: an op screen that enumerates (q_chunk, k_chunk) pairs measures a
configuration the fold never executes. So this calls `_tri_att_sdpa_at` itself -- the one function
every shipped triangle attention routes through -- with `TT_BIO_SDPA_WIDE_K` flipped between arms,
and reads the pair actually served out of `SDPA_CHUNK_PICKS` per arm. An arm that did not take
cannot read as a null.

Arms interleaved off/on/off/on so neither inherits the other's allocator state, and every
padded length gets a bit-exactness check plus rmsd/std of on-vs-off, because k_chunk sets the
online-softmax reduction order and the arms are NOT bit-exact where the ladder fires.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--lengths", type=str, required=True,
                    help="comma-separated PADDED sequence lengths (multiples of 32)")
    ap.add_argument("--outdir", type=Path, default=ROOT / "perf/sdpa_widek/out")
    ap.add_argument("--label", type=str, required=True, help="model label for the record")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.environ["TT_BIO_SDPA_WIDE_K"] = "0"
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__

    H, D = a.heads, a.head_dim
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    a.outdir.mkdir(parents=True, exist_ok=True)

    device = T.get_device()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    scale = D ** -0.5

    for P in lengths:
        out = a.outdir / f"widek_{a.label}_h{H}_d{D}_p{P}.json"
        if out.exists() and not a.force:
            print(f"skip {out.name} (exists)", flush=True)
            continue
        rec = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
               "grid": list(grid), "cores": grid[0] * grid[1], "arch": T.arch_name(),
               "label": a.label, "h": H, "d": D, "padded": P,
               "shipped_chunks": list(T._sdpa_chunks_shipped(P, P)),
               "q_ladder": list(T._tri_att_q_chunks(P, P)),
               "warmup": a.warmup, "iters": a.iters, "blocks": a.blocks, "arms": {}}
        os.environ["TT_BIO_SDPA_WIDE_K"] = "0"
        rec["ladder_off"] = list(T._tri_att_k_chunks(P, P))
        os.environ["TT_BIO_SDPA_WIDE_K"] = "1"
        rec["ladder_on"] = list(T._tri_att_k_chunks(P, P))
        rec["ladder_fires"] = rec["ladder_on"] != rec["ladder_off"]

        try:
            torch.manual_seed(0)
            def mk(shape):
                t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=device)
            q, k, v = mk((P, H, P, D)), mk((P, H, P, D)), mk((P, H, P, D))
            bias = mk((1, H, P, P))
        except Exception as exc:                                        # noqa: BLE001
            rec["error"] = f"alloc {type(exc).__name__}: {exc}"[:300]
            out.write_text(json.dumps(rec, indent=1))
            print(f"{P}: ALLOC FAILED {rec['error']}", flush=True)
            continue

        def call(arm: str):
            os.environ["TT_BIO_SDPA_WIDE_K"] = "1" if arm == "on" else "0"
            T.SDPA_CHUNK_PICKS.pop((P, P), None)
            o = T._tri_att_sdpa_at(q, k, v, bias, scale)
            return o, T.SDPA_CHUNK_PICKS.get((P, P))

        try:
            picks, ms = {}, {"off": [], "on": []}
            for arm in ("off", "on"):
                for _ in range(a.warmup):
                    o, pk = call(arm)
                    ttnn.deallocate(o)
                picks[arm] = pk
            ttnn.synchronize_device(device)
            for _ in range(a.blocks):
                for arm in ("off", "on"):
                    os.environ["TT_BIO_SDPA_WIDE_K"] = "1" if arm == "on" else "0"
                    t0 = time.perf_counter()
                    outs = [T._tri_att_sdpa_at(q, k, v, bias, scale) for _ in range(a.iters)]
                    ttnn.synchronize_device(device)
                    ms[arm].append((time.perf_counter() - t0) * 1e3 / a.iters)
                    for o in outs:
                        ttnn.deallocate(o)
            o_off, pk_off = call("off")
            o_on, pk_on = call("on")
            t_off, t_on = ttnn.to_torch(o_off), ttnn.to_torch(o_on)
            d = t_on.float() - t_off.float()
            rec["bit_exact"] = bool(torch.equal(t_on, t_off))
            rec["max_abs_diff"] = round(float(d.abs().max()), 6)
            rec["rmsd_over_std"] = round(float(d.pow(2).mean().sqrt() / t_off.float().std()), 6)
            ttnn.deallocate(o_off); ttnn.deallocate(o_on)
            del t_off, t_on, d
            for arm in ("off", "on"):
                rec["arms"][arm] = {
                    "pick": picks[arm], "ms_median": round(statistics.median(ms[arm]), 4),
                    "ms_all": [round(x, 4) for x in ms[arm]],
                }
            rec["speedup"] = round(rec["arms"]["off"]["ms_median"]
                                   / rec["arms"]["on"]["ms_median"], 4)
        except Exception as exc:                                        # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            for t in (q, k, v, bias):
                try:
                    ttnn.deallocate(t)
                except Exception:                                       # noqa: BLE001, S110
                    pass

        out.write_text(json.dumps(rec, indent=1))
        if "error" in rec:
            print(f"{a.label} h{H} p{P}: ERROR {rec['error']}", flush=True)
        else:
            print(f"{a.label} h{H} p{P}: fires={rec['ladder_fires']} "
                  f"off={rec['arms']['off']['pick']} {rec['arms']['off']['ms_median']:.3f}ms  "
                  f"on={rec['arms']['on']['pick']} {rec['arms']['on']['ms_median']:.3f}ms  "
                  f"{rec['speedup']:.4f}x  bit_exact={rec['bit_exact']} "
                  f"rmsd/std={rec['rmsd_over_std']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
