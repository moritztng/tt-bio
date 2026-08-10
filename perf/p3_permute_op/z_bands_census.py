#!/usr/bin/env python3
"""H3: which shapes and which destination do the 4352 eligible `_channel_move` calls actually have?

The isolated harness (`bands_probe.py` section F) measures 1.72x into DRAM and 0.99x into L1 at both
298 and 320, so if the fold's calls are all at an L1 destination there is no isolated win to explain
`y-permute-flip`'s +209.3 ms/fold, and either the instrument (H2) or the shape class (H3) carries it.
This counts every call in one wire fold by (shape, destination, eligible) and settles which.

One fold, kernel ON, no timing: the census is a call-site inventory, not a measurement.
"""
from __future__ import annotations

import json, os, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import ttnn

OUT = Path(__file__).resolve().parent


def main() -> int:
    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "w6_gate_msa"

    from tt_baseline import build_fold
    t0 = time.perf_counter()
    one_fold, meta, state = build_fold("protenix-v2", msa_dir, target, a3m, hoist=True)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    census: Counter = Counter()
    orig = T._channel_move

    def _census_channel_move(chunk, memory_config):
        key = (tuple(int(d) for d in chunk.shape), str(memory_config.buffer_type).split(".")[-1],
               bool(RP.eligible(chunk, memory_config)))
        census[key] += 1
        return orig(chunk, memory_config)

    T._channel_move = _census_channel_move
    RP.set_enabled(True)
    RP.STATS[0] = RP.STATS[1] = 0
    RP.REJECTS.clear()

    wall, fold_meta = one_fold()
    plddt = fold_meta.get("plddt")

    rows = []
    for (shape, dest, elig), n in sorted(census.items(), key=lambda kv: -kv[1]):
        rows.append({"shape": list(shape), "dest": dest, "eligible": elig, "calls": n})
    R = {
        "wall_s": round(wall, 2),
        "plddt": plddt,
        "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
        "total_channel_move_calls": sum(census.values()),
        "served": RP.STATS[0],
        "fell_through": RP.STATS[1],
        "rejects": dict(RP.REJECTS),
        "by_shape": rows,
    }
    print(json.dumps(R, indent=2), flush=True)
    (OUT / "z_bands_census.json").write_text(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
