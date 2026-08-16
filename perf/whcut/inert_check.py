#!/usr/bin/env python3
"""Dump every grid-scaled constant and every SDPA chunk pick a tree would use on a
given compute grid, without opening a device.

Run it once per tree and diff the two JSONs. On a Blackhole grid the assembled
branch must be identical to main everywhere except the K3 sizes; that is what
turns "inert by construction" into a check (state/japanfold-wh-cutover.md §4.1).

    python3 perf/whcut/inert_check.py <tree> <out.json>
"""
import json
import os
import sys

tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import tt_bio.tenstorrent as T  # noqa: E402

assert os.path.dirname(os.path.dirname(T.__file__)) == os.path.realpath(tree), T.__file__


def scalars():
    """Every module-level scalar/tuple constant, so an unintended change to one
    that _apply_grid_thresholds does not write is caught too."""
    d = {}
    for k, v in vars(T).items():
        if isinstance(v, (int, float, bool, str, tuple)) and not k.startswith("__"):
            d[k] = list(v) if isinstance(v, tuple) else v
    return d


report = {"tree": tree, "head": os.popen("git -C %s rev-parse HEAD" % tree).read().strip()}

for grid in ((13, 10), (11, 10), (8, 9)):
    try:
        T._apply_grid_thresholds(grid, None)  # device=None -> DRAM probe falls back to 0
    except TypeError:
        T._apply_grid_thresholds(grid)  # main's signature carries no device argument
    report["grid_%dx%d" % grid] = scalars()

# The SDPA picks, over every 32-aligned length the fold path can reach.
picks = {}
for s in range(32, 1089, 32):
    picks[str(s)] = {
        "padded": T._padded_sdpa_len(s),
        "shipped": list(T._sdpa_chunks_shipped(s, s)),
        "capped": T._capped_sdpa_chunk_size(s),
    }
report["sdpa_picks"] = picks
report["switches"] = {
    "_SDPA_DIV_K": getattr(T, "_SDPA_DIV_K", None),
    "_SDPA_BAND_DIV_K": getattr(T, "_SDPA_BAND_DIV_K", None),
}

with open(out, "w") as f:
    json.dump(report, f, indent=1, sort_keys=True)
print("wrote", out, "head", report["head"])
