#!/usr/bin/env python3
"""Does `TT_BIO_MM_SHORT_M_BW` make the output depend on the core grid?

WHY, AND WHY BEFORE THE A/B. Region T was fold-verified at +0.2020 s, accuracy-clear at 0.37848 A
against a 0.60 A bar, and still NO-GO as a default, because the release gate's `l1-budget` arm
found its CIF digest moved with the core grid. "Output that depends on which card ran it" is a
standing hard stop and is not scored against the Angstrom bar. `MM_SHORT_M_BW` is this row's other
landing candidate and it derives its whole plan from `COMPUTE_GRID_MAIN`, which
`tenstorrent.py:5117` assigns from the live device:

    per_core_M = ceil(m_tiles / gy)      per_core_N = ceil(n_tiles / gx)
    in0_block_w = largest divisor d of k_tiles with d * per_core_N <= _MM_IN1_BLOCK_TILES
                  and the circular buffers still fitting

`in0_block_w` is how many K tiles a core folds into one accumulation block, so a grid that moves
it moves the order the contraction is summed in, and bf16 addition is not associative. That is the
same mechanism region T died of, reached by a different route.

WHAT THIS COSTS: nothing. `_short_m_proj_program_config` is pure given `COMPUTE_GRID_MAIN`, and
`_matmul_cb_budget()` returns a static number when no device is open, by its own docstring, which
also says why asking the allocator would open a chip. So this screens every grid on the host.

WHAT IT CAN AND CANNOT SAY. A config that is IDENTICAL across grids cannot move the numerics, so
that is a clean pass. A config that DIFFERS is a red flag and not a verdict: two different foldings
can still coincide bit for bit. The decider stays the gate's `l1-budget` arm on a device. This
screen exists to say whether that arm is worth waiting for a quiet box to run, and whether the
18-rep A/B is worth a quiet window at all.

  python3 perf/c14_land/mmshort_grid_screen.py --out perf/c14_land/mmshort_grid_screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Grids this part and its siblings actually present. 11x10 is qb2's p300c, 13x10 is qb1's p150a,
# and 8x8 plus the narrow rows are what the gate's l1-budget arm walks.
GRIDS = [(11, 10), (13, 10), (8, 8), (10, 10), (11, 9), (7, 10)]

# The shapes the lever was fitted on, from `_short_m_proj_program_config`'s own docstring and the
# ladder in perf/c14_matmul_ceiling/bwladder_b{3,4}.json. m_tiles = 16 is the diffusion
# transformer's token projection at 512 residues, which is the site the lever exists for.
SHAPES = [
    ("dit_token_768x768",   16, 24, 24),
    ("dit_token_768x1536",  16, 24, 48),
    ("dit_token_768x3072",  16, 24, 96),
    ("dit_token_1536x768",  16, 48, 24),
    ("dit_token_768x2304",  16, 24, 72),
]
FIELDS = ("in0_block_w", "out_subblock_h", "out_subblock_w",
          "out_block_h", "out_block_w", "per_core_M", "per_core_N")


def cfg_tuple(pc):
    if pc is None:
        return None
    return {f: int(getattr(pc, f)) for f in FIELDS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import tt_bio.tenstorrent as TT
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    assert TT._device is None, (
        "a device is open in this process, so _matmul_cb_budget would ask the allocator and the "
        "screen would be reading one card's budget rather than the static one")

    out = {
        "git_head": __import__("os").popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "cb_budget_bytes": int(TT._matmul_cb_budget()),
        "mm_in1_block_tiles": int(TT._MM_IN1_BLOCK_TILES),
        "grids": [list(g) for g in GRIDS],
        "shapes": [],
    }

    orig = TT.COMPUTE_GRID_MAIN
    try:
        for name, mt, kt, nt in SHAPES:
            per_grid = {}
            for g in GRIDS:
                TT.COMPUTE_GRID_MAIN = g
                per_grid[f"{g[0]}x{g[1]}"] = cfg_tuple(
                    TT._short_m_proj_program_config(mt, kt, nt, 2))
            served = {k: v for k, v in per_grid.items() if v is not None}
            bws = {k: v["in0_block_w"] for k, v in served.items()}
            out["shapes"].append({
                "shape": name, "m_tiles": mt, "k_tiles": kt, "n_tiles": nt,
                "per_grid": per_grid,
                "grids_served": sorted(served),
                "grids_declined": sorted(k for k, v in per_grid.items() if v is None),
                "in0_block_w_by_grid": bws,
                "in0_block_w_constant": len(set(bws.values())) <= 1,
                "config_constant_across_serving_grids":
                    len({json.dumps(v, sort_keys=True) for v in served.values()}) <= 1,
            })
    finally:
        TT.COMPUTE_GRID_MAIN = orig

    moving_bw = [s["shape"] for s in out["shapes"] if not s["in0_block_w_constant"]]
    moving_cfg = [s["shape"] for s in out["shapes"]
                  if not s["config_constant_across_serving_grids"]]

    # NEGATIVE CONTROL. The screen's claim is "it would have noticed a difference". Give it a
    # grid pair that must disagree -- a 2x2 grid cannot serve m_tiles = 16 at all, since the
    # function refuses when m_tiles >= gx*gy -- and require it to report the disagreement.
    TT.COMPUTE_GRID_MAIN = (2, 2)
    tiny = cfg_tuple(TT._short_m_proj_program_config(16, 24, 24, 2))
    TT.COMPUTE_GRID_MAIN = orig
    native = cfg_tuple(TT._short_m_proj_program_config(16, 24, 24, 2))
    out["negative_control"] = {
        "grid_2x2": tiny, "grid_native": native,
        "screen_separates_them": tiny != native,
        "why": "if the screen cannot tell a refusing grid from a serving one it cannot tell two "
               "serving grids apart either, and every 'constant' above would be vacuous",
    }

    out["in0_block_w_moves_with_grid"] = moving_bw
    out["config_moves_with_grid"] = moving_cfg
    out["verdict"] = (
        "GRID-SENSITIVE: the K folding moves with the core grid on " + ", ".join(moving_bw)
        + ". The gate's l1-budget arm is the decider and must run before this lever is "
          "considered for a default." if moving_bw else
        "CONFIG-STABLE on in0_block_w across every serving grid, so the contraction is folded "
        "the same way everywhere and the grid cannot move the numerics through this lever"
        if not moving_cfg else
        "PARTIAL: in0_block_w is constant but another field moves; that does not change the "
        "accumulation order, so it is not the region-T mechanism")
    if not out["negative_control"]["screen_separates_them"]:
        out["verdict"] = "SCREEN INVALID: negative control did not separate"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "cb_budget_bytes", "in0_block_w_moves_with_grid", "config_moves_with_grid",
        "negative_control", "verdict")}, indent=1))
    for s in out["shapes"]:
        print(f"  {s['shape']:22s} bw={s['in0_block_w_by_grid']} declined={s['grids_declined']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
