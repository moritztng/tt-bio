#!/usr/bin/env python3
"""Fold one target in one (K3, lever C) corner, and prove from inside the process that the
corner actually took effect.

The gate this serves compares four corners on identical input. Its one fatal failure mode is
an arm that never reaches the constant it is supposed to move, which turns an A/B into an A/A
and makes every downstream number meaningless. So this driver wraps `_apply_grid_thresholds`
and writes the post-device-open value of both switches to the probe file before the fold
starts. Report which arm a CIF came from by reading that file, never by reading the command
line that launched it.

Lever C moves `SEQ_LEN_MORE_CHUNKING` and nothing else, so `--force-slmc` moves exactly that
one constant. The shipped `TT_BIO_SEQ_LEN_MORE_CHUNKING` env hook lives at the end of
`_apply_grid_thresholds`, which returns early on a full-size grid, so it is unreachable on
Blackhole; `--force-slmc` runs after the original either way and is therefore the one
mechanism that works on both architectures. K3 is `TT_BIO_SDPA_DIV_K`, read at import time,
so it needs no help.

Usage:
  run_arm.py --tree <tt-bio checkout> --probe <arm.json> [--force-slmc N] -- <tt-bio argv...>
"""
import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="tt-bio checkout to import tt_bio from")
    ap.add_argument("--probe", required=True, help="where to write the arm's resolved settings")
    ap.add_argument("--force-slmc", type=int, default=None,
                    help="lever C: force SEQ_LEN_MORE_CHUNKING after _apply_grid_thresholds")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    sys.path.insert(0, a.tree)
    import tt_bio.tenstorrent as T

    orig = T._apply_grid_thresholds

    def probed(grid, device=None):
        orig(grid, device)
        if a.force_slmc is not None:
            T.SEQ_LEN_MORE_CHUNKING = a.force_slmc
        json.dump({
            "tree": a.tree,
            "grid": [int(grid[0]), int(grid[1])],
            "is_small_grid": bool(T._IS_SMALL_GRID),
            "forced_slmc": a.force_slmc,
            "SEQ_LEN_MORE_CHUNKING": int(T.SEQ_LEN_MORE_CHUNKING),
            "TRANSITION_BATCH_CHUNKING_THRESHOLD": int(T.TRANSITION_BATCH_CHUNKING_THRESHOLD),
            "TRANSITION_W_CHUNKING_THRESHOLD": int(T.TRANSITION_W_CHUNKING_THRESHOLD),
            "SDPA_DIV_K": bool(T._SDPA_DIV_K),
            "k_chunk_640": int(T._dividing_sdpa_chunk_size(640)),
            "k_chunk_768": int(T._dividing_sdpa_chunk_size(768)),
            "tt_bio_env": {k: v for k, v in sorted(os.environ.items())
                           if k.startswith(("TT_BIO_", "TT_VISIBLE_"))},
        }, open(a.probe, "w"), indent=1)

    T._apply_grid_thresholds = probed

    # `_configure_active_compute_grid` skips the call when the device grid already equals the
    # module default (11x10). Neither part in play here does -- Blackhole p150a snaps to 13x10
    # and Wormhole adopts 8x8 -- but if the probe file is missing after a run, that is why, and
    # the run must not be scored.
    from tt_bio.main import main as cli
    sys.argv = ["tt-bio"] + rest
    return cli()


if __name__ == "__main__":
    sys.exit(main())
