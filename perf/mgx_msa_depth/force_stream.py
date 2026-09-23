#!/usr/bin/env python3
"""`tt_bio.main` with esmfold2's MSA encoder sent down its depth-streamed path from the start.

    force_stream.py ROWS predict ...      (chain.py: MSAD_ENTRY="perf/mgx_msa_depth/force_stream.py 512")

The streamed path normally runs only after DRAM refuses the whole alignment, so at a size where
both run this is the A/B that scores what the depth-chunked OPM costs in Angstrom.
"""
import os
import runpy
import sys

import tt_bio.esmfold2 as e

# predict folds in a spawned worker, which re-imports this file as __mp_main__ with the parent's
# (already shortened) argv, so the row count travels by environment and only the parent runs the CLI.
if __name__ == "__main__":
    os.environ["MSAD_FORCE_ROWS"] = sys.argv.pop(1)
rows = int(os.environ["MSAD_FORCE_ROWS"])


class _Refused(dict):
    def __contains__(self, key):
        return True

    def get(self, key, default=None):
        return rows


e._MSA_DEPTH_REFUSED = _Refused()
if __name__ == "__main__":
    sys.argv[0] = "tt_bio.main"
    runpy.run_module("tt_bio.main", run_name="__main__")
