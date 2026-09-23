#!/usr/bin/env python3
"""`tt_bio.main` with esmfold2's MSA encoder sent down its depth-streamed path from the start.

    force_stream.py ROWS predict ...      (chain.py: MSAD_ENTRY="perf/mgx_msa_depth/force_stream.py 512")

The streamed path normally runs only after DRAM refuses the whole alignment, so at a size where
both run this is the A/B that scores what the depth-chunked OPM costs in Angstrom.
"""
import runpy
import sys

import tt_bio.esmfold2 as e

rows = int(sys.argv.pop(1))


class _Refused(dict):
    def __contains__(self, key):
        return True

    def get(self, key, default=None):
        return rows


e._MSA_DEPTH_REFUSED = _Refused()
sys.argv[0] = "tt_bio.main"
runpy.run_module("tt_bio.main", run_name="__main__")
