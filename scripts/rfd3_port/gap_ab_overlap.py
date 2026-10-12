#!/usr/bin/env python3
"""A/B of the attn_indices overlap (RFD3_ATTN_OVERLAP) on the page fixture, through fold_ab.

    gap_ab_overlap.py OUT.json [STEPS] [ARMS]

`on` dispatches the recycle-1 token encoder first and builds the atom neighbour index while the
card runs it. `off` builds the index first with the card idle. The digest must not move: both arms run the same ops on the same inputs.
"""
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio.rfd3 import model as M                                        # noqa: E402
from fold_ab import fold_ab                                               # noqa: E402

args = sys.argv[1:]
out = pathlib.Path(args[0])
steps = int(args[1]) if len(args) > 1 else 200
arms = (args[2] if len(args) > 2 else "off,off,on,off,on,on").split(",")

counter = [0]
_recycle = M.RFD3DiffusionModule._forward_with_recycle


def _counted(self, n_recycle, first_tok=None, **kw):
    counter[0] += first_tok is not None
    return _recycle(self, n_recycle, first_tok=first_tok, **kw)


M.RFD3DiffusionModule._forward_with_recycle = _counted


def set_enabled(on):
    M._ATTN_OVERLAP = on


fold_ab(flag="attn_overlap", set_enabled=set_enabled, counter=counter,
        fixture="perf/dsfix/fixtures/rfd3_R4.json", out=out, steps=steps, arms=arms, tag="gap_overlap")
