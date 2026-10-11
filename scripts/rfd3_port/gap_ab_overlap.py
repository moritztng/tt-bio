#!/usr/bin/env python3
"""A/B of the attn_indices overlap (RFD3_ATTN_OVERLAP) on the page fixture, through fold_ab.

    gap_ab_overlap.py OUT.json [STEPS] [ARMS] [--tokfirst]

`on` builds the atom neighbour index on a host thread while the step's first device work runs;
with --tokfirst it also dispatches the recycle-1 token encoder before the atom encoder. `off` is
the serial shipped order. The digest must not move: both arms run the same ops on the same inputs.
"""
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio.rfd3 import model as M                                        # noqa: E402
from fold_ab import fold_ab                                               # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
TOKFIRST = "--tokfirst" in sys.argv
out = pathlib.Path(args[0])
steps = int(args[1]) if len(args) > 1 else 200
arms = (args[2] if len(args) > 2 else "off,off,on,off,on,on").split(",")

counter = [0]
_pool = M._index_pool


def _counted():
    counter[0] += 1
    return _pool()


M._index_pool = _counted


def set_enabled(on):
    M._ATTN_OVERLAP = on
    M._TOKEN_ENC_FIRST = on and TOKFIRST


fold_ab(flag="attn_overlap" + ("+tokfirst" if TOKFIRST else ""), set_enabled=set_enabled,
        counter=counter, fixture="perf/dsfix/fixtures/rfd3_R4.json", out=out, steps=steps, arms=arms,
        tag="gap_overlap" + ("_tf" if TOKFIRST else ""))
