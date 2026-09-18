#!/usr/bin/env python3
"""A/A floor for the B1 fold A/B, through the SAME rig.

`perf/k10_b1_permute/fold_ab.py` reports an `aa_floor` that is the off arm min/max spread, not an
A/A. This runs that rig unmodified with `set_trimul_gp_bank_split` replaced by a function that
ignores its argument and always pins the shipped `on` order, so the two labelled arms are the same
arm: same weight-cache key, same kernel, same runtime args. Whatever separates them is the box, and
that is the floor the fold ratio has to clear.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_fold_ab_aa", REPO / "perf" / "k10_b1_permute" / "fold_ab.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

import tt_bio.tenstorrent as TT

_real = TT.set_trimul_gp_bank_split
TT.set_trimul_gp_bank_split = lambda on: _real(True)
assert TT.set_trimul_gp_bank_split(False) is not None or True
assert TT.gp_roles() == ("p_a", "g_a", "p_b", "g_b"), \
    f"the A/A pin did not take: {TT.gp_roles()}"

sys.exit(M.main())
