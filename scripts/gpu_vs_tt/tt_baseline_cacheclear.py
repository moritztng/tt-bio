#!/usr/bin/env python3
"""tt_baseline with the z-size sweep's post-load cache clears replicated.

Bisects why the tt_baseline CLI path hangs pc card 0 at 298 aa while the sweep
harness folds the same input/flags/card (gpu-vs-tt-precision-fairness, pass 5).
The sweep calls set_arm() after build_fold, which clears
_pair_proj_program_config's lru_cache and _L1_OUT_REFUSED. This shim adds
exactly that to the tt_baseline path and nothing else: no monkeypatches, no
import-order change, same measure() loop.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T
import tt_baseline as B

_orig_build_fold = B.build_fold


def _build_fold_then_clear(*a, **k):
    out = _orig_build_fold(*a, **k)
    T._pair_proj_program_config.cache_clear()
    T._L1_OUT_REFUSED.clear()
    return out


B.build_fold = _build_fold_then_clear

if __name__ == "__main__":
    sys.exit(B.main())
