#!/usr/bin/env python3
"""Fold-level paired A/B of the Transition row-block height, Boltz-2 512 aa.

`pairtrack.py` measures the shipped pair-track Transition at 7.9455 ms/call and the same call at
a forced row block of 32 at 6.5745 ms, with identical bytes and identical FLOPs: half the ops and
twice the per-core work. x280 calls/fold that projects 0.384 s. This turns the projection into a
fold number and a CIF digest, through `tt_baseline.measure`'s paired interleaved A/B (both arms
in one process, alternating within-pair order, each arm's cold fold discarded).

Arm `16` forces the value production already derives, so it doubles as the control that the knob
itself changes nothing at the shipped height.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import tt_baseline as B                                                       # noqa: E402
from tt_bio.main import _resolve_recycling_steps                              # noqa: E402

B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
B.SAMPLING_STEPS = 200
from fold_ab_multi import patch_boltz2_cfg                                    # noqa: E402
patch_boltz2_cfg()

HERE = Path(__file__).resolve().parent
FIX = ROOT / "perf" / "size512" / "fixtures"
out = Path(sys.argv[1])
vals = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("32", "16")
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
B.measure("boltz2", reps, HERE / ".msa_512", out,
          FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_512.a3m", "512 aa",
          keep_cif=HERE / "cif", ab_env="TT_BIO_TRANSITION_H_CHUNK", ab_values=vals)
