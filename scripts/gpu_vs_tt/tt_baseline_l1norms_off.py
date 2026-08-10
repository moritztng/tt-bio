#!/usr/bin/env python3
"""tt_baseline with the h=1.5 L1 layer_norm class disabled.

Workaround for the pc card-0 298-aa hang (gpu-vs-tt-precision-fairness, 2026-08-10):
with production defaults the 320-token fold silently wedges the device on this host
across ttnn 0.67.4/0.68.0 and KMD 2.7.0/2.8.0, while the same shape folds on qb1/qb2.
The z-size sweep arms isolate the cause to the h=1.5 L1 layer_norm class: `off` (all
five capacity-gated wins) and `norms_off` (only this class) both fold fine. Numbers
taken through this wrapper are production-current-main minus this one class, which the
z-size-robustness leg prices at ~0.65 s/fold at 298 aa.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T

T._PAIR_BIAS_L1_NORM = False
T._PWA_L1_NORM = False
T._TEMPLATE_L1_NORM = False

import tt_baseline

if __name__ == "__main__":
    sys.exit(tt_baseline.main())
