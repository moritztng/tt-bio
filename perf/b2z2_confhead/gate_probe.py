#!/usr/bin/env python3
"""Why TT_BIO_DEVICE_CONFIDENCE did or did not take: every term of the gate, on the live model.

Loads the fold state and prints the gate's operands without folding, so a silent fallback is a
named operand rather than a fold that came back the same speed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
FIX = REPO / "perf" / "size512" / "fixtures"


def main() -> int:
    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    import tt_bio.boltz2 as boltz2
    import tt_bio.tenstorrent as T
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    tgt, a3m = FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_512.a3m"
    msa_dir = Path(__file__).resolve().parent / ".msa_512"
    _one_fold, _meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)

    m = state.model
    print("state.model", type(m).__module__ + "." + type(m).__name__)
    inner = getattr(m, "model", m)
    print("inner      ", type(inner).__module__ + "." + type(inner).__name__)
    for obj, label in ((m, "state.model"), (inner, "inner")):
        conf = getattr(obj, "confidence_module", None)
        if conf is None:
            continue
        print(f"--- {label} ---")
        print("  use_tenstorrent      ", getattr(obj, "use_tenstorrent", "?"))
        print("  run_trunk_and_structure", getattr(obj, "run_trunk_and_structure", "?"))
        print("  is_msa_compiled      ", getattr(obj, "is_msa_compiled", "?"))
        print("  is_pairformer_compiled", getattr(obj, "is_pairformer_compiled", "?"))
        print("  affinity_trunk_fp32  ", getattr(obj, "affinity_trunk_fp32", "?"))
        print("  confidence_prediction", getattr(obj, "confidence_prediction", "?"))
        print("  conf type            ", type(conf).__module__ + "." + type(conf).__name__)
        print("  add_z_input_to_z     ", getattr(conf, "add_z_input_to_z", "MISSING"))
        print("  bond_type_feature    ", getattr(conf, "bond_type_feature", "MISSING"))
        ps = getattr(conf, "pairformer_stack", None)
        print("  pairformer_stack     ", type(ps).__module__ + "." + type(ps).__name__)
        print("  isinstance(T.Pairformer", isinstance(ps, T.PairformerModule))
        print("  supports()           ", T.ConfidencePairDevice.supports(conf))
        print("  _device_confidence() ", boltz2._device_confidence())
        print("  env                  ", os.environ.get("TT_BIO_DEVICE_CONFIDENCE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
