#!/usr/bin/env python3
"""One rank of a cadence arm. `--host-features` restores the pre-change path IN THIS BUILD.

Both arms run the same binary and the same repro script; the only difference is whether the two
input one-hots are expanded on the card or built on the host and uploaded. Keeping the switch
here rather than in `scripts/abb3_port/repro.py` keeps an A/B knob out of the shipped launcher,
and keeping it in-process means the two arms cannot differ by anything else.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _host_features(aatype, is_heavy, residue_index, *, device, **kw):
    """The baseline: build both one-hots on the host and upload them, 138 MB a micro-batch."""
    from tt_bio.abodybuilder3 import to_device_fp32
    from tt_bio.abodybuilder3_reference import single_and_pair_features
    single, pair = single_and_pair_features(aatype, is_heavy, residue_index)
    return to_device_fp32(single), to_device_fp32(pair)


if "--host-features" in sys.argv:
    sys.argv.remove("--host-features")
    import tt_bio.train.abb3_features_device as F
    F.input_features_device = _host_features
    print("[arm] HOST features: the reference one-hot, uploaded", flush=True)
else:
    print("[arm] DEVICE features: the one-hot expanded on the card", flush=True)

sys.argv[0] = str(ROOT / "scripts" / "abb3_port" / "repro.py")
runpy.run_path(sys.argv[0], run_name="__main__")
