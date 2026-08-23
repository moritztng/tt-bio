"""Capture the structure module's own input and output tensors whole, not subsampled.

`capture_ref_jax.py` stores anything over 32768 elements as an 8192-element subsample, which is
enough to score a tap and not enough to *drive* a module. So the host structure module has only
ever been fed a trunk arm's own output, and every reading of its error is confounded with the
trunk error it was handed. This capture stores recycle 0's `single`, `pair` and the whole
structure-module output whole, so the host module can be run at JAX's exact input and its own
float32 math scored on its own.

Everything else is stored exactly as `capture_ref_jax.py` stores it, same config, same recycle
count, so the recycle-0 subsamples in the committed `ref_taps.npz` have to reproduce element for
element from this capture. `structure_ref_isolate.py` checks that before it reports anything.

Runs in the external CPU-only JAX env, never inside tt-bio, same contract as the capture it
wraps. The artifact is ~40 MB and is not committed.

    ~/pxd_af2_cpu/bin/python scripts/af2_port/capture_ref_structure.py \\
        --cif perf/pxdesign/targets/laczc_128.cif --binder 80 --stage complex \\
        --out /tmp/af2_ref_structure_full.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_ref_jax as C  # noqa: E402

#: Recycle 0 only. The structure module is what is being isolated and its two inputs are what
#: drive it; storing all four recycles whole would quadruple a 40 MB artifact for nothing.
FULL_PREFIXES = ("evoformer#0/", "linear/single_activations#0/",
                 "structure_module#0/", "predicted_lddt_head#0/")

_subsampling_store = C._store


def _store(key: str, arr) -> None:
    """Store `key` whole if the isolation needs it, otherwise exactly as the capture would."""
    if not key.startswith(FULL_PREFIXES):
        _subsampling_store(key, arr)
        return
    saved, C.FULL_MAX = C.FULL_MAX, 1 << 40
    try:
        _subsampling_store(key, arr)
    finally:
        C.FULL_MAX = saved


C._store = _store

if __name__ == "__main__":
    C.main()
