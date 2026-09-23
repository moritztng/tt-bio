#!/usr/bin/env python3
"""ESMFold2 input features: tt-bio's path against upstream esm 3.4.1, tensor by tensor, on CPU.

    PYTHONPATH=.:<dir holding upstream `esm`> python perf/mgx/acc/featparity.py 7aqx_1024 [...]

tt-bio side is the path `tt-bio predict` takes: `_read_bio_chains` -> `resolve_msa` ->
`build_spi` -> the vendored `prepare_esmfold2_input`. Upstream side is what
perf/mgx/ref/ref_fold.py::run_esmfold2 feeds the GPU reference: one `ProteinInput` per chain
copy with `MSA.from_a3m(max_sequences=16384)`, through upstream's own `prepare_esmfold2_input`.
The upstream package only needs its featurizer, so `esm/__init__.py`, `esm/models/__init__.py`
and `esm/models/esmfold2/__init__.py` may be empty stubs (the full package pulls in the model).

Prints one line per feature: identical, or the first differing index and the count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perf" / "mgx" / "ref"))


def tt_features(fixture: str):
    from tt_bio._vendor.esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from tt_bio.esmfold2_runtime import build_spi, resolve_msa
    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(ROOT / "perf/mgx/ref/fixtures" / f"{fixture}.yaml")
    chains = [(c[0], c[1], resolve_msa(c[2], c[1]), *c[3:]) for c in chains]
    return prepare_esmfold2_input(build_spi(chains), seed=0)[0]


def upstream_features(fixture: str):
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
    from esm.utils.msa.msa import MSA
    from ref_fold import load_fixture

    spi = StructurePredictionInput(sequences=[
        ProteinInput(id=cid, sequence=c["sequence"], msa=MSA.from_a3m(c["msa"], max_sequences=16384))
        for c in load_fixture(fixture) for cid in c["ids"]])
    return prepare_esmfold2_input(spi, seed=0)[0]


def compare(a: dict, b: dict) -> int:
    bad = 0
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f"  {k}: only on the {'tt-bio' if k in a else 'upstream'} side")
            bad += 1
            continue
        x, y = a[k], b[k]
        if not isinstance(x, torch.Tensor):
            continue
        if x.shape != y.shape:
            print(f"  {k}: shape tt-bio {tuple(x.shape)} upstream {tuple(y.shape)}")
            bad += 1
            continue
        d = (x.double() - y.double()).abs() if x.is_floating_point() else (x != y).double()
        n = int((d > 0).sum())
        if n:
            idx = tuple(int(i) for i in np.unravel_index(int(d.flatten().argmax()), d.shape))
            print(f"  {k}: {n}/{d.numel()} differ, max |d| {float(d.max()):.4g} at {idx}, "
                  f"tt-bio nonzero {int((x != 0).sum())} upstream nonzero {int((y != 0).sum())}")
            bad += 1
        else:
            print(f"  {k}: identical {tuple(x.shape)}")
    return bad


def main() -> int:
    bad = 0
    for fx in sys.argv[1:]:
        print(fx)
        bad += compare(tt_features(fx), upstream_features(fx))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
