"""RFD4's host path stays exact where it was made faster: the writer's per-residue-type template
and the invariant check's neighbour compare.

Card-free. Needs the PD-L1 benchmark specs (`RFD4_SPEC`, default ~/rfd4-inputs/pdl1-128.yaml, and
pdl1-224.yaml beside it). Upstream's path is the reference: the writer with its template memo off,
the invariant check with an `np.unique` per segment.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from test_rfd4_write_designs import _stand_in

SPEC = Path(os.environ.get("RFD4_SPEC", Path.home() / "rfd4-inputs" / "pdl1-128.yaml"))
SPECS = [SPEC, SPEC.with_name("pdl1-224.yaml")]

needs_specs = pytest.mark.skipif(not all(s.exists() for s in SPECS), reason=f"no RFD4 specs beside {SPEC}")


def _segment_reference(starts, values):
    """Upstream's check: one np.unique per segment, NaN equal to NaN."""
    out = []
    for a, b in zip(starts[:-1], starts[1:]):
        seg = values[a:b]
        if seg.ndim == 1:
            out.append(np.unique(seg).size <= 1)
        else:
            equal = seg == seg[:1]
            if np.issubdtype(seg.dtype, np.floating):
                equal |= np.isnan(seg) & np.isnan(seg[:1])
            out.append(bool(equal.all()))
    return np.array(out, dtype=bool)


def test_uniform_per_segment_matches_a_unique_per_segment():
    from tt_bio._vendor.rfd4_prep._cif.transforms_invariants import _uniform_per_segment

    rng = np.random.default_rng(0)
    pools = [np.array([0, 1]), np.array([0.0, -0.0, 1.0, np.nan]), np.array(["A", "B", "."])]
    for trial in range(2000):
        n = int(rng.integers(0, 30))
        starts = np.concatenate([[0], np.sort(rng.integers(0, n + 1, size=int(rng.integers(0, 6)))), [n]])
        values = rng.choice(pools[trial % 3], n) if trial % 4 < 3 else rng.choice([0.0, np.nan], (n, 3))
        np.testing.assert_array_equal(_uniform_per_segment(starts, values), _segment_reference(starts, values))


@needs_specs
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.stem)
def test_featurize_is_exact(spec, tmp_path):
    """Every annotation the invariant check reads gets the same verdict per segment as upstream's,
    on the real atom array, so the features (which the check only inspects) cannot move."""
    from tt_bio import rfd4
    from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
    from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts
    from tt_bio._vendor.atomworks.io.utils.standard_annotations import STANDARD_ANNOTATIONS
    from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import Level
    from tt_bio._vendor.rfd4_prep._cif.transforms_invariants import _uniform_per_segment

    ex = rfd4.prepare_examples(str(spec), str(tmp_path))[0]
    _, aa = rfd4.featurize(ex.cif, seed=0)
    categories = set(aa.get_annotation_categories())
    checked = 0
    for sa in STANDARD_ANNOTATIONS:
        if sa.n_body != 1 or sa.full_name not in categories:
            continue
        values = aa.get_annotation(sa.full_name)
        for starts in (get_residue_starts(aa, add_exclusive_stop=True), get_token_starts(aa, add_exclusive_stop=True)):
            np.testing.assert_array_equal(_uniform_per_segment(starts, values), _segment_reference(starts, values))
            checked += 1
    assert checked > 20


@needs_specs
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.stem)
def test_writer_is_byte_identical_to_upstream(spec, tmp_path, monkeypatch):
    """A CIF written with one template per residue type is the file upstream's path writes."""
    from tt_bio import rfd4
    from tt_bio._vendor.rfd4_prep._cif import utils_coordinate_transfer as CT

    ex = rfd4.prepare_examples(str(spec), str(tmp_path / "inputs"))[0]
    features, atom_array = rfd4.featurize(ex.cif, seed=0)
    output = _stand_in(features, 2)
    for k in range(2):
        rfd4.write_design(output, features, atom_array, tmp_path / "memo" / "d.cif", row=k)
        with monkeypatch.context() as m:
            m.setattr(CT, "_template_key", lambda *a: None)
            rfd4.write_design(output, features, atom_array, tmp_path / "upstream" / "d.cif", row=k)
        memo, upstream = ((tmp_path / d / "d.cif").read_text().splitlines() for d in ("memo", "upstream"))
        # The files differ only in the second the header was stamped.
        assert [line for line in memo if not line.startswith("_entry.time")] == \
               [line for line in upstream if not line.startswith("_entry.time")]
        for d in ("memo", "upstream"):
            (tmp_path / d / "d.cif").unlink()
