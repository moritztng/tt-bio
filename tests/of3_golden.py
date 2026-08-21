"""The OpenFold3 device tests' golden capture, read so a missing key skips.

``~/of3_ref_out.pkl`` is an UNCOMMITTED per-host artifact written by
``scripts/of3_trunk_golden.py``, so it can exist and still not carry the key a test asks
for. Every one of those tests guards on existence only
(``skipif(not (exists(_CKPT) and exists(_GOLD)))``), so on a host whose capture is
partial the guard passes and the test then dies on ``KeyError`` deep inside itself.

Measured on qb1 (2026-08-19): 7 of the 17 keys present, so 11 tests went red -- and a red
test looks exactly like a regression in whatever branch happens to be checked out, which
is what it looked like while re-running these regressions for an unrelated change. The
two hosts' captures are also not nested: qb1 and qb2 disagree on all 7 shared keys, so
copying one over the other would silently re-baseline the tests that do pass.

Skipping on a missing key states the real precondition. It does not decide which of the
two divergent captures is the right reference -- that belongs to whoever owns the capture
script -- it only stops a missing fixture from being reported as a code defect.
"""
from __future__ import annotations

import pickle

import pytest


class _SkipOnMissing(dict):
    def __missing__(self, key):
        have = ", ".join(repr(k) for k in sorted(map(str, self))[:4])
        pytest.skip(f"golden has no {key!r} -- partial per-host capture "
                    f"({len(self)} keys: {have}...). Regenerate ~/of3_ref_out.pkl "
                    f"with scripts/of3_trunk_golden.py.")

    def __getitem__(self, key):
        v = super().__getitem__(key)
        # A capture is partial at every depth, not only the top: on qb1
        # `input_embedder_real` is present but carries no 'relpos'. Wrapping on the way
        # out makes a missing sub-key skip for the same reason a missing key does.
        return _SkipOnMissing(v) if type(v) is dict else v


def intermediates(path):
    """``golden["intermediates"]``, as a mapping whose missing keys skip the test.

    Nested plain dicts are wrapped the same way, so a partial entry skips rather than
    raising KeyError from inside the test body.
    """
    with open(path, "rb") as fh:
        return _SkipOnMissing(pickle.load(fh)["intermediates"])
