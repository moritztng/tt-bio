"""Every `_latch` name in tenstorrent.py has a row in LATCH_STATS.

Not a tidiness test. Each `_latch` call sits inside an except block recovering from a device
refusal, so a KeyError raised by the counter turns a handled refusal into a dead fold -- which is
exactly what an unregistered `pair_bias_ln_l1` did on 2026-09-08: the DRAM retry in
`_pair_bias_from_z` never ran and a 576 aa fold died on its own instrumentation. `_latch` now
`setdefault`s so a gap cannot be fatal; this keeps the dict complete anyway, so the run still
reports the latch under the name the code uses.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

from tt_bio import tenstorrent as tt  # noqa: E402

SRC = Path(tt.__file__).read_text()
NAMES = sorted(set(re.findall(r'_latch\(\s*"([^"]+)"', SRC)))
FIELDS = sorted(set(re.findall(r'_latch\(\s*"[^"]+"\s*,\s*"([^"]+)"', SRC)))


def test_there_are_latches_to_check():
    assert len(NAMES) >= 5, NAMES          # the check is vacuous if the regex stops matching


@pytest.mark.parametrize("name", NAMES)
def test_name_is_registered(name):
    assert name in tt.LATCH_STATS, f"_latch(\"{name}\", ...) has no LATCH_STATS row"


@pytest.mark.parametrize("field", FIELDS)
def test_field_is_a_counter_on_every_row(field):
    for name, row in tt.LATCH_STATS.items():
        assert field in row, f"{name} has no '{field}' counter"


def test_an_unregistered_name_counts_instead_of_throwing():
    """The guard that made the failure survivable. Negative control: it must not KeyError."""
    tt._latch("a_name_no_row_exists_for", "refused", RuntimeError("x"))
    row = tt.LATCH_STATS.pop("a_name_no_row_exists_for")
    assert row["refused"] == 1 and len(row["why"]) == 1
