"""The OpenFold3 MSA row cap must actually cut rows, and the fold must see the cut copy.

`0174d8d0` fixed a vendoring omission: tt-bio's `parse_a3m` / `parse_stockholm` called
`parsed_msa.truncate(max_seq_count)`, and `truncate` defaults to `inplace=False`, so the truncated
copy was thrown away and the full alignment returned. The featurizer's own `max_rows` still cut the
model's input to the same depth, so nothing crashed and no existing parity fixture noticed: every
committed MSA sits under its cap. A deep alignment simply reached the model as a different set of
rows than upstream feeds it.

Card-free, no weights: it parses strings and counts rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tt_bio._vendor.openfold3.core.data.io.sequence.msa import (
    MSA_PARSER_REGISTRY,
    parse_a3m,
    parse_msas_direct,
    parse_stockholm,
)

QUERY = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _a3m(n_rows: int) -> str:
    # n_rows total including the query. Each hit differs from the query so a dedup step,
    # if one ever appears, cannot mask a truncation bug by collapsing them.
    out = [f">query\n{QUERY}"]
    for i in range(1, n_rows):
        seq = list(QUERY)
        seq[i % len(QUERY)] = "A" if seq[i % len(QUERY)] != "A" else "G"
        out.append(f">hit{i}\n{''.join(seq)}")
    return "\n".join(out) + "\n"


def _sto(n_rows: int) -> str:
    lines = ["# STOCKHOLM 1.0", f"query {QUERY}"]
    for i in range(1, n_rows):
        seq = list(QUERY)
        seq[i % len(QUERY)] = "A" if seq[i % len(QUERY)] != "A" else "G"
        lines.append(f"hit{i} {''.join(seq)}")
    lines.append("//")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("cap", [1, 3, 17])
def test_parse_a3m_returns_exactly_the_cap(cap):
    msa = parse_a3m(_a3m(64), cap)
    assert msa.msa.shape[0] == cap, f"cap {cap} returned {msa.msa.shape[0]} rows"
    assert len(msa.metadata) == cap, "metadata must be truncated with the rows"


@pytest.mark.parametrize("cap", [1, 3, 17])
def test_parse_stockholm_returns_exactly_the_cap(cap):
    msa = parse_stockholm(_sto(64), cap)
    assert msa.msa.shape[0] == cap, f"cap {cap} returned {msa.msa.shape[0]} rows"
    assert len(msa.metadata) == cap


def test_the_query_row_survives_the_cap():
    # Truncation keeps the head of the file, so row 0 is still the query. A cap that dropped
    # it would fold a hit as the target.
    msa = parse_a3m(_a3m(64), 4)
    assert "".join(msa.msa[0].astype(str)).replace("-", "") == QUERY


def test_an_msa_under_its_cap_is_untouched():
    # The committed parity fixtures all live here, which is why they were blind to the bug.
    assert parse_a3m(_a3m(9), 16384).msa.shape[0] == 9


def test_no_cap_keeps_every_row():
    assert parse_a3m(_a3m(40), None).msa.shape[0] == 40


def test_parse_msas_direct_applies_the_per_source_cap(tmp_path: Path):
    # The path a real fold takes: a per-source cap dict keyed on the file stem.
    (tmp_path / "uniref90_hits.a3m").write_text(_a3m(64))
    (tmp_path / "bfd_uniclust_hits.sto").write_text(_sto(64))
    msas = parse_msas_direct(
        [tmp_path / "uniref90_hits.a3m", tmp_path / "bfd_uniclust_hits.sto"],
        max_seq_counts={"uniref90_hits": 5, "bfd_uniclust_hits": 7},
    )
    assert msas["uniref90_hits"].msa.shape[0] == 5
    assert msas["bfd_uniclust_hits"].msa.shape[0] == 7


def test_every_registered_parser_honours_the_cap():
    # A new extension added to the registry must not reintroduce the omission.
    bodies = {".a3m": _a3m(64), ".sto": _sto(64)}
    for ext, parser in MSA_PARSER_REGISTRY.items():
        assert ext in bodies, f"no cap coverage for parser {ext}; add a body above"
        assert parser(bodies[ext], 6).msa.shape[0] == 6, f"{ext} ignored its cap"
