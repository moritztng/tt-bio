"""The tri-attention work split must not silently fall back to one q chunk per core.

At 1024 aa on a 130-core grid the persistent-mask SDPA's split granularity is
`n_heads * q_num_chunks = 48` cores, so 96 of 130 cores get work and each owns 512 batch rows.
Letting a core own two q chunks drops the granularity to 24, uses 120 cores at 205 rows, and
205 * 2 = 410 < 512 -- measured 32.837 -> 25.734 ms, bit-exact
(`perf/oddel1/qpercore_parity_pc0.json`). A lever that quietly returns to one chunk per core is
exactly the regression this file exists to catch, so the pick is asserted and not just the ranking.

Host-only, no device: `_q_split` is arithmetic over (q chunk count, heads, cores, batch).
"""
from __future__ import annotations

import pytest

from tt_bio import triatt_sdpa as PM


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(PM, "_Q_PER_CORE", True)


def test_1024_aa_on_130_cores_owns_two_q_chunks():
    assert PM._q_split(4, 12, 130, 1024)[0] == (2, 2)


def test_ties_go_to_the_pre_lever_split():
    # 768 aa, 130 cores: c=1 is 154 units and c=2 is 154 too, so main's split must win.
    assert PM._q_split(2, 12, 130, 768)[0] == (2, 1)


def test_candidates_are_ranked_not_singular():
    # 1024 aa on 110 cores prefers c=4, whose mask CB refuses L1; the caller has to be able to
    # walk down to a narrower split instead of retiring the shape.
    cands = PM._q_split(4, 12, 110, 1024)
    assert cands[0] == (1, 4)
    assert (4, 1) in cands


def test_disabled_reproduces_mains_split(monkeypatch):
    monkeypatch.setattr(PM, "_Q_PER_CORE", False)
    assert PM._q_split(4, 12, 130, 1024) == [(4, 1)]


def test_mask_cb_scales_with_q_chunks_per_core():
    # The CB holds one block per (q chunk, k chunk) pair the core owns. 1024 aa at q_chunk 256
    # and k_chunk 256: 2 * 4 * 8 * 8 = 512 tiles, against 256 for main's split.
    q_pf, q_per_core = PM._q_split(4, 12, 130, 1024)[0]
    assert q_per_core * 4 * 8 * 8 == 512
