"""`results.json` must say which chain each `pair_chains_iptm` key belongs to.

`pair_chains_iptm` and `chains_ptm` are keyed by chain position (boltz2's asym_id), while
`interface_score_distribution.pairs` in the same entry is keyed by chain name. Without a mapping
a consumer cannot join the two, and on a two-chain target cannot tell a binder-vs-target score
from its mirror image. `tt_bio.main._chain_ids` emits that mapping as `chain_ids`.

Host-only: pure dict/list work on a stub structure, no device, no network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tt_bio.data.types import Chain
from tt_bio.main import _chain_ids


@dataclass
class _Struct:
    chains: np.ndarray


def _struct(pairs):
    """A structure whose `chains` is the real `Chain` structured array, not a dict stub: the
    production object is numpy, and a stub would keep passing if that dtype changed."""
    arr = np.zeros(len(pairs), dtype=Chain)
    for i, (asym_id, name) in enumerate(pairs):
        arr[i]["asym_id"] = asym_id
        arr[i]["name"] = name
    return _Struct(arr)


def test_asym_id_keys_resolve_to_names():
    # boltz2: the matrix is keyed by asym_id, 0-based.
    s = _struct([(0, "A"), (1, "B")])
    assert _chain_ids(s, {0: {}, 1: {}}) == {0: "A", 1: "B"}


def test_position_keys_resolve_when_asym_id_is_not_zero_based():
    # OpenFold3-family: asym_id starts at 1 and the matrix is keyed by position instead.
    s = _struct([(1, "H"), (2, "L"), (3, "G")])
    assert _chain_ids(s, {0: {}, 1: {}, 2: {}}) == {0: "H", 1: "L", 2: "G"}


def test_chains_are_ordered_by_asym_id_not_by_input_order():
    s = _struct([(2, "G"), (0, "A"), (1, "B")])
    assert _chain_ids(s, {0: {}, 1: {}, 2: {}}) == {0: "A", 1: "B", 2: "G"}


def test_a_key_outside_the_chain_set_is_dropped_not_guessed():
    s = _struct([(0, "A")])
    assert _chain_ids(s, {0: {}, 7: {}}) == {0: "A"}
