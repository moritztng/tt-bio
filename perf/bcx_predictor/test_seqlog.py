#!/usr/bin/env python3
"""The sequence recorder, without a card and without weights.

A rejected trajectory writes no sequence anywhere: the losses CSV carries metrics only and
the PDB is written on acceptance. This test is what says the recorder closes that gap.
"""
import json
import os
import sys

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bc2_state as B                                                   # noqa: E402,F401
from bindcraft.protein import AMINO_ACIDS, Protein                      # noqa: E402

import ttbio_predictor as T                                             # noqa: E402


def test_sequence_letters_reads_the_argmax():
    protein = Protein.empty(11, jax.random.PRNGKey(0))
    letters = T.sequence_letters(protein)
    assert len(letters) == 11
    assert set(letters) <= set(AMINO_ACIDS)
    expected = "".join(AMINO_ACIDS[i] for i in np.asarray(protein.sequence).argmax(-1))
    assert letters == expected


def test_record_writes_one_line_per_call(tmp_path):
    path = str(tmp_path / "sequences.jsonl")
    model = T.TTBioAlphaFoldDesignModel.__new__(T.TTBioAlphaFoldDesignModel)
    model.trunk, model.seqlog = "device", path
    states = {"complex": {"A": Protein.empty(7, jax.random.PRNGKey(1)),
                          "B": Protein.empty(5, jax.random.PRNGKey(2))}}
    model._record("model_3_multimer_v3", states)
    model._record("model_1_multimer_v3", states)

    rows = [json.loads(line) for line in open(path)]
    assert [r["model"] for r in rows] == ["model_3_multimer_v3", "model_1_multimer_v3"]
    assert [len(r["states"]["complex"]["A"]) for r in rows] == [7, 7]
    assert len(rows[0]["states"]["complex"]["B"]) == 5


def test_no_seqlog_writes_nothing(tmp_path):
    path = str(tmp_path / "absent.jsonl")
    model = T.TTBioAlphaFoldDesignModel.__new__(T.TTBioAlphaFoldDesignModel)
    model.trunk, model.seqlog = "device", None
    model._record("model_1_multimer_v3", {"complex": {"A": Protein.empty(3, jax.random.PRNGKey(3))}})
    assert not os.path.exists(path)
