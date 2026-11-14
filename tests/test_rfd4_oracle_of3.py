"""The OpenFold3 backend behind the oracle seam, with the fold stubbed.

The device half cannot run here, so this covers the two places the adapter can be wrong without
a card: the per-ATOM to per-TOKEN pLDDT reduction (AF3's summary is per token, OF3 reports per
atom), and the mapping from OF3's confidence dict to the reward keys a beam config weights.
"""
import types

import numpy as np
import pytest

from tt_bio.rfd4 import oracle_of3 as adapter
from tt_bio.rfd4.oracle import OracleRequest
from tests.test_rfd4_oracle_af2ig import _pdb


def test_token_plddt_is_the_mean_over_a_tokens_atoms():
    atom = np.array([0.2, 0.4, 0.9, 0.1, 0.5])
    index = np.array([0, 0, 1, 1, 1])
    assert adapter.token_plddt(atom, index, 2).tolist() == pytest.approx([0.3, 0.5])


def test_a_token_with_no_atoms_is_zero_not_a_divide_by_zero():
    out = adapter.token_plddt(np.array([0.5]), np.array([0]), 3)
    assert out.tolist() == [0.5, 0.0, 0.0]


def test_structure_chains_are_in_token_order():
    assert adapter.structure_chains(_pdb()) == ["A", "B"]


def _stub_result(n_token, n_chain=2):
    rng = np.random.default_rng(0)
    pae = rng.uniform(1.0, 20.0, size=(n_token, n_token))
    conf = {"plddt_atom": rng.uniform(0.4, 0.9, size=n_token), "pae": pae,
            "ptm": 0.6, "iptm": 0.3, "ranking_score": 0.55, "disorder": 0.1, "has_clash": 0.0,
            "pair_chains_iptm": {i: {j: 0.3 + 0.1 * (i == j) for j in range(n_chain)}
                                 for i in range(n_chain)},
            "chain_ptm": {i: 0.5 for i in range(n_chain)}}
    return types.SimpleNamespace(confidence=[conf], samples=[None])


def test_score_one_emits_the_reward_keys_from_openfold3s_confidence_dict(monkeypatch):
    import tt_bio.worker as worker

    text = _pdb()
    n_token = 10                                   # 5 + 5 residues, one atom per token here
    host = types.SimpleNamespace(
        features={"restype": np.zeros((n_token, 32)),
                  "asym_id": np.array([0] * 5 + [1] * 5)},
        aux={"atom_to_token_index": np.arange(n_token)}, seed=0)
    monkeypatch.setattr(worker, "openfold3_host_inputs", lambda *a, **k: host)
    monkeypatch.setattr(worker, "openfold3_device_fold",
                        lambda *a, **k: _stub_result(n_token))

    oracle = adapter.OpenFold3Oracle(msa_dir="/tmp/of3_oracle_msa",
                                     template_structures="/tmp/of3_oracle_tmpl",
                                     name="oracle_ppi")
    oracle.model = object()                        # setup() would open a device; stand in for it
    metrics = oracle.fold_batch([OracleRequest(structure=text, example_id="c0",
                                               chain_of_interest="B")])[0]

    for key in ("oracle_ppi.overall_plddt", "oracle_ppi.plddt_coi", "oracle_ppi.pae_coi",
                "oracle_ppi.ipae_coi", "oracle_ppi.min_ipae_coi", "oracle_ppi.min_ipae_A_B",
                "oracle_ppi.overall_ptm", "oracle_ppi.overall_iptm"):
        assert key in metrics, key
    assert 40.0 <= metrics["oracle_ppi.overall_plddt"] <= 90.0      # 0..1 in, 0..100 out
    assert metrics["oracle_ppi.min_ipae_coi"] <= metrics["oracle_ppi.ipae_coi"]


def test_a_failed_fold_yields_no_keys(monkeypatch):
    import tt_bio.worker as worker

    def boom(*a, **k):
        raise RuntimeError("device fold failed")

    monkeypatch.setattr(worker, "openfold3_host_inputs", boom)
    oracle = adapter.OpenFold3Oracle(msa_dir="/tmp/x", template_structures="/tmp/y")
    oracle.model = object()
    assert oracle.fold_batch([OracleRequest(structure=_pdb(), example_id="c")]) == [{}]


def test_fold_batch_before_setup_is_an_error():
    oracle = adapter.OpenFold3Oracle(msa_dir="/tmp/x", template_structures="/tmp/y")
    with pytest.raises(RuntimeError, match="setup"):
        oracle.fold_batch([])


def test_the_protocol_defaults_are_upstreams_oracle_not_a_cheaper_one():
    """A beam scored by a cheaper fold than the recipe's is a different algorithm."""
    oracle = adapter.OpenFold3Oracle(msa_dir="/tmp/x", template_structures="/tmp/y")
    assert (oracle.recycles, oracle.sampling_steps, oracle.diffusion_samples) == (5, 200, 1)
