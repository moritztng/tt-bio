"""The RFD4 oracle seam's confidence reductions, against an independent float64 reference.

The reward keys here are upstream's (RFD4-Proteina @ 2db07bb4,
`rfproteina/metrics/oracles/oracle_metric.py`), so the names and the 0..1 -> 0..100 scales are
part of the contract: a beam-search config weights `oracle.plddt_coi` by name, and a key that is
missing or renamed scores the candidate -inf and drops it from the beam.
"""
import numpy as np
import pytest

from tt_bio.rfd4 import oracle as orc


def _case(seed=0, sizes=(7, 5, 4)):
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    chains = np.array([c for c, size in zip("ABC", sizes) for _ in range(size)])
    return {
        "plddt": rng.uniform(0.3, 0.98, n),
        "pae": rng.uniform(0.5, 30.0, (n, n)),     # directional: not symmetrised
        "chains": chains,
        "ptm": 0.71,
        "iptm": 0.42,
        "chain_ptm": [0.8, 0.6, 0.5],
        "chain_pair_iptm": rng.uniform(0.1, 0.9, (3, 3)),
    }


def test_per_chain_reductions_match_a_loop_reference():
    case = _case()
    summary, order = orc.summary_confidences(**case)
    assert order == ["A", "B", "C"]

    chains, pae, plddt = case["chains"], case["pae"], case["plddt"]
    for i, ci in enumerate(order):
        rows = chains == ci
        assert summary["chain_plddt"][i] == pytest.approx(plddt[rows].mean(), abs=1e-12)
        assert summary["chain_pae"][i] == pytest.approx(pae[np.ix_(rows, rows)].mean(), abs=1e-12)
        for j, cj in enumerate(order):
            block = pae[np.ix_(rows, chains == cj)]
            assert summary["chain_pair_pae"][i][j] == pytest.approx(block.mean(), abs=1e-12)
            assert summary["chain_pair_pae_min"][i][j] == pytest.approx(block.min(), abs=1e-12)

    assert summary["overall_plddt"] == pytest.approx(plddt.mean(), abs=1e-12)
    assert summary["overall_pae"] == pytest.approx(pae.mean(), abs=1e-12)


def test_pae_stays_directional():
    """AF3's chain-pair PAE is not symmetric; averaging the two directions is the reader's job."""
    summary, _ = orc.summary_confidences(**_case())
    m = np.array(summary["chain_pair_pae"])
    assert not np.allclose(m, m.T)


def test_reward_keys_and_scales():
    case = _case()
    summary, order = orc.summary_confidences(ranking_score=0.63, has_clash=0.0, **case)
    metrics = orc.extract_confidence_metrics(summary, "B", order)

    assert metrics["overall_plddt"] == pytest.approx(summary["overall_plddt"] * 100)
    assert metrics["overall_ptm"] == pytest.approx(71.0)
    assert metrics["overall_iptm"] == pytest.approx(42.0)
    assert metrics["overall_pae"] == pytest.approx(summary["overall_pae"])   # Angstrom, unscaled
    assert metrics["ranking_score"] == pytest.approx(0.63)
    assert metrics["has_clash"] == 0.0
    assert "fraction_disordered" not in metrics          # absent input, absent key

    coi = order.index("B")
    assert metrics["plddt_coi"] == pytest.approx(summary["chain_plddt"][coi] * 100)
    assert metrics["ptm_coi"] == pytest.approx(60.0)
    assert metrics["pae_coi"] == pytest.approx(summary["chain_pae"][coi])

    pae = np.array(summary["chain_pair_pae"])
    pae_min = np.array(summary["chain_pair_pae_min"])
    others = [i for i in range(3) if i != coi]
    assert metrics["ipae_coi"] == pytest.approx(
        np.mean([(pae[coi, j] + pae[j, coi]) / 2 for j in others]))
    assert metrics["min_ipae_coi"] == pytest.approx(
        np.min([(pae_min[coi, j] + pae_min[j, coi]) / 2 for j in others]))

    # One key per unordered pair, so a two-interface design does not hide its worse interface.
    assert metrics["min_ipae_A_B"] == pytest.approx((pae_min[0, 1] + pae_min[1, 0]) / 2)
    assert set(k for k in metrics if k.startswith("min_ipae_") and k != "min_ipae_coi") == {
        "min_ipae_A_B", "min_ipae_A_C", "min_ipae_B_C"}


def test_monomer_has_no_interface_keys():
    n = 9
    summary, order = orc.summary_confidences(
        plddt=np.full(n, 0.9), pae=np.full((n, n), 4.0), chains=["A"] * n, ptm=0.8)
    metrics = orc.extract_confidence_metrics(summary, "A", order)
    assert metrics["plddt_coi"] == pytest.approx(90.0)
    assert not [k for k in metrics if "ipae" in k or "iptm" in k]


def test_unknown_chain_of_interest_drops_only_the_coi_keys():
    case = _case()
    summary, order = orc.summary_confidences(**case)
    metrics = orc.extract_confidence_metrics(summary, "Z", order)
    assert not [k for k in metrics if k.endswith("_coi")]
    assert metrics["overall_ptm"] == pytest.approx(71.0)   # the global keys still stand


def test_prefix_namespaces_under_the_config_key():
    summary, order = orc.summary_confidences(**_case())
    named = orc.prefix("oracle_ppi", orc.extract_confidence_metrics(summary, "A", order))
    assert "oracle_ppi.min_ipae_coi" in named
