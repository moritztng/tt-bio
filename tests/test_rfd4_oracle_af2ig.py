"""The AF2-IG adapter behind the RFD4 oracle seam, on the host with a stand-in trunk.

The trunk is stubbed, so this is about the adapter: reading the designed chain's sequence back
off the candidate, the token-order chain labels, and the reward keys beam search weights. It
runs on CPU with no AlphaFold2 weights, which is the point of the seam having a reference arm.
"""
import numpy as np
import pytest

from tt_bio.rfd4 import oracle_af2ig as adapter
from tt_bio.rfd4.oracle import OracleRequest

TARGET = "GAVLS"
BINDER = "KEWTA"
THREE = {v: k for k, v in adapter._THREE_TO_ONE.items()}


def _pdb(target=TARGET, binder=BINDER, target_chain="A", binder_chain="B") -> str:
    """A backbone-only two-chain PDB: enough for the adapter's readers and the featurizer."""
    lines, serial = [], 1
    for chain, seq in ((target_chain, target), (binder_chain, binder)):
        for i, letter in enumerate(seq, start=1):
            for atom, element, offset in (("N", "N", 0.0), ("CA", "C", 1.5), ("C", "C", 3.0),
                                          ("O", "O", 4.0), ("CB", "C", 2.2)):
                x = 3.8 * i + offset
                y = 7.0 if chain == binder_chain else 0.0
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} {THREE[letter]} {chain}{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}")
                serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_chain_sequence_reads_the_designed_chain_back():
    text = _pdb()
    assert adapter.chain_sequence(text, "A") == TARGET
    assert adapter.chain_sequence(text, "B") == BINDER
    with pytest.raises(ValueError, match="not in the candidate structure"):
        adapter.chain_sequence(text, "Z")


def test_non_protein_residues_are_refused():
    text = _pdb().replace("GLY A", "ATP A")
    with pytest.raises(ValueError, match="non-protein"):
        adapter.chain_sequence(text, "A")


def test_residue_view_is_token_order():
    import io

    import biotite.structure.io.pdb as pdb

    atoms = pdb.PDBFile.read(io.StringIO(_pdb())).get_structure(model=1)
    first, chains = adapter.residue_view(atoms)
    assert len(first) == len(TARGET) + len(BINDER)
    assert chains.tolist() == ["A"] * len(TARGET) + ["B"] * len(BINDER)


def test_target_chain_is_inferred_and_a_third_chain_is_refused():
    oracle = adapter.AF2IGOracle(params="unused")
    job = OracleRequest(structure=_pdb(), example_id="x", chain_of_interest="B")
    assert oracle._target_chain(job) == "A"

    three = _pdb() + _pdb(target="AA", target_chain="C").split("END")[0]
    with pytest.raises(ValueError, match="exactly one target chain"):
        oracle._target_chain(OracleRequest(structure=three, example_id="x",
                                           chain_of_interest="B"))


def test_pair_matrix_from_the_chain_confidence_dict():
    pci = {0: {0: 0.9, 1: 0.4}, 1: {0: 0.4, 1: 0.7}}
    assert adapter._pair_matrix(pci, 2).tolist() == [[0.9, 0.4], [0.4, 0.7]]
    assert adapter._pair_matrix({}, 2) is None


class _StubTrunk:
    """A trunk of the right shapes: the adapter's contract, not AlphaFold2's."""

    def __init__(self, tokens):
        self.n = tokens
        self.calls = 0

    def __call__(self, feats, prev):
        import torch

        self.calls += 1
        n = self.n
        rng = torch.Generator().manual_seed(0)
        return {
            "msa_first_row": torch.zeros(n, 256),
            "pair": torch.zeros(n, n, 128),
            "structure": {"final_atom_positions": torch.zeros(n, 37, 3)},
            "plddt_logits": torch.rand(n, 50, generator=rng),
            "pae_logits": torch.rand(n, n, 64, generator=rng),
            "pae_breaks": torch.linspace(0.0, 31.0, 63),
        }

    def distogram_logits(self, pair):
        import torch

        return torch.zeros(self.n, self.n, 64)


def test_score_one_emits_the_reward_keys_beam_search_weights():
    tokens = len(TARGET) + len(BINDER)
    oracle = adapter.AF2IGOracle(params="unused", name="oracle_ppi", recycles=1)
    oracle.model = _StubTrunk(tokens)

    job = OracleRequest(structure=_pdb(), example_id="cand0", chain_of_interest="B")
    metrics = oracle.score_one(job)

    assert oracle.model.calls == 2                       # recycles + 1 passes
    for key in ("overall_plddt", "overall_ptm", "plddt_coi", "pae_coi", "ipae_coi",
                "min_ipae_coi", "min_ipae_A_B"):
        assert key in metrics, key
    assert 0.0 <= metrics["overall_plddt"] <= 100.0
    assert metrics["min_ipae_coi"] <= metrics["ipae_coi"]  # a minimum, not a mean

    named = oracle.fold_batch([job])[0]
    assert named["oracle_ppi.min_ipae_coi"] == pytest.approx(metrics["min_ipae_coi"])


def test_a_failed_candidate_returns_no_keys_so_the_reward_is_minus_inf():
    """Upstream drops a candidate whose oracle failed; it must not rank by accident."""
    from tt_bio.rfd4.search import RewardModel

    oracle = adapter.AF2IGOracle(params="unused")
    oracle.model = _StubTrunk(len(TARGET) + len(BINDER))
    bad = OracleRequest(structure="not a structure", example_id="bad", chain_of_interest="B")
    assert oracle.fold_batch([bad]) == [{}]
    assert RewardModel({"oracle.plddt_coi": {"weight": 1.0}}).aggregate_metrics_output({}) == -np.inf


def test_fold_batch_before_setup_is_an_error():
    with pytest.raises(RuntimeError, match="setup"):
        adapter.AF2IGOracle(params="unused").fold_batch([])


def test_setup_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        adapter.AF2IGOracle(params="unused", backend="gpu").setup()
