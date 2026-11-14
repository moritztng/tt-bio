"""OpenFold3's host half, extracted out of the serving path so an oracle can share it.

`openfold3_host_inputs` is the featurization `_WorkerState._predict_openfold3_one` used to carry
inline: query -> MSA -> features -> block aux, template slots, relpos. It touches no device, which
is both why RFD4's beam-search oracle can call it per candidate and why this test runs on a box
holding no card.
"""
import textwrap

import pytest

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


@pytest.fixture(scope="module")
def host_inputs(tmp_path_factory):
    from tt_bio.worker import openfold3_host_inputs

    work = tmp_path_factory.mktemp("of3host")
    yaml = work / "probe.yaml"
    yaml.write_text(textwrap.dedent(f"""\
        sequences:
          - protein:
              id: A
              sequence: {SEQ}
        """))
    cfg = {"model": "openfold3", "msa_dir": str(work / "msas"),
           "template_structures": str(work / "templates"), "seed": 0,
           "single_sequence": True, "output_format": "cif"}
    (work / "msas").mkdir()
    (work / "templates").mkdir()
    return openfold3_host_inputs(yaml, cfg, model_name="openfold3")


def test_the_host_half_runs_with_no_device(host_inputs):
    assert int(host_inputs.features["restype"].shape[0]) == len(SEQ)
    assert len(host_inputs.chains) == 1


def test_single_sequence_is_a_one_row_alignment_not_a_disabled_stack(host_inputs):
    """Upstream substitutes the query sequence rather than turning the MSA stack off; folding
    with the stack off instead costs 1.5 A CA-RMSD on ubiquitin (tt_bio/worker.py)."""
    assert host_inputs.msa_feat.shape[1] == len(SEQ)
    assert host_inputs.msa_feat.shape[0] >= 1
    assert host_inputs.of3_query.use_msas is True


def test_it_carries_everything_the_device_half_is_handed(host_inputs):
    """The fold's kwargs come from exactly these fields; a missing one is a silent divergence
    between the serving path and the oracle, which is the whole reason for one callable."""
    assert host_inputs.template_feat is not None and host_inputs.template_slots is not None
    assert host_inputs.relpos is not None
    for key in ("atom_mask", "atom_to_token_index", "ca_mask", "n_atom", "n_token",
                "max_atom_per_token_mask", "NP", "nb"):
        assert key in host_inputs.aux, key
    assert int(host_inputs.aux["n_token"]) == len(SEQ)
    for key in ("restype", "profile", "deletion_mean", "token_bonds", "token_mask",
                "atom_array", "asym_id", "is_protein", "is_rna", "is_dna"):
        assert key in host_inputs.features, key


def test_the_full_pae_never_needed_a_file(host_inputs):
    """The oracle reads confidence off the fold result in memory. `write_pae` only mirrors it to
    disk, so an oracle does not need a per-candidate file round trip."""
    import inspect

    from tt_bio.worker import _WorkerState

    src = inspect.getsource(_WorkerState._predict_openfold3_one)
    assert "result.confidence" in src
    assert 'confs[order[0]]' in src


def test_the_serving_path_goes_through_both_extracted_callables():
    """If a future edit inlines either half back into the method, the oracle and the service
    start drifting silently. Cheaper to fail here."""
    import inspect

    from tt_bio.worker import _WorkerState

    src = inspect.getsource(_WorkerState._predict_openfold3_one)
    assert "openfold3_host_inputs(" in src
    assert "openfold3_device_fold(" in src
    assert "build_openfold3_features(" not in src      # the host half left the method
    assert "run_input_atom_encoder(" not in src        # so did the device half
