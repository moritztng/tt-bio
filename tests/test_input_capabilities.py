"""Every model's answer to every input feature, driven off the one capability table.

The bug class this exists to stop is not a crash. It is a fold that accepts an input, drops
it, and reports success: ESMFold2 folded a ligand-bearing YAML as bare protein, `cyclic:
true` folded linear on four models, and `modifications:` still reaches Protenix, OpenDDE and
the OF3 family only to be discarded by a featurizer that has no argument for it. Those were
found one at a time because the enforcement was one validator per model.

So the enforcement is one function over one table now, and this test is the full cross
product: for every shipped --model and every feature the reader can express, the table's
verdict has to be what the check actually does. A model added without a row fails
test_every_shipped_model_has_a_row; a predict path that skips the check fails
test_every_predict_path_calls_check_input. Host-only, no card: reading YAML and raising.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tt_bio.capabilities import (CAPABILITY, FEATURES, HONOURED, NOTED, REFUSED, check_input,
                                 detect, honoured_by)
from tt_bio.main import PREDICT_MODELS, _read_bio_chains

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
_HEAD = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n"

#: feature -> a YAML that expresses THAT feature and nothing else in the table. Kept minimal
#: on purpose: a fixture that also carried a ligand would make a ligand refusal look like a
#: constraint refusal.
INPUTS: dict[str, str] = {
    "ligand": _HEAD + "  - ligand:\n      id: B\n      ccd: ATP\n",
    "rna": _HEAD + "  - rna:\n      id: R\n      sequence: GAUC\n",
    "dna": _HEAD + "  - dna:\n      id: D\n      sequence: GATC\n",
    "cyclic": _HEAD + "      cyclic: true\n",
    "modifications": _HEAD + "      modifications:\n        - position: 5\n          ccd: TPO\n",
    "templates": _HEAD + "      templates: /nonexistent/tmpl.npz\n",
    "bond": _HEAD + ("constraints:\n  - bond:\n      atom1: [A, 5, SG]\n"
                     "      atom2: [A, 9, SG]\n"),
    "pocket": _HEAD + ("constraints:\n  - pocket:\n      binder: A\n"
                       "      contacts: [[A, 5]]\n"),
    "affinity": _HEAD + "properties:\n  - affinity:\n      binder: A\n",
}

PLAIN = _HEAD


def _yaml(tmp_path, text, name="q.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _check(tmp_path, text, model, notes=None):
    """Run the real reader and the real check, collecting NOTED messages."""
    p = _yaml(tmp_path, text)
    return check_input(p, _read_bio_chains(p), model,
                       echo=(notes.append if notes is not None else None))


def test_every_shipped_model_has_a_row():
    """A new port with no row is invisible: nothing would tell you it drops `modifications:`."""
    assert set(PREDICT_MODELS) <= set(CAPABILITY), \
        f"no capability row for {sorted(set(PREDICT_MODELS) - set(CAPABILITY))}"
    for model, caps in CAPABILITY.items():
        assert set(caps) == set(FEATURES), \
            f"{model} row does not cover {sorted(set(FEATURES) ^ set(caps))}"
        assert set(caps.values()) <= {HONOURED, REFUSED, NOTED}


def test_boltz2_honours_the_whole_input_language():
    """The negative control for the table: if every row were REFUSED the cross product below
    would still pass. Boltz-2 has the upstream parser, the constraint embedder, the template
    pipeline and the affinity head, so nothing in the reader is beyond it."""
    assert set(CAPABILITY["boltz2"].values()) == {HONOURED}


@pytest.mark.parametrize("model", sorted(CAPABILITY))
@pytest.mark.parametrize("feature", sorted(INPUTS))
def test_the_table_verdict_is_what_the_check_does(tmp_path, model, feature, capsys):
    verdict = CAPABILITY[model][feature]
    label = FEATURES[feature][0]
    notes: list[str] = []
    if verdict == REFUSED:
        with pytest.raises(RuntimeError) as e:
            _check(tmp_path, INPUTS[feature], model, notes)
        msg = str(e.value)
        assert model in msg and label in msg
        # A refusal that names nowhere else to go leaves the user stuck.
        others = [m for m in honoured_by(feature) if m != model]
        if others:
            assert any(m in msg for m in others), msg
        return
    assert _check(tmp_path, INPUTS[feature], model, notes), "the feature was not even detected"
    if verdict == NOTED:
        assert any(label in n for n in notes), notes
    else:
        assert not any(label in n for n in notes), notes


@pytest.mark.parametrize("model", sorted(CAPABILITY))
def test_a_plain_protein_input_is_accepted_by_every_model(tmp_path, model):
    """The control for the whole file: a check that refused everything would pass the arms
    above."""
    notes: list[str] = []
    assert _check(tmp_path, PLAIN, model, notes) == {}
    assert notes == []


@pytest.mark.parametrize("model", sorted(CAPABILITY))
def test_cyclic_false_is_not_a_refusal(tmp_path, model):
    _check(tmp_path, _HEAD + "      cyclic: false\n", model)


def test_every_refused_feature_is_named_at_once(tmp_path):
    """A user fixing one key should not have to run again to find the next."""
    text = (_HEAD + "      cyclic: true\n      modifications:\n        - position: 5\n"
            "          ccd: TPO\n")
    with pytest.raises(RuntimeError) as e:
        _check(tmp_path, text, "protenix-v2")
    msg = str(e.value)
    assert "cyclic" in msg and "modifications" in msg


def test_every_offending_chain_id_is_named_and_no_other(tmp_path):
    text = (f"version: 1\nsequences:\n  - protein:\n      id: [A, B]\n      sequence: {SEQ}\n"
            f"      cyclic: true\n  - protein:\n      id: C\n      sequence: {SEQ}\n")
    with pytest.raises(RuntimeError) as e:
        _check(tmp_path, text, "openbind")
    msg = str(e.value)
    assert "A" in msg and "B" in msg
    assert "C" not in msg.split("Honoured by")[0]


def test_a_fasta_input_still_sees_its_molecule_types(tmp_path):
    """The keyed features are YAML-only, but a FASTA expresses molecule types, so a ligand
    record in a FASTA has to reach the same refusal a YAML ligand does."""
    p = tmp_path / "q.fasta"
    p.write_text(f">A|protein\n{SEQ}\n>B|ccd\nATP\n")
    chains = _read_bio_chains(p)
    assert detect(p, chains) == {"ligand": "chain(s) B"}
    with pytest.raises(RuntimeError, match="polymer-only"):
        check_input(p, chains, "openfold3", echo=None)
    check_input(p, chains, "openbind", echo=None)


def test_an_empty_or_odd_yaml_does_not_raise_by_accident(tmp_path):
    for text in ("", "sequences:\n  - notadict\n", "version: 1\n"):
        p = _yaml(tmp_path, text)
        assert detect(p, []) == {}


def test_the_committed_cyclic_example_is_refused():
    p = Path(__file__).resolve().parent.parent / "examples" / "cyclic_prot.yaml"
    if not p.exists():
        pytest.skip("examples/cyclic_prot.yaml not in this checkout")
    with pytest.raises(RuntimeError, match="cyclic"):
        check_input(p, _read_bio_chains(p), "openbind", echo=None)


def test_every_predict_path_calls_check_input():
    """A check a path does not call is not a guard, and a missing call is exactly how the
    ESMFold2 cyclic hole sat unnoticed."""
    from tt_bio.worker import _WorkerState

    for method in ("_predict_esmfold2_one", "_predict_opendde_one", "_protenix_inputs",
                   "_predict_rf3_one", "_predict_openfold3_one"):
        src = inspect.getsource(getattr(_WorkerState, method))
        assert "check_input(path, chains," in src, f"{method} does not call check_input"
    # protenix-v1/v2 reach it one level down, through the shared input builder.
    assert "self._protenix_inputs(" in inspect.getsource(_WorkerState._predict_protenix_one)


@pytest.mark.parametrize("model", ["esmfold2", "esmfold2-fast"])
def test_the_dispatch_path_refuses_before_any_device_work(tmp_path, model):
    """Proves a real job reaches the check through predict_one, not just that the function
    raises. A bare instance with no loaded model is enough because the check runs first."""
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    with pytest.raises(RuntimeError) as e:
        state.predict_one(_yaml(tmp_path, INPUTS["cyclic"]), {"model": model})
    assert model in str(e.value) and "cyclic" in str(e.value)


def test_a_plain_job_still_gets_past_the_dispatch_guard(tmp_path):
    """Control for the test above: a linear chain must fail on the missing config, not the
    guard."""
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    with pytest.raises(KeyError, match="msa_dir"):
        state.predict_one(_yaml(tmp_path, PLAIN), {"model": "esmfold2"})


def test_the_vendored_of3_tree_still_has_no_cyclic_field():
    """Why the OF3/OpenBind cyclic rows are REFUSED. If a vendor bump restores Chain.cyclic
    and the cyclic_mask feature, this fails and the row should be revisited rather than left
    refusing something the tree now supports."""
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Chain,
    )
    assert "cyclic" not in Chain.model_fields


def test_the_reader_still_drops_what_the_table_refuses():
    """Why the cyclic and modifications rows are REFUSED rather than honoured: there is no
    path from the key to the featurizer. Each of these failing means a port gained the
    feature and its row is now wrong."""
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.worker import _WorkerState

    assert "cyclic" not in inspect.getsource(_read_bio_chains), \
        "the reader now carries `cyclic` -- revisit the cyclic rows"
    assert "modifications" not in inspect.signature(build_complex_features).parameters, \
        "build_complex_features now takes modifications -- revisit the Protenix/OpenDDE rows"
    of3 = inspect.getsource(_WorkerState._predict_openfold3_one)
    assert '"non_canonical_residues": None' in of3, \
        "the OF3 query now carries non_canonical_residues -- revisit the OF3 modifications rows"
