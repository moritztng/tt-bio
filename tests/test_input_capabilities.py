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

from tt_bio.capabilities import (CAPABILITY, CHAIN_FEATURES, CHAINS_ELSEWHERE, FEATURES,
                                 FLAG_READERS, FLAG_WHY, HONOURED, NOTED, REFUSED,
                                 check_capabilities, detect, honoured_by, how, unread_flags)
from tt_bio.main import AFFINITY_MODELS, PREDICT_MODELS, _read_bio_chains

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
    return check_capabilities(p, _read_bio_chains(p), model,
                       echo=(notes.append if notes is not None else None))


def test_every_shipped_model_has_a_row():
    """A new port with no row is invisible: nothing would tell you it drops `modifications:`."""
    shipped = set(PREDICT_MODELS) | set(AFFINITY_MODELS)
    assert shipped <= set(CAPABILITY), \
        f"no capability row for {sorted(shipped - set(CAPABILITY))}"
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
    if model in CHAINS_ELSEWHERE and feature in CHAIN_FEATURES:
        pytest.skip(f"{model} has its own reader; the chain columns record its verdict, they "
                    f"are not applied here (see CHAINS_ELSEWHERE)")
    verdict = CAPABILITY[model][feature]
    label = FEATURES[feature][0]
    notes: list[str] = []
    if verdict == REFUSED:
        with pytest.raises(RuntimeError) as e:
            _check(tmp_path, INPUTS[feature], model, notes)
        msg = str(e.value)
        assert how(model) in msg and label in msg
        # A refusal that names nowhere else to go leaves the user stuck.
        others = [m for m in honoured_by(feature) if m != model]
        if others:
            assert any(how(m) in msg for m in others), msg
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
    text = (_HEAD + "      cyclic: true\n"
            + "constraints:\n  - pocket:\n      binder: A\n      contacts: [[A, 5]]\n")
    with pytest.raises(RuntimeError) as e:
        _check(tmp_path, text, "protenix-v2")
    msg = str(e.value)
    assert "cyclic" in msg and "pocket" in msg


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
        check_capabilities(p, chains, "openfold3", echo=None)
    check_capabilities(p, chains, "openbind", echo=None)


def test_an_empty_or_odd_yaml_does_not_raise_by_accident(tmp_path):
    for text in ("", "sequences:\n  - notadict\n", "version: 1\n"):
        p = _yaml(tmp_path, text)
        assert detect(p, []) == {}


def test_the_committed_cyclic_example_is_refused():
    p = Path(__file__).resolve().parent.parent / "examples" / "cyclic_prot.yaml"
    if not p.exists():
        pytest.skip("examples/cyclic_prot.yaml not in this checkout")
    with pytest.raises(RuntimeError, match="cyclic"):
        check_capabilities(p, _read_bio_chains(p), "openbind", echo=None)


def test_every_predict_path_calls_check_capabilities():
    """A check a path does not call is not a guard, and a missing call is exactly how the
    ESMFold2 cyclic hole sat unnoticed."""
    from tt_bio.worker import _WorkerState

    for method in ("_predict_esmfold2_one", "_predict_opendde_one", "_protenix_inputs",
                   "_predict_rf3_one", "_predict_openfold3_one"):
        src = inspect.getsource(getattr(_WorkerState, method))
        assert "check_capabilities(path, chains," in src, f"{method} does not call check_input"
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
    """Why the cyclic rows are REFUSED rather than honoured: there is no path from the key to
    the featurizer. This failing means a port gained the feature and its row is now wrong."""
    assert "cyclic" not in inspect.getsource(_read_bio_chains), \
        "the reader now carries `cyclic` -- revisit the cyclic rows"


def test_modifications_reach_every_featurizer_that_honours_them():
    """The other half of the same guard, now that the key is wired: a model whose row says
    `yes` has to have a path from `modifications:` into its features."""
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.worker import _WorkerState

    assert "modifications" in inspect.signature(build_complex_features).parameters
    for name in ("_predict_opendde_one", "_protenix_inputs"):
        assert "modifications=[mods" in inspect.getsource(getattr(_WorkerState, name)), \
            f"{name} builds features without the chain's modifications"
    of3 = inspect.getsource(_WorkerState._predict_openfold3_one)
    assert '"non_canonical_residues": ({m["position"]' in of3, \
        "the OF3 query stopped carrying non_canonical_residues"
    for model in ("protenix-v1", "protenix-v2", "opendde", "opendde-abag", "openfold3",
                  "openbind", "esmfold2"):
        assert CAPABILITY[model]["modifications"] == HONOURED
    assert CAPABILITY["rf3"]["modifications"] == REFUSED, \
        "rf3 reads modified residues from its own JSON/CIF spec, not from this YAML"


def test_the_nesso1_row_covers_the_command_that_is_not_predict():
    """Nesso-1 reads the same Boltz-2 affinity yaml through its own parser, so it needs a row
    in the same table: its `constraints:` block used to go by in silence while its protein
    keys were already warned about. It answers `properties: affinity`, which is why the
    affinity hint now points at `tt-bio affinity --model nesso1` and not only at boltz2."""
    from tt_bio.capabilities import COMMAND

    assert CAPABILITY["nesso1"]["affinity"] == HONOURED
    assert "nesso1" in honoured_by("affinity")
    assert COMMAND["nesso1"].startswith("tt-bio affinity")
    assert CAPABILITY["nesso1"]["bond"] == NOTED


def test_the_affinity_command_runs_the_check():
    import inspect

    from tt_bio.main import affinity_cmd

    src = inspect.getsource(affinity_cmd.callback)
    assert "check_capabilities(yp, None, model)" in src


# --- output flags a model does not read --------------------------------------------------
# Same failure as the yaml keys, one layer out: --write_pae was accepted and written by
# nobody on esmfold2 and rf3, --write_pde did nothing on protenix (--write_pae writes both),
# and --max_msa_seqs did nothing on protenix/opendde/rf3.

ALL_FLAGS = {f: True for f in FLAG_READERS}


@pytest.mark.parametrize("model", sorted(CAPABILITY))
def test_a_flag_is_either_read_or_reported(model):
    notes = unread_flags(model, ALL_FLAGS)
    for flag, readers in FLAG_READERS.items():
        # match the flag in its own slot: a reason may name another flag
        named = [n for n in notes if f"ignores {flag}:" in n]
        if model in readers:
            assert not named, f"{model} reads {flag} but is warned about it"
        else:
            assert len(named) == 1, f"{model} does not read {flag} and says nothing"
            assert model in named[0]


def test_no_flag_is_reported_when_none_was_passed():
    """The control: a warning that always fires is noise, not information."""
    for model in CAPABILITY:
        assert unread_flags(model, {f: False for f in FLAG_READERS}) == []


def test_boltz2_reads_every_output_flag():
    assert unread_flags("boltz2", ALL_FLAGS) == []


def test_every_reason_names_a_model_that_exists():
    for (flag, model) in FLAG_WHY:
        assert flag in FLAG_READERS, f"{flag} has a reason but no reader list"
        assert model in CAPABILITY, f"{flag}/{model}: no such model"
        assert model not in FLAG_READERS[flag], f"{flag}/{model} both reads it and explains why not"


def test_predict_reports_them():
    import inspect

    from tt_bio.main import predict

    src = inspect.getsource(predict.callback)
    assert "unread_flags(model," in src


# --- --max_msa_seqs actually caps depth ---------------------------------------------------
# It was listed as read by boltz2/esmfold2/openfold3/openbind and unread by protenix/opendde/
# rf3. Two of those were wrong: the OF3 family read OF3_MAX_MSA_SEQS, never the flag. All of
# them cap now, through one truncation function, and only when the user asks -- protenix,
# opendde, rf3 and the OF3 family fold the resolved alignment whole by default, so inheriting
# boltz2's 8192 would silently change every fold they have already produced.

A3M = ">q\nMKTA\n>h1\nMKTS\n>h2\nMKSA\n>h3\nMATA\n"


def test_cap_a3m_text_counts_records_from_the_top():
    from tt_bio.main import cap_a3m_text

    assert cap_a3m_text(A3M, 2) == ">q\nMKTA\n>h1\nMKTS\n"
    assert cap_a3m_text(A3M, 1) == ">q\nMKTA\n"


def test_an_uncapped_a3m_is_returned_untouched():
    """The control: no cap, or a cap the file already fits under, must not rewrite the input."""
    from tt_bio.main import cap_a3m_text

    for cap in (None, 0, 4, 99):
        assert cap_a3m_text(A3M, cap) is A3M


def test_cap_a3m_file_only_writes_a_copy_when_it_has_to(tmp_path):
    from tt_bio.main import cap_a3m_file

    src = tmp_path / "aln.a3m"
    src.write_text(A3M)
    assert cap_a3m_file(src, None, tmp_path) == src
    assert cap_a3m_file(src, 99, tmp_path) == src
    out = cap_a3m_file(src, 2, tmp_path)
    assert out != src and out.read_text().count(">") == 2


def test_the_cap_reaches_every_msa_path():
    """Each model's MSA loading path reads the cap the CLI put in the config. Without this
    the flag is accepted and dropped, which is exactly what it did on protenix/opendde/rf3."""
    from tt_bio import worker
    from tt_bio.worker import _WorkerState

    assert 'cfg.get("msa_cap")' in inspect.getsource(worker._build_chain_specs), \
        "protenix/opendde chain specs no longer apply the cap"
    for name in ("_predict_rf3_one", "_predict_openfold3_one"):
        assert 'cfg.get("msa_cap")' in inspect.getsource(getattr(_WorkerState, name)), \
            f"{name} no longer applies the cap"
    src = inspect.getsource(_WorkerState._predict_opendde_one)
    assert "cap_a3m_text(paired.get(" in src, "the opendde paired MSA is no longer capped"


def test_the_default_does_not_travel():
    """8192 is boltz2's and esmfold2's shipped default, not a cap the other models ever had.
    predict must pass the cap on only when the flag was set, or every protenix/opendde/rf3/
    OF3 fold silently changes depth."""
    from tt_bio.main import predict

    src = inspect.getsource(predict.callback)
    assert "ParameterSource.DEFAULT" in src and '"msa_cap": msa_cap' in src


@pytest.mark.parametrize("model", sorted(CAPABILITY))
def test_every_folding_model_reports_the_depth_it_used(model):
    """`msa: true` says an alignment was used, not how deep. --max_msa_seqs is only checkable
    from the outside if the depth is in the row."""
    from tt_bio.worker import _WorkerState

    paths = {"protenix-v1": "_protenix_emit", "protenix-v2": "_protenix_emit",
             "opendde": "_predict_opendde_one", "opendde-abag": "_predict_opendde_one",
             "openfold3": "_predict_openfold3_one", "openbind": "_predict_openfold3_one",
             "rf3": "_predict_rf3_one"}
    if model not in paths:
        pytest.skip(f"{model} has no MSA depth of its own to report")
    assert '"msa_depth"' in inspect.getsource(getattr(_WorkerState, paths[model]))
