"""`cyclic: true` must be refused by every model that cannot honour it, not folded linear.

OF3/OpenBind: upstream's query format carries `Chain.cyclic` and its structure featurizer
derives a `cyclic_mask` from it; tt-bio's vendored copy has neither, both dropped when the
tree was vendored.

Protenix (v1/v2) and OpenDDE: there is no cyclic input path to drop. `_read_bio_chains` never
reads the flag, and upstream Protenix v0.5.0 has no cyclic chain flag either -- its only
"cyclic" is the `cyclic-pseudo-peptide` LIGAND entity label. Folding
examples/cyclic_prot.yaml with --model protenix-v1 SUCCEEDED and returned a linear 13-token
structure, which is how this was found during the v1 bring-up sweep.

ESMFold2 / ESMFold2-Fast: same shape. `_read_protein_chains` returns (chain_id, sequence,
msa_spec, modifications) and never reads the flag either, so examples/cyclic_prot.yaml folded
to status=ok on a straight chain. This was the last predict path missing the guard.

Cyclisation changes the structure, so this is a hard error, the same treatment `constraints:`
gets and unlike `properties: affinity`, which only omits an extra output. boltz2 and rf3 do
honour the flag and must never be routed here. Card-free: yaml reading and a raise.
"""
from pathlib import Path

import pytest

from tt_bio.worker import _validate_cyclic_unsupported

LINEAR = """version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKG
"""
CYCLIC = """version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKG
      cyclic: true
"""
CYCLIC_LIST_IDS = """version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: QLEDSEVEAVAKG
      cyclic: true
  - protein:
      id: C
      sequence: QLEDSEVEAVAKG
"""
CYCLIC_COMMA_IDS = """version: 1
sequences:
  - protein:
      id: A,B
      sequence: QLEDSEVEAVAKG
      cyclic: true
"""
CYCLIC_FALSE = """version: 1
sequences:
  - protein:
      id: A
      sequence: QLEDSEVEAVAKG
      cyclic: false
"""


def _yaml(tmp_path, text, name="q.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


@pytest.mark.parametrize("model", ["openfold3", "openbind", "protenix-v1", "protenix-v2",
                                   "opendde", "opendde-abag", "esmfold2", "esmfold2-fast"])
def test_cyclic_chain_is_refused_for_every_model_that_cannot_honour_it(tmp_path, model):
    with pytest.raises(RuntimeError) as e:
        _validate_cyclic_unsupported(_yaml(tmp_path, CYCLIC), model)
    msg = str(e.value)
    assert model in msg
    assert "cyclic" in msg
    # The message has to name a model that DOES honor it, or the user is stuck.
    assert "rf3" in msg or "boltz2" in msg


def test_a_linear_chain_passes(tmp_path):
    _validate_cyclic_unsupported(_yaml(tmp_path, LINEAR), "openbind")


def test_cyclic_false_is_not_a_refusal(tmp_path):
    _validate_cyclic_unsupported(_yaml(tmp_path, CYCLIC_FALSE), "openbind")


def test_every_cyclic_chain_id_is_named(tmp_path):
    with pytest.raises(RuntimeError) as e:
        _validate_cyclic_unsupported(_yaml(tmp_path, CYCLIC_LIST_IDS), "openbind")
    msg = str(e.value)
    assert "A" in msg and "B" in msg
    # C is linear and must not be blamed.
    assert "C" not in msg.split("in q.yaml")[0]


def test_a_comma_separated_id_names_both_chains(tmp_path):
    """`id: A,B` is the other spelling the chain readers accept, and this guard used to blame
    a single chain called "A,B" -- it only expanded the list form. Both forms go through
    `yaml_input.chain_ids` now, so a user with a comma-separated id gets told which chains to
    fix."""
    with pytest.raises(RuntimeError) as e:
        _validate_cyclic_unsupported(_yaml(tmp_path, CYCLIC_COMMA_IDS), "openbind")
    msg = str(e.value)
    assert "A, B" in msg, f"expected both chains named separately, got {msg!r}"
    assert "A,B" not in msg


def test_a_non_yaml_input_is_skipped_not_crashed(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">A|protein\nQLEDSEVEAVAKG\n")
    _validate_cyclic_unsupported(fasta, "openbind")


def test_an_empty_or_odd_yaml_does_not_raise_by_accident(tmp_path):
    _validate_cyclic_unsupported(_yaml(tmp_path, ""), "openbind")
    _validate_cyclic_unsupported(_yaml(tmp_path, "sequences:\n  - notadict\n"), "openbind")


def test_the_committed_cyclic_example_is_refused():
    p = Path(__file__).resolve().parent.parent / "examples" / "cyclic_prot.yaml"
    if not p.exists():
        pytest.skip("examples/cyclic_prot.yaml not in this checkout")
    with pytest.raises(RuntimeError):
        _validate_cyclic_unsupported(p, "openbind")


def test_every_caller_of_a_reader_that_drops_the_flag_runs_the_guard():
    """A validator nothing calls is not a guard, and the missing esmfold2 call sat unnoticed
    for exactly that reason.

    The parametrized cases above prove the FUNCTION refuses; they say nothing about whether a
    fold path calls it. The invariant that covers all of them at once comes from why the guard
    exists: neither chain reader carries the flag -- `_read_bio_chains` returns (chain_id,
    sequence, msa_spec, mol_type) and `_read_protein_chains` returns (chain_id, sequence,
    msa_spec, modifications), both pinned by the two `_really_drops_the_flag` tests below. So
    ANY function that builds a spec from either one drops `cyclic: true` on the floor and has
    to refuse first. Discovered from the source rather than listed, so a new model whose spec
    builder uses one of those readers is covered the day it lands instead of the day someone
    remembers this file.

    Reachability is one hop, which is enough today: `_predict_protenix_one` gets the guard
    through `_protenix_inputs`, and the other four call it directly.
    """
    import ast
    import inspect

    from tt_bio import worker

    tree = ast.parse(inspect.getsource(worker))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def called(fn):
        out = set()
        for c in ast.walk(fn):
            if isinstance(c, ast.Call):
                f = c.func
                out.add(f.id if isinstance(f, ast.Name)
                        else f.attr if isinstance(f, ast.Attribute) else None)
        return out

    def reaches_guard(name, depth=1):
        direct = called(fns[name])
        if "_validate_cyclic_unsupported" in direct:
            return True
        return depth > 0 and any(reaches_guard(c, depth - 1)
                                 for c in direct if c in fns)

    readers = {"_read_bio_chains", "_read_protein_chains"}
    builders = [n for n, fn in fns.items() if called(fn) & readers]
    assert len(builders) >= 5, (
        f"expected every model spec builder to be found, got {sorted(builders)}; if a reader "
        f"was renamed this invariant needs rewriting")
    unguarded = [n for n in builders if not reaches_guard(n)]
    assert not unguarded, (
        f"these build a spec from a chain reader that drops `cyclic`, without refusing it "
        f"first: {sorted(unguarded)}. A cyclic input would fold linear and return status=ok.")


def test_the_vendored_tree_really_has_no_cyclic_support():
    """The reason this gate exists. If a future vendor bump restores `Chain.cyclic` and
    the `cyclic_mask` feature, this test fails and the gate should be reconsidered rather
    than left refusing something the tree now supports."""
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
        Chain,
    )
    assert "cyclic" not in Chain.model_fields, \
        "the vendored query format now has Chain.cyclic -- revisit _validate_cyclic_unsupported"


@pytest.mark.parametrize("model", ["protenix-v1", "protenix-v2", "opendde"])
def test_the_protenix_reader_really_drops_the_flag(model):
    """The reason the Protenix/OpenDDE arms of this gate exist. `_read_bio_chains` returns
    (chain_id, sequence, msa_spec, mol_type) and carries no cyclic field, so the flag cannot
    reach the featurizer. If that ever changes, this fails and the gate should be reconsidered
    rather than left refusing something the featurizer now supports."""
    import inspect

    from tt_bio.main import _read_bio_chains

    src = inspect.getsource(_read_bio_chains)
    assert "cyclic" not in src, \
        "_read_bio_chains now reads `cyclic` -- revisit _validate_cyclic_unsupported"


@pytest.mark.parametrize("model", ["esmfold2", "esmfold2-fast"])
def test_the_esmfold2_dispatch_path_really_refuses_it(tmp_path, model):
    """The parametrized test above proves the validator refuses the name; this proves the
    ESMFold2 path reaches the validator at all, through the same `predict_one` dispatch a real
    job takes. Card-free: the guard raises before any model or device work, so a bare instance
    with no loaded model is enough. Against the pre-fix worker this walked past the guard and
    died on the missing `msa_dir` config key instead."""
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    with pytest.raises(RuntimeError) as e:
        state.predict_one(_yaml(tmp_path, CYCLIC), {"model": model})
    msg = str(e.value)
    assert model in msg and "cyclic" in msg


def test_a_linear_esmfold2_job_still_gets_past_the_guard(tmp_path):
    """The control for the test above: a guard that refused everything would pass it. A linear
    chain must reach the work and fail on the missing config, not on cyclic."""
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    with pytest.raises(KeyError) as e:
        state.predict_one(_yaml(tmp_path, LINEAR), {"model": "esmfold2"})
    assert "msa_dir" in str(e.value)


def test_the_esmfold2_reader_really_drops_the_flag():
    """The reason the ESMFold2 arm of this gate exists. `_read_protein_chains` returns
    (chain_id, sequence, msa_spec, modifications) and carries no cyclic field, so the flag
    cannot reach the folder. If that ever changes, this fails and the gate should be
    reconsidered rather than left refusing something ESMFold2 now supports."""
    import inspect

    from tt_bio.main import _read_protein_chains

    src = inspect.getsource(_read_protein_chains)
    assert "cyclic" not in src, \
        "_read_protein_chains now reads `cyclic` -- revisit _validate_cyclic_unsupported"
