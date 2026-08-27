"""`cyclic: true` must be refused by every model that cannot honour it, not folded linear.

OF3/OpenBind: upstream's query format carries `Chain.cyclic` and its structure featurizer
derives a `cyclic_mask` from it; tt-bio's vendored copy has neither, both dropped when the
tree was vendored.

Protenix (v1/v2) and OpenDDE: there is no cyclic input path to drop. `_read_bio_chains` never
reads the flag, and upstream Protenix v0.5.0 has no cyclic chain flag either -- its only
"cyclic" is the `cyclic-pseudo-peptide` LIGAND entity label. Folding
examples/cyclic_prot.yaml with --model protenix-v1 SUCCEEDED and returned a linear 13-token
structure, which is how this was found during the v1 bring-up sweep.

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
                                   "opendde", "opendde-abag"])
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
