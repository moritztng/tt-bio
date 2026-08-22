"""`cyclic: true` must be refused by OpenFold3, not folded as a linear chain.

Upstream's query format carries `Chain.cyclic` and its structure featurizer derives a
`cyclic_mask` from it. tt-bio's vendored copy has neither -- both were dropped when the
tree was vendored -- so a cyclic chain used to reach the model as an ordinary linear one
and the fold returned status=ok. Cyclisation changes the structure, so this is a hard
error, the same treatment `constraints:` gets and unlike `properties: affinity`, which
only omits an extra output. Card-free: yaml reading and a raise.
"""
from pathlib import Path

import pytest

from tt_bio.worker import _validate_openfold3_cyclic

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


def test_a_cyclic_chain_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as e:
        _validate_openfold3_cyclic(_yaml(tmp_path, CYCLIC))
    msg = str(e.value)
    assert "openfold3" in msg
    assert "cyclic" in msg
    # The message has to name a model that DOES honor it, or the user is stuck.
    assert "rf3" in msg or "boltz2" in msg


def test_a_linear_chain_passes(tmp_path):
    _validate_openfold3_cyclic(_yaml(tmp_path, LINEAR))


def test_cyclic_false_is_not_a_refusal(tmp_path):
    _validate_openfold3_cyclic(_yaml(tmp_path, CYCLIC_FALSE))


def test_every_cyclic_chain_id_is_named(tmp_path):
    with pytest.raises(RuntimeError) as e:
        _validate_openfold3_cyclic(_yaml(tmp_path, CYCLIC_LIST_IDS))
    msg = str(e.value)
    assert "A" in msg and "B" in msg
    # C is linear and must not be blamed.
    assert "C" not in msg.split("in q.yaml")[0]


def test_a_non_yaml_input_is_skipped_not_crashed(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">A|protein\nQLEDSEVEAVAKG\n")
    _validate_openfold3_cyclic(fasta)


def test_an_empty_or_odd_yaml_does_not_raise_by_accident(tmp_path):
    _validate_openfold3_cyclic(_yaml(tmp_path, ""))
    _validate_openfold3_cyclic(_yaml(tmp_path, "sequences:\n  - notadict\n"))


def test_the_committed_cyclic_example_is_refused():
    p = Path(__file__).resolve().parent.parent / "examples" / "cyclic_prot.yaml"
    if not p.exists():
        pytest.skip("examples/cyclic_prot.yaml not in this checkout")
    with pytest.raises(RuntimeError):
        _validate_openfold3_cyclic(p)


def test_the_vendored_tree_really_has_no_cyclic_support():
    """The reason this gate exists. If a future vendor bump restores `Chain.cyclic` and
    the `cyclic_mask` feature, this test fails and the gate should be reconsidered rather
    than left refusing something the tree now supports."""
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
        Chain,
    )
    assert "cyclic" not in Chain.model_fields, \
        "the vendored query format now has Chain.cyclic -- revisit _validate_openfold3_cyclic"
