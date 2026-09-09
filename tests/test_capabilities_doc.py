"""docs/model-capabilities.md is generated from the capability table, so it cannot drift.

The published matrix is what a user reads before choosing a model. Hand-maintained, it goes
stale the first time a port gains a feature, and a doc that promises support the code refuses
is worse than no doc. So the doc carries a generated block and this test regenerates it.

Run `python3 -m tt_bio.capabilities` and paste the output between the two markers to fix a
failure here.
"""
from pathlib import Path

import pytest

from tt_bio.capabilities import CAPABILITY, DOC_COLUMNS, FEATURES, markdown_table
from tt_bio.main import PREDICT_MODELS

DOC = Path(__file__).resolve().parents[1] / "docs" / "model-capabilities.md"
BEGIN = "<!-- BEGIN CAPABILITY TABLE (generated: python3 -m tt_bio.capabilities) -->"
END = "<!-- END CAPABILITY TABLE -->"


def _published() -> str:
    text = DOC.read_text()
    assert BEGIN in text and END in text, f"{DOC.name} lost its generated-table markers"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def test_the_published_matrix_matches_the_table():
    assert _published() == markdown_table(), (
        "docs/model-capabilities.md disagrees with CAPABILITY. Regenerate it:\n"
        "  python3 -m tt_bio.capabilities")


def test_a_table_change_would_fail_this(monkeypatch):
    """Negative control: the check compares content, it does not just find the markers."""
    from tt_bio import capabilities

    flipped = {m: dict(caps) for m, caps in CAPABILITY.items()}
    flipped["boltz2"]["ligand"] = capabilities.REFUSED
    monkeypatch.setattr(capabilities, "CAPABILITY", flipped)
    assert _published() != markdown_table()


def test_every_column_and_model_is_published():
    assert {f for f, _h in DOC_COLUMNS} == set(FEATURES)
    published = _published()
    for model in PREDICT_MODELS:
        assert f"`{model}`" in published


def test_the_doc_is_linked_from_the_readme():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "docs/model-capabilities.md" in readme, \
        "a capability doc nobody links to is a doc nobody reads"


@pytest.mark.parametrize("model", sorted(CAPABILITY))
def test_the_doc_never_promises_what_the_code_refuses(model):
    """The specific failure this guards: a row that reads `yes` for something check_input
    raises on."""
    row = [ln for ln in _published().splitlines() if ln.startswith(f"| `{model}` |")]
    assert len(row) == 1, f"{model} has {len(row)} rows in the published matrix"
    cells = [c.strip() for c in row[0].strip("|").split("|")][1:]
    for (feature, _h), cell in zip(DOC_COLUMNS, cells):
        if CAPABILITY[model][feature] != "honoured":
            assert cell != "yes", f"{model}/{feature} is published as supported"
