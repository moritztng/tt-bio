"""The RFD4 host path: spec loading, the CLI surface, and the engine boundary.

The exactness bars live with the parity harness in scripts/rfd4_port/, because they need
the upstream clone and a reference env: 391/391 inputs exact against upstream's
`prepare_cif`, 385/385 against RFD4's own feature goldens, and upstream's 39 condition-spec
and contig tests run unchanged against the vendored modules. What is checked here is what
can be checked from a clean tt-bio install with no device and no network.
"""
import textwrap

import pytest
from click.testing import CliRunner

from tt_bio.main import DESIGN_MODELS, cli


def test_rfd4_is_a_design_model():
    assert "rfd4" in DESIGN_MODELS


def test_design_help_describes_the_real_rfd4_flags():
    """--help has to answer "what can I pass" before the model lands, not after."""
    out = CliRunner().invoke(cli, ["design", "--help"]).output
    for flag in ("--nsteps", "--annotated_cif", "--no_validate", "--spec", "--seed"):
        assert flag in out
    assert "rfd4" in out


def test_legacy_dialect_is_refused_not_half_read(tmp_path):
    """RFD4's old contig-JSON spec is a different language; reading it halfway is worse
    than refusing it."""
    from tt_bio.rfd4 import load_spec

    spec = tmp_path / "legacy.yaml"
    spec.write_text(textwrap.dedent("""
        design-1:
          contig: "A1-10,20"
          length: "70"
          select_hotspots: "A52"
    """))
    with pytest.raises(NotImplementedError, match="legacy contig-JSON"):
        load_spec(spec)


def test_global_args_fold_into_every_example_and_the_example_wins(tmp_path):
    from tt_bio.rfd4 import load_spec

    spec = tmp_path / "spec.yaml"
    spec.write_text(textwrap.dedent("""
        global_args:
          input: shared.cif
          contig: "A1-10"
        a: {}
        b:
          contig: "A1-20"
    """))
    examples = load_spec(spec)
    assert set(examples) == {"a", "b"}
    assert examples["a"] == {"input": "shared.cif", "contig": "A1-10"}
    assert examples["b"]["input"] == "shared.cif"
    assert examples["b"]["contig"] == "A1-20"


def test_unknown_design_is_named(tmp_path):
    from tt_bio.rfd4 import prepare_examples

    spec = tmp_path / "spec.yaml"
    spec.write_text("a:\n  input: x.cif\n")
    with pytest.raises(ValueError, match=r"\['nope'\]"):
        prepare_examples(spec, tmp_path / "out", only=["nope"])


def test_preprocess_adds_the_two_derived_keys():
    """The network reads coords in nm and the atom37 mask under its own name
    (`Proteina.preprocess_batch_for_generation`); featurize gives Angstroms."""
    import torch

    from tt_bio.rfd4.engine import preprocess

    features = {"coords": torch.full((1, 4, 37, 3), 10.0),
                "atom37_mask": torch.ones((1, 4, 37), dtype=torch.bool)}
    batch = preprocess(features)
    assert torch.equal(batch["coords_nm"], torch.ones((1, 4, 37, 3)))
    assert batch["coord_mask"] is features["atom37_mask"]
    assert "coords" in batch, "preprocess adds keys, it does not consume the features"


def test_engine_names_are_not_guessed():
    from tt_bio.rfd4 import get_engine

    with pytest.raises(ValueError, match="unknown RFD4 engine"):
        get_engine("ttnn")


def test_every_design_of_a_batch_writes(tmp_path, monkeypatch):
    """The features are one input shared by every design (a leading axis of 1), and
    write_cif indexes them by row like the output; row 1 of a 2-design batch must write."""
    import torch

    from tt_bio._vendor.rfd4_prep import processing
    from tt_bio.rfd4 import write_design
    from tt_bio.rfd4.engine import DesignOutput

    seen = []
    monkeypatch.setattr(processing, "write_cif",
                        lambda out, f, aa, path, row=0: seen.append(f["padding_mask"][row].sum()) or str(path))
    output = DesignOutput(coords_nm=torch.zeros(2, 4, 37, 3), residue_type=torch.zeros(2, 4),
                          atom_mask=torch.ones(2, 4, 37), insertion_logits=torch.zeros(2, 4, 1))
    features = {"padding_mask": torch.ones(1, 4, dtype=torch.bool)}
    for row in range(2):
        write_design(output, features, None, tmp_path / f"d{row}.cif", row=row)
    assert [int(x) for x in seen] == [4, 4]
