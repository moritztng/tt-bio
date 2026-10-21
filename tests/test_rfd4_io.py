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


def test_engine_stub_says_what_it_prepared():
    """Until the model lands the run must stop with the shape of the work, not a shrug."""
    import torch

    from tt_bio.rfd4 import get_engine

    engine = get_engine()
    features = {"padding_mask": torch.ones((1, 225), dtype=torch.bool)}
    with pytest.raises(NotImplementedError, match="225 tokens"):
        engine.design(features, num_designs=2, nsteps=60, seed=0)


def test_engine_names_are_not_guessed():
    from tt_bio.rfd4 import get_engine

    with pytest.raises(ValueError, match="unknown RFD4 engine"):
        get_engine("ttnn")
