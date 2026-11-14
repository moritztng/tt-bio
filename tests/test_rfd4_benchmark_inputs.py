"""Upstream's shipped benchmark inputs, the port's parity fixtures.

`data/benchmarks/` is 389 annotated CIFs across ten families and every how-to starts from one of
them. They are upstream's data and are not vendored, so `perf/rfd4/benchmark_manifest.json`
(built by `perf/rfd4/benchmarks.py`) records what each one is and what featurizing it produces.
The manifest checks run anywhere; the inputs themselves are re-featurized where the library is on
the box (`RFD4_BENCHMARKS`, or the clone's default path).
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "perf" / "rfd4" / "benchmark_manifest.json").read_text())
LIBRARY = Path(os.environ.get("RFD4_BENCHMARKS",
                              Path.home() / "rfd4-ref-env/src/RFD4-Proteina/data/benchmarks"))
HAVE = LIBRARY.is_dir()

#: Every family the feature inventory names, as upstream's directory for it.
FAMILIES = {"monomer", "ppi", "mcsa", "motif_scaffolding", "small_molecule_binder", "dna_binder",
            "structure_prediction", "adherence", "sequence_bias"}


def test_the_manifest_covers_every_family_the_library_ships():
    assert FAMILIES <= set(MANIFEST["families"])
    assert len(MANIFEST["inputs"]) + len(MANIFEST["failed"]) == 389


def test_every_recorded_input_featurizes_to_a_real_conditioned_example():
    """A family whose inputs come out with no conditions set is a library the port reads but
    does not understand, which is invisible once a design runs."""
    for row in MANIFEST["inputs"]:
        assert row["tokens"] > 0, row["path"]
        assert row["conditions"], row["path"]
    tokens = [row["tokens"] for row in MANIFEST["inputs"]]
    # The smallest is a bare 3-residue motif (motif_data_aa/1qjg_aa), the largest a 1000+ token
    # complex: the library is meant to span the whole size range the model is run at.
    assert min(tokens) == 3 and max(tokens) > 500


def test_the_six_inputs_the_port_refuses_are_upstreams_own_refusals():
    """rfd4-io graded 391/391 against upstream's `prepare_cif` with 6 refusals identical. All six
    are PLINDER training examples whose residues lack the atom37 slot-1 reference, so upstream's
    own transform stack raises on them too."""
    assert len(MANIFEST["failed"]) == 6
    for row in MANIFEST["failed"]:
        assert row["path"].startswith("tip_atom_plinder_training_examples/")
        assert "Transforms failed at stage" in row["error"]


@pytest.mark.skipif(not HAVE, reason=f"the benchmark library is not at {LIBRARY}")
@pytest.mark.parametrize("row", MANIFEST["inputs"][::40], ids=lambda r: r["path"])
def test_a_sampled_input_still_featurizes_to_what_the_manifest_records(row):
    import hashlib

    from tt_bio.rfd4 import featurize

    path = LIBRARY / row["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "the library moved"
    features, _ = featurize(path, seed=0)
    assert int(features["padding_mask"].shape[-1]) == row["tokens"]
    present = {key for key, value in features.items()
               if key.startswith("condition_") and hasattr(value, "any") and bool(value.any())}
    collapsed = {"condition_sequence-probability-*" if k.startswith("condition_sequence-probability-")
                 else k for k in present}
    assert collapsed == set(row["conditions"])
