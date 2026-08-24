"""The two guards `accuracy_cell.py` grew when the 997 aa cell was scored, both card-free.

Neither is checkable from the numbers the harness prints, which is why they are here:

* The crystal token map. `score_crystal` places deposited residue numbering on fixture
  tokens; an off-by-one there reads as a slightly worse RMSD and no superposition can
  catch it. On `7eip_997` the modelled residues happen to span the whole sequence, so a
  shift is caught by the range check alone -- the identity check is what covers the
  ordinary case of a crystal missing a terminal residue, so it is tested on exactly that.
* The reference backend. `--rescore` used to report its own unused `--ref-device` default
  as the backend the reference had run on, which is how a report claimed "cpu torch" for a
  reference generated on an H200. A cache with no manifest must read `unrecorded` rather
  than inherit a default, and a cache finished across two backends must say so.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "rf3_port"))

FIXTURE = REPO / "scripts/rf3_port/size_ladder/7eip_997"


@pytest.fixture(scope="module")
def ac():
    sys.argv = [sys.argv[0]]  # the module parses nothing at import, but argparse is armed
    return pytest.importorskip("accuracy_cell", reason="needs torch and the tt_bio tree")


def _fixture_copy(ac, monkeypatch, tmp_path, mutate):
    """A copy of the 7EIP fixture with a deliberately wrong token map."""
    d = tmp_path / "shifted"
    d.mkdir()
    shutil.copy(FIXTURE / "input.json", d)
    g = json.loads((FIXTURE / "ground_truth_ca.json").read_text())
    mutate(g)
    (d / "ground_truth_ca.json").write_text(json.dumps(g))
    monkeypatch.setattr(ac, "fixture_dir", lambda name: d)
    return d


def test_crystal_map_is_read_and_asserted(ac):
    tok, xyz = ac.crystal_ca("7eip_997")
    assert len(tok) == len(xyz) == 966
    assert tok.min() == 0 and tok.max() == 996


def test_crystal_absent_is_not_an_error(ac):
    assert ac.crystal_ca("ubq_76") is None


def test_token_map_off_the_end_is_refused(ac, tmp_path, monkeypatch):
    _fixture_copy(ac, monkeypatch, tmp_path, lambda g: g.update(token0_auth_seq_id=26))
    with pytest.raises(SystemExit, match="runs off the 997-residue"):
        ac.crystal_ca("shifted")


def test_one_residue_shift_is_refused_by_identity(ac, tmp_path, monkeypatch):
    def mutate(g):
        # Drop the C-terminal CA so the token range no longer pins the offset by itself,
        # which is the ordinary case rather than a contrived one.
        g["ca"] = g["ca"][:-1]
        g["token0_auth_seq_id"] = 24

    _fixture_copy(ac, monkeypatch, tmp_path, mutate)
    with pytest.raises(SystemExit, match="disagrees with the fixture sequence"):
        ac.crystal_ca("shifted")


def test_committed_997_reference_is_recorded_as_the_h200_it_ran_on(ac):
    r = {}
    d = REPO / "perf/rf3/results/accuracy_7eip_997"
    ac.adopt_ref_provenance(r, d, d)
    assert r["ref_device"] == "cuda"
    assert r["ref_provenance"]["device_names"] == ["NVIDIA H200"]
    assert "cuda torch" in r["reference"]


def test_a_cache_with_no_manifest_reads_unrecorded(ac, tmp_path):
    r = {}
    ac.adopt_ref_provenance(r, tmp_path, tmp_path / "nope")
    assert r["ref_device"] == "unrecorded"
    assert "not recorded" in r["reference"]


def test_a_reference_split_across_backends_says_so(ac, tmp_path):
    ac.ref_manifest_record(tmp_path, {})
    assert not (tmp_path / ac.REF_MANIFEST).exists(), "nothing computed, nothing to record"
    ac.ref_manifest_record(tmp_path, {0: {"ref_device": "cuda"}})
    ac.ref_manifest_record(tmp_path, {1: {"ref_device": "cpu"}})
    m = json.loads((tmp_path / ac.REF_MANIFEST).read_text())
    assert sorted(m["per_seed"]) == ["0", "1"]
    r = {}
    ac.adopt_ref_provenance(r, tmp_path, tmp_path)
    assert r["ref_device"] == "mixed: cpu,cuda"
