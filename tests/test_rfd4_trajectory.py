"""The trajectory a saved run writes: one model per snapshot, CA only, in a CIF a viewer reads."""
import numpy as np
import pytest
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure

from tt_bio.rfd4.sampler import BB_CA
from tt_bio.rfd4.trajectory import write_trajectory


def features(n_tokens=8, n_designs=2):
    """Enough of an example for the writer: which tokens are emitted, and their chains."""
    emit = torch.ones(1, n_tokens, dtype=torch.bool)
    emit[:, -2:] = False
    chain = torch.zeros(1, n_tokens, dtype=torch.long)
    chain[:, n_tokens // 2:] = 1
    return {"padding_mask": torch.ones(1, n_tokens, dtype=torch.bool),
            "flow_emit_mask": emit, "chain_iid": chain}, emit[0]


def snapshots(n_frames=3, n_designs=2, n_tokens=8):
    return [{"step": i * 5,
             "x_1": {BB_CA: torch.full((n_designs, n_tokens, 3), float(i))}}
            for i in range(n_frames)]


@pytest.fixture(autouse=True)
def emit_mask_from_the_fixture(monkeypatch):
    """`compute_flow_emit_masks` is upstream's and wants a full example; the writer only needs
    the backbone's mask, so the fixture supplies it directly."""
    monkeypatch.setattr("tt_bio.rfd4.trajectory._emitted",
                        lambda f: np.asarray(f["flow_emit_mask"][0], dtype=bool))


def test_a_trajectory_is_one_model_per_saved_step(tmp_path):
    feats, emit = features()
    path = write_trajectory(snapshots(3), feats, tmp_path / "t.cif")
    models = get_structure(CIFFile.read(str(path)))
    assert models.stack_depth() == 3
    assert models.array_length() == int(emit.sum())
    assert set(models.atom_name) == {"CA"}


def test_the_frames_are_the_snapshots_in_angstrom_for_the_requested_design(tmp_path):
    feats, _ = features()
    frames = snapshots(3)
    frames[1]["x_1"][BB_CA][1] = 4.2          # design 1 only
    path = write_trajectory(frames, feats, tmp_path / "t.cif", row=1)
    models = get_structure(CIFFile.read(str(path)))
    assert np.allclose(models.coord[0], 0.0)
    assert np.allclose(models.coord[1], 42.0), "nanometres are written as Angstrom"


def test_chains_survive_so_a_viewer_can_tell_binder_from_target(tmp_path):
    feats, _ = features()
    path = write_trajectory(snapshots(2), feats, tmp_path / "t.cif")
    models = get_structure(CIFFile.read(str(path)))
    assert set(models.chain_id) == {"A", "B"}
    assert list(models.res_id[models.chain_id == "A"]) == [1, 2, 3, 4]


def test_writing_nothing_is_an_error_not_an_empty_file(tmp_path):
    feats, _ = features()
    with pytest.raises(ValueError, match="save_trajectory_every"):
        write_trajectory([], feats, tmp_path / "t.cif")
