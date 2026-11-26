"""The DiffDock-L weights: the pin, the verification, and finding both checkpoints.

Upstream fetches `releases/latest/download/diffdock_models.zip`, which 404s today, and then
tries a second URL and carries on. The row here names the v1.1 asset and its sha256, and a
mismatch is an error, not a fallback. The network is mocked; nothing here downloads.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tt_bio import weights
from tt_bio.diffdock import checkpoints

ROW = weights.ARTIFACTS["diffdock"]
MEMBERS = ("score_model/best_ema_inference_epoch_model.pt",
           "score_model/model_parameters.yml",
           "confidence_model/best_model_epoch75.pt",
           "confidence_model/model_parameters.yml")


def test_download_manifest_and_sha(tmp_path, monkeypatch):
    """The pin is the v1.1 asset, and a wrong byte fails loudly instead of being loaded."""
    assert ROW.sources == (
        "https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip",), \
        "the pin must name v1.1 explicitly; releases/latest 404s"
    assert ROW.sha256 == "5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d"
    assert ROW.approx_bytes == 129825226
    assert ROW.derived is not None and ROW.derived.expect == MEMBERS

    archive = tmp_path / "diffdock_models.zip"
    _fake_release_zip(archive)
    served = archive.read_bytes()
    calls = []

    def fake_download(source, dest, **kw):
        calls.append(source)
        Path(dest).write_bytes(served)
        return True, False        # (a tool reported success, worth another try)

    monkeypatch.setattr(weights, "_download_to", fake_download)

    dest = tmp_path / "cache" / "diffdock" / "diffdock_models.zip"
    dest.parent.mkdir(parents=True)
    with pytest.raises(weights.DownloadFailed) as bad:
        weights.fetch_file(ROW.sources, dest, sha256=ROW.sha256, quiet=True)
    assert "sha256" in str(bad.value).lower()
    assert set(calls) == set(ROW.sources), "one pinned URL, no silent second source"

    # The same fetch with the sha of what is actually served keeps the file.
    good = hashlib.sha256(served).hexdigest()
    weights.fetch_file(ROW.sources, dest, sha256=good, quiet=True)
    assert dest.read_bytes() == served


def test_both_checkpoints_resolve(tmp_path):
    """Score and confidence, from the cached release, from a directory, from a file."""
    root = tmp_path / "models"
    _fake_extracted_tree(root)

    for kind, (subdir, name) in (("score", (checkpoints.SCORE_DIR, checkpoints.SCORE_CKPT)),
                                 ("confidence", (checkpoints.CONFIDENCE_DIR,
                                                 checkpoints.CONFIDENCE_CKPT))):
        ckpt, params = checkpoints.resolve(kind, root / subdir)
        assert ckpt == root / subdir / name
        assert params == root / subdir / checkpoints.PARAMETERS
        # Upstream's own layout, a directory, and a bare .pt file both resolve.
        assert checkpoints.resolve(kind, root / subdir / name)[0] == ckpt

    with pytest.raises(checkpoints.CheckpointMissing, match="no score checkpoint"):
        checkpoints.resolve("score", tmp_path / "empty")

    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / checkpoints.SCORE_CKPT).write_bytes(b"x")
    with pytest.raises(checkpoints.CheckpointMissing, match="model_parameters.yml"):
        checkpoints.resolve("score", lonely)


def test_a_host_without_the_weights_is_told_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "cached_path", lambda *a, **k: None)
    with pytest.raises(checkpoints.CheckpointMissing, match="tt-bio weights --download diffdock"):
        checkpoints.models_root(tmp_path)


def test_the_row_is_listed_for_the_model():
    assert ROW in weights.artifacts_for("diffdock")
    assert "diffdock" in weights.models_known()


def _fake_release_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for member in MEMBERS:
            zf.writestr(member, b"not the real weights")


def _fake_extracted_tree(root: Path) -> None:
    for subdir, ckpt in ((checkpoints.SCORE_DIR, checkpoints.SCORE_CKPT),
                         (checkpoints.CONFIDENCE_DIR, checkpoints.CONFIDENCE_CKPT)):
        (root / subdir).mkdir(parents=True)
        (root / subdir / ckpt).write_bytes(b"x")
        (root / subdir / checkpoints.PARAMETERS).write_text("all_atoms: false\n")
