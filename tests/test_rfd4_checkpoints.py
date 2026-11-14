"""RFD4 always loads two checkpoints together, out of one pinned bundle.

The denoiser and the autoencoder it was trained against are not interchangeable across releases
(upstream `docs/reference/checkpoints.md`), so the port never resolves them separately: the
registry pins one HF revision, `load_checkpoint` reads one export out of it, and the engine
splits that state into the `nn.` and `autoencoder.` halves.

Everything here is host-side. The bundle-shaped tests skip where the snapshot is not on the box.
"""
import json
from pathlib import Path

import pytest
import torch

from tt_bio import weights as W
from tt_bio.rfd4.network import _sub, load_checkpoint

RFD4 = W.ARTIFACTS["rfd4"]


def test_the_registry_pins_one_revision_and_takes_only_the_two_files_the_engine_reads():
    assert RFD4.repo == "nvidia/RFD4-Proteina"
    assert RFD4.revision == "48377e45eb25a4fce8d2aff56c1a659b726d2106"
    # The repo also ships AF2, OpenFold3 and container images (~100 GB). The allow list is what
    # keeps `tt-bio weights` from pulling them.
    assert set(RFD4.allow) == {"config.json", "model.safetensors"}


def _bundle(tmp_path, *, with_export=True):
    """A bundle shaped like the release: config.json plus one or both weight files."""
    path = tmp_path / "bundle"
    (path / "flow_checkpoints").mkdir(parents=True)
    config = {"config": {"nn": {"nlayers": 2, "sequence_readout": {"num_classes": 147}},
                         "autoencoder_config": {"nn_ae": {"latent_z_dim": 32, "n_residue_types": 147}}}}
    (path / "config.json").write_text(json.dumps(config))
    state = {"nn.block.weight": torch.ones(2, 2), "autoencoder.decoder.weight": torch.full((2, 2), 2.0)}
    torch.save({"state_dict": {k: v * 10 for k, v in state.items()}},
               path / "flow_checkpoints" / "rfd4_proteina_flow-ema.ckpt")
    if with_export:
        from safetensors.torch import save_file

        save_file(state, str(path / "model.safetensors"))
    return path


def test_the_export_is_preferred_over_the_training_checkpoint(tmp_path):
    """Both files carry the same weights; NVIDIA's export has the LoRA folded in already, and
    merging it ourselves lands 2.5e-7 away in the model's output (network.load_checkpoint)."""
    state, config = load_checkpoint(_bundle(tmp_path))
    assert state["nn.block.weight"][0, 0] == 1.0          # the export, not the 10x ckpt
    assert config["nn"]["sequence_readout"]["num_classes"] == 147


def test_without_the_export_the_training_checkpoint_is_read(tmp_path):
    state, _ = load_checkpoint(_bundle(tmp_path, with_export=False))
    assert state["nn.block.weight"][0, 0] == 10.0


def test_one_bundle_carries_both_checkpoints(tmp_path):
    """The engine splits one state into the denoiser and the autoencoder; neither half may be
    empty, because a bundle missing one of them would otherwise load and design noise."""
    state, _ = load_checkpoint(_bundle(tmp_path))
    assert _sub(state, "nn.") and _sub(state, "autoencoder.")


def _released_bundle():
    path = W.resolve("rfd4")
    return path if path is not None and (path / "config.json").is_file() else None


@pytest.mark.skipif(_released_bundle() is None, reason="the RFD4 bundle is not on this box")
def test_the_released_config_is_the_pair_the_engine_assumes():
    config = json.loads((_released_bundle() / "config.json").read_text())["config"]
    ae = config["autoencoder_config"]["nn_ae"]
    assert config["nn"]["nlayers"] == 14
    assert config["nn"]["sequence_readout"]["num_classes"] == ae["n_residue_types"] == 147
    assert ae["latent_z_dim"] == 32
