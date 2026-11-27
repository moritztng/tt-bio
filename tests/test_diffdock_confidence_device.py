"""The confidence network with its convolutions on a card.

The device form is not a second implementation: :func:`tt_bio.diffdock.device.convs_on_device`
swaps each of the 45 :class:`~tt_bio.diffdock.conv.TensorProductConvLayer` modules for the same
:class:`~tt_bio.diffdock.device.DeviceConvModule` the score network runs, and the host glue
(graph build, encoders, pooled head) is untouched. Two card-free tests pin the wiring; the device
test grades the assembled head against the goldens ``dfd-confidence`` captured from upstream.

The acceptance criterion is the ORDER, because ranking is all the confidence model is for:
identical top-1 and identical full ranking on every captured pose set. The logit agreement is
the diagnostic beside it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.diffdock import config as dfd_config
from tt_bio.diffdock.confidence_model import ConfidenceModel, load_released
from tt_bio.diffdock.conv import TensorProductConvLayer

CKPT_DIR = Path.home() / "scratch/dfd/ckpt/extracted/confidence_model"
#: pc keeps the goldens in the campaign's state tree, the card hosts in scratch.
GOLDEN_DIRS = (Path.home() / ".coworker/state/dfd/goldens", Path.home() / "scratch/dfd/goldens")
GOLDEN_IDS = ("8G43_ZU6", "8AEU_M0L", "7A9E_R4W")


def _released():
    ckpt = CKPT_DIR / "best_model_epoch75.pt"
    if not ckpt.exists():
        pytest.skip(f"no confidence checkpoint at {ckpt}")
    cfg = dfd_config.load(CKPT_DIR, old_model=True, name="confidence")
    return load_released(torch.load(ckpt, map_location="cpu", weights_only=False), cfg)


def _golden_sets():
    for root in GOLDEN_DIRS:
        sets = [root / f"{c}__s0" for c in GOLDEN_IDS
                if (root / f"{c}__s0" / "confidence_inputs.npz").exists()]
        if sets:
            return sets
    pytest.skip(f"no confidence goldens under {GOLDEN_DIRS}")


def _order(a) -> list[int]:
    return np.argsort(-np.asarray(a, np.float64)[:, 0], kind="stable").tolist()


def score_golden(model, d: Path) -> np.ndarray:
    """``[N, 2]`` for every pose of one captured set, on upstream's own graph tensors."""
    inp = {k: torch.from_numpy(v) for k, v in np.load(d / "confidence_inputs.npz").items()}
    poses = np.load(d / "poses.npz")
    lig_pos = torch.from_numpy(poses["pos"] - poses["center"]).float()
    with torch.no_grad():
        return torch.cat([
            model(inp["lig_x"].float(), pose, inp["bond_index"].long(),
                  inp["bond_attr"].float(), inp["rec_x"].float(), inp["rec_pos"].float(),
                  inp["rec_edge_index"].long(), inp["atom_x"].float(),
                  inp["atom_pos"].float(), inp["atom_edge_index"].long(),
                  inp["atom_rec_edge_index"].long())
            for pose in lig_pos]).double().numpy()


# -- card-free: the swap reaches every convolution, and the product's seam reaches the swap --

class _Stub(torch.nn.Module):
    def __init__(self, device, layer, **_):
        super().__init__()
        self.layer = layer


def test_convs_on_device_swaps_all_45_confidence_convolutions(monkeypatch):
    from tt_bio.diffdock import device

    if not (CKPT_DIR / "model_parameters.yml").exists():
        pytest.skip("no confidence model_parameters.yml")
    model = ConfidenceModel(dfd_config.load(CKPT_DIR, old_model=True, name="confidence"))
    monkeypatch.setattr(device, "DeviceConvModule", _Stub)
    swapped = device.convs_on_device(model, "a-card")
    assert len(swapped) == len(model.conv_layers) == 45
    assert all(isinstance(m, _Stub) for m in model.conv_layers)


def test_device_score_network_puts_the_confidence_convolutions_on_the_card(monkeypatch):
    """`tt-bio dock --device tt` reaches the card confidence path through `DeviceScoreNetwork`."""
    from tt_bio.diffdock import device

    calls = []
    monkeypatch.setattr(device, "convs_on_device",
                        lambda model, dev, **k: calls.append((model, dev)) or {})
    host = type("Host", (), dict(model="score", config="cfg", confidence_model="conf"))()
    net = device.DeviceScoreNetwork(host, "a-card")
    assert calls == [("score", "a-card"), ("conf", "a-card")]
    assert net.confidence.__func__ is device.DeviceScoreNetwork.confidence

    calls.clear()
    host.confidence_model = None                       # --no_confidence
    device.DeviceScoreNetwork(host, "a-card")
    assert calls == [("score", "a-card")]


# -- on a card: the assembled head against upstream's goldens ------------------------------

@pytest.mark.device
def test_device_confidence_ranks_every_golden_pose_set_like_upstream():
    from tt_bio.diffdock.device import convs_on_device
    from tt_bio.main import ensure_p300_mesh_descriptor
    from tt_bio.tenstorrent import get_device

    sets = _golden_sets()
    model = _released()
    ensure_p300_mesh_descriptor()
    convs_on_device(model, get_device())
    for d in sets:
        got = score_golden(model, d)
        rebuild = np.load(d / "confidence.npz")["confidence_raw"]
        rollout = np.load(d / "poses.npz")["confidence_raw"]
        assert _order(got) == _order(rebuild), f"{d.name}: ranking differs from upstream"
        assert _order(got) == _order(rollout), f"{d.name}: ranking differs from the rollout"
        pcc = np.corrcoef(got.ravel(), rebuild.ravel())[0, 1]
        assert pcc > 0.999, f"{d.name}: PCC {pcc:.6f} against upstream"
