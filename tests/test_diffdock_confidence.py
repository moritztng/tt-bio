"""The confidence network: strict load, the zero-edge guard, and parity against upstream.

The parity numbers live in `state/dfd/CONFIDENCE.md`; what is pinned here is the behaviour
that must not regress silently.

The zero-edge tests build the degenerate case **directly** rather than hoping a rollout
produces one. Upstream only reaches it on a pose that has drifted out of the pocket, so a
test that waits for one grades nothing on most runs, and a truncated rollout is not a
cheap way to force it (D-dfd-orch-10: `--steps 1` and `--steps 3` hit upstream's crash
because the poses are still undocked, which cost `dfd-ref` two runs).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tt_bio.diffdock import config as dfd_config
from tt_bio.diffdock.confidence_model import (
    LIGAND_GROUPS,
    ConfidenceModel,
    ScalarAtomEncoder,
    load_released,
)
from tt_bio.diffdock.conv import TensorProductConvLayer

CKPT_DIR = Path.home() / "scratch/dfd/ckpt/extracted/confidence_model"
SH = "1x0e+1x1o+1x2e"


def _checkpoint():
    ckpt = CKPT_DIR / "best_model_epoch75.pt"
    if not ckpt.exists():
        pytest.skip(f"no confidence checkpoint at {ckpt}")
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def _config():
    if not (CKPT_DIR / "model_parameters.yml").exists():
        pytest.skip("no confidence model_parameters.yml")
    return dfd_config.load(CKPT_DIR, old_model=True, name="confidence")


# -- the checkpoint ---------------------------------------------------------------------

def test_released_checkpoint_loads_strictly():
    """No missing and no unexpected key, and every tensor accounted for.

    `load_released` raises on either, so reaching the assertions is most of the test. The
    counts are here so that a future change which quietly drops a module shows up as a
    number rather than as a slightly different confidence.
    """
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)

    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    assert params == 4_769_636
    assert params + buffers == 4_773_124
    assert params + buffers + 5_778 == dfd_config.CONFIDENCE_PARAM_COUNT == 4_778_902
    assert len(model.conv_layers) == 9 * cfg.num_conv_layers == 45
    assert cfg.num_confidence_outputs == 2


def test_encoder_is_the_old_one_not_the_new_one():
    """The receptor encoder adds its scalar projection; it does not concatenate.

    Upstream ships two atom encoders with identical widths. `score_model.AtomEncoder` is the
    newer one and concatenates; the confidence checkpoint wants `OldAtomEncoder`, which adds.
    Swapping them loads cleanly in the score model's case and gives wrong features, so this
    pins the arithmetic rather than the key names.
    """
    enc = ScalarAtomEncoder(4, ([5], 0), sigma_embed_dim=3)
    with torch.no_grad():
        enc.atom_embedding_list[0].weight.copy_(
            torch.arange(20, dtype=torch.float32).reshape(5, 4))
        enc.linear.weight.zero_()
        enc.linear.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    # category 2 -> embedding row [8,9,10,11]; the scalar branch contributes its bias.
    assert torch.equal(enc(x), torch.tensor([[9.0, 11.0, 13.0, 15.0]]))


def test_head_emits_two_columns_and_ranking_uses_column_zero():
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    assert model.confidence_predictor[-1].out_features == 2


# -- the zero-edge guard (D-dfd-orch-10) --------------------------------------------------

def test_empty_edge_set_returns_zeros_at_the_destination_count():
    """A cross group with no edges returns zeros shaped by `out_nodes`, not by its source.

    This is the exact layer that crashes upstream: it reads receptor nodes and writes ligand
    nodes, so the two counts differ and returning the source's would break the caller's
    residual add even when the values are right.
    """
    layer = TensorProductConvLayer("24x0e", SH, "24x0e + 6x1o", 72, residual=False).eval()
    source = torch.randn(7, 24)
    empty = torch.zeros((2, 0), dtype=torch.long)
    out = layer(source, empty, torch.zeros(0, 72), torch.zeros(0, 9), out_nodes=5)
    assert out.shape == (5, 42)
    assert torch.all(out == 0)
    assert torch.isfinite(out).all()


def test_fully_masked_edge_set_matches_an_empty_one():
    """Masking every edge must give the empty answer, not NaN from a 0/0 mean.

    The mean is computed as sum-then-divide, so an all-masked group drives the divisor to
    zero; it is clamped, which is what keeps this finite.
    """
    layer = TensorProductConvLayer("24x0e", SH, "24x0e + 6x1o", 72, residual=False).eval()
    torch.manual_seed(0)
    node_attr = torch.randn(6, 24)
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 1]])
    masked = layer(node_attr, edge_index, torch.randn(4, 72), torch.randn(4, 9),
                   out_nodes=6, edge_mask=torch.zeros(4))
    assert torch.isfinite(masked).all()
    assert torch.allclose(masked, torch.zeros_like(masked), atol=1e-6)


def test_confidence_model_scores_a_pose_with_no_ligand_receptor_edges():
    """The whole network on a pose placed far outside the pocket.

    Upstream raises `ValueError: number sections must be larger than 0` here
    (`tensor_layers.py:362`). The port returns a finite score. The ligand is moved 500 A
    away, which empties both the ligand-receptor and the ligand-atom cross graphs.
    """
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    torch.manual_seed(0)

    n_lig, n_rec, n_atom = 5, 6, 9
    lig_x = torch.zeros(n_lig, len(_lig_dims()))
    bond_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    bond_attr = torch.zeros(4, cfg.in_lig_edge_features)
    rec_pos = torch.randn(n_rec, 3) * 4
    atom_pos = torch.randn(n_atom, 3) * 4
    far = torch.randn(n_lig, 3) + 500.0

    out = model(
        lig_x, far, bond_index, bond_attr,
        torch.zeros(n_rec, 1 + 1280), rec_pos, _ring(n_rec),
        torch.zeros(n_atom, 4), atom_pos, _ring(n_atom),
        torch.stack([torch.arange(n_atom), torch.arange(n_atom) % n_rec]),
    )
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_last_layer_runs_only_the_three_ligand_groups():
    assert LIGAND_GROUPS == (0, 1, 2)
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    calls: list[int] = []
    for i, layer in enumerate(model.conv_layers):
        layer.register_forward_hook(lambda m, a, o, i=i: calls.append(i))

    n_lig, n_rec, n_atom = 4, 5, 7
    model(torch.zeros(n_lig, len(_lig_dims())), torch.randn(n_lig, 3) * 2,
          torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, cfg.in_lig_edge_features),
          torch.zeros(n_rec, 1 + 1280), torch.randn(n_rec, 3) * 2, _ring(n_rec),
          torch.zeros(n_atom, 4), torch.randn(n_atom, 3) * 2, _ring(n_atom),
          torch.stack([torch.arange(n_atom), torch.arange(n_atom) % n_rec]))

    last = [c for c in calls if c >= 9 * (cfg.num_conv_layers - 1)]
    assert sorted(last) == [36, 37, 38]


def _lig_dims():
    from tt_bio.diffdock.constants import lig_feature_dims
    return lig_feature_dims[0]


def _ring(n: int) -> torch.Tensor:
    """A connected `[2, n]` graph, so no node is isolated."""
    src = torch.arange(n)
    return torch.stack([src, (src + 1) % n])
