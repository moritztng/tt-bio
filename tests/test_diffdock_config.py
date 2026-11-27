"""The DiffDock-L irrep algebra, graded against the released checkpoint's own shapes.

tt-bio does not depend on e3nn (see tt_bio/diffdock/irreps.py), so the first thing that
has to hold is that our irrep bookkeeping agrees with e3nn's where it matters: the number
of per-edge weights each tensor product consumes. That number is observable without e3nn,
because the FC block that produces the weights ends in a Linear whose output width *is*
``tp.weight_numel``. Nineteen tensor products in the score checkpoint, nineteen row counts
to match, no tolerance.

CPU only; no Tenstorrent device is opened here.
"""

import os
from pathlib import Path

import pytest

from tt_bio.diffdock import config as cfgmod
from tt_bio.diffdock.irreps import (
    Irreps,
    irrep_seq,
    irrep_to_size,
    spherical_harmonics_irreps,
    weight_numel,
)


def test_irreps_parse_and_layout():
    ir = Irreps.parse("48x0e + 10x1o + 10x1e + 10x0o")
    assert ir.dim == 48 + 30 + 30 + 10 == irrep_to_size("48x0e + 10x1o + 10x1e + 10x0o")
    assert [s.start for s in ir.slices()] == [0, 48, 78, 108]
    assert ir.mul_of(1, -1) == 10 and ir.mul_of(2, 1) == 0
    assert str(ir) == "48x0e + 10x1o + 10x1e + 10x0o"


def test_spherical_harmonics_parity():
    # o3.Irreps.spherical_harmonics(2) is 1x0e + 1x1o + 1x2e: parity (-1)**l.
    assert str(spherical_harmonics_irreps(2)) == "1x0e + 1x1o + 1x2e"


def test_irrep_seq_matches_upstream_strings():
    assert irrep_seq(48, 10, False, True) == [
        "48x0e",
        "48x0e + 10x1o",
        "48x0e + 10x1o + 10x1e",
        "48x0e + 10x1o + 10x1e + 10x0o",
    ]
    # reduce_pseudoscalars off keeps the pseudoscalar track at ns, not nv.
    assert irrep_seq(48, 10, False, False)[-1].endswith("48x0o")


def test_selection_rule():
    # 0e x 1o can only make 1o; 1o x 1o makes 0e, 1e and 2e.
    assert weight_numel("1x0e", "1x1o", "1x1o") == 1
    assert weight_numel("1x0e", "1x1o", "1x0e") == 0
    # odd x odd is even, so 1o x 1o lands on 0e/1e/2e and nothing odd.
    for out, n in (("1x0e", 1), ("1x1e", 1), ("1x2e", 1), ("1x3e", 0), ("1x1o", 0)):
        assert weight_numel("1x1o", "1x1o", out) == n


def _checkpoint_dir():
    env = os.environ.get("DIFFDOCK_MODELS")
    for p in (Path(env) if env else None,
              Path.home() / "scratch/dfd/ckpt/extracted"):
        if p is not None and (p / "score_model/model_parameters.yml").exists():
            return p
    try:
        from tt_bio import weights
        p = weights.cache_root() / "diffdock/models"
        if (p / "score_model/model_parameters.yml").exists():
            return p
    except Exception:
        pass
    return None


needs_ckpt = pytest.mark.skipif(_checkpoint_dir() is None,
                                reason="DiffDock-L v1.1 checkpoint not downloaded")


@needs_ckpt
def test_score_config_is_the_released_model():
    cfg = cfgmod.load(_checkpoint_dir() / "score_model", old_model=False)
    assert cfg.name == "diffdockL"
    assert (cfg.ns, cfg.nv, cfg.num_conv_layers, cfg.num_prot_emb_layers) == (48, 10, 11, 3)
    assert cfg.all_atoms is False          # the score network is CGModel, not AAModel
    assert cfg.differentiate_convolutions is True
    # sh_lmax is 2, so FasterTensorProduct (which asserts 1x0e+1x1o) is unreachable:
    # every tensor product in the shipped model is the fully connected one.
    assert cfg.faster_tensor_product is False
    assert [cfg.conv_edge_groups(i) for i in range(11)] == [4] * 10 + [2]


@needs_ckpt
def test_confidence_config_is_the_released_model():
    cfg = cfgmod.load(_checkpoint_dir() / "confidence_model", old_model=True)
    assert (cfg.ns, cfg.nv, cfg.num_conv_layers) == (24, 6, 5)
    assert cfg.all_atoms is True and cfg.old_model is True
    assert cfg.parallel_aggregators == ("mean", "max", "min", "std")


@needs_ckpt
def test_weight_numel_matches_every_fc_block_in_the_checkpoint():
    torch = pytest.importorskip("torch")
    ckpt = _checkpoint_dir() / "score_model/best_ema_inference_epoch_model.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = cfgmod.load(_checkpoint_dir() / "score_model", old_model=False)

    plan = cfg.layer_plan()
    assert len(plan) == 19                   # 3 rec + 3 lig + 11 conv + final + torsion
    assert sum(v.numel() for v in sd.values()) == cfgmod.SCORE_PARAM_COUNT

    last = 3 * (cfg.tp_weights_layers - 1)   # FCBlock: Linear, act, drop, ..., Linear
    for entry in plan:
        for g in range(entry["edge_groups"]):
            stem = f"{entry['name']}.fc"
            key = (f"{stem}.{g}.{last}.weight" if entry["edge_groups"] > 1
                   else f"{stem}.{last}.weight")
            rows, cols = sd[key].shape
            assert rows == entry["weight_numel"], (
                f"{key}: checkpoint wants {rows} per-edge weights, "
                f"the irrep algebra says {entry['weight_numel']}")
            assert cols == entry["hidden_features"]


@needs_ckpt
def test_load_refuses_a_checkpoint_that_is_not_the_pin(tmp_path):
    import yaml

    src = _checkpoint_dir() / "score_model/model_parameters.yml"
    raw = yaml.safe_load(src.read_text())
    raw["ns"] = 16                            # the constructor default, not the release
    (tmp_path / "model_parameters.yml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="expected 48, got 16"):
        cfgmod.load(tmp_path, old_model=False)
