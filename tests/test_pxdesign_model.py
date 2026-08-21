"""PXDesign-d generator: the arithmetic PXDesign adds over what tt-bio already runs.

Device-free. The two new blocks (the 65-row conditioning embedding and `input_map`) and the
eta schedule are scored against upstream's own definitions; the end-to-end device run lives
in `scripts/pxdesign_port/design_e2e.py`.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from tt_bio.protenix import step_scale_schedule
from tt_bio.pxdesign.featurize import condition_template_index

REPO = Path(__file__).resolve().parent.parent
E2E = REPO / "scripts" / "pxdesign_port" / "design_e2e.py"
ART = REPO / "scripts" / "pxdesign_port" / "parity_artifacts" / "pdl1"
CKPT = Path("~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt").expanduser()


@pytest.fixture(scope="module")
def e2e():
    spec = importlib.util.spec_from_file_location("pxdesign_design_e2e", E2E)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def feats(e2e):
    if not (ART / "ref_design_inputs.pt").exists():
        pytest.skip("committed PD-L1 model inputs missing")
    return e2e.load_design_inputs()


# --- the eta schedule ---------------------------------------------------------------

def test_a_float_step_scale_is_the_constant_it_always_was():
    """Protenix-v2 and OpenDDE must be untouched: same float, every step."""
    assert step_scale_schedule(1.5, 200) == [1.5] * 200


def test_piecewise_65_switches_where_upstream_switches():
    """`eta = min if step_t / T < 0.65 else max`, T = len(noise_schedule) = n_step + 1."""
    n = 400
    got = step_scale_schedule({"type": "piecewise_65", "min": 1.0, "max": 2.5}, n)
    want = [1.0 if k / (n + 1) < 0.65 else 2.5 for k in range(n)]
    assert got == want
    assert got.count(1.0) == 261 and got.count(2.5) == 139


def test_the_shipped_default_is_the_one_a_pxdesign_run_actually_uses():
    """`configs_base.py` declares piecewise_65 but `cli.py common_run_options` defaults
    --eta_type/--eta_min/--eta_max to const 2.5 and ALIASES remaps them onto
    sample_diffusion.eta_schedule, so const 2.5 is what every `pxdesign infer` /
    `pxdesign pipeline` invocation runs. The config's declared schedule stays reachable."""
    from tt_bio.pxdesign.model import DESIGN_ETA_SCHEDULE, DESIGN_ETA_SCHEDULE_CONFIG
    assert DESIGN_ETA_SCHEDULE == {"type": "const", "min": 2.5, "max": 2.5}
    assert step_scale_schedule(DESIGN_ETA_SCHEDULE, 400) == [2.5] * 400
    assert DESIGN_ETA_SCHEDULE_CONFIG == {"type": "piecewise_65", "min": 1.0, "max": 2.5}
    assert step_scale_schedule(DESIGN_ETA_SCHEDULE_CONFIG, 400) != [2.5] * 400


@pytest.mark.parametrize("kind", ["linear", "poly", "cos", "piecewise", "piecewise_70"])
def test_every_upstream_schedule_kind_is_monotone_between_its_bounds(kind):
    got = step_scale_schedule({"type": kind, "min": 1.0, "max": 2.5}, 100)
    assert got[0] == 1.0 and max(got) <= 2.5 and min(got) >= 1.0
    assert all(b >= a for a, b in zip(got, got[1:]))


def test_an_unknown_schedule_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        step_scale_schedule({"type": "piecewise_99", "min": 1.0, "max": 2.5}, 10)
    with pytest.raises(ValueError):
        step_scale_schedule({"type": "const", "min": 1.0, "max": 2.5}, 10)


# --- the conditioning embedding ------------------------------------------------------

def test_condition_index_fits_the_65_row_embedding(feats):
    """64 bins + the reserved unconditioned row. An index of 65 would silently index out."""
    idx = condition_template_index(feats["conditional_templ"], feats["conditional_templ_mask"])
    assert idx.min() == 0 and idx.max() <= 64
    assert idx.shape == (196, 196)


def test_unconditioned_pairs_all_take_row_zero(feats):
    """Every pair outside the conditioned sub-block must read the SAME embedding row, or
    'no condition here' is not what the model is told."""
    idx = condition_template_index(feats["conditional_templ"], feats["conditional_templ_mask"])
    mask = feats["conditional_templ_mask"].bool()
    assert bool((idx[~mask] == 0).all())
    assert bool((idx[mask] > 0).all())


# --- the captured model inputs -------------------------------------------------------

def test_model_inputs_are_the_pdl1_anchor(feats):
    assert feats["ref_pos"].shape == (1250, 3)
    assert int(feats["atom_to_token_idx"].max()) + 1 == 196
    assert feats["restype"].shape == (196, 36)


def test_binder_tokens_are_the_placeholder_and_are_80_of_them(feats):
    """The 80-residue binder is written as `xpb`, column 32 of the 36-way vocabulary."""
    assert int((feats["restype"].argmax(-1) == 32).sum()) == 80


def test_compact_storage_round_trips_to_one_hots(feats):
    """The fixture stores the two big one-hots as uint8; they must come back as one-hots."""
    for k, width in (("ref_element", 128), ("ref_atom_name_chars", 64)):
        v = feats[k].reshape(-1, width)
        assert bool(((v == 0) | (v == 1)).all())
        assert bool((v.sum(-1) == 1).all())


# --- the class, without a card -------------------------------------------------------

def test_a_protenix_checkpoint_is_refused_as_a_design_checkpoint():
    """Handed protenix-v2, this must fail at load rather than build a generator with no
    conditioning embedding and fail with a shape error 16 DiT blocks later."""
    from tt_bio.pxdesign.model import ProtenixDesign
    v2_shaped = {"model": {"module.trunk.pairformer_stack.blocks.0.x": torch.zeros(1)}}
    with pytest.raises(ValueError, match="not a PXDesign generator checkpoint"):
        ProtenixDesign.design_state_dict(v2_shaped, "protenix-v2.pt")


def test_the_module_prefix_is_stripped():
    from tt_bio.pxdesign.model import ProtenixDesign
    sd = ProtenixDesign.design_state_dict({"model": {
        "module.design_condition_embedder.condition_template_embedder.embedder.weight":
            torch.zeros(65, 128)}})
    assert list(sd) == ["design_condition_embedder.condition_template_embedder.embedder.weight"]


@pytest.mark.skipif(not CKPT.exists(), reason="pxdesign generator checkpoint not fetched")
def test_the_checkpoint_carries_exactly_the_two_new_weights():
    """The port's whole claim: PXDesign-d adds one Embedding(65,128) and one Linear(430->449)
    over modules tt-bio already runs."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    assert sd["design_condition_embedder.condition_template_embedder.embedder.weight"].shape \
        == (65, 128)
    assert sd["design_condition_embedder.input_embedder.input_map.weight"].shape == (449, 430)
    assert 384 + 36 + 1 + 1 + 4 + 4 == 430
    assert not [k for k in sd if k.startswith(("trunk", "confidence_head", "pairformer"))]
