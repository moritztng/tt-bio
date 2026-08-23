"""Card-free, checkpoint-free gate on the AF2 `.npz` -> tt-bio weight remap.

`params_model_1_ptm.npz` is 373 MB and not committed, so these tests rebuild a zero-filled
source from the committed key/shape manifest under `scripts/af2_port/parity_artifacts/` and run
the real remap over it. That gates coverage and every output shape with no download and no
device.

Shapes cannot see the one substantive transformation in the remap: AF2's *incoming* triangle
multiplication has left and right swapped relative to tt-bio's, and both orderings produce
identically shaped weights. `test_incoming_swaps_left_and_right` asserts it numerically, and
`test_block_slices_are_not_aliased` asserts each of the 48 blocks reads its own slice of the
stacked array rather than block 0 forty-eight times.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import pytest
import torch

from tt_bio.af2_weights import (
    NUM_EVOFORMER_BLOCKS,
    UNUSED_SCOPES,
    load_af2_state_dict,
    remap_af2_params,
)

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "scripts" / "af2_port" / "parity_artifacts"
CHECKPOINT_SHAPES = ARTIFACTS / "params_model_1_ptm_shapes.json"
REMAPPED_SHAPES = ARTIFACTS / "params_model_1_ptm_remapped_shapes.json"

EVO_BLOCK = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"


def _shapes(path: Path) -> dict[str, list[int]]:
    return json.loads(path.read_text())


def _zero_source(fill: float = 0.0) -> dict[str, np.ndarray]:
    return {
        key: np.full(shape, fill, dtype=np.float32)
        for key, shape in _shapes(CHECKPOINT_SHAPES).items()
    }


def test_remap_matches_committed_shape_manifest():
    remapped = {k: list(v.shape) for k, v in remap_af2_params(_zero_source()).items()}
    assert remapped == _shapes(REMAPPED_SHAPES)


def test_parameter_count_balances():
    """Every checkpoint parameter is either remapped or in a deliberately unused head."""
    checkpoint = _shapes(CHECKPOINT_SHAPES)
    total = sum(int(np.prod(s)) for s in checkpoint.values())
    remapped = sum(int(np.prod(s)) for s in _shapes(REMAPPED_SHAPES).values())
    unused = sum(
        int(np.prod(s)) for k, s in checkpoint.items() if k.startswith(UNUSED_SCOPES)
    )
    assert remapped + unused == total


def test_unconsumed_arrays_are_only_the_unused_heads():
    consumed = {k for k in _shapes(CHECKPOINT_SHAPES) if not k.startswith(UNUSED_SCOPES)}
    # The remap raises if anything outside UNUSED_SCOPES is left over, so reaching here with a
    # non-empty consumed set is the assertion. Guard the count so a manifest that lost the heads
    # cannot make this vacuous.
    remap_af2_params(_zero_source())
    assert len(_shapes(CHECKPOINT_SHAPES)) - len(consumed) == 6


def test_an_unconsumed_array_fails_loudly():
    """The coverage assertion has to be able to fire, or it is not a check."""
    source = _zero_source()
    source["alphafold/alphafold_iteration/evoformer/a_block_nobody_reads//weights"] = np.zeros(
        (4, 4), dtype=np.float32
    )
    with pytest.raises(AssertionError, match="not consumed by the remap"):
        remap_af2_params(source)


def test_incoming_swaps_left_and_right():
    """AF2's incoming trimul is `kjc,kic->ijc`; tt-bio's is `bkid,bkjd->bijd`.

    So tt-bio's first `p_in` half must be AF2's *right* projection for the incoming block and
    AF2's *left* for the outgoing one. Both orderings are shape-identical, and getting it wrong
    transposes every incoming pair update.
    """
    source = _zero_source()
    marks = {"left_projection": 1.0, "right_projection": 2.0, "left_gate": 3.0, "right_gate": 4.0}
    for direction in ("outgoing", "incoming"):
        for name, value in marks.items():
            key = f"{EVO_BLOCK}triangle_multiplication_{direction}/{name}//weights"
            source[key] = np.full(source[key].shape, value, dtype=np.float32)

    sd = remap_af2_params(source)
    hidden = sd["evoformer.0.tri_mul_out.p_in.weight"].shape[0] // 2

    for slot, (tt_key, expected) in enumerate(
        (
            ("evoformer.0.tri_mul_out.p_in.weight", (1.0, 2.0)),
            ("evoformer.0.tri_mul_out.g_in.weight", (3.0, 4.0)),
            # Swapped: `a` is AF2's right, `b` is AF2's left.
            ("evoformer.0.tri_mul_in.p_in.weight", (2.0, 1.0)),
            ("evoformer.0.tri_mul_in.g_in.weight", (4.0, 3.0)),
        )
    ):
        weight = sd[tt_key]
        assert torch.all(weight[:hidden] == expected[0]), f"{tt_key} first half"
        assert torch.all(weight[hidden:] == expected[1]), f"{tt_key} second half"


def test_block_slices_are_not_aliased():
    """Each of the 48 Evoformer blocks must read its own slice of the stacked array."""
    source = _zero_source()
    key = f"{EVO_BLOCK}pair_transition/transition1//weights"
    stacked = np.zeros(source[key].shape, dtype=np.float32)
    for i in range(NUM_EVOFORMER_BLOCKS):
        stacked[i] = float(i)
    source[key] = stacked

    sd = remap_af2_params(source)
    for i in range(NUM_EVOFORMER_BLOCKS):
        weight = sd[f"evoformer.{i}.pair_transition.fc1.weight"]
        assert torch.all(weight == float(i)), f"block {i} read the wrong slice"


def test_haiku_linear_is_transposed_to_torch_layout():
    """`[in, out]` in the checkpoint must come out `[out, in]`, not reshaped in place."""
    source = _zero_source()
    key = f"{EVO_BLOCK}pair_transition/transition1//weights"
    stacked = np.zeros(source[key].shape, dtype=np.float32)
    stacked[0, 3, 7] = 1.0  # in-channel 3, out-channel 7
    source[key] = stacked

    weight = remap_af2_params(source)["evoformer.0.pair_transition.fc1.weight"]
    assert weight.shape == (512, 128)
    assert weight[7, 3] == 1.0
    assert weight.abs().sum() == 1.0


BANNED_FRAMEWORKS = frozenset(
    ("jax", "jaxlib", "dm-haiku", "haiku", "colabdesign", "optax", "chex", "flax"))


def test_no_jax_dependency():
    """The port's approval rests on no new framework entering tt-bio.

    Matched on the parsed requirement NAME, not as a substring of the whole file. tt-bio
    already depends on `jaxtyping` (atomworks, reached from rf3 data/pipelines.py), which is a
    shape-annotation library that does not depend on jax; a substring check fails on it and
    cannot tell it apart from a real `jax` entry, so it would have to be deleted rather than
    kept honest.
    """
    text = (REPO / "pyproject.toml").read_text()
    names = {re.split(r"[<>=!~;\[\s]", q, 1)[0].strip().lower().replace("_", "-")
             for q in re.findall(r"""["']([^"'\n]+)["']""", text)}
    hit = names & BANNED_FRAMEWORKS
    assert not hit, f"{sorted(hit)} must not be a tt-bio dependency"


@pytest.mark.skipif(
    not Path("~/pxd_tool_weights/af2/params_model_1_ptm.npz").expanduser().is_file(),
    reason="params_model_1_ptm.npz not present (373 MB, not committed)",
)
def test_real_checkpoint_matches_the_manifest():
    """When the checkpoint is on the host, the committed manifest must still describe it."""
    sd = load_af2_state_dict(
        str(Path("~/pxd_tool_weights/af2/params_model_1_ptm.npz").expanduser())
    )
    assert {k: list(v.shape) for k, v in sd.items()} == _shapes(REMAPPED_SHAPES)
    assert all(v.dtype == torch.float32 for v in sd.values())
