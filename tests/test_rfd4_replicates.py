"""Per-replicate sampling transforms, and the classifier-free-guidance refusal.

A transform runs once per design, which is how upstream's VHH recipe draws a different hotspot
patch for each replicate instead of repeating one condition. Classifier-free guidance is the
other half of the same config surface and is not available on the release, so it is refused with
the reason rather than ignored.
"""
import pytest
import torch

from tt_bio.rfd4 import sampler as S
from tt_bio.rfd4.replicates import apply_transforms, resample_hotspots


def example(n_tokens=6, n_hotspots=5):
    """One featurized example: leading axis 1, a hotspot mask with `n_hotspots` atoms marked."""
    hotspots = torch.zeros(1, n_tokens, 37, dtype=torch.bool)
    flat = hotspots.reshape(-1)
    flat[torch.arange(n_hotspots) * 7] = True
    return {"padding_mask": torch.ones(1, n_tokens, dtype=torch.bool),
            "condition_hotspot_1_atom": hotspots,
            "nested": {"coords": torch.zeros(1, n_tokens, 3)},
            "not_a_tensor": "carried through"}


def test_without_transforms_the_batch_is_the_broadcast_the_engine_already_does():
    batch = apply_transforms(example(), num_designs=4, transforms=(), seed=0)
    assert batch["condition_hotspot_1_atom"].shape[0] == 4
    assert torch.equal(batch["condition_hotspot_1_atom"][0], batch["condition_hotspot_1_atom"][3])


def test_each_replicate_gets_its_own_hotspot_patch():
    batch = apply_transforms(example(), num_designs=4, transforms=[resample_hotspots(2)], seed=0)
    patches = batch["condition_hotspot_1_atom"]
    assert patches.shape == (4, 6, 37)
    assert all(int(patch.sum()) == 2 for patch in patches)
    assert any(not torch.equal(patches[0], other) for other in patches[1:]), \
        "a batch that draws the same patch every time is the broadcast, not a transform"


def test_a_patch_is_drawn_from_the_atoms_the_input_marks():
    marked = example()["condition_hotspot_1_atom"]
    batch = apply_transforms(example(), num_designs=3, transforms=[resample_hotspots(3)], seed=7)
    for patch in batch["condition_hotspot_1_atom"]:
        assert torch.all(marked[0][patch]), "a drawn hotspot the input did not mark is invented"


def test_a_replicate_is_the_same_design_whatever_the_batch_asked_for():
    """Design 2 of a 3-design run and of a 9-design run are the same: the generator is seeded
    per replicate index, so a bigger batch extends a run rather than re-rolling it."""
    small = apply_transforms(example(), num_designs=3, transforms=[resample_hotspots(2)], seed=11)
    large = apply_transforms(example(), num_designs=9, transforms=[resample_hotspots(2)], seed=11)
    assert torch.equal(small["condition_hotspot_1_atom"][2], large["condition_hotspot_1_atom"][2])


def test_nested_and_non_tensor_values_survive_the_stack():
    batch = apply_transforms(example(), num_designs=2, transforms=[resample_hotspots(1)], seed=0)
    assert batch["nested"]["coords"].shape[0] == 2
    assert batch["not_a_tensor"] == "carried through"


def test_asking_for_more_hotspots_than_the_input_marks_is_an_error():
    with pytest.raises(ValueError, match="cannot draw 9"):
        apply_transforms(example(), num_designs=1, transforms=[resample_hotspots(9)], seed=0)


def test_a_transform_that_resizes_the_example_is_refused():
    def resize(features, index, generator):
        return dict(features, padding_mask=torch.ones(1, 99, dtype=torch.bool))

    with pytest.raises(ValueError, match="changed the shape of padding_mask"):
        apply_transforms(example(), num_designs=2, transforms=[resize], seed=0)


# ------------------------------------------------------------------ classifier-free guidance


def test_guidance_w_is_one_on_every_shipped_recipe():
    for name in S.RECYCLE_SCHEDULES:
        assert S.recycle_strategy(name).guidance_w == 1.0


def test_a_guidance_weight_other_than_one_is_refused_with_the_reason():
    spec = S.replace(S.RELEASED, guidance_w=0.5)
    with pytest.raises(ValueError, match="cath_code/x_motif"):
        S.check_guidance_w(spec)


def test_the_refusal_reaches_a_run_rather_than_being_advisory():
    from tests.test_rfd4_sampler import Net, groups, masks

    spec = S.replace(S.RELEASED, nsteps=4, n_recycle_schedule=((4, 1),), guidance_w=0.0)
    with pytest.raises(ValueError, match="guidance_w=0.0"):
        S.generate(spec, Net(), n_designs=2, n_tokens=48, seed=1,
                   emit_masks=masks(2, 48), groups=groups(2, 48))
