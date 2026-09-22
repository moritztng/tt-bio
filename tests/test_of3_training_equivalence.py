"""OpenFold3 training-equivalence gates that need no card and no checkpoint.

The full instruments live in `perf/of3t_equivalence/` and are run by hand; what is worth a
place in the suite is the part that will silently rot: the LR schedule's OF3 form, and the
per-(stage, dataset) loss weight table rebuilt from upstream's own configs.

Deliberately NOT here: the bijection manifest (loads a 2.3 GB checkpoint) and the optimizer
drive (200 steps against torch). Both are in `perf/of3t_equivalence/` with their JSON output.
"""

import pytest

from tt_bio.train.losses import (OF3_CROP_TOKENS, OF3_LOSS_BASE, OF3_LOSS_OVERRIDES,
                                 of3_loss_weights)
from tt_bio.train.optim import af3_lr


# OF3's AlphaFoldLRScheduler, `openfold3/core/utils/lr_schedulers.py`, shipped defaults.
OF3 = dict(base_lr=0.0, max_lr=1e-3, warmup_no_steps=1000,
           start_decay_after_n_steps=50000, decay_every_n_steps=50000, decay_factor=0.95)


def _ours(step):
    return af3_lr(step, OF3["max_lr"], warmup_steps=OF3["warmup_no_steps"],
                  decay_every_n_steps=OF3["decay_every_n_steps"],
                  decay_factor=OF3["decay_factor"], base_lr=OF3["base_lr"],
                  plateau_until=OF3["start_decay_after_n_steps"])


def _theirs(step):
    """Their `get_lr` body, `lr_schedulers.py:79-91`. Independent of our branch structure."""
    if step <= OF3["warmup_no_steps"]:
        return OF3["base_lr"] + (step / OF3["warmup_no_steps"]) * OF3["max_lr"]
    if step > OF3["start_decay_after_n_steps"]:
        since = step - OF3["start_decay_after_n_steps"]
        return OF3["max_lr"] * (OF3["decay_factor"] ** (since // OF3["decay_every_n_steps"] + 1))
    return OF3["max_lr"]


# The knees PROTOCOL SS4 names, and both sides of each.
@pytest.mark.parametrize("step", [0, 1, 999, 1000, 1001, 49999, 50000, 50001,
                                  99999, 100000, 100001])
def test_of3_lr_matches_upstream_at_every_knee(step):
    assert _ours(step) == _theirs(step), f"step {step}"


def test_of3_lr_matches_upstream_across_the_domain():
    bad = [k for k in range(0, 100002, 97) if _ours(k) != _theirs(k)]
    assert not bad, f"{len(bad)} mismatching steps, first {bad[:5]}"


def test_of3_and_protenix_forms_differ_exactly_at_the_decay_boundary():
    """Where the two closed forms actually part, measured rather than assumed.

    With OF3's shipped defaults start_decay_after_n_steps == decay_every_n_steps == 50000,
    the plateau form and the Protenix form coincide at every step but ONE: at k = 50000
    Protenix has already applied a decay (9.5e-4) while OF3 is still on its plateau (1e-3),
    a 5 % difference in the rate at that step. Easy to mistake for "the two schedules are
    the same", which is why it is pinned here.
    """
    def protenix(k):
        return af3_lr(k, OF3["max_lr"], warmup_steps=OF3["warmup_no_steps"],
                      decay_every_n_steps=OF3["decay_every_n_steps"],
                      decay_factor=OF3["decay_factor"])
    differing = [k for k in range(0, 150001) if protenix(k) != _ours(k)]
    assert differing == [50000]
    assert protenix(50000) == pytest.approx(9.5e-4)
    assert _ours(50000) == pytest.approx(1e-3)


def test_the_two_forms_diverge_broadly_when_the_knees_are_not_equal():
    """And when start_decay != decay_every, they disagree over half the domain.

    This is what makes `plateau_until` load-bearing rather than decoration: it is not a
    one-step correction in general, it is a different schedule.
    """
    kw = dict(warmup_steps=100, decay_every_n_steps=200, decay_factor=0.95)
    differing = [k for k in range(0, 4001)
                 if af3_lr(k, 1e-3, **kw) != af3_lr(k, 1e-3, plateau_until=300, **kw)]
    assert len(differing) > 1800, len(differing)


def test_protenix_schedule_unchanged_by_the_of3_extension():
    """`plateau_until=None` must still be the pre-extension closed form, exactly."""
    lr, warm, every, factor = 1.8e-3, 1000, 50000, 0.95
    for k in (0, 1, 999, 1000, 1001, 49999, 50000, 50001, 99999, 100000):
        want = (k / warm * lr) if k <= warm else lr * (factor ** (k // every))
        assert af3_lr(k, lr, warmup_steps=warm, decay_every_n_steps=every,
                      decay_factor=factor) == want, f"step {k}"


def test_of3_stages_and_crops_are_upstreams():
    assert sorted(OF3_LOSS_OVERRIDES) == ["finetune_1", "finetune_2", "finetune_3",
                                          "initial_training"]
    assert OF3_CROP_TOKENS == {"initial_training": 384, "finetune_1": 640,
                               "finetune_2": 768, "finetune_3": 768}
    # R4: OF3 training never exceeds a 768-token crop.
    assert max(OF3_CROP_TOKENS.values()) == 768


def test_distillation_sets_zero_the_confidence_terms():
    """The error this table exists to prevent: training confidence heads on distillation."""
    for stage in ("initial_training", "finetune_1", "finetune_2"):
        w = of3_loss_weights(stage, "short-monomer-distillation")
        for term in ("resolved", "plddt", "pae", "pde"):
            assert w[term] == 0.0, f"{stage}/{term}"
        # ...while the real PDB set in the same stage keeps them on.
        assert of3_loss_weights(stage, "weighted-pdb")["plddt"] == 1e-4


def test_finetune_3_is_the_confidence_only_stage():
    w = of3_loss_weights("finetune_3", "weighted-pdb")
    assert w["mse"] == 0.0 and w["distogram"] == 0.0 and w["smooth_lddt"] == 0.0
    assert w["pae"] == 1e-4 and w["plddt"] == 1e-4


def test_of3_base_differs_from_protenix_pretrain_in_pae():
    """Recorded because it is one term and it is easy to assume the two tables agree."""
    from tt_bio.train.losses import LOSS_WEIGHTS
    assert OF3_LOSS_BASE["pae"] == 1e-4
    assert LOSS_WEIGHTS["pretrain"]["pae"] == 0.0


def test_unknown_stage_and_dataset_are_refused_by_name():
    with pytest.raises(KeyError, match="training stage"):
        of3_loss_weights("finetune_4")
    with pytest.raises(KeyError, match="train dataset"):
        of3_loss_weights("finetune_3", "RNA-monomer-distillation")


def test_per_example_override_wins_and_accepts_their_term_name():
    w = of3_loss_weights("initial_training", "weighted-pdb",
                         overrides={"experimentally_resolved": 0.0})
    assert w["resolved"] == 0.0
