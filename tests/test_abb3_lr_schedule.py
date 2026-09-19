"""The learning-rate schedule the reproduction runs, and the property that makes a resume exact.

`params.yaml`'s `optimiser:` block is RAdam at 5e-4 *plus* `T_0: 50`, `T_mult: 1` and
`eta_min: 0`, which `lightning_module.py:144` turns into `CosineAnnealingWarmRestarts` stepped
once per epoch. The reproduction's recipe once carried lr and weight decay alone, so nothing
wrapped the optimizer and the run held a constant 5e-4: 1.96x upstream's mean over a full
schedule, and none of its troughs. Four things are checked, and each one is a way that failure
stayed invisible:

1. the recipe carries the whole `optimiser:` block, compared against `params.yaml` itself
   rather than against a second copy of its numbers;
2. the lr is a pure function of the global step, so a resume lands where the step index says
   and no scheduler state has to survive in the checkpoint;
3. it agrees with upstream's per-epoch staircase at the epoch boundaries, which is the one
   place a per-step schedule and a per-epoch one are comparable;
4. the cycle is 50 epochs: the trough falls at the end of it and the restart is a full reset to
   the base lr.

No device: all of it is host arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tt_bio.train.abb3_run import CosineRestartsByStep
from tt_bio.train.abodybuilder3_step import RECIPE
from tt_bio.train.sharding import steps_per_epoch

#: Upstream's own config, as they ship it. Machine-local, so the comparison skips where the
#: checkout is absent rather than asserting numbers copied out of it.
PARAMS = Path("/home/ttuser/abb3_src/ABodyBuilder3/params.yaml")

#: 8,395 training structures at global batch 64 with the short final batch kept. The released
#: checkpoint's own `global_step` 193,512 is 132 x 1466, which is where the 132 is confirmed.
STEPS_PER_EPOCH = 132


def sched(steps_per_epoch_=STEPS_PER_EPOCH) -> CosineRestartsByStep:
    p = torch.zeros(1, requires_grad=True)
    return CosineRestartsByStep(torch.optim.RAdam([p], lr=RECIPE["lr"]),
                                steps_per_epoch=steps_per_epoch_, T_0=RECIPE["T_0"],
                                T_mult=RECIPE["T_mult"], eta_min=RECIPE["eta_min"])


@pytest.mark.skipif(not PARAMS.is_file(), reason=f"{PARAMS} is not on this host")
def test_the_recipe_carries_the_whole_optimiser_block():
    yaml = pytest.importorskip("yaml")
    block = yaml.safe_load(PARAMS.read_text())["optimiser"]
    assert block["optimiser"] == "RAdam", (
        f"params.yaml selects {block['optimiser']!r}, and the schedule this module installs is "
        f"the one lightning_module.py builds on the RAdam branch")
    for key in ("lr", "weight_decay", "T_0", "eta_min", "T_mult"):
        assert float(RECIPE[key]) == float(block[key]), (
            f"RECIPE[{key!r}] is {RECIPE[key]} and params.yaml says {block[key]}. The three "
            f"scheduler keys went missing once and the run trained at a constant lr")


def test_the_number_of_steps_in_an_epoch_is_the_one_the_batcher_counts():
    assert steps_per_epoch(8395, 64, drop_last=False) == STEPS_PER_EPOCH
    assert STEPS_PER_EPOCH * 1466 == 193_512, (
        "upstream's released checkpoint sits at global_step 193,512, which is 1,466 epochs of "
        "132 steps only if the short final batch is kept")


def test_the_learning_rate_is_a_pure_function_of_the_global_step():
    """So a resume needs no scheduler state, which is why the fractional form was taken.

    A call-counted scheduler restarts at the top of the cosine on every resume, and this run
    expects 9 to 47 of them.
    """
    walked = sched()
    for step in range(1, 4001):
        value = walked.set_step(step)
    jumped = sched().set_step(4000)
    assert jumped == value, (
        "the lr after walking to step 4,000 differs from the lr set by jumping there, so a "
        "resumed run would not be on the same schedule as an uninterrupted one")
    # And out of order, which is what a resume onto an earlier checkpoint does.
    assert sched().set_step(1) == pytest.approx(RECIPE["lr"], rel=1e-15)
    back = walked.set_step(1)
    assert back == pytest.approx(RECIPE["lr"], rel=1e-15)


def test_it_agrees_with_upstreams_per_epoch_staircase_at_the_epoch_boundaries():
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.RAdam([p], lr=RECIPE["lr"])
    upstream = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=RECIPE["T_0"], T_mult=RECIPE["T_mult"], eta_min=RECIPE["eta_min"])
    ours = sched()
    for epoch in range(120):
        theirs = opt.param_groups[0]["lr"]
        mine = ours.set_step(epoch * STEPS_PER_EPOCH + 1)
        assert mine == pytest.approx(theirs, rel=1e-12), (
            f"at the first step of epoch {epoch} the schedules disagree: {mine} against "
            f"upstream's {theirs}")
        upstream.step()


def test_the_cycle_is_fifty_epochs_and_the_restart_resets_to_the_base_lr():
    cycle = RECIPE["T_0"] * STEPS_PER_EPOCH
    assert cycle == 6600
    s = sched()
    assert s.set_step(1) == pytest.approx(RECIPE["lr"], rel=1e-15)
    assert s.set_step(cycle // 2 + 1) == pytest.approx(RECIPE["lr"] / 2, rel=1e-3)
    assert s.set_step(cycle) < 1e-8, "the trough sits at the end of the cycle"
    assert s.set_step(cycle + 1) == pytest.approx(RECIPE["lr"], rel=1e-15), (
        "a warm restart returns to the base lr, which is what makes the troughs a feature of "
        "the recipe rather than the end of it")


def test_the_schedule_round_trips_through_the_file_a_reader_checks_it_with():
    spec = sched().spec
    assert CosineRestartsByStep.load(spec).set_step(3_000) == sched().set_step(3_000)
    assert spec["steps_per_epoch"] == STEPS_PER_EPOCH and spec["lr"] == RECIPE["lr"]


# ------------------------------------------------------------------ the wiring, not the class

class _FakeStep:
    """The host half of a ``TrainStep``: a mirror, an optimizer, a dropout RNG and the recipe.

    The defect was never in the schedule, it was in nobody calling one, so this checks the loop
    and not the arithmetic: what lr does the optimizer actually see on step ``k``, and does the
    history say so.
    """

    def __init__(self, n=2, size=4, accumulate=16):
        from tt_bio.abodybuilder3 import Dropout
        self.mirror = [torch.zeros(size, size, requires_grad=True) for _ in range(n)]
        self.params = []
        self.optimizer = torch.optim.RAdam(self.mirror, lr=RECIPE["lr"],
                                           weight_decay=RECIPE["weight_decay"])
        self.dropout = Dropout.__new__(Dropout)
        self.dropout.rate, self.dropout.calls = RECIPE["dropout_rate"], 0
        self.dropout.generator = torch.Generator().manual_seed(0)
        self.recipe = dict(RECIPE)
        self.accumulate = accumulate
        self.loss_terms = {}
        #: The lr in the parameter group at the moment the update is taken, per step.
        self.seen = []

    def step(self, micros):
        from tt_bio.train.abodybuilder3_step import StepTiming
        self.seen.append(float(self.optimizer.param_groups[0]["lr"]))
        for i, m in enumerate(self.mirror):
            m.grad = torch.full(m.shape, 0.01 * (i + 1))
        self.optimizer.step()
        return {"loss": 1.0}, StepTiming(micro_batches=len(micros))


class _FakeData:
    def __init__(self, n=8395):
        self.n = n

    def __len__(self):
        return self.n

    # `host`/`upload` is the seam `prefetch.host_stream` cuts at, and the loop pulls its
    # micro-batches through it. `batch` stays because `catalogue.REQUIRED_DATASET_MEMBERS`
    # names it and `recipes` calls it, and it is the composition of the other two, same as
    # `SyntheticFvs`.
    def host(self, indices):
        return list(indices)

    def upload(self, indices):
        return {"indices": list(indices)}

    def batch(self, indices):
        return self.upload(self.host(indices))


def test_the_run_loop_drives_the_schedule_and_writes_it_into_every_row(tmp_path):
    """The check the missing scheduler would have failed: read the lr out of the run.

    Both halves matter. ``seen`` is what the optimizer's parameter group held when it stepped,
    so a scheduler that is built and never called fails here even though every test of the
    class above passes. The history is the other half: the lr went unlogged for the run's first
    444 steps, which is why the defect survived two readers.
    """
    import json

    from tt_bio.train.abb3_run import RunConfig, run

    step = _FakeStep()
    cfg = RunConfig(out_dir=tmp_path, steps=6, global_batch=64, micro_batch=4,
                    rendezvous=tmp_path / "rv", checkpoint_minutes=1e6)
    out = run(step, _FakeData(), cfg, resume=False)

    want = [sched().set_step(k) for k in range(1, 7)]
    assert step.seen == want, (
        "the lr the optimizer stepped at does not follow the recipe's cosine, so the schedule "
        "is built but not driven")
    rows = out["history"]
    assert [r["lr"] for r in rows] == want
    assert all(r["grad_norm"] > 0 for r in rows), "the gradient norm is not reaching the history"
    assert step.seen[0] == pytest.approx(RECIPE["lr"], rel=1e-15)
    assert step.seen[-1] < step.seen[0], "the cosine is flat, which is the defect itself"

    spec = json.loads((tmp_path / "schedule.json").read_text())
    assert spec["steps_per_epoch"] == STEPS_PER_EPOCH
    assert CosineRestartsByStep.load(spec).set_step(6) == want[-1]


# ------------------------------- every key of the four training blocks has to land somewhere

#: Where each key of `params.yaml`'s four training blocks lands in this port, `loss` flattened
#: with dots. The value is the destination in prose, so a failure reads as a sentence; what is
#: asserted is that the key APPEARS here. Dropping a key from its destination later then means
#: editing this map, rather than editing nothing, which is what happened to `T_0`, `T_mult` and
#: `eta_min`: the recipe carried two of six keys and its docstring claimed to be the block.
DESTINATIONS = {
    "optimiser": {
        "optimiser": "selects the RAdam branch of lightning_module.py:144, and with it the "
                     "CosineAnnealingWarmRestarts this module installs",
        "lr": "RECIPE['lr'], the base of the cosine",
        "weight_decay": "RECIPE['weight_decay'], passed to torch.optim.RAdam",
        "T_0": "RECIPE['T_0'], the cycle in epochs",
        "T_mult": "RECIPE['T_mult']",
        "eta_min": "RECIPE['eta_min'], the floor of the cosine",
    },
    "model": {
        "c_s": "ABB3Config.c_s",
        "embed_dim": "ABB3Config.embed_dim",
        "c_ipa": "ABB3Config.c_ipa",
        "c_resnet": "ABB3Config.c_resnet",
        "no_heads_ipa": "ABB3Config.no_heads_ipa",
        "no_qk_points": "ABB3Config.no_qk_points",
        "no_v_points": "ABB3Config.no_v_points",
        "dropout_rate": "ABB3Config.dropout_rate and RECIPE['dropout_rate']",
        "no_blocks": "ABB3Config.no_blocks",
        "no_transition_layers": "ABB3Config.no_transition_layers",
        "no_resnet_blocks": "ABB3Config.no_resnet_blocks",
        "no_angles": "ABB3Config.no_angles",
        "trans_scale_factor": "ABB3Config.trans_scale_factor",
        "epsilon": "ABB3Config.epsilon",
        "inf": "ABB3Config.inf",
        "rel_pos_dim": "folded into ABB3Config.c_z: stages/train.py:41-46 builds c_z as "
                       "2 * rel_pos_dim + 1, then +3 for edge_chain_feature, which is 132",
        "edge_chain_feature": "folded into ABB3Config.c_z, the +3 above",
        "use_original_sm": "true, and this port implements that branch",
        "rotation_propagation": "true, and we match it by default: the reference has no detach, "
                                "no stop_rot_gradient and no no_grad near the frames",
    },
    "loss": {
        "fape.weight": "RECIPE['fape_weight'], the coefficient on the FAPE pair in the total",
        "fape.backbone.weight": "losses_geometry.fape_loss(backbone_weight=...)",
        "fape.sidechain.weight": "losses_geometry.fape_loss(sidechain_weight=...)",
        "final_output_backbone_loss.weight": "RECIPE['final_backbone_weight']",
        "supervised_chi.chi_weight": "RECIPE['chi_weight']",
        "supervised_chi.angle_norm_weight": "RECIPE['angle_norm_weight']",
        "supervised_chi.weight": "the coefficient on the chi term in abodybuilder3_step's total, "
                                 "which is a literal 1 because upstream's weight is 1.0",
    },
    "train": {
        "batch_size": "RunConfig.global_batch, and the 132 steps per epoch derived from it",
    },
}

#: Keys deliberately not carried, each with the reason it is not a gap. An exclusion has to be
#: written down to be an exclusion; a key that is simply absent from both maps fails the audit.
EXCLUDED = {
    "loss.violation.clash_overlap_tolerance": "violation terms are finetune-only in upstream's "
                                              "own gating and base-loss carries none",
    "loss.violation.violation_tolerance_factor": "same, finetune-only",
    "loss.violation_loss_bondangle.weight": "same, finetune-only",
    "loss.violation_loss_bondlength.weight": "same, finetune-only",
    "loss.violation_loss_clash.weight": "same, finetune-only",
    "loss.plddt.weight": "base-loss has no pLDDT head, so there is no term to weight",
    "train.epochs": "params.yaml says 1000, and the released checkpoint sits at global_step "
                    "193,512, which is 1,466 epochs of 132. The reproduction targets the "
                    "checkpoint's own step count rather than the config's epoch count",
    "train.early_stopping": "10,000 epochs, which never fired: the released checkpoint ran to "
                            "1,466. A fixed-step reproduction has nothing to stop early",
}

#: The four blocks this port is a transcription of. `base`, `cluster`, `filter`, `split`,
#: `finetune`, `inference` and `language` describe data preparation, the finetune stage and
#: inference, none of which this leg runs.
AUDITED_BLOCKS = ("optimiser", "model", "loss", "train")


def flatten(block, prefix: str = "") -> dict:
    out = {}
    for k, v in block.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def audit(params: dict) -> list:
    """Keys with no destination and destinations with no key, for all four blocks.

    Both directions on purpose. A map that only ever grows is a map nobody prunes, and a
    destination for a key upstream deleted is a claim about a config that no longer exists.
    """
    problems = []
    for block in AUDITED_BLOCKS:
        if block not in params:
            problems.append(f"params.yaml has no {block!r} block")
            continue
        keys = set(flatten(params[block]))
        known = set(DESTINATIONS[block])
        excluded = {k.split(".", 1)[1] for k in EXCLUDED if k.startswith(f"{block}.")}
        for k in sorted(keys - known - excluded):
            problems.append(
                f"{block}.{k} is in params.yaml and has no destination in this port. Either "
                f"carry it, or add it to EXCLUDED with the reason it is not a gap")
        for k in sorted((known | excluded) - keys):
            problems.append(
                f"{block}.{k} is claimed here but is not in params.yaml any more")
    return problems


@pytest.mark.skipif(not PARAMS.is_file(), reason=f"{PARAMS} is not on this host")
def test_every_key_of_the_four_training_blocks_has_a_destination():
    """Key set, not values: the failure was a key that silently had nowhere to go.

    This starts green on `model`, `loss` and `train` -- they were audited by hand on
    2026-09-19 and they are complete. The defect it catches in anger is the one it was written
    for, the three scheduler keys, and that one is already fixed. So its negative control is
    necessarily synthetic: `test_the_audit_fails_when_a_key_loses_its_destination` deletes a
    destination and watches the audit name it.
    """
    yaml = pytest.importorskip("yaml")
    problems = audit(yaml.safe_load(PARAMS.read_text()))
    assert not problems, "\n".join(problems)


def test_the_audit_fails_when_a_key_loses_its_destination():
    """The negative control, and it is synthetic because the real defect is already repaired.

    Both directions, because a one-directional check rots: a destination for a key upstream
    deleted is as wrong as a key with no destination.
    """
    params = {"optimiser": {"optimiser": "RAdam", "lr": 5e-4, "weight_decay": 1e-4, "T_0": 50,
                            "eta_min": 0, "T_mult": 1},
              "model": {k: 1 for k in DESTINATIONS["model"]},
              "loss": {"fape": {"weight": 1.0, "backbone": {"weight": 0.5},
                                "sidechain": {"weight": 1.0}},
                       "final_output_backbone_loss": {"weight": 0.5},
                       "supervised_chi": {"weight": 1.0, "chi_weight": 0.5,
                                          "angle_norm_weight": 0.02},
                       "plddt": {"weight": 0.01},
                       "violation": {"clash_overlap_tolerance": 1.5,
                                     "violation_tolerance_factor": 12.0},
                       "violation_loss_bondangle": {"weight": 0.1},
                       "violation_loss_bondlength": {"weight": 0.1},
                       "violation_loss_clash": {"weight": 0.1}},
              "train": {"epochs": 1000, "batch_size": 64, "early_stopping": 10000}}
    assert audit(params) == [], "the fixture is supposed to be the clean case"

    # The defect, reproduced: a scheduler key in the config with nowhere to go.
    dropped = dict(params, optimiser={**params["optimiser"], "T_5": 7})
    assert any("optimiser.T_5" in p and "no destination" in p for p in audit(dropped))

    # And the other direction: a destination for a key that is no longer in the config.
    gone = dict(params, optimiser={k: v for k, v in params["optimiser"].items() if k != "T_0"})
    assert any("optimiser.T_0" in p and "not in params.yaml" in p for p in audit(gone))


@pytest.mark.skipif(not PARAMS.is_file(), reason=f"{PARAMS} is not on this host")
def test_the_model_and_loss_values_are_the_ones_upstream_ships():
    """The key-set audit above says a key has somewhere to go. This says it arrived."""
    yaml = pytest.importorskip("yaml")
    import inspect

    from tt_bio.abodybuilder3_reference import ABB3Config
    from tt_bio.train import losses_geometry

    params = yaml.safe_load(PARAMS.read_text())
    model, loss = params["model"], params["loss"]
    cfg = ABB3Config()
    for key in DESTINATIONS["model"]:
        if hasattr(cfg, key):
            assert float(getattr(cfg, key)) == float(model[key]), (
                f"ABB3Config.{key} is {getattr(cfg, key)} and params.yaml says {model[key]}")
    assert cfg.c_z == 2 * model["rel_pos_dim"] + 1 + (3 if model["edge_chain_feature"] else 0), (
        "c_z is built from rel_pos_dim and edge_chain_feature at stages/train.py:41-46, and "
        "that derivation is the only place those two keys land")
    assert model["use_original_sm"] is True and model["rotation_propagation"] is True

    assert float(RECIPE["fape_weight"]) == float(loss["fape"]["weight"])
    assert float(RECIPE["final_backbone_weight"]) == \
        float(loss["final_output_backbone_loss"]["weight"])
    assert float(RECIPE["chi_weight"]) == float(loss["supervised_chi"]["chi_weight"])
    assert float(RECIPE["angle_norm_weight"]) == \
        float(loss["supervised_chi"]["angle_norm_weight"])
    assert float(loss["supervised_chi"]["weight"]) == 1.0, (
        "abodybuilder3_step adds the chi term with a literal coefficient of 1")
    sig = inspect.signature(losses_geometry.fape_loss).parameters
    assert float(sig["backbone_weight"].default) == float(loss["fape"]["backbone"]["weight"])
    assert float(sig["sidechain_weight"].default) == float(loss["fape"]["sidechain"]["weight"])
    assert params["train"]["batch_size"] == 64


def test_the_audit_says_so_when_its_input_is_missing_rather_than_passing():
    """A gate that is green in CI because its input is absent is not a gate.

    `params.yaml` is not vendored: it lives on qb2 and nowhere in CI, so the two tests above
    skip with the path in the reason. This one runs everywhere and asserts the skip is the
    reason they are quiet.
    """
    if PARAMS.is_file():
        pytest.skip(f"{PARAMS} is present, so the audit above actually ran")
    problems = audit({})
    assert len(problems) == len(AUDITED_BLOCKS)
    assert all("has no" in p for p in problems)
