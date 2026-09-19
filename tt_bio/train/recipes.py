"""Every Tier-1 body, importable, and ``source(name)`` returning its text.

**This module IS the escape hatch, and the hatch is a mechanism with a test rather than a
promise.** r4's central finding is that everyone ships a hatch and almost nobody ships one
that still works a year later, because the tier above quietly grows a dependency on something
the tier below cannot reach. The durable form of the lesson, worth keeping where it will be
read: *an escape hatch is only real if the tier above is expressible in the tier below's
public API, and it only stays real if a test asserts it.*

So the arrangement here is deliberately rigid.

* ``tt_bio.train.finetune`` does not implement a loop. It looks a recipe up in this module and
  calls it. There is exactly one code path, so "the hatch reproduces the Tier-1 result" is
  identity rather than a claim about two implementations agreeing.
* Every function here is written against ``tt_bio.train``'s public Tier-2 names ONLY. The
  tuple ``tt_bio.train.TIER2`` is the whole permitted vocabulary, plus builtins and the
  recipe's own arguments. The imports below are from the submodules rather than the package
  so there is no import cycle, and the test checks them by IDENTITY against ``TIER2`` -- an
  import that reached a private helper would resolve to an object that is not in the tuple.
* ``source(name)`` returns the text, and ``tests/test_train_interface.py`` execs it in a
  namespace containing nothing but ``TIER2`` and asserts the resulting function's bytecode is
  identical to the shipped one. A recipe that reaches for a private helper fails that exec,
  and then the helper becomes public or the recipe changes. There is no third option and no
  way to satisfy the test with a docstring.

The bodies read as a user would write them, because that is the point: what you get from
``source()`` is a starting file, not a generated shim.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict

from . import launcher, objectives, provenance
from ..autograd import backward, install, uninstall
from .sharding import batches
from .checkpoint import Checkpointer
from .lora import LoraConfig, attach, trainable
from .mesh import Mesh
from .optim import AdamW, af3_lr
from .dryrun import plan
from .tensors import to_device, to_host

__all__ = ["source", "names", "recipe", "train_loop"]


def train_loop(forward, dataset, *, out_dir, global_batch, steps, objective="af3",
               train="adapters", mesh=None, lora=None, seed=0, lr=3e-4, warmup_steps=1000,
               checkpoint_every=100, tokens=None, weights=None, model=None):
    """Fine-tune or pre-train a shipped forward. The Tier-1 default, and a Tier-2 program.

    ``train`` is what the optimizer owns, and it is a NAME for the same reason ``objective``
    is: it decides the memory arithmetic, so Tier 0 has to be able to check it before a
    device opens. ``"adapters"`` trains a LoRA factor pair beside each site on a frozen
    trunk. ``"weights"`` trains the model's own weights at those same sites, which is what a
    pre-training run does. Nothing else in this function changes between them -- the loop,
    the objective, the optimizer, the checkpointer and the data-parallel axis are the same
    code, which is why going from a fine-tune to a pre-training run is one argument and not a
    rewrite.

    ``forward(batch) -> {name: device tensor}`` is the SHIPPED forward, handed in rather than
    constructed here: the training path calls the same forward inference calls, and the way to
    guarantee that is to not have a second one to call.

    ``model`` is the built model object, and passing it is strongly preferred: discovery
    then WALKS it, so the optimizer owns every device weight the model reaches rather than the
    weights its routed ``ops.linear`` call sites pass. Leave it out and discovery is the
    call-site census, which cannot see a weight a module fused in its own ``__init__`` --
    measured at 2119 of 2531 on OpenFold3's trunk.

    ``dataset`` needs ``__len__`` and ``batch(indices) -> dict`` carrying the labels the
    objective row names. No featurizer is imposed -- per-model featurisation is the one thing
    the campaign refused to generalise, because each family's cropping and MSA handling is
    exactly the part that is genuinely different.

    A data-parallel axis wider than one chip is one process per chip, and this function is one
    process. So it hands the run to :mod:`tt_bio.train.launcher`, which re-runs the calling
    program once per chip and comes back with the aggregate; inside each of those processes
    this same function runs, takes its own shard and hands its own gradient to ``step()``.
    Two lines below know about it -- which shard is read, and that ``step()`` is given
    something to reduce -- and the loop is otherwise the loop it was on one chip.
    """
    dp = (mesh or Mesh({"dp": [0]})).axis("dp")
    if dp.width > 1 and launcher.driving():
        return launcher.drive(dp, out_dir=out_dir, steps=steps)
    dp_rank, dp = launcher.rank(), launcher.reducer(dp)
    # `None` here means "train the weights themselves", and it is the only line that reads
    # `train`. Everything downstream branches on `cfg is None` or not at all.
    cfg = (lora or LoraConfig()) if train == "adapters" else None
    row = objectives.objective(objective)

    fit = plan(tokens=tokens or dataset.tokens, chips=dp.width,
               global_batch=global_batch, frozen_trunk=cfg is not None)
    if fit.verdict == "refused":
        raise RuntimeError(f"refusing to start: {fit}")

    install()
    try:
        # One census pass on a real batch discovers the trainable sites. It costs an
        # inference, and it is how the parameter set comes from the forward instead of from a
        # list somebody maintains by hand. The same pass serves both modes.
        plan_order = batches(len(dataset), global_batch=global_batch, steps=steps, seed=seed,
                             data_parallel=dp)
        first = next(plan_order)
        # The seed reaches the ADAPTER INIT, not just the batch order. A is random and B is
        # zero, so without it every rank of a data-parallel run starts from different weights
        # and trains a different model -- which the launcher's master-hash check catches at
        # the end of the run rather than at the start of it. Training the weights themselves
        # initialises nothing, so there the seed reaches the batch order only.
        # `model` is the BUILT model, and handing it over is what makes the parameter set
        # every weight it reaches instead of the weights its four routed `ops.linear` call
        # sites happen to pass. Without it discovery falls back to the call-site census, which
        # is blind to every weight a module fuses in its own `__init__`.
        installed, params = trainable(forward, cfg, dataset.device,
                                      dataset.batch(first.per_chip[dp_rank]),
                                      model=model, rng=seed)
        opt = AdamW(params, lr=lr, data_parallel=dp,
                    schedule=lambda s: af3_lr(s, lr, warmup_steps=warmup_steps))
        # Rank 0 owns out_dir and the others get a subdirectory of it. The masters are
        # bit-identical across ranks, so one copy is the run's checkpoint; the reason not to
        # let them share the path is that two writers make a truncated safetensors file.
        ckpt = Checkpointer(launcher.out_dir(out_dir), every=checkpoint_every, metric="loss")
        history, last = [], None

        with provenance.during(seed=seed, config={
                "objective": objective, "global_batch": global_batch, "steps": steps,
                "lr": lr, "train": train, "chips": dp.width,
                "rank": cfg.rank if cfg else None, "alpha": cfg.alpha if cfg else None,
                "dp_rank": dp_rank, "rollout": row.rollout, "sites": sorted(params)}) as prov:
            with attach(installed, cfg):
                for batch in [first, *plan_order]:
                    opt.zero_grad()
                    # This rank's shard of the global batch, never the whole of it. On one
                    # chip per_chip has one entry and this is the global batch.
                    data = dataset.batch(batch.per_chip[dp_rank])
                    outputs = forward(data)
                    total, breakdown, seeds = row(
                        data, {k: to_host(v.value) for k, v in outputs.items()},
                        weights=weights)
                    # One backward over the union of the seeded roots' ancestors. Calling
                    # Tensor.backward once per root would replay every shared ancestor once
                    # per root and land the fan-in sums partial.
                    backward([outputs[k] for k in seeds],
                             [to_device(g, dataset.device) for g in seeds.values()])
                    # The gradient this replica holds, summed across the axis inside step().
                    # Empty on one chip, and the optimizer refuses a wide axis without it
                    # rather than stepping on one replica's gradient.
                    opt.step(replicas=launcher.replicas(params))
                    # The optimizer replaced each leaf's value with a new device tensor; this
                    # puts those tensors back where the walk found them, so the next forward
                    # reads what the optimizer moved. A no-op for a call-site census, whose
                    # hook hands the forward the leaf itself.
                    params.rebind()
                    last = {"step": batch.step, "loss": total, "breakdown": breakdown,
                            "grad_norm": opt.last_grad_norm, "lr": opt.last_lr,
                            "s": launcher.tick()}
                    history.append(last)
                    ckpt.save(batch.step, opt, metrics={"loss": total},
                              provenance=prov.as_dict())
            if last is not None:
                ckpt.save(last["step"], opt, metrics={"loss": last["loss"]},
                          provenance=prov.as_dict(), force=True)
        # The step control, on the cumulative ratio over the whole run. Raises rather than
        # warns: a run whose updates never reached the weight the forward reads produced
        # nothing, and it produced nothing while every number above looked healthy.
        prov.config["displacement"] = opt.check_displacement()
        # A rank's result leaves with the rank. The driver has no device and no optimizer, so
        # the masters it compares for divergence only exist inside this process. No-op when
        # nothing launched us as a rank.
        launcher.report(opt, params, history, ckpt, prov, plan=fit)
    finally:
        uninstall()
    return {"history": history, "params": params, "optimizer": opt,
            "checkpointer": ckpt, "provenance": prov, "plan": fit,
            "displacement": prov.config["displacement"]}


# One body, both modes. `train=` selects what the optimizer owns; there is no second
# recipe to drift from this one, and no pair of implementations to keep agreeing.
_RECIPES: Dict[str, Callable] = {"default": train_loop}


def recipe(name: str) -> Callable:
    try:
        return _RECIPES[name]
    except KeyError:
        raise KeyError(f"no recipe {name!r}; recipes are {sorted(_RECIPES)}") from None


def names() -> list:
    return sorted(_RECIPES)


def source(name: str) -> str:
    """The recipe's text, ready to paste into a file and edit.

    ``inspect.getsource`` of the same object :func:`recipe` returns, so what you read is what
    ran. A hand-written copy kept beside the implementation would be a second implementation,
    and the whole argument of this module is that a second one drifts.
    """
    return inspect.getsource(recipe(name))
