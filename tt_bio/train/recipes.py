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

from . import objectives, provenance
from ..autograd import backward, install, uninstall
from .sharding import batches
from .checkpoint import Checkpointer
from .lora import LoraConfig, attach, lora_factors_for
from .mesh import Mesh
from .optim import AdamW, af3_lr
from .dryrun import plan
from .tensors import to_device, to_host

__all__ = ["source", "names", "recipe", "lora_finetune"]


def lora_finetune(forward, dataset, *, out_dir, global_batch, steps, objective="af3",
                  mesh=None, lora=None, seed=0, lr=3e-4, warmup_steps=1000,
                  checkpoint_every=100, tokens=None, weights=None):
    """LoRA fine-tuning on a frozen trunk. The Tier-1 default, and a Tier-2 program.

    ``forward(batch) -> {name: device tensor}`` is the SHIPPED forward, handed in rather than
    constructed here: the training path calls the same forward inference calls, and the way to
    guarantee that is to not have a second one to call.

    ``dataset`` needs ``__len__`` and ``batch(indices) -> dict`` carrying the labels the
    objective row names. No featurizer is imposed -- per-model featurisation is the one thing
    the campaign refused to generalise, because each family's cropping and MSA handling is
    exactly the part that is genuinely different.
    """
    dp = (mesh or Mesh({"dp": [0]})).axis("dp")
    if dp.width > 1:
        raise NotImplementedError(
            f"axis {dp} is {dp.width} chips wide and this recipe is a single process, so it "
            f"only ever holds one replica's gradient. Data parallelism needs one process per "
            f"chip and each of them handing its own gradient to opt.step(replicas=...), which "
            f"is the interface -- the optimizer already refuses an unreduced step. The launcher "
            f"for it is not built, and running this recipe on a wide axis would train one "
            f"replica and call it four")
    cfg = lora or LoraConfig()
    row = objectives.objective(objective)

    fit = plan(tokens=tokens or dataset.tokens, chips=dp.width,
               global_batch=global_batch, frozen_trunk=True)
    if fit.verdict == "refused":
        raise RuntimeError(f"refusing to start: {fit}")

    install()
    try:
        # One census pass on a real batch discovers the adaptable sites. It costs an
        # inference, and it is how the adapter set comes from the forward instead of from a
        # list somebody maintains by hand.
        plan_order = batches(len(dataset), global_batch=global_batch, steps=steps, seed=seed,
                             data_parallel=dp)
        first = next(plan_order)
        factors = lora_factors_for(forward, cfg, dataset.device, dataset.batch(first.indices))
        params = {f"{site}.{which}": t
                  for site, pair in factors.items()
                  for which, t in zip(("A", "B"), pair)}
        opt = AdamW(params, lr=lr, data_parallel=dp,
                    schedule=lambda s: af3_lr(s, lr, warmup_steps=warmup_steps))
        ckpt = Checkpointer(out_dir, every=checkpoint_every, metric="loss")
        history, last = [], None

        with provenance.during(seed=seed, config={
                "objective": objective, "global_batch": global_batch, "steps": steps,
                "lr": lr, "rank": cfg.rank, "alpha": cfg.alpha, "chips": dp.width,
                "rollout": row.rollout, "sites": sorted(factors)}) as prov:
            with attach(factors, cfg):
                for batch in [first, *plan_order]:
                    opt.zero_grad()
                    data = dataset.batch(batch.indices)
                    outputs = forward(data)
                    total, breakdown, seeds = row(
                        data, {k: to_host(v.value) for k, v in outputs.items()},
                        weights=weights)
                    # One backward over the union of the seeded roots' ancestors. Calling
                    # Tensor.backward once per root would replay every shared ancestor once
                    # per root and land the fan-in sums partial.
                    backward([outputs[k] for k in seeds],
                             [to_device(g, dataset.device) for g in seeds.values()])
                    opt.step()
                    last = {"step": batch.step, "loss": total, "breakdown": breakdown,
                            "grad_norm": opt.last_grad_norm, "lr": opt.last_lr}
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
    finally:
        uninstall()
    return {"history": history, "params": params, "optimizer": opt,
            "checkpointer": ckpt, "provenance": prov, "plan": fit,
            "displacement": prov.config["displacement"]}


_RECIPES: Dict[str, Callable] = {"lora": lora_finetune}


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
