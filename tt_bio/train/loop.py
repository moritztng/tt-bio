"""Tier 1: ``finetune(...) -> Run``. We own the loop; the objective is a named row.

The cut line is the unit of user authorship, not the amount of configuration: at Tier 1 there
is **no ``for`` over steps in user code**. Cross into Tier 2 and you own the ``for`` statement.
That boundary is one a user can name before they start, which is the property the surveyed
frameworks lack -- HF cuts at quantity of configuration so its boundary lands mid-loop,
Lightning cuts at inversion of control so crossing it costs the whole loop at once.

``finetune`` implements nothing. It resolves a recipe and calls it, so the Tier-1 result and
the Tier-2 program are the same code rather than two implementations that agree today. See
``tt_bio/train/recipes.py`` for why that is the only arrangement whose escape hatch survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import objectives, recipes

__all__ = ["finetune", "Run"]


@dataclass
class Run:
    """What a Tier-1 call returns: the history, the artifacts and the provenance.

    ``provenance`` is not optional and is not a flag. Every run carries the AICLK sampled
    DURING it, the seed, the git sha, the config and the accuracy triple, because a training
    number whose clock was read before the work started is not a measurement of the work.
    """

    recipe: str
    history: list
    params: dict
    optimizer: object
    checkpointer: object
    provenance: object
    plan: object
    displacement: dict
    #: The data-parallel record when the run was one: per-rank timings, the chips the ranks
    #: actually held, and the count of distinct master-weight hashes. ``None`` on one chip.
    dp: Optional[dict] = None

    @property
    def steps(self) -> int:
        return len(self.history)

    @property
    def loss(self) -> Optional[float]:
        return self.history[-1]["loss"] if self.history else None

    @property
    def best(self):
        return self.checkpointer.best()

    def __str__(self) -> str:
        loss = "no steps ran" if self.loss is None else f"loss {self.loss:.6f}"
        out = (f"Run({self.recipe}) {self.steps} steps, {loss}, "
               f"displacement ratio {self.displacement.get('ratio', float('nan')):.4f}\n"
               f"  {self.provenance.summary()}")
        if self.dp:
            d = self.dp
            out += (f"\n  {d['world']} ranks on /dev/tenstorrent {d['nodes']}, "
                    f"{d['distinct_master_sha']} distinct master hash"
                    f"{'' if d['distinct_master_sha'] == 1 else 'es'}, "
                    f"median step {d['median_step_s']:.3f} s, "
                    f"all-reduce {d['comm_bytes'] / 1e6:.1f} MB in "
                    f"{1e3 * (d['median_transfer_s'] or 0):.1f} ms "
                    f"(+{1e3 * (d['median_barrier_wait_s'] or 0):.1f} ms at the barrier)")
        return out


def finetune(forward, dataset, *, out_dir, global_batch, steps, objective="af3",
             recipe="lora", **kw) -> Run:
    """Fine-tune ``forward`` on ``dataset``. One call, no loop, no callables in the objective.

    ``global_batch`` is required and is never derived from the chip count. It is the axis a
    published recipe pins, and a framework that redefines it per box makes every comparison
    against a published number ambiguous.

    ``objective`` is a NAME. That is the cut line, not a style preference: a named row is
    decidable before a device opens, so Tier 0 can pass it through from a flag, and a run's
    log can say which objective it trained against without serialising a closure.

    Everything else forwards to the recipe. ``tt_bio.train.recipes.source(recipe)`` prints the
    body that is about to run, and running that text yourself is Tier 2.
    """
    if objective not in objectives.names():
        raise KeyError(f"no objective {objective!r}; rows are {objectives.names()}. An "
                       f"objective is a name at this tier -- pass a callable at Tier 2, where "
                       f"you own the loop that calls it")
    out = recipes.recipe(recipe)(forward, dataset, out_dir=out_dir,
                                 global_batch=global_batch, steps=steps,
                                 objective=objective, **kw)
    return Run(recipe=recipe, **out)
