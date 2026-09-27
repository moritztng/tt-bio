"""``batches`` -- global batch in, per-chip micro-batches out, and the sharding.

**``global_batch`` is required and is never derived from device count.** It is the one axis a
published recipe pins, so it is the last thing that should change when you move to a bigger
box. ``accelerate`` redefines it in terms of device count at ``data_loader.py:347-348`` and
its scheduler follows at ``scheduler.py:69-76``, which means the same script trains a
different recipe on a different machine and the difference never appears in the config. Here
the arithmetic runs the other way: you pin the global batch and the chip count decides the
micro-batch, and a global batch that does not divide by the axis width is an error rather
than a rounding.

No featurizer, deliberately. r3's advice and the biggest trap in the design: a shared data
layer across AF2, Protenix, OpenFold3, Boltz and BoltzGen would have to model every family's
cropping and MSA handling, and each family's is the part that is genuinely different. This
module shards and orders indices. Turning an index into features is the model's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from .mesh import Axis

__all__ = ["batches", "Batch"]


@dataclass(frozen=True)
class Batch:
    """One optimizer step's worth of work, already split per chip.

    ``per_chip`` is one list of dataset indices per chip on the axis, so a caller reads
    ``batch.per_chip[rank]`` and never has to know the global batch to find its own slice.
    ``step`` counts optimizer steps, not micro-batches: it is the number a schedule and a
    checkpoint cadence are written against, and off-by-a-factor-of-chips is the bug that
    makes a warmup end at the wrong place.
    """

    step: int
    epoch: int
    per_chip: tuple
    global_batch: int

    @property
    def micro_batch(self) -> int:
        return len(self.per_chip[0])

    @property
    def indices(self) -> list:
        return [i for chunk in self.per_chip for i in chunk]


def batches(n_examples: int, *, global_batch: int, steps: Optional[int] = None,
            epochs: Optional[int] = None, seed: int = 0, shuffle: bool = True,
            data_parallel: Optional[Axis] = None,
            drop_last: bool = True) -> Iterator[Batch]:
    """Deterministic index batches, sharded across the data-parallel axis.

    Exactly one of ``steps`` and ``epochs``. Both is refused: they are two ways of saying the
    same number and a run given both would silently obey one, which makes every comparison
    against a published recipe ambiguous.

    Determinism is per epoch off ``seed``, so a run resumed at step k sees the same order it
    would have seen without the interruption. Seeding once and consuming the generator would
    not survive a restart.
    """
    if n_examples < 1:
        raise ValueError(f"n_examples must be at least 1, got {n_examples}")
    if global_batch < 1:
        raise ValueError(f"global_batch must be at least 1, got {global_batch}")
    if (steps is None) == (epochs is None):
        raise ValueError("pass exactly one of steps= or epochs=; both name the same quantity "
                         "and a run given both obeys one of them silently")
    width = 1 if data_parallel is None else data_parallel.width
    if global_batch % width:
        raise ValueError(
            f"global_batch {global_batch} does not divide by the {width} chips on axis "
            f"{data_parallel}. Rounding it would change the recipe on a box with a different "
            f"chip count, which is the substitution this module exists to refuse -- pick a "
            f"global batch that divides, or a narrower axis")
    micro = global_batch // width
    if drop_last and n_examples < global_batch:
        raise ValueError(f"{n_examples} examples is less than one global batch of "
                         f"{global_batch}; pass drop_last=False to train on a partial batch")

    import numpy as np
    per_epoch = (n_examples // global_batch if drop_last
                 else -(-n_examples // global_batch))
    total = steps if steps is not None else epochs * per_epoch
    step = 0
    epoch = 0
    while step < total:
        order = (np.random.default_rng([seed, epoch]).permutation(n_examples)
                 if shuffle else np.arange(n_examples))
        for b in range(per_epoch):
            if step >= total:
                return
            chunk = order[b * global_batch:(b + 1) * global_batch].tolist()
            # A short final batch is padded by wrapping the epoch's own order rather than by
            # repeating its last example: repeating one example weights it k times in that
            # step's gradient, and at batch 64 that is a 1/64 bias nobody would find.
            if len(chunk) < global_batch:
                chunk += order[:global_batch - len(chunk)].tolist()
            per_chip = tuple(tuple(chunk[r * micro:(r + 1) * micro]) for r in range(width))
            yield Batch(step=step, epoch=epoch, per_chip=per_chip,
                        global_batch=global_batch)
            step += 1
        epoch += 1
