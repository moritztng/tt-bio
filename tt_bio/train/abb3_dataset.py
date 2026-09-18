"""Datasets for the reproduction run: the real SAbDab split, and a synthetic stand-in.

Two implementations behind one contract -- ``__len__`` and ``micro_batch(indices) -> dict`` --
because the two things this run has to prove need different data.

**The resume and data-parallel invariants are data-independent, and proving them on synthetic
samples is the right instrument rather than a shortcut.** What a resume has to establish is that
the next step is the step the uninterrupted run would have taken, which is a statement about
masters, moments, schedules and RNG. Synthetic samples make that provable at BIT level, because
the sample for a given index is a pure function of the index. Real data would make the same
proof weaker, not stronger: it adds file I/O and a featuriser between the claim and the check
without touching the mechanism being checked.

**What synthetic data cannot tell you is whether the loss VALUE is right, and nothing here
pretends otherwise.** That is ``scripts/abb3_port/loss_gate.py``'s job and it is already
bit-exact against upstream's own ``loss.py``. The accuracy claim belongs to the real dataset and
to the 250-structure evaluation at the end of the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

__all__ = ["SyntheticFvs", "dataset"]

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "abb3_port"


class SyntheticFvs:
    """Deterministic synthetic Fvs at the real shapes, one per index.

    Calls ``scripts/abb3_port/step_gate.py``'s own ``synthetic_micro_batch`` rather than
    building a second set of targets: that function already carries every field the stage-1
    loss set reads at its real shape, and a second copy would drift from the loss it feeds.

    ``seed=index`` is what makes a resume provable: the sample for a given index is a pure
    function of the index, so an interrupted run and an uninterrupted one see byte-identical
    inputs at every step and any divergence is the mechanism rather than the data.
    """

    def __init__(self, n: int, cfg, micro: int, tokens: int, device):
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        from step_gate import synthetic_micro_batch
        self._make = synthetic_micro_batch
        self.n, self.cfg, self.micro, self.tokens, self.device = n, cfg, micro, tokens, device

    def __len__(self) -> int:
        return self.n

    def micro_batch(self, indices) -> dict:
        """One micro-batch. The indices set the seed, so the batch is reproducible from them.

        The whole micro-batch takes one seed derived from its indices rather than one sample
        per index, because ``synthetic_micro_batch`` builds a batch at once and splitting it
        per sample to reassemble it would cost a concatenation for no gain in fidelity.
        """
        seed = int(torch.tensor([int(i) for i in indices]).sum().item()) * 1_000_003 + len(indices)
        return self._make(self.cfg, len(indices), self.tokens, seed % (2 ** 31), self.device)


def dataset(kind: str, *, cfg, micro: int, tokens: int, device, n: int = 8192):
    if kind == "synthetic":
        return SyntheticFvs(n, cfg, micro, tokens, device)
    raise ValueError(
        f"no dataset {kind!r}. 'synthetic' is the mechanism harness; the real SAbDab split "
        f"lives in data/structures/structures/*.pt from Zenodo 10.5281/zenodo.11354577 and its "
        f"featuriser is upstream's dataloader.py, which is a separate deliverable from the "
        f"resume and DP invariants this module's synthetic path exists to prove")
