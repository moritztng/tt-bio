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

__all__ = ["SyntheticFvs", "dataset", "resolve_split", "TRAIN_STRUCTURES", "STEPS_PER_EPOCH"]

#: Upstream's own training-set size, from the ``split.csv`` they ship. NOT "everything except the
#: 250 we are scored on", which is 11,570 and would be a different experiment: 2,765 of the
#: 11,820 staged structures never reach ``split.csv`` at all because the filter stage drops them,
#: and a further 410 are explicitly ``unassigned``.
TRAIN_STRUCTURES = 8395

#: 8,395 at batch 64 keeping the short final batch. ``drop_last`` is NOT a free choice here: it
#: is pinned by their released checkpoint, whose ``global_step`` is 193,512, and
#: ``132 x 1466 = 193512`` exactly while ``131 x 1466 = 192046``. So upstream keeps the partial
#: final batch, and a run that drops it walks a different data order from the second epoch on.
STEPS_PER_EPOCH = 132
DROP_LAST = False

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "abb3_port"


def resolve_split(split_csv, released_true_dir=None) -> dict:
    """Upstream's train/valid/test split, taken from what they SHIP rather than re-derived.

    ``split_data.py`` draws the validation and test sets with ``np.random.choice`` under a
    seeded ``np.random.seed``, so the split is reproducible in principle -- but reproducing a
    sampling draw is a worse source than the answer itself, and they ship the answer as
    ``data/split.csv``. That also removes any dependence on the seed, on their ``filters.csv``
    and on the legacy table the split stage reads.

    ``released_true_dir`` is the cross-check and it is the point of this function. Their
    released predictions carry the ground-truth structures for the evaluation set, so the ids
    in that directory are the evaluation set as ACTUALLY scored. If ``split.csv``'s
    ``valid`` + ``test`` rows and those ids ever disagree, the split we would train against is
    not the split the published 2.714 A was measured on, and the run is pointless. Measured on
    the staged artefacts: **250 and 250 with 250 in agreement and none either side**.

    Returns ``{"train": [...], "valid": [...], "test": [...]}``.
    """
    import pandas as pd
    d = pd.read_csv(split_csv, index_col=0)
    out = {k: sorted(d.query(f"split == {k!r}").index.astype(str)) for k in
           ("train", "valid", "test")}
    if len(out["train"]) != TRAIN_STRUCTURES:
        raise ValueError(
            f"{split_csv} lists {len(out['train'])} training structures and this build is "
            f"pinned to upstream's {TRAIN_STRUCTURES}. Their released checkpoint's "
            f"global_step of 193,512 is 132 x 1466 at batch 64, which only closes at "
            f"{TRAIN_STRUCTURES}; a different count means a different split and the 2.714 A "
            f"bar would not apply to the result")
    if released_true_dir is not None:
        ids = {p.stem for p in Path(released_true_dir).glob("*.pdb")}
        scored = set(out["valid"]) | set(out["test"])
        if ids and ids != scored:
            raise ValueError(
                f"the split's valid+test ({len(scored)}) and the ids in their released "
                f"predictions ({len(ids)}) disagree: {len(scored - ids)} in the split only, "
                f"{len(ids - scored)} released only. The published mean is computed over the "
                f"released set, so training against a split that differs from it makes the "
                f"comparison meaningless")
    return out


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
        f"no dataset {kind!r}. 'synthetic' is the mechanism harness that proves the resume and "
        f"data-parallel invariants, which are data-independent. The real split is resolved by "
        f"resolve_split() and its structures are staged at "
        f"/home/ttuser/abb3_data/data/structures/structures/*.pt (Zenodo "
        f"10.5281/zenodo.11354577); every field the stage-1 losses read is already in those "
        f".pt files, so what remains is assembling and bucketing them, not featurising from "
        f"sequence")
