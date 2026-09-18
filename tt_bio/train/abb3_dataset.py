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

__all__ = ["SyntheticFvs", "SabdabFvs", "dataset", "resolve_split",
           "TRAIN_STRUCTURES", "STEPS_PER_EPOCH", "bucket_tokens"]

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


def dataset(kind: str, *, cfg, micro: int, tokens: int, device, n: int = 8192,
            ids=None, root=None):
    if kind == "synthetic":
        return SyntheticFvs(n, cfg, micro, tokens, device)
    if kind == "sabdab":
        if ids is None or root is None:
            raise ValueError("the sabdab dataset needs ids= from resolve_split() and root= "
                             "pointing at structures/structures/")
        return SabdabFvs(ids, root, cfg, device, tokens=tokens)
    raise ValueError(
        f"no dataset {kind!r}. 'synthetic' is the mechanism harness that proves the resume and "
        f"data-parallel invariants, which are data-independent. The real split is resolved by "
        f"resolve_split() and its structures are staged at "
        f"/home/ttuser/abb3_data/data/structures/structures/*.pt (Zenodo "
        f"10.5281/zenodo.11354577); every field the stage-1 losses read is already in those "
        f".pt files, so what remains is assembling and bucketing them, not featurising from "
        f"sequence")


# --------------------------------------------------------------------- the real split

#: Padded token axes are rounded up to this, the standing rule for a tiled device: a token count
#: that is not a multiple of 32 leaves a ragged tile whose tail the kernels handle differently.
TOKEN_MULTIPLE = 32

#: What each target is padded WITH, and the two that are not zero are the interesting ones.
#: A padded residue is masked out of every loss, so the values below can only matter through an
#: intermediate computed before the mask is applied -- which is exactly where a degenerate value
#: turns into a NaN that the mask then cannot remove, because 0 * NaN is NaN.
#:
#: * ``rigidgroups_*_frames`` and ``backbone_rigid_tensor`` get the IDENTITY, not zeros. A 4x4
#:   with a zero 3x3 block is not a rotation, and FAPE inverts these frames before it multiplies
#:   by the mask.
#: * ``chi_angles_sin_cos`` gets ``[1, 0]``, a unit vector, not ``[0, 0]``. The angle terms
#:   normalise their argument, and a zero vector normalises to NaN.
_IDENTITY_FRAMES = ("backbone_rigid_tensor", "rigidgroups_gt_frames",
                    "rigidgroups_alt_gt_frames")


def bucket_tokens(n: int, multiple: int = TOKEN_MULTIPLE) -> int:
    """Round a token count up to a whole number of tiles."""
    return max(multiple, -(-int(n) // multiple) * multiple)


class SabdabFvs:
    """Upstream's own per-structure tensors, assembled into the micro-batch B2's step reads.

    **Nothing is featurised from sequence here, because nothing needs to be.** Their
    ``structures/*.pt`` already carry every field the stage-1 losses read -- the rigid-group
    frames, both atom14 naming alternatives, the chi targets, the CDR mask -- at upstream's own
    shapes, produced by upstream's own pipeline. Re-deriving any of it would be re-implementing
    their featuriser and inviting a silent disagreement with the checkpoint we are reproducing.
    What is left is stacking, padding and the two input feature maps, which come from B2's
    ``single_and_pair_features``.

    **Padding is to a multiple of 32 and is masked everywhere**, and the two non-zero pad values
    above are load-bearing rather than defensive: ``0 * NaN`` is ``NaN``, so a mask cannot remove
    a degenerate value that was already computed. Frames pad to the identity and chi targets to a
    unit vector for that reason.

    ``residue_index`` continues past the real residues rather than resetting to zero. The
    relative-position feature is a clamped difference, so a reset would put padded columns at a
    large negative offset from the real ones; continuing keeps them adjacent, and the square mask
    removes them from attention either way.
    """

    def __init__(self, ids, root, cfg, device, *, tokens: int | None = None):
        self.ids = list(ids)
        self.root = Path(root)
        self.cfg, self.device = cfg, device
        #: Fixed token axis when given, else per-micro-batch bucketing off the longest member.
        #: Fixed costs one compile and wastes work on short Fvs; bucketing costs one compile per
        #: distinct bucket and there are few, since the Fv length distribution is narrow (median
        #: 230, max 281 over 11,820).
        self.tokens = tokens
        missing = [i for i in self.ids[:64] if not (self.root / f"{i}.pt").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of the first 64 ids have no .pt under {self.root}, e.g. "
                f"{missing[:3]}. The structures come from data.tar.gz of Zenodo "
                f"10.5281/zenodo.11354577")

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, index: int) -> dict:
        return torch.load(self.root / f"{self.ids[int(index)]}.pt", weights_only=False)

    def micro_batch(self, indices) -> dict:
        """Stack, pad and featurise. Returns the dict ``TrainStep.micro_batch`` reads."""
        from ..abodybuilder3 import to_device_fp32
        from ..abodybuilder3_reference import single_and_pair_features

        raw = [self.load(i) for i in indices]
        b = len(raw)
        n_tok = self.tokens or bucket_tokens(max(int(d["aatype"].shape[0]) for d in raw))
        if any(int(d["aatype"].shape[0]) > n_tok for d in raw):
            longest = max(int(d["aatype"].shape[0]) for d in raw)
            raise ValueError(
                f"an Fv of {longest} tokens does not fit the fixed axis of {n_tok}. Cropping it "
                f"would be a different experiment than the one being reproduced -- raise "
                f"tokens= or leave it None to bucket per batch")

        def pad(key, fill=0.0, dtype=None):
            """Stack field ``key`` across the batch, padded to ``n_tok`` on the residue axis."""
            out = None
            for j, d in enumerate(raw):
                t = d[key]
                if out is None:
                    shape = (b, n_tok) + tuple(t.shape[1:])
                    out = torch.full(shape, float(fill),
                                     dtype=dtype or (t.dtype if t.is_floating_point()
                                                     else torch.int64))
                out[j, :t.shape[0]] = t.to(out.dtype)
            return out

        aatype = pad("aatype", fill=UNKNOWN_AATYPE, dtype=torch.int64)
        seq_mask = pad("seq_mask", dtype=torch.float32)
        is_heavy = pad("is_heavy", dtype=torch.int64)
        # Continued, not reset: see the class docstring.
        residue_index = pad("residue_index", dtype=torch.int64)
        for j, d in enumerate(raw):
            n = int(d["aatype"].shape[0])
            if n < n_tok:
                last = int(d["residue_index"][-1])
                residue_index[j, n:] = torch.arange(last + 1, last + 1 + (n_tok - n))
        single, pair = single_and_pair_features(aatype, is_heavy, residue_index)

        targets = {
            "aatype": aatype,
            "seq_mask": seq_mask,
            "chi_mask": pad("chi_mask", dtype=torch.float32),
            "chi_angles_sin_cos": _pad_unit_sin_cos(raw, "chi_angles_sin_cos", b, n_tok),
            "backbone_rigid_mask": pad("backbone_rigid_mask", dtype=torch.float32),
            "rigidgroups_gt_exists": pad("rigidgroups_gt_exists", dtype=torch.float32),
            "atom14_gt_positions": pad("atom14_gt_positions", dtype=torch.float32),
            "atom14_alt_gt_positions": pad("atom14_alt_gt_positions", dtype=torch.float32),
            "atom14_gt_exists": pad("atom14_gt_exists", dtype=torch.float32),
            "atom14_alt_gt_exists": pad("atom14_alt_gt_exists", dtype=torch.float32),
            "atom14_atom_is_ambiguous": pad("atom14_atom_is_ambiguous", dtype=torch.float32),
            "atom14_atom_exists": pad("atom14_atom_exists", dtype=torch.float32),
            "cdr_mask": pad("cdr_mask", dtype=torch.int64),
            "resolution": torch.tensor([float(d["resolution"]) for d in raw]),
        }
        for key in _IDENTITY_FRAMES:
            targets[key] = _pad_identity_frames(raw, key, b, n_tok)

        square = (seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)).unsqueeze(1)
        return {
            "device": self.device,
            "single_d": to_device_fp32(single),
            "pair_d": to_device_fp32(pair),
            "square_d": to_device_fp32(square),
            "bias_d": to_device_fp32(self.cfg.inf * (square - 1.0)),
            "aatype": aatype,
            "targets": targets,
            "ids": [self.ids[int(i)] for i in indices],
            "tokens": n_tok,
        }


#: Upstream's unknown/gap residue type. Padded positions take it so the 21-way one-hot stays
#: legal; they are removed from attention by the square mask and from every loss by seq_mask.
UNKNOWN_AATYPE = 20


def _pad_identity_frames(raw, key, b, n_tok):
    """Stack 4x4 frame fields, padding with the IDENTITY rather than zeros."""
    sample = raw[0][key]
    out = torch.zeros((b, n_tok) + tuple(sample.shape[1:]), dtype=torch.float32)
    out[..., 0, 0] = out[..., 1, 1] = out[..., 2, 2] = out[..., 3, 3] = 1.0
    for j, d in enumerate(raw):
        t = d[key]
        out[j, :t.shape[0]] = t.to(torch.float32)
    return out


def _pad_unit_sin_cos(raw, key, b, n_tok):
    """Stack (sin, cos) targets, padding with the unit vector [1, 0] rather than [0, 0]."""
    sample = raw[0][key]
    out = torch.zeros((b, n_tok) + tuple(sample.shape[1:]), dtype=torch.float32)
    out[..., 0] = 1.0
    for j, d in enumerate(raw):
        t = d[key]
        out[j, :t.shape[0]] = t.to(torch.float32)
    return out
