"""``plan()`` -- the dry run. Will this fit, and how long will it take.

Every number here was measured on our own hardware and carries where it was measured. That
is the whole point of the module: the five perf campaigns that died in this repo died on
premises taken from uncalibrated sources, so a planner that guesses is worse than no planner
at all. A configuration this module has no measurement for comes back ``UNMEASURED`` with the
reason, and ``UNMEASURED`` is a first-class answer rather than an error.

Where a number was measured is load-bearing rather than decorative, and this module learned
that the expensive way. It used to refuse Protenix's own 384-token crop on an OOM taken against
a differentiable TWIN of the pair track, a module that has since been deleted. The allocation
that twin was refused, 75,497,472 B, is byte for byte one ``[1,384,384,256]`` bf16 pair tensor;
the shipped forward allocates that tensor and 37 more of the same block's intermediates inside
2.283 GB and peaks at 0.877 GB of 34.23 GB over all 48 blocks. The wall was the twin, not the
card, and the refusal outlived the module it described by two campaigns. So every entry in
every table below names its own source, and :func:`provenance_gap` is what keeps that true.

The other line this module holds is between what is measured and what is inferred from it. A
trained trunk's RETENTION is measured; the peak of a real backward is not, because no taped
pairformer exists to run one. Reporting the first in the shape of the second is how a bound
becomes a commitment, so ``plan()`` reports the gigabytes and withholds the verdict.

No ttnn import, no device. ``plan()`` has to answer before a card opens, because Tier 0's cut
line is that a flag's legality is decidable without one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["plan", "Plan", "UNMEASURED", "CARD_DRAM_BYTES", "MEASURED", "FORWARD_FITS",
           "FORWARD_OOM", "TRAINED_TRUNK_BOUND", "provenance_gap"]


UNMEASURED = "UNMEASURED"

# Card DRAM, read live off a qb1 Blackhole chip as 8 banks x 4,278,190,016 B
# (train-r5-distributed-tt, REPLICA-FITS). Not the board spec, the allocator's own budget.
CARD_DRAM_BYTES = 8 * 4_278_190_016
CARD_DRAM_GB = CARD_DRAM_BYTES / 1e9          # 34.23 GB

# Everything below is measured, and each entry names its source. `where` is what makes a
# number auditable a year from now; a number without one does not belong in this table.
MEASURED = {
    "replica_256aa_gb": (5.06, "train-r5 REPLICA-FITS: 0.929 GB bf16 weights over 464,442,431 "
                               "censused parameters + 0.429 GB fp32 trunk gradients + 3.70 GB "
                               "activations at the backward sync point, per-block checkpointing"),
    "replica_256aa_opt_resident_gb": (10.63, "train-r5: the same replica with masters and both "
                                             "Adam moments in DRAM. They are host-side today "
                                             "because ttnn.moreh_adamw refuses an fp32 master"),
    "dp2_speedup": (1.87, "train-r5 THROUGHPUT: 8.08 tokens/s on one chip, 15.11 on two, qb1 "
                          "chips 1 and 3, 1350 MHz sampled during on both"),
    "dp2_efficiency": (0.935, "train-r5: the 6.5 % that does not scale is load imbalance, not "
                              "bandwidth"),
}

# A source has to be something a reader can open a year from now: a repo path, a state doc or a
# commit. Everything else reads like provenance without being any.
_CITES = re.compile(r"state/[\w./-]+|[\w./-]+\.(?:py|md|json|txt|yaml)"
                    r"|\b(?=[0-9a-f]{7,40}\b)[0-9a-f]*[a-f][0-9a-f]*\b")


def provenance_gap(where):
    """``None`` if ``where`` cites something openable, otherwise why it does not.

    This exists because of what it would have caught. ``FORWARD_OOM``'s 384 and 512 entries were
    two bare tuples under a shared comment, and when the module they were measured on was
    deleted the entries stayed, refusing the recipe's own crop for another two campaigns. A
    per-entry citation is what makes that visible: a number whose source names a file can be
    checked against the file, and a number that names nothing cannot be checked at all.
    """
    if not isinstance(where, str) or not where.strip():
        return "no source at all"
    if not _CITES.search(where):
        return f"names no file, state doc or commit: {where[:60]!r}"
    return None


# Peak device DRAM of the SHIPPED 48-block pairformer at the checkpoint's own widths (c_z 256,
# 8 triangle heads of 32), forward only and untaped -- which is the state per-block
# checkpointing runs the forward in, so this is the figure the crop question turns on.
FORWARD_FITS = {
    256: (0.699, "ptx-crop measured 699,301,888 B on qb1 card 1 (wk/ptx-crop 9c567b590, "
                 "state/concluded/ptx-crop); train-x-cropunblock reproduced 699,318,272 B on "
                 "qb1 card 2, perf/train_x_crop/out/trunk48_qb1c2.json"),
    384: (0.877, "ptx-crop measured 877,076,480 B on qb1 card 1 (wk/ptx-crop 9c567b590, "
                 "state/concluded/ptx-crop); train-x-cropunblock reproduced 877,092,864 B on "
                 "qb1 card 2, 16 KB apart, perf/train_x_crop/out/trunk48_qb1c2.json. Both at "
                 "1350 MHz median sampled during. This is the crop Protenix's own recipe uses"),
    512: (1.055, "ptx-crop measured 1,055,121,408 B on qb1 card 1 (wk/ptx-crop 9c567b590, "
                 "state/concluded/ptx-crop); train-x-cropunblock reproduced 1,055,154,176 B on "
                 "qb1 card 2, perf/train_x_crop/out/trunk48_qb1c2.json"),
    640: (1.284, "ptx-crop, 1,284,096,000 B on qb1 card 1, wk/ptx-crop 9c567b590, "
                 "state/concluded/ptx-crop. Not re-measured by train-x-cropunblock"),
    768: (1.614, "ptx-crop, 1,614,217,216 B on qb1 card 1, wk/ptx-crop 9c567b590, "
                 "state/concluded/ptx-crop. Not re-measured by train-x-cropunblock"),
}

# What a TRAINED trunk RETAINS at a crop under per-block checkpointing, and an upper bound
# rather than a peak. Four terms: the 48 block-boundary (z, s) pairs and one block's inner set
# are read off the allocator on the card, the weight and gradient terms are the checkpoint's own
# censused parameter counts (464,442,431 whole model x 2 B bf16, 227,960,832 trunk x 4 B fp32).
# What is NOT in it is the backward's own working set inside the block being recomputed, because
# no taped pairformer exists to allocate one. That missing term is why plan() reports these
# gigabytes and still answers UNMEASURED.
TRAINED_TRUNK_BOUND = {
    256: (4.074, "1.620 GB boundaries + 0.613 GB inner set, both measured; ptx-crop on qb1 "
                 "card 1, wk/ptx-crop 4ab2aad2a, state/concluded/ptx-crop"),
    384: (7.762, "3.638 GB boundaries + 2.283 GB inner set (38 DRAM intermediates), both "
                 "measured; ptx-crop on qb1 card 1 (wk/ptx-crop 4ab2aad2a) and reproduced byte "
                 "for byte by train-x-cropunblock on qb1 card 2, 3,638,034,432 B and "
                 "2,283,307,008 B, perf/train_x_crop/out/tape_qb1c2.json, 1350 MHz sampled "
                 "during. Replaces the 27.58 GB projection this module used to refuse on"),
    512: (13.047, "6.461 GB boundaries + 4.745 GB inner set, both measured; ptx-crop on qb1 "
                  "card 1, wk/ptx-crop 4ab2aad2a, state/concluded/ptx-crop"),
}

# The crop sizes whose forward is MEASURED to OOM: allocated GB, bytes refused, and where the
# measurement was taken. Empty, and kept rather than deleted, because the mechanism is right and
# it was the entries that were wrong -- refusing on a measurement beats extrapolating a slope
# through two failures. 384 and 512 sat here on the deleted twin's numbers (see the module
# docstring) until both were measured to PASS on the shipped forward; they are in FORWARD_FITS
# now. Any future entry carries its own `where`, which the two that left did not.
FORWARD_OOM: dict = {}

# The largest crop with a measured training replica. Nothing above this has one.
MEASURED_CROP = 256


@dataclass
class Plan:
    """What ``plan()`` answers. ``fits`` and ``seconds_per_step`` are ``None`` when UNMEASURED.

    ``verdict`` is one of ``"fits"``, ``"refused"`` or ``UNMEASURED``, and ``why`` carries the
    measurement or the missing measurement behind it. A caller that only prints ``verdict``
    still cannot mistake a projection for a number, because there is no number to print.
    """

    verdict: str
    why: str
    tokens: int
    chips: int
    global_batch: Optional[int] = None
    fits: Optional[bool] = None
    replica_gb: Optional[float] = None
    card_gb: float = CARD_DRAM_GB
    occupancy: Optional[float] = None
    seconds_per_step: Optional[float] = None
    sources: list = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.verdict != UNMEASURED

    def __str__(self) -> str:
        head = f"plan: {self.verdict} at {self.tokens} tokens on {self.chips} chip(s)"
        if self.replica_gb is not None:
            head += (f" -- {self.replica_gb:.2f} GB of {self.card_gb:.2f} GB"
                     f" ({self.occupancy * 100:.1f} %)")
        if self.seconds_per_step is not None:
            head += f", {self.seconds_per_step:.2f} s/step"
        return f"{head}\n  {self.why}" + "".join(f"\n  - {s}" for s in self.sources)


def plan(*, tokens: int, chips: int = 1, global_batch: Optional[int] = None,
         frozen_trunk: bool = True, optimizer_resident: bool = False,
         seconds_per_step_1chip: Optional[float] = None) -> Plan:
    """Will a run of this shape fit, and how long is a step.

    ``frozen_trunk=False`` means full fine-tuning. It comes back UNMEASURED, but with the
    measured retention bound in the reason rather than a projection: what is unmeasured is the
    backward's own working set, not the whole arithmetic. A crop whose FORWARD is measured to
    OOM is refused before that, because the forward is what both modes run and "we measured
    this failing" is a different answer from "we have no measurement".

    ``seconds_per_step_1chip`` is the caller's own measurement of a single-chip step. Given
    one, the multi-chip step time is that number divided by the MEASURED two-chip speedup;
    without one there is no step time to report, because we have not measured a Protenix-v2
    training step on this hardware and dividing a projection by 1.87 keeps it a projection.
    """
    if chips < 1:
        raise ValueError(f"chips must be at least 1, got {chips}")
    if global_batch is not None and global_batch < 1:
        raise ValueError(f"global_batch must be at least 1, got {global_batch}")

    if tokens in FORWARD_OOM:
        allocated, refused, where = FORWARD_OOM[tokens]
        return Plan(
            verdict="refused", tokens=tokens, chips=chips, global_batch=global_batch,
            fits=False,
            why=f"the forward OOMs at {tokens} aa today: {allocated:.2f} GB allocated and "
                f"{refused:,} B refused, measured under per-block checkpointing where the "
                f"forward is untaped, so what fails is one block's working set. Distribution "
                f"does not fix it -- 8 chips each OOM at {tokens} aa exactly as one does. "
                f"A refusal, not an estimate.",
            sources=[where])

    if not frozen_trunk:
        bound = TRAINED_TRUNK_BOUND.get(tokens)
        crops = ", ".join(str(k) for k in sorted(TRAINED_TRUNK_BOUND))
        if bound is None:
            return Plan(
                verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
                why=f"full fine-tuning: a trained trunk's cost is measured at {crops} aa and "
                    f"not at {tokens} aa, and both of its measured terms move with the crop "
                    f"across the triangle ops' chunking thresholds. Measure it.",
                sources=[f"TRAINED_TRUNK_BOUND covers {crops} aa"])
        gb, src = bound
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why=f"full fine-tuning: a trained trunk RETAINS {gb:.3f} GB of "
                f"{CARD_DRAM_GB:.2f} GB ({gb / CARD_DRAM_GB * 100:.1f} %) at {tokens} aa under "
                f"per-block checkpointing, measured on the card. That is a bound on retention "
                f"and not a peak: no taped pairformer exists yet, so the backward's own working "
                f"set inside the block being recomputed has never been allocated and is not in "
                f"the figure. UNMEASURED is that one missing term rather than the whole "
                f"arithmetic, and memory is no longer the reason to expect this not to fit.",
            sources=[src])

    if tokens > MEASURED_CROP:
        fwd = FORWARD_FITS.get(tokens)
        sources = [f"MEASURED replica exists only at {MEASURED_CROP} aa"]
        if fwd is None:
            forward = (f"The forward has no measurement at {tokens} aa either; it is measured "
                       f"to fit at {', '.join(str(k) for k in sorted(FORWARD_FITS))} aa.")
        else:
            forward = (f"The forward is not the obstacle: it is measured to fit at {tokens} aa, "
                       f"peaking at {fwd[0]:.3f} GB of {CARD_DRAM_GB:.2f} GB "
                       f"({fwd[0] / CARD_DRAM_GB * 100:.1f} %) over all 48 blocks.")
            sources.append(f"forward: {fwd[1]}")
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why=f"{tokens} tokens is above the largest crop with a measured training replica "
                f"({MEASURED_CROP} aa). {forward} What is missing is the replica's activation "
                f"term at this crop: activation volume is neither linear nor quadratic in "
                f"tokens across the triangle ops' chunking thresholds, so scaling the "
                f"{MEASURED_CROP} aa replica up would be a guess with a plausible shape. "
                f"Measure it.",
            sources=sources)

    replica_gb, replica_src = MEASURED[
        "replica_256aa_opt_resident_gb" if optimizer_resident else "replica_256aa_gb"]
    if tokens < MEASURED_CROP:
        # A smaller crop cannot need more than the measured one: same weights, same optimizer
        # state, strictly fewer activations. So the 256 aa figure is a sound upper bound below
        # 256 and is reported as one, not scaled down -- scaling it down would be the guess.
        why = (f"fits, bounded above by the measured {MEASURED_CROP} aa replica. {tokens} aa "
               f"carries the same weights and optimizer state and strictly fewer activations, "
               f"so {replica_gb:.2f} GB is an upper bound rather than an estimate for this "
               f"size. The exact figure at {tokens} aa is not measured.")
    else:
        why = f"fits, at the measured {MEASURED_CROP} aa replica size."

    occupancy = replica_gb / CARD_DRAM_GB
    sources = [f"replica: {replica_src}"]

    sps = None
    if seconds_per_step_1chip is not None:
        if seconds_per_step_1chip <= 0:
            raise ValueError("seconds_per_step_1chip must be positive")
        speedup, speed_src = _dp_speedup(chips)
        if speedup is None:
            sources.append(speed_src)
        else:
            sps = seconds_per_step_1chip / speedup
            sources.append(speed_src)
    elif chips > 1:
        sources.append("no step time: pass seconds_per_step_1chip from your own measurement. "
                       "We have not measured a Protenix-v2 training step on this hardware")

    return Plan(verdict="fits", why=why, tokens=tokens, chips=chips,
                global_batch=global_batch, fits=True, replica_gb=replica_gb,
                occupancy=occupancy, seconds_per_step=sps, sources=sources)


def _dp_speedup(chips: int):
    """The measured DP speedup, or ``(None, why)`` above where it was measured.

    Two chips is 1.87x at 93.5 % efficiency and that is the only multi-chip point we have. The
    tempting move is to carry 93.5 % forward as a per-chip efficiency and report 3.74x at 4
    chips; it is refused here for the reason the campaign already wrote down twice -- two
    points cannot measure a scaling exponent, and r5 named shard balance and the host reduce's
    O(N) per-rank volume as the terms that grow with rank count. They do not show up at N=2.
    """
    if chips == 1:
        return 1.0, "single chip: no collective, speedup 1.0 by definition"
    if chips == 2:
        return MEASURED["dp2_speedup"][0], f"DP speedup: {MEASURED['dp2_speedup'][1]}"
    return None, (f"{chips}-chip step time is {UNMEASURED}: 1.87x on two chips is the only "
                  f"multi-chip point measured, and extrapolating 93.5 % efficiency per chip "
                  f"assumes the host reduce's O(N) per-rank volume and shard imbalance stay "
                  f"flat in rank count, which is exactly what is unmeasured")
