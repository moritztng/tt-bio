"""``plan()`` -- the dry run. Will this fit, and how long will it take.

Every number here was measured on our own hardware and carries where it was measured. That
is the whole point of the module: the five perf campaigns that died in this repo died on
premises taken from uncalibrated sources, so a planner that guesses is worse than no planner
at all. A configuration this module has no measurement for comes back ``UNMEASURED`` with the
reason, and ``UNMEASURED`` is a first-class answer rather than an error.

The line between the two is not "small versus large". It is **a LoRA adapter on a frozen
trunk, versus anything above it**. Below that line the arithmetic is a replica measured off a
live chip. Above it the only source is ``perf/hall_grad/DECISION.md``, which is a feasibility
memo -- its 27.58 GB of tape and its 40 engineer-days are projections of work that is not
built, and it says so itself. Reporting a projection in the same shape as a measurement is
how a projection becomes a commitment.

No ttnn import, no device. ``plan()`` has to answer before a card opens, because Tier 0's cut
line is that a flag's legality is decidable without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["plan", "Plan", "UNMEASURED", "CARD_DRAM_BYTES", "MEASURED"]


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
    # The four-chip point is ABodyBuilder3's, not Protenix's: it is the model we can run a
    # four-rank step on today. The same ladder's two-chip rung is 1.984x, so the two models'
    # two-chip points agree within 6 % and the ladder is not silently a different scaling law.
    "dp4_speedup": (3.686, "train-w-fourchip LADDER: ABodyBuilder3 on four qb1 p150a chips, "
                           "global batch 64, micro 4, 256 tokens, arms interleaved from cold "
                           "starts, 7.647 s a step against the 28.189 s one-chip cadence of "
                           "the same session; 1350 MHz median and minimum sampled during every "
                           "arm. A second take at three rounds gave 3.742x"),
    "dp4_efficiency": (0.922, "train-w-fourchip ATTRIBUTION: the 7.8 % that does not scale is "
                              "host torch, not the exchange -- the reduce moves 127.87 MB in "
                              "0.293 s, 3.8 % of the step. Conditional on the host's cores "
                              "being divided across the ranks, which tt_bio/train/launcher.py "
                              "now does by default: with torch's own default width every rank "
                              "claims all 16 cores and the same step takes 931 s"),
}

# The crop sizes whose forward is measured to OOM, with the allocation that was refused.
# train-r5 measured both; PLAN.md P2 corrected r5's causal attribution (production's trimul
# chunking is built and the training path simply does not get it) and scheduled A2 to
# re-measure on the shipped forward. The OOM itself stands, so plan() refuses on it -- and
# refuses by naming the measurement rather than by extrapolating a slope through it.
FORWARD_OOM = {
    384: (4.14, 75_497_472),
    512: (7.15, 536_870_912),
}

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

    ``frozen_trunk=False`` means full fine-tuning, which is the line this module will not
    cross: the memory arithmetic for a trained trunk exists only as a projection, so it comes
    back UNMEASURED at any crop the forward itself fits at. A crop whose FORWARD is measured
    to OOM is refused before that, because the forward is what both modes run and "we measured
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
        allocated, refused = FORWARD_OOM[tokens]
        return Plan(
            verdict="refused", tokens=tokens, chips=chips, global_batch=global_batch,
            fits=False,
            why=f"the forward OOMs at {tokens} aa today: {allocated:.2f} GB allocated and "
                f"{refused:,} B refused, measured under per-block checkpointing where the "
                f"forward is untaped, so what fails is one block's working set. Distribution "
                f"does not fix it -- 8 chips each OOM at {tokens} aa exactly as one does. "
                f"Under re-measure by the A2 crop row on the shipped forward; until that "
                f"lands this is a refusal, not an estimate.",
            sources=["train-r5-distributed-tt REPLICA-FITS -- measured",
                     "state/train/PLAN.md P2 -- r5's attribution corrected, the OOM stands"])

    if not frozen_trunk:
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why="full fine-tuning: the only source for a trained trunk's tape is "
                "perf/hall_grad/DECISION.md, a feasibility memo whose 27.58 GB (15.76 GB of "
                "tape + 11.82 GB live, per-block checkpointing, at 800 aa) and 40 engineer-day "
                "estimate are projections of unbuilt work. Measured planning stops at a LoRA "
                "adapter on a frozen trunk.",
            sources=["perf/hall_grad/DECISION.md:38,107 -- projection, not measurement"])

    if tokens > MEASURED_CROP:
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why=f"{tokens} tokens is above the largest crop with a measured training replica "
                f"({MEASURED_CROP} aa) and is not one of the two sizes whose forward OOM was "
                f"measured ({', '.join(str(k) for k in sorted(FORWARD_OOM))} aa). Activation "
                f"volume is neither linear nor quadratic in tokens across the triangle ops' "
                f"chunking thresholds, so interpolating between 256 and 384 would be a guess "
                f"with a plausible shape. Measure it.",
            sources=[f"MEASURED replica exists only at {MEASURED_CROP} aa"])

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
    """The measured DP speedup, or ``(None, why)`` at a width nobody has measured.

    One, two and four chips are measured. Three is not, and neither is anything above four:
    the fourth chip is the last one a QuietBox has, so a wider world crosses a host boundary
    and pays a link that the same-box points say nothing about.

    The four-chip point is worth reading with its condition attached. Carrying two chips'
    efficiency forward would have predicted 3.74x, which is within 2 % of the measurement --
    and it would still have been the wrong number to trust, because the same four-chip step
    takes 931 s when every rank takes torch's default thread width and 7.647 s when the host's
    cores are divided across the ranks. What scales is a measured configuration, not a chip
    count.
    """
    if chips == 1:
        return 1.0, "single chip: no collective, speedup 1.0 by definition"
    if chips == 2:
        return MEASURED["dp2_speedup"][0], f"DP speedup: {MEASURED['dp2_speedup'][1]}"
    if chips == 4:
        return MEASURED["dp4_speedup"][0], f"DP speedup: {MEASURED['dp4_speedup'][1]}"
    return None, (f"{chips}-chip step time is {UNMEASURED}: one, two and four chips are "
                  f"measured on this hardware and {chips} is not. Interpolating between them "
                  f"assumes the host reduce's per-rank volume and the host torch that owns "
                  f"the 7.8 % gap at four chips both move smoothly in rank count, and above "
                  f"four the world leaves the box and pays a link these points never crossed")
