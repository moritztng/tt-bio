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

__all__ = ["plan", "Plan", "UNMEASURED", "CARD_DRAM_BYTES", "MEASURED",
           "FORWARD_OOM_BY_MODEL"]


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

# The crop sizes measured to refuse, PER MODEL, with the allocation that was refused.
#
# Keyed by model, and that is the whole point. This table used to be one flat dict applied to
# whatever `tt-bio finetune --model X` was given, and its two entries were Protenix-v2's. A
# memory wall is a property of a model's activation shapes, not of a token count, so a flat
# table answers a question it was never asked: `--model openfold3 --tokens 512` was REFUSED on
# Protenix-v2's number for a crop OpenFold3 is measured to RUN, and 544/576/640/768 came back
# UNMEASURED for OpenFold3 when all four are measured to refuse. Wrong in both directions, on a
# documented command.
#
# A model with no entry here gets no refusal from this table. That is deliberate: borrowing a
# neighbour's wall is what produced the defect, and UNMEASURED is a first-class answer.
FORWARD_OOM_BY_MODEL = {
    # train-r5 measured both; PLAN.md P2 corrected r5's causal attribution (production's trimul
    # chunking is built and the training path simply does not get it) and scheduled A2 to
    # re-measure on the shipped forward. The OOM itself stands, so plan() refuses on it -- and
    # refuses by naming the measurement rather than by extrapolating a slope through it.
    "protenix-v2": {
        384: (4.14, 75_497_472, "train-r5-distributed-tt REPLICA-FITS -- measured"),
        512: (7.15, 536_870_912, "train-r5-distributed-tt REPLICA-FITS -- measured"),
    },
    # of3t-crop768, concluded 2026-09-21, on Blackhole at a median 1350 MHz polled DURING every
    # rung. Forward AND backward peaks on the taped training path, cards 0 and 1 of qb1 (p150a)
    # for 544/576/640/768 and qb2 (p300c) for the 512 baseline; both boards present the same
    # 8 x 4,278,190,016 B = 34,225,520,128 B per-chip DRAM read off the allocator's own refusal
    # lines. 512 RUNS and is the largest that does. The three refusals above it are not one
    # wall: 640 and 768 die with the card full, 576 dies with 6,671,522,304 B still free,
    # refused for CONTIGUITY inside `ttnn::concat`, short by 77,930,560 B per bank. A capacity
    # extrapolation cannot locate that frontier -- the row's own 2.08 fit said 576 would clear
    # with 14 % of margin, and it did not -- which is exactly why these are table entries and
    # not a slope.
    "openfold3": {
        544: (None, None, "of3t-crop768 -- measured to refuse, qb1 p150a, 1350 MHz during"),
        576: (None, 77_930_560, "of3t-crop768 -- CONTIGUITY inside ttnn::concat with "
                                "6,671,522,304 B still free, short by this much per bank; "
                                "reproduces byte for byte on cards 0 and 1"),
        640: (None, None, "of3t-crop768 -- card full, 23,710,208 B free device-wide"),
        768: (None, None, "of3t-crop768 -- card full, 6,231,552 B free device-wide; the "
                          "levered fit puts 768 at 1.558x the card and the fit UNDER-predicts, "
                          "so that is a floor on the overshoot"),
    },
}

# The largest crop a model's forward+backward is measured to COMPLETE, per model. This is a
# weaker fact than a training replica -- it says the memory fits, not how long a step takes --
# and the two are kept apart because conflating them is how a fit becomes a projected step time.
LARGEST_MEASURED_TO_FIT = {
    "openfold3": (512, "of3t-crop768 -- 512 is the largest crop measured to run; 544 is the "
                       "first that refuses"),
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


def plan(*, tokens: int, model: Optional[str] = None, chips: int = 1,
         global_batch: Optional[int] = None, frozen_trunk: bool = True,
         optimizer_resident: bool = False,
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

    # Only this model's own table. A model we have not measured gets UNMEASURED, never a
    # neighbour's wall -- see FORWARD_OOM_BY_MODEL.
    oom = FORWARD_OOM_BY_MODEL.get(model, {})
    if tokens in oom:
        allocated, refused, source = oom[tokens]
        detail = []
        if allocated is not None:
            detail.append(f"{allocated:.2f} GB allocated")
        if refused is not None:
            detail.append(f"{refused:,} B refused")
        measured = f": {' and '.join(detail)}" if detail else ""
        return Plan(
            verdict="refused", tokens=tokens, chips=chips, global_batch=global_batch,
            fits=False,
            why=f"{model} is measured to run out of memory at {tokens} aa{measured}. "
                f"Distribution does not fix it -- each chip OOMs at {tokens} aa exactly as one "
                f"does. This is a refusal, not an estimate, and it is {model}'s own "
                f"measurement: no other model's wall is applied here.",
            sources=[source])

    if not frozen_trunk:
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why="full fine-tuning: the only source for a trained trunk's tape is "
                "perf/hall_grad/DECISION.md, a feasibility memo whose 27.58 GB (15.76 GB of "
                "tape + 11.82 GB live, per-block checkpointing, at 800 aa) and 40 engineer-day "
                "estimate are projections of unbuilt work. Measured planning stops at a LoRA "
                "adapter on a frozen trunk.",
            sources=["perf/hall_grad/DECISION.md:38,107 -- projection, not measurement"])

    # A measured memory fit is not a measured step time, but it is not nothing either: saying
    # "no measurement" when we have watched this exact crop complete would throw away the one
    # fact the user needs to decide whether to try it.
    fit_note, fit_src = "", []
    largest_fit, fit_source = LARGEST_MEASURED_TO_FIT.get(model, (None, None))
    if largest_fit is not None and tokens <= largest_fit:
        fit_note = (f" What IS measured for {model}: the memory fits up to {largest_fit} aa, "
                    f"so this crop is expected to run -- there is no step time for it, which "
                    f"is what UNMEASURED means here.")
        fit_src = [fit_source]

    if tokens > MEASURED_CROP:
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why=f"{tokens} tokens is above the largest crop with a measured training replica "
                f"({MEASURED_CROP} aa) and is not a size "
                f"{model or 'this model'}'s own forward OOM was measured at "
                f"({', '.join(str(k) for k in sorted(oom)) or 'none recorded'}). Activation "
                f"volume is neither linear nor quadratic in tokens across the triangle ops' "
                f"chunking thresholds, so interpolating between 256 and 384 would be a guess "
                f"with a plausible shape. Measure it." + fit_note,
            sources=[f"MEASURED replica exists only at {MEASURED_CROP} aa"] + fit_src)

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
