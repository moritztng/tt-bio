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

__all__ = ["plan", "Plan", "UNMEASURED", "CARD_DRAM_BYTES", "MEASURED", "needs_exact",
           "FORWARD_OOM_BY_MODEL", "WEIGHTS_STEP_S", "FORWARD_FITS_BY_MODEL",
           "TRAINED_TRUNK_BOUND_BY_MODEL", "provenance_gap"]


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

    This exists because of what it would have caught. Protenix-v2's 384 and 512 refusals were
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


# Peak device DRAM of Protenix-v2's SHIPPED 48-block pairformer at the checkpoint's own widths (c_z 256,
# 8 triangle heads of 32), forward only and untaped -- which is the state per-block
# checkpointing runs the forward in, so this is the figure the crop question turns on.
FORWARD_FITS_BY_MODEL = {
    "protenix-v2": {
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
    },
}

# What a TRAINED Protenix-v2 trunk RETAINS at a crop under per-block checkpointing, and an upper bound
# rather than a peak. Four terms: the 48 block-boundary (z, s) pairs and one block's inner set
# are read off the allocator on the card, the weight and gradient terms are the checkpoint's own
# censused parameter counts (464,442,431 whole model x 2 B bf16, 227,960,832 trunk x 4 B fp32).
# What is NOT in it is the backward's own working set inside the block being recomputed, because
# no taped pairformer exists to allocate one. That missing term is why plan() reports these
# gigabytes and still answers UNMEASURED.
TRAINED_TRUNK_BOUND_BY_MODEL = {
    "protenix-v2": {
        256: (4.074, "1.620 GB boundaries + 0.613 GB inner set, both measured; ptx-crop on qb1 "
                     "card 1, wk/ptx-crop 4ab2aad2a, state/concluded/ptx-crop"),
        384: (7.762, "3.638 GB boundaries + 2.283 GB inner set (38 DRAM intermediates), both "
                     "measured; ptx-crop on qb1 card 1 (wk/ptx-crop 4ab2aad2a) and reproduced byte "
                     "for byte by train-x-cropunblock on qb1 card 2, 3,638,034,432 B and "
                     "2,283,307,008 B, perf/train_x_crop/out/tape_qb1c2.json, 1350 MHz sampled "
                     "during. Replaces the 27.58 GB projection this module used to refuse on"),
        512: (13.047, "6.461 GB boundaries + 4.745 GB inner set, both measured; ptx-crop on qb1 "
                      "card 1, wk/ptx-crop 4ab2aad2a, state/concluded/ptx-crop"),
    },
}

# The crop sizes measured to refuse, PER MODEL, with the allocation that was refused.
#
# Keyed by model, and that is the whole point. This table used to be one flat dict applied to
# whatever `tt-bio train --model X` was given, and its two entries were Protenix-v2's. A
# memory wall is a property of a model's activation shapes, not of a token count, so a flat
# table answers a question it was never asked: `--model openfold3 --tokens 512` was REFUSED on
# Protenix-v2's number for a crop OpenFold3 is measured to RUN, and 544/576/640/768 came back
# UNMEASURED for OpenFold3 when all four are measured to refuse. Wrong in both directions, on a
# documented command.
#
# A model with no entry here gets no refusal from this table. That is deliberate: borrowing a
# neighbour's wall is what produced the defect, and UNMEASURED is a first-class answer.
FORWARD_OOM_BY_MODEL = {
    # Empty, and kept so the retraction stays visible. 384 and 512 sat here, measured by
    # train-r5 against a differentiable TWIN of the pair track that has since been deleted (see
    # the module docstring), until train-x-cropunblock measured both to PASS on the shipped
    # forward: they are in FORWARD_FITS_BY_MODEL now. Any future entry cites its own source,
    # which `provenance_gap` checks and the two that left did not.
    "protenix-v2": {},

    # of3t-crop768, concluded 2026-09-21, on Blackhole at a median 1350 MHz polled DURING every
    # rung. Forward AND backward peaks on the taped training path, cards 0 and 1 of qb1 (p150a)
    # for 544/576/640/768 and qb2 (p300c) for the 512 baseline; both boards present the same
    # 8 x 4,278,190,016 B = 34,225,520,128 B per-chip DRAM read off the allocator's own refusal
    # lines. At the time: 512 RAN and was the largest that did, and the refusals above it were
    # not one wall -- 640 and 768 died with the card full, 576 died with 6,671,522,304 B still
    # free, refused for CONTIGUITY inside `ttnn::concat`, short by 77,930,560 B per bank. A
    # capacity extrapolation cannot locate that frontier -- the row's own 2.08 fit said 576
    # would clear with 14 % of margin, and it did not -- which is exactly why these are table
    # entries and not a slope.
    #
    # SUPERSEDED IN PART, 2026-09-25: of3t-cropwall's concat-heads split landed on main
    # (e558bfb06, 2026-09-23) and moved the frontier one rung, 512 -> 576. 640 re-measured
    # post-fix and still refuses. A measured refusal is only true of the code it was measured
    # on, so an entry here has to be re-read whenever a fix touches the allocation it names.
    "openfold3": {
        # 544 and 576 USED to be entries here and were removed on 2026-09-25. Their refusals
        # were real, and of3t-cropwall's concat-heads split fix (e558bfb06, on main since
        # 2026-09-23) removed the thing they measured: 576 refused for CONTIGUITY inside
        # `ttnn::concat`, which is exactly the allocation that fix reshapes. Post-fix, 576
        # COMPLETES -- twice, at two separate main-ancestor commits. Keeping a refusal whose
        # cause is fixed is how plan() came to refuse a crop the engine can run, which is the
        # defect this whole table exists to have stopped.
        640: (None, None, "of3t-crop768 -- card full, 23,710,208 B free device-wide; "
                          "RE-CONFIRMED post-concat-fix at e558bfb06 (perf/of3t_cropwall/out/"
                          "split_640_fix2_card3.json), so this wall is not the concat one"),
        768: (None, None, "of3t-crop768 (state/concluded/of3t-crop768) -- card full, "
                          "6,231,552 B free device-wide; the "
                          "levered fit puts 768 at 1.558x the card and the fit UNDER-predicts, "
                          "so that is a floor on the overshoot"),
    },
}

# The largest crop a model's forward+backward is measured to COMPLETE, per model. This is a
# weaker fact than a training replica -- it says the memory fits, not how long a step takes --
# and the two are kept apart because conflating them is how a fit becomes a projected step time.
LARGEST_MEASURED_TO_FIT = {
    "openfold3": (576, "of3t-cropwall -- 576 is the largest crop measured to COMPLETE, on "
                       "main since e558bfb06 (2026-09-23). Backward DRAM high-water "
                       "30,230,471,680 B of the card's 34,225,520,128 B (88.3 %), qb2 p300c "
                       "card 3, median 1350 MHz polled DURING: perf/of3t_cropwall/out/"
                       "split_576_fix2_card3.json, and again at 63b51514d in split_576_post"
                       ".json. 640 still refuses. 544 is unmeasured post-fix -- it sits below "
                       "a crop that runs, so it is not claimed as a refusal either"),
}

# The largest crop the DEVICE path (float64 instrument off, the default) is measured to complete,
# per model. LARGEST_MEASURED_TO_FIT above was measured with the instrument on, which computes
# softmax and layer norm on the host and so keeps the attention probabilities off the card. On
# the device path OpenFold3's triangle-attention backward holds them there in fp32, and 576 runs
# out of memory with the fp32 softmax backward on and off alike. Between the two numbers only the
# instrument fits, so `needs_exact` turns it on there rather than refusing a crop that runs.
DEVICE_PATH_LARGEST_FIT = {
    "openfold3": (512, "of3t-p10default -- 512 completes, backward DRAM high-water 25,983,800,320 B of 34,225,520,128 B (75.9 %), qb1 p150a card 0, median 1350 MHz polled DURING: perf/of3t_p10default/out/split_512_default.json. 576 runs out of memory on a 3,057,647,616 B request: split_576_default.json, split_576_fp32off.json"),
}


#: One training step of a model's own weights, one sample, on one chip, by crop.
WEIGHTS_STEP_S = {
    "openfold3": {
        384: (42.27, "of3t-p10land -- 42.272 and 42.234 s on the merged tree, qb2 p300c card 3, "
                     "1350 MHz sampled DURING; 54.2 s on qb1's p150a (of3t-p10default)"),
    },
}


def _weights_plan(model: str, tokens: int, chips: int, global_batch: Optional[int]) -> Plan:
    """Training the weights themselves, answered from this model's own measurements."""
    device_fit, device_src = DEVICE_PATH_LARGEST_FIT[model]
    exact_fit, exact_src = LARGEST_MEASURED_TO_FIT.get(model, (device_fit, device_src))
    per_chip = (global_batch or chips) / chips
    common = dict(tokens=tokens, chips=chips, global_batch=global_batch)
    if tokens <= device_fit:
        step, step_src = WEIGHTS_STEP_S.get(model, {}).get(tokens, (None, None))
        secs = None if step is None else step * per_chip
        timing = (f" A step takes about {secs:.1f} s: {per_chip:g} sample(s) per chip at "
                  f"{step:.2f} s each on one Blackhole chip." if secs else
                  f" The step time is measured at {sorted(WEIGHTS_STEP_S.get(model, {}))} "
                  f"tokens only.")
        return Plan(verdict="fits", fits=True, seconds_per_step=secs,
                    why=f"training {model}'s weights fits one chip up to {device_fit} tokens, "
                        f"measured." + timing,
                    sources=[device_src] + ([step_src] if step_src else []), **common)
    if tokens <= exact_fit:
        return Plan(verdict="fits", fits=True,
                    why=f"{tokens} tokens fits {model} only with --exact, which turns itself "
                        f"on and makes a step many times slower. {device_fit} is the largest "
                        f"crop the device path fits.",
                    sources=[device_src, exact_src], **common)
    return Plan(verdict=UNMEASURED,
                why=f"{tokens} tokens is above the largest {model} crop measured to train "
                    f"({exact_fit}) and not one measured to run out of memory.",
                sources=[exact_src], **common)


def needs_exact(model: Optional[str], tokens: int) -> bool:
    """Whether this crop fits only with the float64 instrument on: above the device path's
    measured fit and at or below the instrument's."""
    device, _ = DEVICE_PATH_LARGEST_FIT.get(model, (None, None))
    exact, _ = LARGEST_MEASURED_TO_FIT.get(model, (None, None))
    return device is not None and exact is not None and device < tokens <= exact


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

    ``frozen_trunk=False`` means full fine-tuning. A model with its own weights-training
    measurements (``DEVICE_PATH_LARGEST_FIT``) is answered from them; otherwise it comes back
    UNMEASURED, with the model's measured retention bound in the reason where one exists: what
    is unmeasured is the backward's own working set, not the whole arithmetic. A crop whose
    FORWARD is measured to OOM is refused before that, because the forward is what both modes run and "we measured
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

    if not frozen_trunk and model in DEVICE_PATH_LARGEST_FIT:
        return _weights_plan(model, tokens, chips, global_batch)

    if not frozen_trunk:
        bounds = TRAINED_TRUNK_BOUND_BY_MODEL.get(model, {})
        crops = ", ".join(str(k) for k in sorted(bounds)) or "no crop"
        if tokens not in bounds:
            return Plan(
                verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
                why=f"full fine-tuning: a trained {model or 'model'} trunk's cost is measured at "
                    f"{crops} and not at {tokens} aa, and both of its measured terms move "
                    f"with the crop across the triangle ops' chunking thresholds. Measure it.",
                sources=[f"TRAINED_TRUNK_BOUND_BY_MODEL covers {crops} for {model}"])
        gb, src = bounds[tokens]
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
        fits = FORWARD_FITS_BY_MODEL.get(model, {})
        fwd = fits.get(tokens)
        sources = [f"MEASURED replica exists only at {MEASURED_CROP} aa"] + fit_src
        if fwd is not None:
            forward = (f" The forward is not the obstacle: it is measured to fit at {tokens} aa, "
                       f"peaking at {fwd[0]:.3f} GB of {CARD_DRAM_GB:.2f} GB "
                       f"({fwd[0] / CARD_DRAM_GB * 100:.1f} %) over all 48 blocks.")
            sources.append(f"forward: {fwd[1]}")
        elif fits:
            forward = (f" The forward has no measurement at {tokens} aa either; it is measured "
                       f"to fit at {', '.join(str(k) for k in sorted(fits))} aa.")
        else:
            forward = ""
        return Plan(
            verdict=UNMEASURED, tokens=tokens, chips=chips, global_batch=global_batch,
            why=f"{tokens} tokens is above the largest crop with a measured training replica "
                f"({MEASURED_CROP} aa) and is not a size "
                f"{model or 'this model'}'s own forward OOM was measured at "
                f"({', '.join(str(k) for k in sorted(oom)) or 'none recorded'}).{forward} "
                f"Activation volume is neither linear nor quadratic in tokens across the "
                f"triangle ops' chunking thresholds, so scaling the {MEASURED_CROP} aa replica "
                f"up would be a guess with a plausible shape. Measure it." + fit_note,
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
