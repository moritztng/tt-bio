#!/usr/bin/env python3
"""The C10 ladder: every candidate, what it is worth, and how good its evidence is.

One place for what is otherwise scattered across a dozen artifacts. Every row carries an evidence
class, because a number measured on Blackhole at a recorded clock and a number transferred from
Wormhole at an unrecorded one are not the same kind of thing and must not be summed as if they
were. CPU only; this file measures nothing.
"""
import json
import sys

BASELINE_S = 14.8813252375     # c10-bare-baseline bc66f7d6d, 512 aa, pinned 1350 MHz, 16 folds
BASELINE_298_S = 9.6800812565
CLOCK_MHZ = 1350.0
TARGET_S = 10.0

# evidence classes, worst to best
MEASURED_BH_CLOCKED = "measured on Blackhole at a recorded clock"
MEASURED_BH_UNCLOCKED = "measured on Blackhole, clock not recorded"
MEASURED_WH = "measured on Wormhole, clock not recorded"
TRANSFERRED = "transferred across architectures, not measured here"
DERIVED = "arithmetic on measured artifacts, not itself measured"
UNMEASURED = "not measured anywhere"

LADDER = [
    {
        "item": "ttnn trace of the diffusion loop",
        "mechanism": "replay one captured graph per sampling step, removing the host's per-call "
                     "dispatch for 219,200 of the fold's ~287,000 program-launching calls -- which it does, and the fold does not get shorter",
        "fold_s": 0.0, "evidence": MEASURED_BH_CLOCKED,
        "basis": "REFUTED. Sized at 3.02 s from the clock-immune term times the diffusion loop's "
                 "76 % share of launching calls; c10-trace-lever then measured -0.0214 s at 512 aa "
                 "and -0.0161 s at 298 aa at a pinned 1350 MHz, both inside their own A/A floors "
                 "of 0.055 s and 0.043 s, and the null reproduces at 1000 MHz. The call arithmetic "
                 "was right and the inference from it was wrong",
        "accuracy_spend": "none, and it was free to check: one CIF digest per size across all 40 "
                          "accepted folds, identical in both arms, max Kabsch RMSD 3.7e-15 A",
        "row": "c10-trace-lever", "status": "concluded NO-GO",
    },
    {
        "item": "cut the rest of the clock-immune term",
        "mechanism": "featurization, MSA handling, output writing and whatever dispatch the trace "
                     "does not reach",
        "fold_s": None, "evidence": UNMEASURED,
        "basis": "the WHOLE 3.9830 s clock-immune term is now unattributed, not 0.93 s of it: the "
                 "trace reaches none of it. Two independent lines say it is not host dispatch -- "
                 "the trace null, and F scaling at N^1.32 +- 0.07, 19 sigma from size-independent. "
                 "A term that grows at N^1.3 and survives dispatch removal looks like DRAM-bound "
                 "device time, which AICLK does not drive. No value can be quoted until something "
                 "measures what it is made of",
        "accuracy_spend": "unknown, depends entirely on what it turns out to be",
        "row": "c10-fold-census owes the split; no row attacks it yet",
        "status": "unowned",
    },
    {
        "item": "fuse the arithmetic-free elementwise traffic into its producers",
        "mechanism": "ttnn.multiply_, ttnn.layer_norm and ttnn.add_ move 0.881 TB, 30.8 % of the "
                     "fold's entire byte traffic, while performing essentially none of its "
                     "arithmetic -- pure DRAM round trips over the pair and single "
                     "representations that a producer's epilogue can absorb",
        "fold_s": 0.69, "evidence": DERIVED,
        "basis": "2.074 to 2.523 s of modelled traffic depending on which of the two in-house "
                 "roofs is used, times the one-third return this project has already measured for "
                 "this class of fusion. Roof-free part: the byte and call shares, on which the two "
                 "capture artifacts agree to 3 ppm on fold totals. Attacks F, the clock-immune "
                 "term, so it does not compete with arithmetic and does not shrink at burst clock",
        "accuracy_spend": "a fused epilogue changes accumulation order; digest will move, so it "
                          "needs Angstrom against the 0.60 A bar at 512 and 0.35 A at 298",
        "row": "not rowed yet", "status": "unowned",
    },
    {
        "item": "per-class grid sizing, trimul and triangle attention",
        "mechanism": "fewer cores for a class whose parallel efficiency falls off, more for one "
                     "whose does not",
        "fold_s": None, "evidence": TRANSFERRED,
        "basis": "the ledger's 810 Mcycles came from a Wormhole 8x9 sweep where the two classes "
                 "move in OPPOSITE directions; no Blackhole core sweep exists. Withdrawn as a "
                 "value. Blackhole does show trimul at 11.47 TFLOP/s against a same-session dense "
                 "cube's 67.59",
        "accuracy_spend": "a grid change can alter reduction order; digest check needed",
        "row": "c10-grid-sweep", "status": "queued",
    },
    {
        "item": "TT_BIO_SDPA_GRID_Q_CHUNK sign",
        "mechanism": "a default-ON flag whose value is not known to be positive",
        "fold_s": None, "evidence": UNMEASURED,
        "basis": "the ledger puts it between -623 and +281 Mcycles, i.e. -0.46 s to +0.21 s at "
                 "1350 MHz. The downside is a live loss on a shipped default",
        "accuracy_spend": "digest check; it routes a grid decision",
        "row": "c10-qchunk-sign", "status": "queued",
    },
    {
        "item": "TT_BIO_DIT_FUSED_QKV",
        "mechanism": "fuse the DiT QKV projections",
        "fold_s": 0.12, "evidence": MEASURED_WH,
        "basis": "1.04124x on the Wormhole step, about 162 Mcycles if it transfers at full value. "
                 "Never measured on Blackhole",
        "accuracy_spend": "0.346 A of the 0.60 A bar alone; 0.713 A jointly with HEAD_PAD_TAIL, "
                          "over the bar, so the two cannot ship together",
        "row": None, "status": "not rowed",
        "excludes": ["TT_BIO_HEAD_PAD_TAIL"],
    },
    {
        "item": "TT_BIO_HEAD_PAD_TAIL",
        "mechanism": "pad the head tail so the token DiT head widths stop splitting",
        "fold_s": 0.21, "evidence": MEASURED_WH,
        "basis": "0.211 s on a Wormhole fold with 72 of its 96 sites OUTSIDE the diffusion step, "
                 "so pricing it as a step-only 1.02549x lever under-prices it by roughly 4x",
        "accuracy_spend": "0.3696 A at 512 aa, not bit-exact; excludes DIT_FUSED_QKV",
        "row": None, "status": "not rowed",
        "excludes": ["TT_BIO_DIT_FUSED_QKV"],
    },
    {
        "item": "the byte axis, all remaining levers together",
        "mechanism": "delete bytes in the elementwise tail",
        "fold_s": None, "evidence": MEASURED_BH_UNCLOCKED,
        "basis": "CEILING, not a value: deleting EVERY byte takes the floor 12.706 -> 8.543 s, "
                 "1.487x, and the headroom is concentrated in multiply_ 0.770 s, layer_norm "
                 "0.665 s, add_ 0.640 s and permute 0.256 s. The corpus independently caps the "
                 "byte axis at 1.470x. Levers already shipped have consumed an unknown part of it",
        "accuracy_spend": "varies per lever",
        "row": None, "status": "closed as a ceiling, not a candidate",
    },
]


def compatible_best(priced):
    """Largest subset of priced rows with no mutual exclusion, and what it is worth.

    Two rows here cannot ship together: jointly they are 0.713 A against a 0.60 A bar. Summing
    both would overstate the ladder, so drop the cheaper side of every exclusion pair.
    """
    names = {r["item"] for r in priced}
    dropped = set()
    for r in priced:
        for other in r.get("excludes", ()):
            if other in names and other not in dropped and r["item"] not in dropped:
                loser = min((r, next(x for x in priced if x["item"] == other)),
                            key=lambda x: x["fold_s"])
                dropped.add(loser["item"])
    kept = [r for r in priced if r["item"] not in dropped]
    return kept, sorted(dropped), sum(r["fold_s"] for r in kept)


def main():
    priced = [r for r in LADDER if r["fold_s"] is not None and r["status"] != "closed as a ceiling, not a candidate"]
    kept, dropped, total = compatible_best(priced)
    naive = sum(r["fold_s"] for r in priced)
    out = {
        "scope": "CPU consolidation. Every number is cited to an artifact; none is measured here.",
        "baseline": {"512_aa_s": BASELINE_S, "298_aa_s": BASELINE_298_S, "clock_MHz": CLOCK_MHZ,
                     "source": "c10-bare-baseline bc66f7d6d, 16 folds per size, min=max=1350 MHz"},
        "target_s": TARGET_S,
        "gap_s": BASELINE_S - TARGET_S,
        "ladder": [dict(r, Mcycles=None if r["fold_s"] is None else r["fold_s"] * CLOCK_MHZ)
                   for r in LADDER],
        "priced_total": {
            "fold_s": total, "Mcycles": total * CLOCK_MHZ,
            "items": [r["item"] for r in kept],
            "dropped_as_mutually_exclusive": dropped,
            "naive_sum_including_excluded_s": naive,
            "would_read_s": BASELINE_S - total,
            "covers_pct_of_gap": 100.0 * total / (BASELINE_S - TARGET_S),
        },
        "honest_reading": [
            "Nothing in this table has been measured as a fold-level win yet. Cycles saved to "
            "date: zero. Accuracy spent to date: none.",
            "Two of the priced rows are DERIVED, meaning arithmetic over measured artifacts "
            "rather than a measured lever, and two are Wormhole numbers that have never run on "
            "Blackhole. Do not read the total as a forecast.",
            "The two Wormhole rows cannot both ship: jointly they are 0.713 A against a 0.60 A bar.",
            "Perturbations stack strongly sub-additively on this fixture, so a stack must be "
            "measured as a stack and never by summing individual readings. This total is an "
            "inventory, not a prediction.",
            "If every priced row landed at its full value the fold would read %.2f s, which is "
            "above the 10.0 s target. On present evidence the target needs either the "
            "clock-immune term to give up something the trace demonstrably does not reach, "
            "or a device-work lever nobody has found. The one sized candidate left is the "
            "size-independent share of the work term, 3,301 to 8,220 Mcycles, which "
            "c10-size-scaling is measuring. "
            "nobody has found yet." % (BASELINE_S - total),
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
