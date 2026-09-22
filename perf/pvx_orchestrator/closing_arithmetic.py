#!/usr/bin/env python3
"""The arithmetic that closes the PVX charter, computed from the census rather than typed.

The charter was: bring Boltz-2's 1.63x to Protenix-v2. This script partitions every second of
Protenix-v2's 512 aa fold into three classes and asks what the best case is if every second in the
only OPEN class is deleted outright. Nothing here is an estimate: the input is the committed
stage+deep census (`pvx-protenix-specific`, qb1 card 3, p150a, ttnn 0.68.0, pinned and
DURING-sampled 1350 MHz on all five folds), vendored by sha256 so a rewrite of the source row's
branch cannot change what this reads.

The three classes, and why a block lands in one:

  CLOSED-ACCURACY  `fold/sampler/**` -- the fp32 diffusion sampler. Protenix's own GPU reference
                   forces fp32 over the whole sampling score model (`skip_amp.sample_diffusion`),
                   and the bf16 arm, which already ships as PROTENIX_DIFFUSION_FP32_DEVICE=0, was
                   measured and turned down on the pharma HSA benchmark at a tight local-structure
                   floor. Deleting it is doing the model's work wrong, which is a hard stop.

  CLOSED-SURPLUS   the three MAC-carrying bodies of the TRUNK Pairformer, and nothing else.
                   Protenix runs 5.948x Boltz-2's trunk Pairformer arithmetic and converts it
                   1.837x MORE efficiently per MAC on the same shared `PairformerLayer` code.
                   There is no Boltz-2 lever to transfer into a block that already beats Boltz-2,
                   and bucket 2 is resolved 7 of 7 with an upper bound of 1.056x fold-wide.

  OPEN             everything else, which is deliberately generous: stage self time, Pairformer
                   glue, MSA, template and confidence all land here even though MSA and template
                   run the same shared bodies and carry the same per-MAC surplus. `--wide` moves
                   them to CLOSED-SURPLUS and the verdict does not change.

Run anywhere; needs no card and no ttnn.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE / "census" / "protenix_v2_512_qb1c3.json"
CENSUS_SHA = "36917e9e0cad4f8e8bd40070a51fc6418079fee5f496cb7463389962e5647bbc"

# What the campaign was sent to reach, and what its own re-measurement says the target really is.
TARGET_CITED = 1.63     # the published Boltz-2 cell, p300c
TARGET_P150A = 1.5341   # pvx-baseline, same two trees, same pinned clock, p150a -- matched board

# bucket 2's entire screened upper bound, and the part of it already on origin/main.
BUCKET2_UPPER_X = 1.056
SHIPPED_S = 0.789

MAC_BODIES = ("trimul", "triatt", "transition", "pwa", "opm")
TRUNK_PF = "fold/trunk_cond/trunk/pairformer/pf_layer/"

# fp32 -> bf16 on an eltwise op, measured on a relu control (`fold-nontriangle-below-4x`, step E).
# Used only for the sensitivity leg, applied to the WHOLE sampler, which overstates it: not all of
# the sampler is eltwise.
FP32_ELTWISE_X = 2.118


def classify(path: str, wide: bool) -> str:
    if path == "fold/sampler" or path.startswith("fold/sampler/"):
        return "closed-accuracy"
    leaf = path.rsplit("/", 1)[-1]
    if path.startswith(TRUNK_PF) and leaf in MAC_BODIES:
        return "closed-surplus"
    if wide and path.startswith("fold/trunk_cond/trunk/") and leaf in MAC_BODIES:
        return "closed-surplus"
    return "open"


def main() -> int:
    wide = "--wide" in sys.argv
    got = hashlib.sha256(CENSUS.read_bytes()).hexdigest()
    if got != CENSUS_SHA:
        print(f"census sha256 {got} != pinned {CENSUS_SHA}", file=sys.stderr)
        return 2
    d = json.loads(CENSUS.read_text())

    clean = [r["fold_s"] for r in d["runs"] if r["tag"] == "clean"]
    deep = next(r for r in d["runs"] if r["tag"] == "deep")
    fold = sum(clean) / len(clean)
    aa = max(clean) - min(clean)

    buckets = {"closed-accuracy": 0.0, "closed-surplus": 0.0, "open": 0.0}
    for node in deep["tree"]:
        buckets[classify(node["path"], wide)] += node["self_s"]
    instrumented = sum(buckets.values())
    root = next(n["incl_s"] for n in deep["tree"] if n["path"] == "fold")
    if abs(instrumented - root) > 1e-3:
        # a class that silently loses nodes shrinks OPEN and flatters the verdict, so this is a
        # guard on the conclusion and not on the formatting.
        print(f"partition drops {root-instrumented:.4f} s: self_s sums to {instrumented:.4f} "
              f"against a tree root of {root:.4f}", file=sys.stderr)
        return 2
    # the census costs 6 % in probe overhead; every share is deflated to the clean fold so the
    # arithmetic is stated on the number a user actually waits for.
    k = fold / instrumented

    print(f"model {d['model']} {d['size']} aa, {d['host']} card {d['card']}, ttnn {d['ttnn']}, "
          f"{d['recycling_steps']} recycles / {d['sampling_steps']} sampling steps")
    for r in d["runs"]:
        c = r["clock"]["0"]
        print(f"  run {r['ix']:>1} {r['tag']:<6} {r['fold_s']:8.4f} s   AICLK min {c['min']} / "
              f"median {c['median']} MHz over {c['n']} samples, polled DURING the fold")
    print(f"\nclean fold        {fold:8.4f} s   (n={len(clean)}, A/A floor {aa:.4f} s, "
          f"{100*aa/fold:.2f} %)")
    print(f"instrumented sum  {instrumented:8.4f} s   probe overhead {100*(1/k-1):.1f} %, "
          f"deflation {k:.5f}")

    print("\n  class            deflated s   share   why it is in this class")
    why = {
        "closed-surplus": "already 1.837x Boltz-2 per MAC -- no lever to transfer",
        "closed-accuracy": "fp32 sampler, bf16 arm measured and turned down on HSA",
        "open": ("stage self time, Pairformer glue, MSA, template, confidence"
                 if not wide else "stage self time, Pairformer glue, confidence"),
    }
    print(f"  partition: {'WIDE' if wide else 'CONSERVATIVE'} "
          f"({'every shared MAC body' if wide else 'the trunk Pairformer bodies only'} "
          f"counted as surplus)")
    for name in ("closed-surplus", "closed-accuracy", "open"):
        s = buckets[name] * k
        print(f"  {name:<16} {s:9.3f}   {100*s/fold:5.1f} %   {why[name]}")

    open_s = buckets["open"] * k
    unshipped_b2 = fold - fold / BUCKET2_UPPER_X - SHIPPED_S

    print("\nthe best case, and it is not a plan -- it deletes 100 % of every open second:")
    floor_open = fold - open_s
    floor_both = floor_open - unshipped_b2
    print(f"  today                                        {fold:8.3f} s   1.0000x")
    print(f"  delete every OPEN second outright            {floor_open:8.3f} s   "
          f"{fold/floor_open:.4f}x")
    print(f"  and take the whole unshipped rest of bucket 2{floor_both:8.3f} s   "
          f"{fold/floor_both:.4f}x   (+{unshipped_b2:.3f} s, double-counted in the campaign's favour)")

    for label, tgt in (("cited 1.63x (p300c)", TARGET_CITED),
                       ("matched-board 1.5341x (p150a)", TARGET_P150A)):
        need = fold / tgt
        print(f"\n  {label}: needs {need:.3f} s, i.e. {fold-need:.3f} s removed "
              f"({100*(fold-need)/fold:.1f} % of the fold)")
        print(f"    short by {floor_both-need:+.3f} s after the best case above")

    samp = buckets["closed-accuracy"] * k
    saved = samp * (1 - 1 / FP32_ELTWISE_X)
    print(f"\nsensitivity -- what the accuracy stop is worth, if it were lifted, which it is not:")
    print(f"  the whole sampler at the measured fp32->bf16 eltwise ratio {FP32_ELTWISE_X}x saves "
          f"{saved:.3f} s of {samp:.3f} s")
    print(f"  best case then {floor_both-saved:8.3f} s   {fold/(floor_both-saved):.4f}x   "
          f"(still an impossible bound: it also deletes every open second)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
