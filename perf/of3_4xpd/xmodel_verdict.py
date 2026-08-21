#!/usr/bin/env python3
"""Score one model's cross-model A/B and print the verdict the decision turns on.

    xmodel_verdict.py <outdir> [--model M ...]

Three gates, in this order, because only the first is not a judgement call:

  DIGEST   every warm fold of every arm carries the same CIF sha256 and the same plDDT.
           The AdaLN split is bit-exact by construction -- same ops, same dtypes, same
           values, a different issue order -- and this is what turns "by construction"
           into "per consumer". A miss here fails the model outright.
  L1       the two latching L1 gates (`_FP32_SOFTMAX_L1_ROW_CAP`, `_TRANSPOSE_L1_REFUSED`)
           end the run holding the same keys in both arms. A reorder cannot move a value
           but it can move the allocator, and a latched refusal is how that would show up.
  PERF     |median(B) - median(A)| against a noise floor. See below for how the floor is
           built, because the obvious way to build it does not work.

The floor is not one A/A pair. Exactly one model here executes no changed line: ESMFold2,
whose AdaLN is its own ttnn class (esmfold2.py:210) and which never enters the fp32 tail.
Its census measured adaln_calls 0, so its true delta is exactly zero and what it measured
is this harness's noise: +0.064 s on a 31.75 s fold, 0.20 %.

Boltz-2 was predicted to be the second null and is not. boltz2.py:1711 defines an AdaLN,
but that one is the torch reference module; Boltz-2's device path lives in tenstorrent.py
itself (DiffusionTransformerLayer:4511, ConditionedTransitionBlock:4441) and calls
tenstorrent.AdaLN 12000 times per fold, which its census measured. A class of the same name
in the model file is not evidence about the device path. Boltz-2 is a fourth reaching
model, not a control, and its -0.069 s is a real delta inside its own 0.143 floor.

Even with one null the single-pair estimator is visibly unsafe: the A/A pairs came out
0.009 s and 0.143 s on the same afternoon and the same harness, sixteen times apart. One
pair cannot estimate a spread, so the floor a reaching model is judged against is the
widest of

  * its own A/A pair       |median(A1) - median(A2)|
  * its own B/B pair       |median(B1) - median(B2)|,  the arm the A/A floor ignores
  * the null-control band  max |delta| over the models a census proved execute no changed
                           code, scaled to this model's own wall so it is a rate, not a constant

Digests are compared only inside one output directory, which is one host, one card and one
wheel. Nothing here compares against a digest recorded anywhere else.
"""
import argparse, json, statistics, sys
from pathlib import Path

MODELS = ["boltz2", "esmfold2", "protenix-v2", "opendde", "openfold3"]
ARMS = ("A1", "B1", "A2", "B2")


def load(d: Path, model: str):
    return {a: json.loads(p.read_text())
            for a in ARMS if (p := d / f"{model}_{a}.json").exists()}


def digests(rec):
    """(cif sha, plddt) over the warm folds; the cold fold is discarded like its wall."""
    return {(tuple(sorted(f["cif_sha256"].items())), f["plddt"]) for f in rec["folds"]}


def reaches(rec):
    """Whether this run touched a changed line: True, False, or None for "cannot tell".

    `FP32_SOFTMAX_STATS` is the library's own counter and increments whether or not anyone
    is watching, so a zero there is a real zero. `reach` is not: xmodel_ab.py installs those
    wrappers only under --census, so an A/B run reports adaln_calls 0 because nothing was
    counting. Reading that as "AdaLN was never called" is a vacuous zero, and it would have
    classified every model here as a null control, OpenFold3 included.
    """
    if rec["fp32_softmax_stats"]["calls"] > 0:
        return True
    if not rec["census"]:
        return None                     # fp32 tail is a measured null; AdaLN is unmeasured
    return any(rec["reach"].values())


def score(arms, census):
    walls = {k: [f["fold_s"] for f in v["folds"]] for k, v in arms.items()}
    med = lambda *k: statistics.median(sum((walls[x] for x in k), []))
    mA, mB = med("A1", "A2"), med("B1", "B2")
    # The census is the only run that watched AdaLN, so where one exists it is the answer.
    # Folding it together with the A/B records let their vacuous None outvote a measured
    # False, which is how ESMFold2 -- the one real null control -- came out "unknown".
    if census is not None:
        reach = reaches(census)
    else:
        seen = [reaches(v) for v in arms.values()]
        reach = True if any(r is True for r in seen) else None
    return dict(
        mA=mA, mB=mB, delta=mB - mA,
        aa=abs(med("A1") - med("A2")), bb=abs(med("B1") - med("B2")),
        reach=reach,
        seen=set().union(*(digests(v) for v in arms.values())),
        l1={(tuple(sorted(v["fp32_softmax_l1_refused_keys"])),
             tuple(sorted(v["transpose_l1_refused_keys"]))) for v in arms.values()},
        grids={tuple(v["grid"]) for v in arms.values()},
        wheels={v["ttnn"] for v in arms.values()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--model", action="append", default=None)
    a = ap.parse_args()
    models = a.model or MODELS

    done, bad = {}, 0
    for m in models:
        arms = load(a.outdir, m)
        if len(arms) < 4:
            print(f"{m:12s} INCOMPLETE ({sorted(arms)})")
            bad += 1
        else:
            p = a.outdir / f"{m}_census.json"
            done[m] = score(arms, json.loads(p.read_text()) if p.exists() else None)

    # The null controls set the band, as a rate, from the legs whose true delta is zero.
    # A model qualifies only on a census that measured it, never on an unwatched zero.
    nulls = {m: s for m, s in done.items() if s["reach"] is False}
    unknown = sorted(m for m, s in done.items() if s["reach"] is None)
    if unknown:
        print(f"reachability not yet measured (no census): {', '.join(unknown)} "
              f"-- their fp32 tail is a measured null, their AdaLN is not")
    band = max((abs(s["delta"]) / s["mA"] for s in nulls.values()), default=0.0)
    if nulls:
        print("null controls (no changed line executed, so their delta is this harness's noise): "
              + ", ".join(f"{m} {s['delta']:+.3f}s ({100 * s['delta'] / s['mA']:+.2f}%)"
                          for m, s in sorted(nulls.items()))
              + f"  ->  band +-{100 * band:.2f}%")

    for m in models:
        s = done.get(m)
        if not s:
            continue
        floor = max(s["aa"], s["bb"], band * s["mA"])
        ok_d, ok_l1 = len(s["seen"]) == 1, len(s["l1"]) == 1
        ok_p = s["delta"] <= floor
        bad += not (ok_d and ok_l1 and ok_p)
        print(f"{m:12s} A {s['mA']:8.3f}  B {s['mB']:8.3f}  delta {s['delta']:+7.3f}  "
              f"floor {floor:6.3f}  digest {'SAME' if ok_d else 'MOVED'}  "
              f"L1 {'SAME' if ok_l1 else 'MOVED'}  "
              f"perf {'ok' if ok_p else 'REGRESSED'}"
              f"{'   [null control]' if s['reach'] is False else ''}")
        if not ok_d:
            for sha, plddt in sorted(s["seen"]):
                print(f"             {dict(sha)} plddt={plddt}")
        print(f"             floor from A/A {s['aa']:.3f} B/B {s['bb']:.3f} "
              f"band {band * s['mA']:.3f}   grid {s['grids']} ttnn {s['wheels']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
