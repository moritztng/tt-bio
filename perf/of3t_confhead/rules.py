"""Score candidate sample-selection rules against the truth, over folds already run.

The candidate set is FIXED HERE, before any per-sample number exists, because a rule chosen
after seeing which one wins on nine seeds of one target is a rule fitted to nine seeds of one
target. Each candidate is scored on exactly the same five samples per seed -- no extra
diffusion, no extra model work -- so what is being compared is a selection POLICY and its
Angstrom cost, never an accuracy win.

Candidates, and why each is in the set:

  shipped       0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash. AF3 SI 5.9.3, what
                openfold3_fold.py:277 does today. On one chain ipTM and has_clash are
                identically zero, so it reduces to 0.2*ptm + 0.5*disorder;
  family        the same rule with the monomer fallback the other four models already carry
                (boltz2.py:6238, worker.py:1065, rf3/confidence.py:108): iptm := ptm when
                there is no interface. The smallest change that makes OpenFold3 consistent
                with the family, and it restores the 0.8 weight to a term that means something;
  no_disorder   0.8*iptm + 0.2*ptm - 100*has_clash. Isolates the disorder term, so a win for
                `family` can be attributed to the fallback rather than to dropping RASA;
  ptm           pTM alone, the degenerate-interface convention rf3 states in one line;
  plddt         mean pLDDT alone. The head's own per-atom confidence, and the signal Boltz-2
                weights 4/5. It is the natural rival and it must be allowed to win;
  boltz         (4*plddt + ptm)/5, Boltz-2's rule transcribed, as a cross-model control;
  plddt_ptm     0.5*plddt + 0.5*ptm, to see whether the two carry different information;
  gpde          -gPDE, AF3 SI 5.7 Eq 16, contact-weighted predicted distance error. Lower is
                better, so it is negated. Upstream computes it and we currently discard it;
  pae           -mean off-diagonal PAE, the other pair-error reduction.

Reported per rule: the Angstrom a user is served (mean rank 0), against the two policies that
bracket every rule -- picking at random and a perfect selector -- and the seed floor measured
on the same target. A rule that does not beat random is not a selector.

    python3 perf/of3t_confhead/rules.py --root /tmp/of3t/of3t-confhead/fold
    python3 perf/of3t_confhead/rules.py --selftest
"""
import argparse
import glob
import json
import os
import statistics

RULES = {
    "shipped": lambda r: 0.8 * r["iptm"] + 0.2 * r["ptm"] + 0.5 * r["disorder"]
                         - 100.0 * r["has_clash"],
    "family": lambda r: 0.8 * (r["iptm"] or r["ptm"]) + 0.2 * r["ptm"] + 0.5 * r["disorder"]
                        - 100.0 * r["has_clash"],
    "no_disorder": lambda r: 0.8 * r["iptm"] + 0.2 * r["ptm"] - 100.0 * r["has_clash"],
    "ptm": lambda r: r["ptm"],
    "plddt": lambda r: r["plddt"],
    "boltz": lambda r: (4 * r["plddt"] + (r["iptm"] or r["ptm"])) / 5,
    "plddt_ptm": lambda r: 0.5 * r["plddt"] + 0.5 * r["ptm"],
    # The rule this row actually ships (openfold3_fold.sample_ranking_score): ipTM's weight
    # goes to pLDDT when there is no interface. Scored here beside its eight rivals, because
    # shipping a formula that was never on the table would be choosing by taste.
    "of3_fix": lambda r: (0.8 * (r["iptm"] or r["plddt"]) + 0.2 * r["ptm"]
                          + 0.5 * r["disorder"] - 100.0 * r["has_clash"]),
    "gpde": lambda r: -r["gpde"],
    "pae": lambda r: -r["pae_offdiag_mean"],
}


def score(runs):
    """runs: list of per-run lists of per-sample records. Returns rule -> served mean RMSD."""
    out = {}
    for name, f in RULES.items():
        served, chosen = [], []
        for samples in runs:
            k = max(range(len(samples)), key=lambda i: f(samples[i]))
            served.append(samples[k]["rmsd_ca"])
            chosen.append(k)
        out[name] = {"served_mean": statistics.fmean(served),
                     "served_median": statistics.median(served),
                     "served_worst": max(served),
                     "picks_best": sum(s == min(x["rmsd_ca"] for x in r)
                                       for s, r in zip(served, runs)),
                     "chosen": chosen, "served": served}
    return out


def brackets(runs):
    return {"random": statistics.fmean(statistics.fmean(x["rmsd_ca"] for x in r) for r in runs),
            "perfect": statistics.fmean(min(x["rmsd_ca"] for x in r) for r in runs),
            "worst_case": statistics.fmean(max(x["rmsd_ca"] for x in r) for r in runs)}


def report(arm, runs):
    b = brackets(runs)
    s = score(runs)
    print(f"\n{arm}: {len(runs)} runs x {len(runs[0])} samples   "
          f"random {b['random']:.3f} A | perfect {b['perfect']:.3f} A | "
          f"always-worst {b['worst_case']:.3f} A")
    for name in sorted(s, key=lambda n: s[n]["served_mean"]):
        r = s[name]
        flag = "" if r["served_mean"] < b["random"] else "   <- worse than random"
        print(f"  {name:12s} served {r['served_mean']:6.3f} A  median {r['served_median']:6.3f}"
              f"  worst {r['served_worst']:6.3f}  picks best {r['picks_best']}/{len(runs)}{flag}")
    return {"brackets": b, "rules": s}


def _selftest():
    """A rule table needs a case where the rules disagree and the right answer is known.

    Two samples: one good, compact and confident; one bad, looser and slightly less
    confident. Only the RASA term prefers the bad one. The outcome is not what the family
    fallback alone would suggest, and the arithmetic is worth stating BEFORE the fold:

      shipped, one chain:  0.2*ptm + 0.5*disorder  ->  a rival needs 2.5x the disorder gap
      family fallback:     1.0*ptm + 0.5*disorder  ->  a rival needs 0.5x the disorder gap

    so the fallback cuts the pTM edge a better sample needs by 5x but does NOT remove the
    RASA term's leverage. On this case, with a 0.05 pTM gap against a 0.30 disorder gap,
    BOTH still pick the 1.6 A sample. Whether real samples show gaps of that shape is the
    measurement; this only fixes what each rule is committed to.
    """
    good = dict(rmsd_ca=0.5, ptm=0.90, iptm=0.0, disorder=0.10, has_clash=0.0, plddt=0.90,
                gpde=3.0, pae_offdiag_mean=4.0)
    bad = dict(rmsd_ca=1.6, ptm=0.85, iptm=0.0, disorder=0.40, has_clash=0.0, plddt=0.80,
               gpde=5.0, pae_offdiag_mean=6.0)
    s = score([[good, bad]])
    fooled = [n for n in RULES if s[n]["chosen"] == [1]]
    # `of3_fix` is in this list ON PURPOSE and it is a scope statement about the shipped rule:
    # it replaces the DEAD ipTM term, and it leaves AF3's 0.5*disorder alone. Where disorder is
    # zero -- every sample of every seed on 1UBQ -- that is the whole defect; where it is not,
    # a 0.30 disorder gap still outweighs the 0.8*pLDDT + 0.2*pTM edge this case gives it.
    assert fooled == ["shipped", "family", "of3_fix"], f"unexpected fooled set: {fooled}"
    for name in ("no_disorder", "ptm", "plddt", "boltz", "plddt_ptm", "gpde", "pae"):
        assert s[name]["chosen"] == [0], f"{name} should pick the good sample"
    assert brackets([[good, bad]])["perfect"] == 0.5
    # the flip threshold each rule commits to, as a pTM gap per unit of disorder gap
    for name, need in (("shipped", 2.5), ("family", 0.5), ("no_disorder", 0.0)):
        print(f"  {name:12s} a better sample needs pTM higher by {need} x the disorder gap")
    print("selftest: on a 0.05 pTM gap against a 0.30 disorder gap the shipped rule, the "
          "family fallback AND this row's of3_fix all serve the 1.6 A sample over the 0.5 A "
          "one -- of3_fix replaces the dead ipTM term and leaves AF3's disorder weight alone. "
          "The seven rules that do not read RASA at all serve the 0.5 A one. On 1UBQ disorder "
          "is 0.0 on every sample, so this case is a bound on the fix, not a reading of it.")


# The CLI lives under a main guard because analyze.py imports RULES from here; a
# module-level parse_args() would run on ITS argv and refuse ITS flags.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/of3t/of3t-confhead/fold")
    ap.add_argument("--out", default="perf/of3t_confhead/rules.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        raise SystemExit(0)

    rep = {"root": a.root, "candidates": sorted(RULES), "arms": {}}
    for arm in ("ship", "fix"):
        paths = sorted(glob.glob(os.path.join(a.root, f"{arm}_s*", "samples.json")))
        runs = [json.load(open(p))["per_sample"] for p in paths]
        if not runs:
            print(f"{arm}: no runs under {a.root}")
            continue
        rep["arms"][arm] = report(arm, runs)
        rep["arms"][arm]["n_runs"] = len(runs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)
