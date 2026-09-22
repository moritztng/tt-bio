"""Does the OF3 trunk's ending-node bias orientation move a structure a user receives?

`of3t-trunkfwd` (D90) put the shipped OF3 trunk forward at 2.793661e-01 against upstream's
own float64 re-composition, 46.67x upstream's own bf16, and named the site: `transpose_bias`
at `tt_bio/openfold3_trunk.py`, shipped True for every non-OpenBind checkpoint. That is a
relative error on an intermediate pair representation. This script answers the only question
that sets urgency -- how many Angstrom of final coordinates it is -- in the shape
`of3t-pairbias` used for the other flag.

Four quantities, and only the four together mean anything:

  accuracy   Ca-RMSD of each arm against the 1UBQ experimental structure. Reported as rank 0
             (what a user receives) and best-of-5 (what the sampler can reach);
  lever      arm against arm at a MATCHED seed and a matched card, which is what the flag does;
  seed floor same arm, different seed, every pair. Below the floor a lever is noise and above
             it a finding; the bare Angstrom says neither. ubiquitin is 76 aa and gets its OWN
             floor -- the 512 aa cell's 1.84 A is not this target's;
  ranking    pLDDT, pTM, confidence_score and whether rank 0 IS the best sample, because a
             small coordinate move can still reorder which of the five is served.

Controls, all measured rather than asserted:
  zero       a structure against itself, so the instrument's zero is read and not assumed (A16);
  break      the same structure with its residue correspondence reversed, which must be large or
             the metric is not discriminating;
  lever no-op  the shipped arm against the arm that FORCES the shipped value through the new
             environment override, byte-compared -- the override must be inert where it agrees;
  card       the shipped arm re-run on a different card, because an answer that depends on which
             card produced it is a hard stop on this fleet.

    python3 perf/of3t_foldab/fold_ab.py
"""
import argparse
import hashlib
import itertools
import json
import os
import statistics

import gemmi

OUT = "perf/of3t_foldab/fold_ab.json"
GT = "examples/ground_truth_structures/ubiquitin.pdb"
ARMS = ("ship", "fix")


def ca(path):
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    return {(c.name, r.seqid.num): r.find_atom("CA", "*").pos
            for c in st[0] for r in c if r.find_atom("CA", "*") is not None}


def rmsd(a, b):
    """Superposed Ca-RMSD over the shared residue keys. gemmi superposes in double."""
    keys = sorted(set(a) & set(b))
    pa, pb = ([a[k] for k in keys], [b[k] for k in keys]) if keys else (
        list(a.values()), list(b.values()))
    n = min(len(pa), len(pb))
    return gemmi.superpose_positions(pa[:n], pb[:n]).rmsd, n


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/ttuser/of3t_foldab_runs/fold")
ap.add_argument("--out", default=OUT)
ap.add_argument("--gt", default=GT, help="experimental structure, or 'none' for a target "
                                         "with no deposited reference")
ap.add_argument("--results", default="openfold3_results_ubq")
ap.add_argument("--stem", default="ubq")
ap.add_argument("--label", default="1UBQ 76 aa")
ap.add_argument("--kill-bar", type=float, default=0.60)
ap.add_argument("--samples", type=int, default=5)
a = ap.parse_args()
OUT, GT = a.out, a.gt

gt = ca(GT) if GT != "none" else None
runs, conf, files = {}, {}, {}
for d in sorted(os.listdir(a.root)):
    base = os.path.join(a.root, d, a.results)
    sd = os.path.join(base, "structures")
    if not os.path.isdir(sd):
        continue
    fs = [os.path.join(sd, a.stem + ".cif")] + [
        os.path.join(sd, "%s_model_%d.cif" % (a.stem, i)) for i in range(1, a.samples)]
    fs = [f for f in fs if os.path.exists(f)]
    files[d] = fs
    runs[d] = [ca(f) for f in fs]
    conf[d] = json.load(open(os.path.join(base, "results.json")))[0]

seeds = sorted({int(n.split("_s")[1]) for n in runs if n.split("_s")[0] in ARMS})
rep = {"ground_truth": GT, "n_ca_gt": len(gt) if gt else 0, "seeds": seeds,
       "target": a.label, "kill_bar_A": a.kill_bar,
       "msa_depth": conf[next(iter(conf))]["msa_depth"], "runs": {}, "lever": {},
       "seed_floor": {}, "summary": {}, "controls": {}, "confidence": {}}
print("%s   ground truth %s: %s CA   seeds %s   MSA depth %s\n"
      % (a.label, GT, len(gt) if gt else "-", seeds, rep["msa_depth"]))

print("accuracy against %s  (rank 0 is the structure a user receives)" % GT)
for name in sorted(runs, key=lambda n: (n.split("_s")[0], int(n.split("_s")[1]))):
    vs = [rmsd(m, gt)[0] for m in runs[name]] if gt else [float("nan")] * len(runs[name])
    c = conf[name]
    rep["runs"][name] = dict(n_samples=len(vs), rank0=vs[0], best=min(vs),
                             median=statistics.median(vs), all=vs, plddt=c["plddt"],
                             ptm=c["ptm"], confidence_score=c["confidence_score"],
                             ranked_is_best=abs(vs[0] - min(vs)) < 1e-9,
                             rank0_sha=sha(files[name][0]))
    r = rep["runs"][name]
    if not gt:
        print("  %-12s pLDDT %.4f  pTM %.4f  conf %.5f  (no deposited reference)"
              % (name, c["plddt"], c["ptm"], c["confidence_score"]))
        continue
    print("  %-12s rank0 %6.3f A  best %6.3f A  median %6.3f A  pLDDT %.4f  pTM %.4f  "
          "conf %.5f  %s" % (name, vs[0], min(vs), statistics.median(vs), c["plddt"],
                             c["ptm"], c["confidence_score"],
                             "ranked=best" if r["ranked_is_best"] else "ranked>best"))

for arm in ARMS:
    ns = [n for n in runs if n.split("_s")[0] == arm]
    r0 = [rep["runs"][n]["rank0"] for n in ns]
    bst = [rep["runs"][n]["best"] for n in ns]
    rep["summary"][arm] = dict(
        n=len(ns),
        rank0_mean=statistics.fmean(r0) if gt else None,
        rank0_median=statistics.median(r0) if gt else None,
        rank0_min=min(r0) if gt else None, rank0_max=max(r0) if gt else None,
        best_mean=statistics.fmean(bst) if gt else None,
        best_median=statistics.median(bst) if gt else None,
        plddt_mean=statistics.fmean(rep["runs"][n]["plddt"] for n in ns),
        ptm_mean=statistics.fmean(rep["runs"][n]["ptm"] for n in ns),
        conf_mean=statistics.fmean(rep["runs"][n]["confidence_score"] for n in ns),
        ranked_is_best=sum(rep["runs"][n]["ranked_is_best"] for n in ns))
    s = rep["summary"][arm]
    if not gt:
        print("  -> %-5s n=%d  pLDDT %.4f  pTM %.4f  conf %.5f"
              % (arm, s["n"], s["plddt_mean"], s["ptm_mean"], s["conf_mean"]))
        continue
    print("  -> %-5s n=%d  rank0 mean %.3f median %.3f range %.3f-%.3f | best-of-5 mean %.3f"
          " | pLDDT %.4f | ranked==best %d/%d"
          % (arm, s["n"], s["rank0_mean"], s["rank0_median"], s["rank0_min"], s["rank0_max"],
             s["best_mean"], s["plddt_mean"], s["ranked_is_best"], s["n"]))

print("\nlever: shipped vs flipped, SAME seed, SAME card")
lv, lv_all = [], []
for s in seeds:
    sh, fx = "ship_s%d" % s, "fix_s%d" % s
    if sh in runs and fx in runs:
        v, n = rmsd(runs[sh][0], runs[fx][0])
        per = [rmsd(runs[sh][i], runs[fx][i])[0]
               for i in range(min(len(runs[sh]), len(runs[fx])))]
        rep["lever"][str(s)] = dict(rank0=v, n_ca=n, per_sample=per)
        lv.append(v)
        lv_all += per
        print("  seed %d: rank0 %6.3f A   per-sample %s"
              % (s, v, " ".join("%.3f" % x for x in per)))
if lv:
    rep["lever"]["rank0"] = dict(n=len(lv), mean=statistics.fmean(lv),
                                 median=statistics.median(lv), min=min(lv), max=max(lv))
    rep["lever"]["per_sample"] = dict(n=len(lv_all), mean=statistics.fmean(lv_all),
                                      median=statistics.median(lv_all),
                                      min=min(lv_all), max=max(lv_all))
    print("  -> lever rank0   n=%d mean %.3f A  median %.3f A  range %.3f-%.3f A"
          % (len(lv), statistics.fmean(lv), statistics.median(lv), min(lv), max(lv)))
    print("  -> lever sample-paired n=%d mean %.3f A  median %.3f A  range %.3f-%.3f A"
          % (len(lv_all), statistics.fmean(lv_all), statistics.median(lv_all),
             min(lv_all), max(lv_all)))

print("\nseed floor: same arm, different seed, every pair (this target's own)")
for arm in ARMS:
    ss = sorted(int(n.split("_s")[1]) for n in runs if n.split("_s")[0] == arm)
    pairs = {"%dv%d" % (x, y): rmsd(runs["%s_s%d" % (arm, x)][0],
                                    runs["%s_s%d" % (arm, y)][0])[0]
             for x, y in itertools.combinations(ss, 2)}
    if not pairs:
        continue
    vals = list(pairs.values())
    rep["seed_floor"][arm] = dict(pairs=pairs, n_pairs=len(vals), min=min(vals), max=max(vals),
                                  mean=statistics.fmean(vals), median=statistics.median(vals))
    print("  %-5s n=%d pairs  mean %6.3f A  median %6.3f A  range %.3f-%.3f A"
          % (arm, len(vals), statistics.fmean(vals), statistics.median(vals),
             min(vals), max(vals)))

print("\nconfidence deltas, paired by seed (flipped minus shipped)")
for s in seeds:
    sh, fx = "ship_s%d" % s, "fix_s%d" % s
    if sh in runs and fx in runs:
        rep["confidence"][str(s)] = {
            k: dict(ship=rep["runs"][sh][k], fix=rep["runs"][fx][k],
                    delta=rep["runs"][fx][k] - rep["runs"][sh][k])
            for k in (("plddt", "ptm", "confidence_score")
                      + (("rank0", "best") if gt else ()))}
        c = rep["confidence"][str(s)]
        print("  seed %d: pLDDT %.4f -> %.4f (%+.4f)   pTM %+.4f   conf %+.5f"
              % (s, c["plddt"]["ship"], c["plddt"]["fix"], c["plddt"]["delta"],
                 c["ptm"]["delta"], c["confidence_score"]["delta"]))
for k in ("plddt", "ptm", "confidence_score"):
    ds = [rep["confidence"][str(s)][k]["delta"] for s in seeds if str(s) in rep["confidence"]]
    if ds:
        rep["confidence"][k + "_delta"] = dict(n=len(ds), mean=statistics.fmean(ds),
                                               median=statistics.median(ds),
                                               min=min(ds), max=max(ds))
        print("  -> %-17s delta mean %+.5f  range %+.5f..%+.5f"
              % (k, statistics.fmean(ds), min(ds), max(ds)))

print("\nreading: the lever against this target's own floor and against the kill bar")
floor = rep["seed_floor"].get("ship", {}).get("pairs", {})
fv = list(floor.values())
if lv and fv:
    over = [x for x in lv if x >= a.kill_bar]
    # The paired null is NOT the seed floor. With the seed matched and the model unperturbed
    # the arm-vs-arm RMSD is exactly 0 -- the lever no-op control below measures that, byte
    # for byte -- so any non-zero paired number is the flag. The floor says how that size
    # compares with variation a user already accepts.
    rnd = __import__("random").Random(20260920)
    pool = lv + fv
    obs = statistics.fmean(lv) - statistics.fmean(fv)
    hits = 0
    N = 100000
    for _ in range(N):
        rnd.shuffle(pool)
        if statistics.fmean(pool[:len(lv)]) - statistics.fmean(pool[len(lv):]) >= obs:
            hits += 1
    p = (hits + 1) / (N + 1)
    rep["reading"] = dict(
        lever_mean=statistics.fmean(lv), lever_median=statistics.median(lv),
        lever_max=max(lv), floor_mean=statistics.fmean(fv), floor_median=statistics.median(fv),
        floor_max=max(fv), ratio_to_floor=statistics.fmean(lv) / statistics.fmean(fv),
        kill_bar=a.kill_bar, n_seeds_at_or_over_bar=len(over), n_seeds=len(lv),
        mean_over_bar=statistics.fmean(lv) >= a.kill_bar,
        permutation_p_lever_gt_floor=p, permutation_draws=N)
    r = rep["reading"]
    print("  lever  mean %.3f A  median %.3f A  max %.3f A   (n=%d matched seeds)"
          % (r["lever_mean"], r["lever_median"], r["lever_max"], len(lv)))
    print("  floor  mean %.3f A  median %.3f A  max %.3f A   (n=%d seed pairs, shipped arm)"
          % (r["floor_mean"], r["floor_median"], r["floor_max"], len(fv)))
    print("  lever / floor = %.2fx   one-sided permutation p(lever > floor) = %.4f "
          "over %d draws" % (r["ratio_to_floor"], p, N))
    print("  kill bar %.2f A: mean is %s it; %d of %d individual seeds are at or over it"
          % (a.kill_bar, "OVER" if r["mean_over_bar"] else "UNDER", len(over), len(lv)))

print("\ncontrols")
ref = "ship_s0"
z, _ = rmsd(runs[ref][0], runs[ref][0])
rep["controls"]["zero_self"] = z
print("  A16 zero      %s against itself: %.6f A  (measured, not assumed)" % (ref, z))

ks = sorted(runs[ref][0])
rev = dict(zip(ks, reversed([runs[ref][0][k] for k in ks])))
b, _ = rmsd(runs[ref][0], rev)
rep["controls"]["break_reversed_correspondence"] = b
print("  break         %s against itself with the residue correspondence REVERSED: %.3f A"
      % (ref, b))

if "shipforce_s0" in runs:
    same = [sha(x) == sha(y) for x, y in zip(files[ref], files["shipforce_s0"])]
    v, _ = rmsd(runs[ref][0], runs["shipforce_s0"][0])
    rep["controls"]["lever_noop"] = dict(byte_identical=all(same), n_files=len(same), rmsd=v,
                                         ship_sha=sha(files[ref][0]),
                                         forced_sha=sha(files["shipforce_s0"][0]))
    print("  lever no-op   shipped default vs override FORCED to the shipped value: %d/%d "
          "files byte-identical, rank0 RMSD %.6f A (%s / %s)"
          % (sum(same), len(same), v, sha(files[ref][0]), sha(files["shipforce_s0"][0])))

if "shipc1_s0" in runs:
    same = [sha(x) == sha(y) for x, y in zip(files[ref], files["shipc1_s0"])]
    v, _ = rmsd(runs[ref][0], runs["shipc1_s0"][0])
    rep["controls"]["cross_card"] = dict(byte_identical=all(same), n_files=len(same), rmsd=v,
                                         card0_sha=sha(files[ref][0]),
                                         card1_sha=sha(files["shipc1_s0"][0]))
    print("  card identity shipped arm, seed 0, card 0 vs card 1: %d/%d files byte-identical, "
          "rank0 RMSD %.6f A" % (sum(same), len(same), v))

if "fix_s0" in runs:
    rep["controls"]["lever_moves"] = dict(
        ship_sha=sha(files[ref][0]), fix_sha=sha(files["fix_s0"][0]),
        byte_identical=sha(files[ref][0]) == sha(files["fix_s0"][0]))
    print("  lever moves   shipped vs flipped digests differ: %s != %s"
          % (sha(files[ref][0]), sha(files["fix_s0"][0])))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
