"""SPD accuracy harness: grade an arm against a base arm on the ground-truth set, per the SPD charter's bars.

    ~/pfm-accuracy-data/venv/bin/python perf/spd/grade.py RUN_DIR [RUN_DIR ...] --base exact --test lpx \
        --mode normal|fast [--data ~/spd-data] [--out report.md]

Folds come from bench.py run over the accuracy inputs (the 11 PDB ids of perf/pfm_accuracy/set.tsv), 4+ seeds,
on as many chips as you like:
    bench.py --out RUN/<arm>/c<chip> --chip N --arm <spec> --inputs 9TH6,9W89,... --seed 101 --warm 3
Every RUN_DIR holding a bench.jsonl is read; records are grouped by (arch, arm), so one call grades WH and BH.

Per prediction, against the deposited structure (score.py from pfm-accuracy: DockQ v2, iRMSD, LRMSD = binder
RMSD after target fit, TM) plus CA-lDDT of the complex (global, 15 A inclusion, 0.5/1/2/4 A thresholds).
Docking metrics use the top-ranked sample of a fold (what a user keeps); CA-lDDT and pLDDT are the mean over
its samples, which halves their sampling noise.

FLOOR (A/A, base arm only): |base(s) - base(s')| over seed pairs, per metric, and the pose deviation between those
top poses (CA RMSD after a whole-complex fit, resolved residues only). What re-running the base with another seed
moves; every test delta is read against it.
PAIRED: test(s) - base(s), same seed; averaged per complex over seeds, then a percentile bootstrap over complexes
(20,000 resamples) for the 95 % interval. Same-seed pose deviation test vs base is quoted in A beside the floor.

Inputs outside the 11-set (size ladder, c730, PopVax: no deposited structure) get a NO-TRUTH table: the test's top
pose against the base's at the same seed (CA RMSD, CA-lDDT with the base as reference, pLDDT delta) beside the same
numbers for base seed pairs. So a speed run's own folds grade the lever at 256-1536 tokens; it does not enter the verdict.

Verdict, from state/spd/CHARTER.md:
  normal  every fold finite and present; DockQ, CA-lDDT, TM, pLDDT, ipTM intervals reach 0 or lie on the better
          side, LRMSD's likewise (lower is better); pLDDT and ipTM may sit below 0 by at most CONF_TOL
          (orchestrator 2026-10-09: 44 near-deterministic pairs resolve a 1e-4 shift, which is no accuracy loss);
          median same-seed pose deviation <= max(0.60 A kill bar,
          this arch's A/A seed floor median) (orchestrator 2026-10-08: on the 11-set a re-seed moves the top pose
          0.82 A, so the bar as written would fail a re-seed)
  fast    every fold finite; mean paired CA-lDDT and pLDDT drop each <= 0.03; no complex loses > 0.05 pLDDT
          (confidence collapse); docking success (DockQ >= 0.23, top pose) at most 5 points below base
  Either: fewer than 4 paired seeds, or a complex missing from one arm, is INSUFFICIENT, never PASS.
"""
import argparse, itertools, json, os, statistics, sys, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pfm_accuracy"))
from score import ca, chains, kabsch, score  # noqa: E402

SET = {l.split("\t")[0]: l.split("\t")[1:3] for l in (HERE.parent / "pfm_accuracy" / "set.tsv").read_text().splitlines()[1:]}
KILL_BAR = 0.60
CONF_TOL = 0.005  # normal: tolerated mean pLDDT / ipTM drop (fast bar is 0.03)
SUCCESS = 0.23


def lddt_ca(P, Q):
    dq = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
    m = (dq < 15.0) & ~np.eye(len(Q), dtype=bool)
    diff = np.abs(dp - dq)[m]
    return float(np.mean([(diff < t).mean() for t in (0.5, 1.0, 2.0, 4.0)]))


def ref_keys(ref, pdb):
    return [set(c) for c in chains(ref, SET[pdb])]


def pose_ca(cif, keys):
    """CA coordinates of both chains on the reference-resolved residues, concatenated (None if any is absent)."""
    p = chains(cif)
    try:
        return np.vstack([ca(p[i], sorted(keys[i])) for i in range(2)])
    except (KeyError, IndexError):
        return None


def rmsd(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R + t - Q) ** 2).sum(-1).mean()))


def score_one(job):
    cif, ref, pdb = job
    with tempfile.TemporaryDirectory() as tmp:
        s = score(cif, ref, SET[pdb], tmp, 0)
    p, r = chains(cif), chains(ref, SET[pdb])
    keys = [sorted(set(pc) & set(rc)) for pc, rc in zip(p, r)]
    P = np.vstack([ca(p[i], keys[i]) for i in range(2)])
    Q = np.vstack([ca(r[i], keys[i]) for i in range(2)])
    s["lddt_ca"] = lddt_ca(P, Q)
    return cif, {k: float(v) for k, v in s.items()}


def all_ca(cif):
    """(chain, label_seq) -> CA position over every polymer chain; the no-truth inputs have more than two."""
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    return {(ch.name, r.label_seq): np.array(r["CA"][0].pos.tolist()) for ch in st[0] for r in ch.get_polymer()
            if r.label_seq is not None and r.find_atom("CA", "*")}


def self_pair(x, y):
    """Top pose x against top pose y: CA RMSD after a fit, and CA-lDDT of x with y as the reference."""
    k = sorted(set(x) & set(y))
    P, Q = np.array([x[i] for i in k]), np.array([y[i] for i in k])
    return rmsd(P, Q), lddt_ca(P, Q)


def boot(vals, n=20000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def no_truth(free, failed, a):
    """Inputs without a deposited structure (the size ladder, c730, PopVax): the test arm's top pose against the
    base's at the same seed, beside what a re-seed of the base moves. Informational; the verdict is the 11-set's."""
    top = {k: (all_ca(f["cifs"][0]), statistics.mean(c["plddt"] for c in f["conf"])) for k, f in free.items()}
    out = [f"## No ground truth: {a.test} vs {a.base} top pose, same seed (A/A = {a.base} across seeds)\n",
           "| arch | input | A/A pairs | A/A CA RMSD med (A) | A/A CA-lDDT med | paired | CA RMSD med / max (A) | "
           "CA-lDDT med / min | pLDDT delta mean | failed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arch, inp in sorted({(k[0], k[2]) for k in top}):
        B = {k[3]: v for k, v in top.items() if k[0] == arch and k[2] == inp and k[1] == a.base}
        T = {k[3]: v for k, v in top.items() if k[0] == arch and k[2] == inp and k[1] == a.test}
        aa = [self_pair(B[s1][0], B[s2][0]) for s1, s2 in itertools.combinations(sorted(B), 2)]
        pr = [self_pair(T[s][0], B[s][0]) for s in sorted(set(B) & set(T))]
        dp = [T[s][1] - B[s][1] for s in sorted(set(B) & set(T))]
        nf = sum(1 for k in failed if k[0] == arch and k[2] == inp)
        f = lambda v, i, g: f"{g([x[i] for x in v]):.3f}" if v else "-"
        out.append(f"| {arch} | {inp} | {len(aa)} | {f(aa, 0, statistics.median)} | {f(aa, 1, statistics.median)} | "
                   f"{len(pr)} | {f(pr, 0, statistics.median)} / {f(pr, 0, max)} | {f(pr, 1, statistics.median)} / "
                   f"{f(pr, 1, min)} | {f'{statistics.mean(dp):+.4f}' if dp else '-'} | {nf} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--base", default="exact")
    ap.add_argument("--test", required=True)
    ap.add_argument("--mode", required=True, choices=["normal", "fast"])
    ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-seeds", type=int, default=4, help="charter: 4; lower only to test the harness")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    reps = []
    for d in a.runs:
        for f in [d] if d.is_file() else d.rglob("bench.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("ev") == "rep" and r["arm"] in (a.base, a.test):
                    reps.append(dict(r, _dir=f.parent))
    folds, free, failed, free_failed = {}, {}, [], []
    for r in reps:
        key = (r["arch"], r["arm"], r["input"], r["seed"])
        if r["err"] or not r["finite"] or not r.get("samples_conf"):
            (failed if r["input"] in SET else free_failed).append(key); continue
        sd = Path(r["struct_dir"])
        sd = sd if sd.exists() else r["_dir"] / sd.name  # a run copied off its box
        cifs = [sd / (f"{r['input']}.cif" if k == 0 else f"{r['input']}_model_{k}.cif")
                for k in range(len(r["samples_conf"]))]
        (folds if r["input"] in SET else free)[key] = dict(cifs=cifs, conf=r["samples_conf"])

    # Score every sample once; cache next to the data so a re-grade costs nothing.
    cache_f = a.data / "grade_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    jobs = [(str(c), str(a.data / "ref" / f"{k[2]}.cif"), k[2]) for k, f in folds.items() for c in f["cifs"]
            if str(c) not in cache]
    with ProcessPoolExecutor(a.jobs) as ex:
        for cif, s in ex.map(score_one, jobs, chunksize=2):
            cache[cif] = s
    cache_f.write_text(json.dumps(cache))

    refkeys = {p: ref_keys(a.data / "ref" / f"{p}.cif", p) for p in SET}
    fold_m = {}
    for k, f in folds.items():
        s = [cache[str(c)] for c in f["cifs"]]
        top = s[0]  # rank 0 = highest confidence_score
        fold_m[k] = dict(dockq=top["dockq"], irmsd=top["irmsd"], lrmsd=top["lrmsd"], tm=top["tm_complex"],
                         iptm=f["conf"][0]["iptm"], lddt_ca=statistics.mean(x["lddt_ca"] for x in s),
                         plddt=statistics.mean(c["plddt"] for c in f["conf"]),
                         top_ca=pose_ca(f["cifs"][0], refkeys[k[2]]))

    M = ["dockq", "lddt_ca", "tm", "lrmsd", "irmsd", "plddt", "iptm"]
    LOWER_BETTER = {"lrmsd", "irmsd"}
    out = [f"# grade: {a.test} vs {a.base}, {a.mode} mode\n",
           f"Records: {len(reps)} reps from {len(a.runs)} run dirs; failed/non-finite folds: {len(failed)}"
           + (f" {sorted(set(failed))[:12]}" if failed else "") + "\n"]
    verdicts = {}
    for arch in sorted({k[0] for k in fold_m}):
        B = {(k[2], k[3]): v for k, v in fold_m.items() if k[0] == arch and k[1] == a.base}
        T = {(k[2], k[3]): v for k, v in fold_m.items() if k[0] == arch and k[1] == a.test}
        pdbs = sorted({p for p, _ in B} | {p for p, _ in T})
        paired = sorted(set(B) & set(T))
        seeds = sorted({s for _, s in paired})
        P = out.append
        P(f"## {arch}\n")
        P(f"Complexes {len(pdbs)}; paired folds {len(paired)}; paired seeds per complex: "
          + ", ".join(f"{p} {sum(1 for q, _ in paired if q == p)}" for p in pdbs) + "\n")

        # A/A floor of the base arm.
        P(f"### FLOOR (A/A, {a.base} seed vs seed, median / mean of |diff|)\n")
        floor, pose_floor = {}, []
        for p in pdbs:
            ss = sorted(s for q, s in B if q == p)
            for s1, s2 in itertools.combinations(ss, 2):
                for m in M:
                    floor.setdefault(m, []).append(abs(B[(p, s1)][m] - B[(p, s2)][m]))
                x, y = B[(p, s1)]["top_ca"], B[(p, s2)]["top_ca"]
                if x is not None and y is not None:
                    pose_floor.append(rmsd(x, y))
        P("| metric | n pairs | median | mean |\n|---|---|---|---|")
        for m in M:
            v = floor.get(m, [])
            if v:
                P(f"| {m} | {len(v)} | {statistics.median(v):.4f} | {statistics.mean(v):.4f} |")
        if pose_floor:
            P(f"| top-pose CA RMSD (A) | {len(pose_floor)} | {statistics.median(pose_floor):.3f} | "
              f"{statistics.mean(pose_floor):.3f} |")
        P("")

        # Paired test - base.
        P(f"### PAIRED ({a.test} - {a.base}, same seed; mean over complexes [95 % bootstrap CI])\n")
        per = defaultdict(dict)
        for m in M:
            for p in pdbs:
                d = [T[(p, s)][m] - B[(p, s)][m] for q, s in paired if q == p]
                if d:
                    per[m][p] = statistics.mean(d)
        res = {m: boot(list(per[m].values())) for m in M if len(per[m]) >= 2}
        P("| metric | mean delta | 95 % CI | better side |\n|---|---|---|---|")
        for m, (mu, lo, hi) in res.items():
            P(f"| {m} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'lower' if m in LOWER_BETTER else 'higher'} |")
        dev = [rmsd(T[k]["top_ca"], B[k]["top_ca"]) for k in paired
               if T[k]["top_ca"] is not None and B[k]["top_ca"] is not None]
        succ_b = sum(B[k]["dockq"] >= SUCCESS for k in paired)
        succ_t = sum(T[k]["dockq"] >= SUCCESS for k in paired)
        n = max(len(paired), 1)
        if dev:
            P(f"\nSame-seed top-pose deviation {a.test} vs {a.base}: median {statistics.median(dev):.3f} A, max "
              f"{max(dev):.3f} A (n={len(dev)}); A/A seed floor median "
              f"{statistics.median(pose_floor) if pose_floor else float('nan'):.3f} A; kill bar {KILL_BAR} A.")
        P(f"Docking success (DockQ >= {SUCCESS}, top pose): {a.base} {succ_b}/{n} = {100 * succ_b / n:.1f} %, "
          f"{a.test} {succ_t}/{n} = {100 * succ_t / n:.1f} %.")
        pl_drop = {p: -per["plddt"][p] for p in per["plddt"]}
        P(f"Per-complex mean pLDDT change: " + ", ".join(f"{p} {-v:+.3f}" for p, v in sorted(pl_drop.items())) + "\n")

        # Verdict.
        why = []
        missing = [p for p in pdbs if not any(q == p for q, _ in paired)]
        if missing or len(seeds) < a.min_seeds or any(k[0] == arch for k in failed):
            v = "INSUFFICIENT"
            why.append(f"missing complexes {missing}" if missing else "")
            why.append(f"{len(seeds)} paired seeds (< {a.min_seeds})" if len(seeds) < a.min_seeds else "")
            why.append(f"{sum(k[0] == arch for k in failed)} failed/non-finite folds" if any(k[0] == arch for k in failed) else "")
        elif a.mode == "normal":
            for m, (mu, lo, hi) in res.items():
                bad = lo > 0 if m in LOWER_BETTER else hi < (-CONF_TOL if m in ("plddt", "iptm") else 0)
                if bad:
                    why.append(f"{m} worse, CI [{lo:+.4f}, {hi:+.4f}] excludes 0")
            bar = max(KILL_BAR, statistics.median(pose_floor)) if pose_floor else KILL_BAR
            if dev and statistics.median(dev) > bar:
                why.append(f"median pose deviation {statistics.median(dev):.3f} A > bar {bar:.3f} A "
                           f"(max of kill bar {KILL_BAR} A and A/A floor median)")
            v = "FAIL" if why else "PASS"
        else:
            if res["lddt_ca"][0] < -0.03:
                why.append(f"CA-lDDT drop {-res['lddt_ca'][0]:.4f} > 0.03")
            if res["plddt"][0] < -0.03:
                why.append(f"pLDDT drop {-res['plddt'][0]:.4f} > 0.03")
            col = [p for p, d in pl_drop.items() if d > 0.05]
            if col:
                why.append(f"pLDDT collapse (> 0.05) on {col}")
            if 100 * (succ_b - succ_t) / n > 5:
                why.append(f"docking success {100 * (succ_b - succ_t) / n:.1f} points below {a.base}")
            v = "FAIL" if why else "PASS"
        why = [w for w in why if w]
        verdicts[arch] = v
        P(f"### VERDICT {arch}: {v}" + (": " + "; ".join(why) if why else "") + "\n")
    if free or free_failed:
        out.append(no_truth(free, free_failed, a))
    out.append("SUMMARY: " + ", ".join(f"{arch} {v}" for arch, v in verdicts.items()))
    text = "\n".join(out)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
