"""Read the grade2 BC2 campaigns: per arm x seed design metrics and s/iter."""
import csv, glob, os, re, statistics as st, sys
R = sys.argv[1]
rows = {}
TRAJ = {}  # arm -> per-trajectory last-iteration i_pTM, the continuous outcome a test can use
for d in sorted(glob.glob(f"{R}/*_s*/")):
    arm, seed = re.match(r".*/(\w+)_s(\d+)/$", d).groups()
    m = {}
    for s in glob.glob(f"{d}campaign_*/summary.csv"):
        for r in csv.DictReader(open(s)):
            m[(r["scope"], r["metric"])] = (float(r["mean"]), int(r["samples"]))
    tr = glob.glob(f"{d}campaign_*/1_Trajectories/!_Trajectories.csv")
    spi, last_iptm = [], []
    if tr:
        for r in csv.DictReader(open(tr[0])):
            t = dict(kv.split("=") for kv in r["Timing"].split(";") if "=" in kv)
            L = glob.glob(f"{d}campaign_*/1_Trajectories/{r['design']}/*_losses.csv")
            if not L: continue
            lr = list(csv.DictReader(open(L[0])))
            if int(r["trajectory"]) > 1: spi.append(float(t["design"]) / len(lr))
            last_iptm.append(float(lr[-1]["hPDL1.iptm"]))
    TRAJ.setdefault(arm, []).extend(last_iptm)
    if not m: continue
    g = lambda k: m.get(k, (float("nan"), 0))
    rows[(arm, int(seed))] = dict(
        traj=int(g(("campaign", "trajectories"))[0]),
        done=int(g(("campaign", "terminated:completed"))[0] if g(("campaign", "terminated:completed"))[1] else 0),
        redes=int(g(("campaign", "redesign_candidates"))[0]), acc=int(g(("campaign", "accepted_designs"))[0]),
        plddt=g(("final", "pLDDT")), iptm=g(("final", "i_pTM")),
        last_iptm=st.mean(last_iptm) if last_iptm else float("nan"),
        spi=st.median(spi) if spi else float("nan"))
print(f"{'arm':7} {'seed':4} traj done redes acc  pLDDT(n)   i_pTM  traj_last_iptm  s/iter")
for (a, s), r in sorted(rows.items()):
    print(f"{a:7} {s:4} {r['traj']:4} {r['done']:4} {r['redes']:5} {r['acc']:3}  {r['plddt'][0]:.3f}({r['plddt'][1]})  {r['iptm'][0]:.3f}  {r['last_iptm']:.3f}  {r['spi']:.2f}")
for a in sorted({a for a, _ in rows}):
    rs = [r for (x, _), r in rows.items() if x == a]
    pl = [r["plddt"][0] for r in rs if r["plddt"][1]]
    print(f"ARM {a}: campaigns {len(rs)} traj {sum(r['traj'] for r in rs)} completed {sum(r['done'] for r in rs)} "
          f"redesign {sum(r['redes'] for r in rs)} accepted {sum(r['acc'] for r in rs)} "
          f"pLDDT mean-of-campaigns {st.mean(pl) if pl else float('nan'):.3f} (n={len(pl)}) "
          f"traj_last_iptm {st.mean(r['last_iptm'] for r in rs):.3f} s/iter median {st.median([r['spi'] for r in rs if r['spi']==r['spi']]):.2f}")


# Accepted designs are rare, so counts alone cannot tell a regression from a dry patch. Every numerical
# change sends a design trajectory somewhere new, which makes two arms two independent samples of one
# distribution if the lever is harmless. So test that: a two-sided Fisher exact test on accepted per
# trajectory, and a permutation test on the mean last-iteration i_pTM per trajectory.
from math import comb
import random


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]]; p of every table with the same margins at least as unlikely."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def pr(x):
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def perm_mean_diff(x, y, n=20000, seed=0):
    rng = random.Random(seed); obs = abs(st.mean(x) - st.mean(y)); pool = x + y; k = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= obs - 1e-12: k += 1
    return (k + 1) / (n + 1)


arms = sorted({a for a, _ in rows})
base = next((a for a in arms if a.startswith("base")), None)
for a in arms:
    if a == base or base is None:
        continue
    acc_b = sum(r["acc"] for (x, _), r in rows.items() if x == base)
    tr_b = sum(r["traj"] for (x, _), r in rows.items() if x == base)
    acc_a = sum(r["acc"] for (x, _), r in rows.items() if x == a)
    tr_a = sum(r["traj"] for (x, _), r in rows.items() if x == a)
    pf = fisher_two_sided(acc_b, tr_b - acc_b, acc_a, tr_a - acc_a)
    xb, xa = TRAJ.get(base, []), TRAJ.get(a, [])
    line = f"TEST {a} vs {base}: accepted {acc_a}/{tr_a} vs {acc_b}/{tr_b}, Fisher p={pf:.3f}"
    if len(xb) >= 3 and len(xa) >= 3:
        line += (f"; last i_pTM per trajectory {st.mean(xa):.3f} (n={len(xa)}) vs {st.mean(xb):.3f} "
                 f"(n={len(xb)}), permutation p={perm_mean_diff(xa, xb):.3f}")
    print(line)
