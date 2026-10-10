"""Paired per-trajectory end-of-design metrics for the BC2 grade2 campaigns.

The validation ensemble folds on host JAX (bindcraft2.campaign_predictor(validation="jax")), so the
acceptance scores are bit-for-bit the reference's own and the arms can only differ through the
device gradient loop. That makes the trajectory's own last-round losses the place to read a lever.
"""
import csv, glob, re, statistics as st, sys
R = sys.argv[1]
M = ["hPDL1.iptm", "hPDL1.ptm", "hPDL1.plddt_loss", "hPDL1.iptm_loss", "hPDL1.interface_pae"]
arms = {}
for d in sorted(glob.glob(f"{R}/*_s*/")):
    arm, seed = re.match(r".*/(\w+)_s(\d+)/$", d).groups()
    T = glob.glob(f"{d}campaign_*/1_Trajectories/!_Trajectories.csv")
    if not T: continue
    for r in csv.DictReader(open(T[0])):
        L = glob.glob(f"{d}campaign_*/1_Trajectories/{r['design']}/*_losses.csv")
        if not L: continue
        rows = list(csv.DictReader(open(L[0])))
        def lastval(m):
            v = [x[m] for x in rows if x.get(m) not in (None, "")]
            return float(v[-1]) if v else float("nan")
        anneal = [x for x in rows if x["phase"] == "anneal"]
        def annealval(m):
            v = [x[m] for x in anneal if x.get(m) not in (None, "")]
            return float(v[-1]) if v else float("nan")
        vals = {m: lastval(m) for m in M}
        vals.update({"anneal:" + m: annealval(m) for m in M})
        arms.setdefault(arm, {})[(int(seed), int(r["trajectory"]))] = dict(
            rounds=len(rows), term=r["terminated"], **vals)
base = arms["base"]
for arm in sorted(arms):
    if arm == "base": continue
    keys = sorted(set(base) & set(arms[arm]))
    print(f"\n== {arm} vs base, {len(keys)} paired trajectories (seed, trajectory)")
    for m in M + ["anneal:" + x for x in M] + ["rounds"]:
        pr = [(base[k][m], arms[arm][k][m]) for k in keys]
        pr = [(x, y) for x, y in pr if x == x and y == y]
        if len(pr) < 2: continue
        b = [x for x, _ in pr]; a = [y for _, y in pr]
        d = [x - y for x, y in zip(a, b)]
        wins = sum(x > 0 for x in d)
        print(f"  {m:22} base {st.mean(b):7.3f}  {arm} {st.mean(a):7.3f}  delta {st.mean(d):+7.3f} "
              f"(sd {st.stdev(d):.3f}, {wins}/{len(d)} up)")
    same = sum(abs(base[k]["hPDL1.iptm"] - arms[arm][k]["hPDL1.iptm"]) < 1e-6 for k in keys)
    print(f"  identical last-round iptm: {same}/{len(keys)}")
