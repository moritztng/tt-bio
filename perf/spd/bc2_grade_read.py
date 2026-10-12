"""Read the grade2 BC2 campaigns: per arm x seed design metrics and s/iter."""
import csv, glob, os, re, statistics as st, sys
R = sys.argv[1]
rows = {}
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
