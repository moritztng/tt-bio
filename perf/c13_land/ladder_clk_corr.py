import glob, json, os, statistics, sys
base = "/home/ttuser/.coworker/wt/c13-land-first/perf/c13_land/ladder_attrib"
rows = []
for d in sorted(glob.glob(base + "/*/")):
    n = os.path.basename(d.rstrip("/"))
    f = sorted(glob.glob(d + "/*/results.json"))
    if not f:
        continue
    j = json.load(open(f[0])); e = j[0] if isinstance(j, list) else j
    rt = e.get("runtime_s")
    clk = [int(x) for x in open(d + "/clk.txt").read().split() if x.strip().isdigit()]
    if not clk:
        continue
    model, arm, rep = n.split("_")
    rows.append((model, arm, rep, rt, statistics.mean(clk), min(clk), max(clk),
                 100.0 * sum(1 for c in clk if c >= 1300) / len(clk), len(clk)))
print(f"{'run':<18}{'runtime_s':>10}{'mean MHz':>10}{'min':>6}{'max':>6}{'%>=1300':>9}{'n':>5}")
for r in rows:
    print(f"{r[0]+'_'+r[1]+'_'+r[2]:<18}{r[3]:>10}{r[4]:>10.0f}{r[5]:>6}{r[6]:>6}{r[7]:>9.1f}{r[8]:>5}")
print()
for model in ("boltz2", "openbind"):
    sub = [r for r in rows if r[0] == model]
    print(f"--- {model}: runtime vs mean clock")
    for r in sorted(sub, key=lambda x: x[3]):
        print(f"   {r[1]}/{r[2]}  {r[3]:>6} s at {r[4]:>6.0f} MHz mean, {r[7]:>5.1f}% of samples at full clock")
    # implied time at full clock
    print(f"   hoist-off min {min(r[3] for r in sub if r[1]=='h0')}  hoist-on min {min(r[3] for r in sub if r[1]=='h1')}")
