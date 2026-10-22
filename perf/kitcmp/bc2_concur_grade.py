"""Grade an interleaved trajectories-per-card A/B: python3 bc2_concur_grade.py <raw dir>

Each chunk directory holds one bench.py campaign (bench.jsonl's `ev=campaign` line), its leg.txt
(`cpus=`, `seed=`, `N=`) and cpu_sampler.py's cpu.tsv. Chunks pair by (core budget, seed): the A and
B of one pair ran back to back on the same cores and the same campaign seed. Per chunk the metric is
campaign wall net of the predictor build, per campaign iteration. hidden = 1 - s_B / s_A per pair,
median over pairs per budget. A pair is dropped when its two chunks saw other-process load on the
pinned set differing by more than 20 % and by more than 0.5 cores (the floor keeps a 0.1 vs 0.2 core
idle box from counting as a 2x swing).
"""
import json, statistics, sys
from pathlib import Path


def chunk(d):
    meta = dict(kv.split("=", 1) for kv in (d / "leg.txt").read_text().split("\n", 1)[0].split())
    rows = [json.loads(x) for x in open(d / "bench.jsonl") if x.startswith("{")]
    c = next((r for r in rows if r.get("ev") == "campaign"), None)
    if c is None or c.get("err") or not c.get("iterations"):
        return None
    cpu = [l.split("\t") for l in (d / "cpu.tsv").read_text().splitlines()[1:]]
    other = statistics.mean(float(r[4]) for r in cpu) if cpu else None
    clk = next(iter((c.get("aiclk") or {}).values()), {})
    net = c["wall_s"] - (c.get("setup_s") or 0)
    return dict(name=d.name, cpus=meta["cpus"], seed=meta["seed"], N=int(meta["N"]), wall=c["wall_s"], setup=c.get("setup_s"),
                iters=c["iterations"], s_it=net / c["iterations"], s_it_gross=c["wall_per_iter"], other=other,
                load=c.get("loadavg"), aiclk_med=clk.get("median"), aiclk_min=clk.get("min"), sha=c.get("sha"),
                nval=sum(1 for t in c["trajectories"] if t.get("terminated") is None),
                its=[t["iterations"] for t in c["trajectories"]])


raw = Path(sys.argv[1])
C = [c for d in sorted(raw.iterdir()) if (d / "done").exists() and (c := chunk(d))]
out = {"chunks": C, "budgets": {}}
for cpus in sorted({c["cpus"] for c in C}, key=lambda s: int(s.split("-")[-1])):
    pairs = []
    for seed in sorted({c["seed"] for c in C if c["cpus"] == cpus}):
        a = [c for c in C if c["cpus"] == cpus and c["seed"] == seed and c["N"] == 1]
        b = [c for c in C if c["cpus"] == cpus and c["seed"] == seed and c["N"] > 1]
        if not (a and b):
            continue
        a, b = a[0], b[0]
        d = abs(a["other"] - b["other"])
        keep = not (d > 0.2 * max(a["other"], b["other"]) and d > 0.5)
        pairs.append(dict(seed=seed, s_A=round(a["s_it"], 4), s_B=round(b["s_it"], 4), hidden=round(1 - b["s_it"] / a["s_it"], 4),
                          other_A=round(a["other"], 2), other_B=round(b["other"], 2), kept=keep))
    kept = [p["hidden"] for p in pairs if p["kept"]]
    out["budgets"][cpus] = dict(cores=len(range(int(cpus.split("-")[0]), int(cpus.split("-")[-1]) + 1)), pairs=pairs,
                                hidden_median=statistics.median(kept) if kept else None, n_kept=len(kept),
                                s_A_median=statistics.median(p["s_A"] for p in pairs if p["kept"]) if kept else None,
                                s_B_median=statistics.median(p["s_B"] for p in pairs if p["kept"]) if kept else None)
json.dump(out, sys.stdout, indent=1)
print()
