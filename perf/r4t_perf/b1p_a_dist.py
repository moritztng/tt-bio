import json, sys, statistics, torch
sys.path.insert(0, "/home/ttuser/.coworker/wt/r4t-perf/perf/r4t_perf")
import b1prime as G
shapes = json.loads(open("/home/ttuser/.coworker/wt/r4t-perf/perf/r4t_perf/rfd4_denoiser_2d_shapes.txt").read().splitlines()[2])
dev = json.load(open("/home/ttuser/r4t/perf/b1p/dev.json"))
per = {(tuple(s), k): ed for s, k, ed, eh in dev["per"]}
gen = torch.Generator().manual_seed(0)
out, allr = [], []
for (r, c), cnt in shapes:
    w = G.wide(torch.randn(cnt, r, c, generator=gen))
    ee, eh = G.errs(w)
    q = [per[(tuple(w.shape[1:]), k)] / float(ee[k]) for k in range(cnt)]
    allr += q
    out.append(dict(shape=[r, c], n=cnt, med=round(statistics.median(q), 4), min=round(min(q), 3), max=round(max(q), 3)))
for o in out: print(o)
print("ALL n", len(allr), "median", round(statistics.median(allr), 4), "frac>1", round(sum(x > 1 for x in allr) / len(allr), 3))
json.dump(dict(per_shape=out, all_median=statistics.median(allr)), open("/home/ttuser/r4t/perf/b1p/a_dist.json", "w"), indent=1)
