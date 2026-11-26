import sys, torch
sys.path.insert(0, "/home/ttuser/.coworker/wt/r4t-perf/perf/r4t_perf")
from b1prime import errs, wide
for snap, shp, k in [(0, (768, 6144), 13), (1, (256, 768), 4)]:
    for g in torch.load(f"/home/ttuser/r4t/perf/b1h/snap{snap}.pt")["groups"]:
        w = wide(g.float())
        if tuple(w.shape[1:]) == shp:
            x = w[k:k+1]; ee, eh = errs(x)
            s = torch.linalg.svdvals(x[0].double()); s = s / s[0]
            print(snap, shp, k, "e_emu %.4f e_host %.4f" % (float(ee[0]), float(eh[0])), "sv/sv0 min %.2e median %.2e" % (float(s[-1]), float(s.median())))
