import glob
import pathlib
import sys
sys.path.insert(0, ".")
from perf.mgxaccuracy.contact import atoms, conditioning_graph
paths = sorted(set(glob.glob("examples/ground_truth_structures/*.cif")
                   + glob.glob("perf/ceilrfd3/targets/*.cif")
                   + glob.glob("perf/mgxaccuracy/targets/*.cif")
                   + glob.glob("perf/bhdesign/targets/*.cif")))
print("%-42s %7s %6s %11s %s" % ("structure", "tokens", "comps", "interchain", "verdict"))
bad = []
for p in paths:
    try:
        meta, xyz = atoms(pathlib.Path(p))
        g = conditioning_graph(meta, xyz)
    except SystemExit as e:
        print("%-42s  skipped: %s" % (p, e))
        continue
    nc = len(g["components"])
    if nc > 1:
        bad.append(p)
    print("%-42s %7d %6d %11d %s %s"
          % (p.split("/")[-1], g["n_token"], nc, g["inter_chain_edges"],
             "ok" if nc == 1 else "DISCONNECTED", g["components"][:3] if nc > 1 else ""))
print("\ndisconnected: %d of %d  -> %s"
      % (len(bad), len(paths), [b.split("/")[-1] for b in bad]))
