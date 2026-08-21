import json, sys
def walk(o, out):
    if isinstance(o, dict):
        if o.get("metric") == "kabsch_rmsd" and "within_noise_floor" in o:
            out.append(o)
        for v in o.values():
            walk(v, out)
for path in sys.argv[1:]:
    print("===", path)
    try:
        d = json.load(open(path))
    except Exception as e:
        print("   ", e); continue
    hits = []
    walk(d, hits)
    for o in hits:
        X, R, D = o["cross"], o["ref_floor"], o["dev_floor"]
        fl = o["floor_mean"]
        std = max(R.get("std", 0), D.get("std", 0))
        bar = fl + std
        dor = o.get("dev_over_ref_floor")
        print("    X=%.4f (n=%d std=%.4f)  R=%.4f (std=%.4f)  D=%.4f (std=%.4f)"
              % (X["mean"], X["n"], X.get("std", 0), R["mean"], R.get("std", 0), D["mean"], D.get("std", 0)))
        print("    floor=%.4f  +maxstd=%.4f  bar=%.4f   X/floor=%.3f  X/bar=%.3f  D/R=%s  within=%s  floor_inflated_by_dev=%s"
              % (fl, std, bar, X["mean"] / fl, X["mean"] / bar,
                 ("%.3f" % dor) if dor else dor, o["within_noise_floor"], o.get("floor_inflated_by_dev")))
