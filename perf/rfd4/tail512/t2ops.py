"""T2CM calls in an elt_prof ops.csv: kernel ms, per-RISC ms, and the op before each."""
import csv, statistics, sys
r = list(csv.DictReader(open(sys.argv[1])))
ms = lambda x, k: float(x[k] or 0) / 1e6
hit = [i for i, x in enumerate(r) if "compute_ln" in x.get("COMPUTE KERNEL SOURCE", "")]
print(len(r), "ops,", len(hit), "T2 calls")
d = [ms(r[i], "DEVICE KERNEL DURATION [ns]") for i in hit]
if d:
    print("kernel ms median %.3f min %.3f max %.3f" % (statistics.median(d), min(d), max(d)))
    for k in ("DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]", "DEVICE TRISC1 KERNEL DURATION [ns]"):
        print(k, "median %.3f" % statistics.median(ms(r[i], k) for i in hit))
    prev = {}
    for i in hit:
        p = r[i - 1]
        key = (p["OP CODE"], p.get("COMPUTE KERNEL SOURCE", "")[-50:])
        prev.setdefault(key, []).append(ms(r[i], "DEVICE KERNEL DURATION [ns]"))
    for k, v in sorted(prev.items(), key=lambda kv: -len(kv[1])):
        print(len(v), "%.3f" % statistics.median(v), k)
    print("first 12:", ["%.3f" % x for x in d[:12]])
