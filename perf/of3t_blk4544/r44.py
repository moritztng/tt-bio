import json
a64 = json.load(open("perf/of3t_blk4544/COTSCAN_SAT_64.json"))["rungs"]
a384 = json.load(open("perf/of3t_blk4544/COTSCAN_SAT_384.json"))["rungs"]
arms = ["base", "all_4544", "allref_4544"]
print("ds, masked to the 56 real tokens, in-frame float64 reference per width")
print("%-13s %14s %14s %10s %10s" % ("arm", "ds@384", "ds@64", "R", "verdict"))
for k in ("45", "44", "43", "0"):
    print("-- rung %s   ref_norm@384 %.12e   ref_norm@64 %.12e"
          % (k, a384[k]["ds"]["ref_norm_masked"], a64[k]["ds"]["ref_norm_masked"]))
    for arm in arms:
        x = a384[k]["ds"][arm]["masked"]["rel_l2"]
        y = a64[k]["ds"][arm]["masked"]["rel_l2"]
        R = x / y
        v = ""
        if k == "44":
            v = "CARRIER" if R <= 1.33 else ("REFUTED" if R >= 1.90 else "amber")
        print("%-13s %14.10f %14.10f %10.6f %10s" % (arm, x, y, R, v))
print()
print("norm_ratio (ours / reference), masked")
print("%-13s %10s %10s" % ("arm", "nr@384", "nr@64"))
for k in ("44",):
    for arm in arms:
        print("%-13s %10.6f %10.6f" % (arm,
              a384[k]["ds"][arm]["masked"]["norm_ratio"], a64[k]["ds"][arm]["masked"]["norm_ratio"]))
print()
print("dz, same frame")
print("%-13s %14s %14s %10s" % ("arm", "dz@384", "dz@64", "R"))
for k in ("45", "44", "0"):
    print("-- rung", k)
    for arm in arms:
        x = a384[k]["dz"][arm]["masked"]["rel_l2"]
        y = a64[k]["dz"][arm]["masked"]["rel_l2"]
        print("%-13s %14.10f %14.10f %10.6f" % (arm, x, y, x / y))
