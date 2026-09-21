import json
for sc in ("single", "never"):
    d = json.load(open("/home/ttuser/of3t_d10d107/out/SCORE_%s.json" % sc))
    print("==", sc, "| zero-participation steps", d["zero_participation_steps"],
          "| clipped", d["samples_clipped"], "/", d["samples_total"])
    print("   np64 vs up64 (reference selfcheck): %.6e" % d["reference_selfcheck"]["worst_rel"])
    print("   up32 vs up64 (reference fp32 cost): %.6e" % d["reference_fp32_cost"]["worst_rel"])
    for a in d["arms"]:
        print("   %-10s vs %-42s %.6e  over1e-6 %2d/%d  zero_part %s"
              % (a["arm"], a["reference"], a["worst_rel"], a["n_over_1e-06"],
                 a["n_readings"], a["worst_rel_at_zero_participation"]))
