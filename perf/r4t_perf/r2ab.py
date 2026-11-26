import json, statistics as s, sys, os
O = os.path.expanduser("~/r4t/perf/r2")
R = {a: [json.loads(l) for l in open(f"{O}/{a}.log") if l.startswith('{"step"')] for a in ("A1", "A2", "B1", "B2")}
for a, rs in R.items():
    print(a, len(rs), [(round(r["rebuild_split"]["load_s"], 2), round(r["fwd_s"], 2), round(r["step_s"], 2)) for r in rs],
          "aiclk", min(r["aiclk"][0] for r in rs), max(r["aiclk"][1] for r in rs))
for k in ("load_s", "rebuild_s", "fwd_s", "step_s"):
    g = lambda a: [r["rebuild_split"]["load_s"] if k == "load_s" else r[k] for r in R[a]]
    print(k, "A1 %.3f A2 %.3f B1 %.3f B2 %.3f | A %.3f B %.3f" % (*[s.median(g(a)) for a in R], s.median(g("A1") + g("A2")), s.median(g("B1") + g("B2"))))
A, B = R["A1"], R["B1"]
print("A1==A2 digests", [r["master_digest"] for r in A] == [r["master_digest"] for r in R["A2"]], "B1==B2", [r["master_digest"] for r in B] == [r["master_digest"] for r in R["B2"]])
for ra, rb in zip(A, B):
    print("step", ra["step"], "loss A %.6f B %.6f rel %.2e" % (ra["loss"], rb["loss"], abs(ra["loss"] - rb["loss"]) / abs(ra["loss"])),
          "dead", ra["dead"], rb["dead"], "idle", ra["idle_dev_w"], rb["idle_dev_w"], ra["master_digest"], rb["master_digest"])
