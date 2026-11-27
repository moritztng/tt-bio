import json, statistics as st, os
d = os.path.expanduser("~/r4t/perf/r3ab"); dig = {}
print(open(d + "/run.log").read().strip())
for arm in "AB":
    R = []
    for i in (1, 2, 3):
        L = [json.loads(x) for x in open(f"{d}/{arm}{i}.log") if x.startswith('{"step"')]
        dig.setdefault(arm, set()).add(tuple(r["master_digest"] for r in L)); R += [r for r in L if r["step"] >= 2]
        print(arm, i, [r["master_digest"][:8] for r in L], "aiclk", min(min(r["aiclk"]) for r in L), max(max(r["aiclk"]) for r in L))
    print(arm, {k: round(st.median(r[k] for r in R), 3) for k in ("step_s", "fwd_s", "bwd_s", "fold_bwd_s", "opt_s", "rebuild_s")}, "load_s", round(st.median(r["rebuild_split"]["load_s"] for r in R), 3), {0:0})
print("DIGESTS", "IDENTICAL" if dig["A"] == dig["B"] and len(dig["A"]) == 1 else "DIFFER")
