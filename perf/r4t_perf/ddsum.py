import json, statistics as st, os, sys
d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/r4t/perf/ddab"); dig = {}
for arm in "AB":
    R = []
    for i in (1, 2, 3):
        L = [json.loads(x) for x in open(f"{d}/{arm}{i}.log") if x.startswith('{"step"')]
        dig.setdefault(arm, set()).add(tuple(r["master_digest"] for r in L)); R += [r for r in L if r["step"] >= 2]
        print(arm, i, [r["master_digest"][:8] for r in L], "aiclk", min(r["aiclk"][0] for r in L), max(r["aiclk"][1] for r in L),
              "fwd", [r["fwd_s"] for r in L])
    print(arm, "medians", {k: round(st.median(r[k] for r in R), 3) for k in ("step_s", "fwd_s", "bwd_s", "fold_bwd_s", "opt_s", "rebuild_s")}, "n", len(R))
print("DIGESTS", "IDENTICAL" if dig["A"] == dig["B"] and len(dig["A"]) == 1 else "DIFFER", dig)
