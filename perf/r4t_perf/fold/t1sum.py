import json, statistics as st, sys
d = sys.argv[1]; dig = {}
for arm in "AB":
    fb, opt = [], []
    for i in (1, 2, 3):
        L = [json.loads(x) for x in open(f"{d}/{arm}{i}.log") if x.startswith('{"step"')]
        dig.setdefault(arm, set()).add(tuple(r["master_digest"] for r in L))
        fb += [r["fold_bwd_s"] for r in L if r["step"] >= 2]; opt += [r["step_s"] for r in L if r["step"] >= 2]
        print(arm, i, [r["master_digest"][:8] for r in L], "aiclk", min(min(r["aiclk"]) for r in L), max(max(r["aiclk"]) for r in L))
    print(arm, "fold_bwd_s median", round(st.median(fb), 3), "range", min(fb), max(fb), "step_s median", round(st.median(opt), 2))
print("GATE", "PASS" if dig["A"] == dig["B"] and len(dig["A"]) == 1 else "FAIL", dig)
