import json, statistics as st, itertools
A = "/home/ttuser/scratch/uod/ab_%s_r%d.json"
data = {}
for arm in ("base", "patched"):
    for r in (0, 1):
        d = json.load(open(A % (arm, r)))
        pl = [x["fold_s"] for x in d["baseline"] if x["arm"] == "plain"]
        ins = [x["fold_s"] for x in d["baseline"] if x["arm"] == "instr"]
        digs = {list(x["cifs"].values())[0] for x in d["baseline"]}
        data[(arm, r)] = (pl, ins, digs, d["cold_s"], d["env"]["git_head"][:8])
        print("%-8s r%d head=%s cold=%6.3f plain=%s instr=%s" % (
            arm, r, d["env"]["git_head"][:8], d["cold_s"],
            ["%.3f" % v for v in pl], ["%.3f" % v for v in ins]))

digs = set().union(*[v[2] for v in data.values()])
print("\ndistinct CIF sha256 across all %d timed folds: %d  %s" % (
    sum(len(v[0]) + len(v[1]) for v in data.values()), len(digs), [d[:16] for d in digs]))

pool = {a: sorted(itertools.chain(*[data[(a, r)][0] for r in (0, 1)])) for a in ("base", "patched")}
for a in ("base", "patched"):
    print("%-8s plain folds n=%d  median %.4f  min %.4f  max %.4f  spread %.3f %%" % (
        a, len(pool[a]), st.median(pool[a]), min(pool[a]), max(pool[a]),
        100 * (max(pool[a]) / min(pool[a]) - 1)))
mb, mp = st.median(pool["base"]), st.median(pool["patched"])
print("\nRATIO (pooled plain medians): %.5fx   delta %.4f s" % (mb / mp, mb - mp))

# A/A floor: the same arm against itself across the two rounds, worst case of the two arms.
floors = []
for a in ("base", "patched"):
    m0, m1 = st.median(data[(a, 0)][0]), st.median(data[(a, 1)][0])
    f = max(m0, m1) / min(m0, m1)
    floors.append(f)
    print("A/A %-8s r0 %.4f vs r1 %.4f  -> %.5fx" % (a, m0, m1, f))
print("worst-case A/A floor %.5fx" % max(floors))
print("\nper-round ratios (base/patched):")
for r in (0, 1):
    b, p = st.median(data[("base", r)][0]), st.median(data[("patched", r)][0])
    print("  r%d lead=%s  %.4f / %.4f = %.5fx" % (
        r, "base" if r == 0 else "patched", b, p, b / p))
print("\nevery base fold vs every patched fold: %d of %d pairs have base slower" % (
    sum(1 for x in pool["base"] for y in pool["patched"] if x > y),
    len(pool["base"]) * len(pool["patched"])))
