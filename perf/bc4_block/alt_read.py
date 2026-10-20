import json, statistics as S, sys
D = sys.argv[1]
d = json.load(open(f"{D}/alt/devprof.json"))
w, m = d["iteration_s"], d["arms"]
for k in ("0", "1"):
    x = d["by_arm"][k]
    print(k, "n", len(x["iteration_s"]), "iter med", round(S.median(x["iteration_s"]), 3),
          {p: round(v["median_s"], 3) for p, v in x["summary"].items()})
def blocks(v):
    return [(v[i + 1] + v[i + 2] - v[i] - v[i + 3]) / 2 for i in range(0, len(v) - 3, 4)]
for name, v in (("iter", w), ("evo.taped", [sum(r["evo.taped"]) for r in d["reps"]]),
                ("evo.backward", [sum(r["evo.backward"]) for r in d["reps"]])):
    b = blocks(v)
    print(f"{name} ABBA block diffs (on - off)", [round(x, 3) for x in b], "mean", round(S.mean(b), 3),
          "se", round(S.stdev(b) / len(b) ** .5, 3))
rows = [l.split("\t") for l in open(f"{D}/aiclk.tsv")]
ck = sorted(int(r[1]) for r in rows); ld = [float(r[2]) for r in rows]
print("aiclk median", ck[len(ck) // 2], "min", ck[0], "load", min(ld), "-", max(ld), "mean", round(S.mean(ld), 1))
