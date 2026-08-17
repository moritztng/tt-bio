import json, sys
d = json.load(open(sys.argv[1]))
print("host", d["host"], "card", d["card"], "grid", d.get("grid"), "ttnn", d["ttnn"],
      "rec", d["recycling_steps"], "steps", d["sampling_steps"])
for r in d["runs"]:
    if r.get("error"):
        print(r["arm"], "ERROR", r["error"][:100]); continue
    w = r["walls_ms"]
    g = lambda k: w.get(k)
    print("%-8s #%d %8.3f s  cif=%s  DT=%s DTLtok=%s DTLatom=%s AdaLNatom=%s AdaLNtok=%s "
          "APBtok=%s APBatom=%s PFL=%s  %s" % (
        r["arm"], r["ix"], r["fold_s"], list(r["cif_sha256"].values()),
        g("stage:DiffusionTransformer"), g("block:DiffusionTransformerLayer|token"),
        g("block:DiffusionTransformerLayer|atom"), g("body:AdaLN|atom"), g("body:AdaLN|token"),
        g("body:AttentionPairBias|token"), g("body:AttentionPairBias|atom"),
        g("block:PairformerLayer"), r["lever_stats"]))
