import json,sys
d=json.load(open(sys.argv[1]))
print(d["host"],"chip",d["card"],d["board"],"sha",d["sha"],"exact_ops",d.get("exact_ops"))
for k in ("wall","wall_aa","untaped"):
    print(k, [(r.get("fwd_s"),r.get("bwd_s"),r.get("aiclk_fwd") or r.get("aiclk"),r.get("aiclk_bwd"),r.get("grad_digest") or r.get("out_digest")) for r in d["arms"][k]])
dr=d["arms"]["drain"]; print("drain step",dr["step"])
for leg in ("fwd","bwd"):
    t=dr[leg]; print(f"\n{leg}: verb_sum {t['verb_sum_s']} roof_sum {t['roof_sum_s']} host {t['host_verb_s']} ({t['host_verb_calls']} calls)")
    for r in t["by_verb"][:18]: print(f"  {r['verb']:28s} n={r['n']:4d} {r['self_s']*1e3:8.2f} ms share={r['share']} roof={r['roof_s']*1e3:7.2f} ms {r['pct_of_roof']}% {r['bound']} {r['gb']}GB {r['tflop']}TF")
    for r in t["by_closure"][:10]: print(f"    {r['closure'][:60]:60s} x{r['fired']} {r['s']*1e3:.2f} ms")
