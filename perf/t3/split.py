#!/usr/bin/env python3
"""Split the fold census into the trunk Pairformer stack and everything else.

`tenstorrent.py:2306` is `Pairformer.__call__`'s `s, z = block(...)` loop line, so a call whose frame
chain contains it ran inside the 48-block pf_stack. Everything else (the diffusion sampler's DiT, the
atom encoder/decoder, the confidence head) is out of scope and is only counted, not chased.
"""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
PF = "tenstorrent.py:2306"
T3 = {"2038", "2046", "2055", "2064", "2066", "2145", "2148", "2242", "2223", "2227", "2231",
      "2235", "2239", "2255", "2259", "1893", "1900", "1906", "1876", "2001", "2011"}

print("fold_s", d["fold_s"], "recycles", d["recycles"], "tokens", d["n_tokens"],
      "plddt", d["plddt"], "total ttnn calls", d["total_calls"])

inn, out = collections.Counter(), collections.Counter()
shp = {}
for r in d["rows"]:
    if not r["site"].startswith("tenstorrent.py"):
        continue
    line = r["site"].split(":")[-1]
    if line not in T3:
        continue
    k = (r["op"], line)
    if PF in r["chain"]:
        inn[k] += r["calls"]
        shp.setdefault(k, []).append((r["calls"], r["shapes"][:2], ">".join(r["chain"][1:3])))
    else:
        out[k] += r["calls"]

print("\n=== T3 ops INSIDE the 48-block trunk Pairformer stack (in scope) ===")
print("%-12s %-6s %8s   shapes seen (calls, first two operands, chain[1:3])" % ("op", "line", "calls/fold"))
for k, n in sorted(inn.items(), key=lambda kv: -kv[1]):
    print("%-12s %-6s %8d   %s" % (k[0], k[1], n, shp[k][:3]))

print("\n=== the same sites OUTSIDE the trunk pf_stack (counted, out of scope, not chased) ===")
for k, n in sorted(out.items(), key=lambda kv: -kv[1])[:12]:
    print("%-12s %-6s %8d" % (k[0], k[1], n))

blocks = inn[("add_", "2223")]
print(f"\nPairformerLayer executions per fold (add_@2223 inside pf_stack) = {blocks}")
tz = inn[("layer_norm", "2038")] - inn[("layer_norm", "2242")]
print(f"Transition ttnn.linear calls/fold inside the pf_stack = "
      f"{inn[('linear', '2046')] + inn[('linear', '2055')] + inn[('linear', '2066')]}")
print(f"Transition swiglu invocations/fold inside the pf_stack (= layer_norm@2038) = "
      f"{inn[('layer_norm', '2038')]}")
print(f"  of which chunked transition_z sub-calls = {tz}  (transition_s = {inn[('layer_norm','2242')]})")
