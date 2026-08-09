#!/usr/bin/env python3
"""Which records failed to re-run (dt=0.0 + error) — pf_block_ops silently scores those 0.0."""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
R = d["records"]
T3_SITES = {"2038", "2046", "2055", "2064", "2066", "2145", "2148", "2242", "2223", "2227",
            "2231", "2235", "2239", "2255", "2259", "1893", "1900", "1906", "1876", "1884",
            "1936", "1937", "1940", "1941", "1942", "1950", "1951", "1952", "1953", "2001",
            "2009", "2011", "372"}
err = collections.Counter()
for r in R:
    if r.get("error"):
        err[(r["op"], r["site"], r["error"][:90])] += 1
print("=== records with a re-run error (scored 0.0 s) ===")
for k, c in err.most_common(40):
    print(c, k)

print("\n=== T3 slice records, one line each ===")
for r in R:
    site = r["site"].split(":")[-1]
    if site not in T3_SITES:
        continue
    ins = " | ".join("x".join(map(str, i["shape"])) + ":" + i["buf"][:3] + ":" + i["dtype"]
                     for i in r["in"])
    out = ("x".join(map(str, r["out"]["shape"])) + ":" + r["out"]["buf"][:3]) if r["out"] else "-"
    print("i=%-4d %-14s %-8s %9.1f us  chain=%-40s in[%s] -> %s%s"
          % (r["i"], r["op"], site, r["s"] * 1e6, ">".join(r["chain"][1:]), ins, out,
             "  ERR:" + r["error"][:60] if r.get("error") else ""))
