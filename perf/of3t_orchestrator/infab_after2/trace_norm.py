#!/usr/bin/env python3
"""Grade AFTER2's traces against PREREGISTERED.md's addendum bars.

Run on qb2: python3 trace_norm.py /tmp/of3t/of3t-infab/work2
Drops `ttnn.deallocate` (finding 3), the D1 pair-bias scalar 1.0 -> sqrt(24) (finding 1) and the
explicit `compute_kernel_config:None` (finding 5), then compares BEFORE0 with AFTER1 line by line
and as a multiset.
"""
import collections, glob, json, re, sys

S = re.compile(r"(ttnn\.multiply_\t\(T\(\(128, 16\).*\)),(1\.0|4\.898979485566356)\)")


def norm(p):
    out = []
    for f in sorted(glob.glob(p + "/*")):
        for ln in open(f, errors="replace"):
            if "\tttnn.deallocate\t" in ln:
                continue
            ln = S.sub(r"\1,S)", ln)
            for k in (",compute_kernel_config:None", "compute_kernel_config:None,", "compute_kernel_config:None"):
                ln = ln.replace(k, "")
            out.append(ln.rstrip())
    return out


res = {}
for m in ("openfold3", "openbind", "protenix-v2", "opendde", "rf3", "rfd3"):
    a, b = norm(f"{sys.argv[1]}/{m}/_sc_before0/trace"), norm(f"{sys.argv[1]}/{m}/_sc_after1/trace")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    res[m] = {"lines": [len(a), len(b)], "differing_lines": len(diff) + abs(len(a) - len(b)),
              "first_differing": diff[:1], "multiset_equal": collections.Counter(a) == collections.Counter(b)}
print(json.dumps(res, indent=1))
