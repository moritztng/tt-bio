#!/usr/bin/env python3
"""One real 298-aa protenix-v2 fold, counting every ttnn call by site. No timing, no re-run.

The point is the CONVERSION. The ledger multiplies a block-level ms by 480 (48 blocks x 10
recycles); a Transition call is chunked into 10 sub-calls and transition_s is not chunked at all, so
T3's ops do not run at the pair track's multiplicity. This counts them instead of assuming.

Recycles stay at the production 10. The diffusion sampler is out of scope, so SAMPLING_STEPS is
dropped to 8 purely to save wall clock -- it cannot change a trunk call count.
"""
import collections
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                            # noqa: E402
import tt_baseline as B                                                # noqa: E402

OPS = ["linear", "matmul", "add", "add_", "multiply", "multiply_", "layer_norm", "rms_norm",
       "softmax", "permute", "transpose", "concat", "reshape", "chunk", "clone", "slice"]

counts = collections.Counter()
shapes = {}
ON = {"v": False}


def call_site():
    chain = [f"{fr.filename.split('/')[-1]}:{fr.lineno}"
             for fr in reversed(traceback.extract_stack())
             if "tt_bio/" in fr.filename and __file__ not in fr.filename]
    return (chain[0] if chain else "?"), tuple(chain[:5])


def wrap(name, fn):
    def inner(*a, **kw):
        if not ON["v"]:
            return fn(*a, **kw)
        ON["v"] = False
        try:
            site, chain = call_site()
            key = (name, site, chain)
            counts[key] += 1
            if key not in shapes:
                shapes[key] = [("x".join(map(str, v.padded_shape)) + ":"
                                + str(v.memory_config().buffer_type).split(".")[-1][:3])
                               for v in list(a) + list(kw.values()) if isinstance(v, ttnn.Tensor)]
        finally:
            ON["v"] = True
        return fn(*a, **kw)
    return inner


saved = []
for nm in OPS:
    f = getattr(ttnn, nm, None)
    if callable(f):
        saved.append((nm, f))
        setattr(ttnn, nm, wrap(nm, f))

B.SAMPLING_STEPS = 8
one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / ".msa_t3census",
                                      ROOT / "examples" / "prot300.yaml",
                                      ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
ON["v"] = True
t, metrics = one_fold()
ON["v"] = False
for nm, f in saved:
    setattr(ttnn, nm, f)

print(f"\nfold {t:.1f}s  pLDDT {metrics.get('plddt')}  tokens {metrics.get('n_tokens')}  "
      f"recycles {B.RECYCLING_STEPS}", flush=True)

T3 = {"2038", "2046", "2055", "2064", "2066", "2145", "2148", "2242", "2223", "2227", "2231",
      "2235", "2239", "2255", "2259", "1893", "1900", "1906", "1876", "2001", "2011"}
rows = []
for (op, site, chain), n in counts.most_common():
    rows.append({"op": op, "site": site, "chain": list(chain), "calls": n,
                 "shapes": shapes[(op, site, chain)]})
json.dump({"fold_s": t, "recycles": B.RECYCLING_STEPS, "n_tokens": metrics.get("n_tokens"),
           "plddt": metrics.get("plddt"), "total_calls": sum(counts.values()), "rows": rows},
          open(sys.argv[1], "w"), indent=1)

print("\n=== every ttnn call site in T3's slice, whole fold ===", flush=True)
print("%-12s %-8s %7s  %-46s %s" % ("op", "line", "calls", "chain(outer->inner)", "shapes"))
tot = collections.Counter()
for r in rows:
    line = r["site"].split(":")[-1]
    if line not in T3:
        continue
    tot[(r["op"], line)] += r["calls"]
    print("%-12s %-8s %7d  %-46s %s"
          % (r["op"], line, r["calls"], ">".join(x.split(":")[-1] for x in r["chain"][1:]),
             " | ".join(r["shapes"][:3])))
print("\n=== per (op, line) totals ===", flush=True)
for (op, line), n in sorted(tot.items(), key=lambda kv: -kv[1]):
    print("  %-12s %-6s %7d calls/fold" % (op, line, n))
lin = sum(n for (op, line), n in tot.items() if op == "linear" and line in {"2046", "2055", "2066"})
print(f"\nTRANSITION ttnn.linear CALLS PER FOLD = {lin}")
print("PairformerLayer body add_ calls/fold = "
      + str(sum(n for (op, line), n in tot.items()
                if op == "add_" and line in {"2223", "2227", "2231", "2235", "2239"})))
print("total ttnn calls counted (all sites) = " + str(sum(counts.values())))
