"""First host line items for F, derived from c12-profiled-fold's per-unit capture.

Not this row's own instrument and no substitute for it. It exists because `c12-profiled-fold`
already measured, at a pinned and during-sampled 1350 MHz, both the BARE synced wall of six
units grabbed out of a live fold and the device kernel time inside them. The difference is the
exposed host + dispatch time of that unit, weighted by the fold's own integer call count: the
first column of the table this row owes, from an instrument that is not a Python profiler and
therefore cannot book device wait onto whichever host frame happens to be spinning.

Read the bare wall from the `*_bare/unit.json` captures, NOT from `composed.json`'s
`in_kernel_pct`. That field is device divided by the PROFILED wall, so using it to derive host
time prices the device profiler's own overhead as host work: it turns DiffusionModule's
0.8908 ms/call of host into 16.7653 ms/call and makes the six units sum to 3.5635 s of host
against a remainder that bounds all exposed host time at 1.6720 s. An impossible total is the
only reason the substitution was caught, which is why the total is checked against the
remainder here rather than reported on its own.

Usage: host_from_units.py <composed.json> <bare_unit.json...> [--out out.json]
"""
import json, sys

args = [a for a in sys.argv[1:] if a != "--out"]
out_path = None
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]
    args = args[:-1]

d = json.load(open(args[0]))
bare = {}
for p in args[1:]:
    for name, u in json.load(open(p))["units"].items():
        bare[name] = dict(wall_ms=u["synced_wall_ms_per_call"], reps=u.get("reps"), src=p)

rows, missing = [], []
for u in d["units"]:
    name, dev, calls = u["unit"], u["device_ms_per_call"], u["calls"]
    if name not in bare:
        missing.append(name)
        continue
    w = bare[name]["wall_ms"]
    host_ms = w - dev
    rows.append(dict(unit=name, calls=calls, dev_ms=dev, bare_ms=w, host_ms=host_ms,
                     host_s=calls * host_ms / 1000.0, dev_s=u["s_per_fold"],
                     in_kernel_pct_bare=100.0 * dev / w,
                     profiled_ms=u.get("profiled_wall_ms"), src=bare[name]["src"]))
rows.sort(key=lambda r: -r["host_s"])

print("%-24s %6s %9s %9s %9s %9s %9s %7s" % (
    "unit", "calls", "dev_ms", "bare_ms", "host_ms", "host_s", "dev_s", "in_k%"))
for r in rows:
    print("%-24s %6d %9.4f %9.4f %9.4f %9.4f %9.4f %7.2f" % (
        r["unit"], r["calls"], r["dev_ms"], r["bare_ms"], r["host_ms"],
        r["host_s"], r["dev_s"], r["in_kernel_pct_bare"]))
host = sum(r["host_s"] for r in rows)
dev = sum(r["dev_s"] for r in rows)
print("%-24s %6s %9s %9s %9s %9.4f %9.4f" % ("TOTAL", "", "", "", "", host, dev))
if missing:
    print("NO BARE CAPTURE, host not derivable: %s" % ", ".join(missing))

rem = d["remainder_s"]
print()
print("fold_s                     %9.4f" % d["fold_s"])
print("device_total_s             %9.4f  (coverage %.1f %%)" % (d["device_total_s"], d["coverage_pct"]))
print("remainder_s                %9.4f   ceiling on ALL exposed host time" % rem)
print("host named by these units  %9.4f   = %.1f %% of the remainder" % (host, 100 * host / rem))
print("host NOT named by a unit   %9.4f   featurisation, CIF write, recycle glue, confidence" % (rem - host))
if host > rem:
    print("IMPOSSIBLE: named host exceeds the remainder that bounds all host time.")

out = dict(rows=rows, host_named_s=host, remainder_s=rem, host_unnamed_s=rem - host,
           fold_s=d["fold_s"], missing_bare=missing,
           source="c12-profiled-fold runs/composed.json + *_bare/unit.json")
if out_path:
    json.dump(out, open(out_path, "w"), indent=2)
