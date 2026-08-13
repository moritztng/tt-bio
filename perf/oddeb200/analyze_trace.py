#!/usr/bin/env python3
"""Decide the diffusion trace at 512 aa from the four uninstrumented legs of run_trace2.sh.

Answers three questions, in the order that matters:
  1. PARITY. The trace claim is bit-identical. Every fold writes a CIF; if the trace-ON digests
     differ from the trace-OFF digests the lever is dead regardless of its wall.
  2. THE CROSS-PROCESS A/A FLOOR. offA vs offB and onA vs onB are identical configurations in
     different processes. Whatever they disagree by is the floor any delta must clear. Section 10
     lists this as owed: pass 2's -0.202 s glue delta sat below a 0.67 s process spread.
  3. THE DELTA, quoted only if it clears that floor.

Schema note: tt_baseline writes `warm_times_s` (the warm folds) and `warm_folds` (a per-fold list
whose entry 0 is the COLD fold, arm_fold="cold"). Digests come from every entry, walls only from
`warm_times_s`.
"""
import json, statistics
from pathlib import Path

O = Path("/home/ttuser/.coworker/wt/opendde-beat-b200/perf/oddeb200")
LEGS = [("offA", "base_notrace_512_a.json"), ("onA", "base_trace_512_a.json"),
        ("offB", "base_notrace_512_b.json"), ("onB", "base_trace_512_b.json")]
H200 = 20.640          # MEASURED, gpu-h200-b200-5model-512aa, same fixture, predict_one boundary
TARGET = 82.56         # 3600/82.56*32 = 1395.4 = the H200's per-server throughput
ANCHOR = 85.606        # MEASURED pass 2, origin/main behaviour, uninstrumented, card 0
GLUE = 85.404          # MEASURED pass 2, this branch's byte-identical host glue

def pps(s):            # predictions/hour/server, 32 AI Processors
    return 3600.0 / s * 32

rows = []
for name, fn in LEGS:
    p = O / fn
    if not p.exists():
        print(f"MISSING {name}: {p}")
        continue
    d = json.loads(p.read_text())
    folds = d.get("warm_folds", [])
    digests = sorted({v for f in folds for v in (f.get("cif_sha256") or {}).values()})
    plddts = sorted({round(f["plddt"], 6) for f in folds if f.get("plddt") is not None})
    rows.append(dict(name=name, trace=name.startswith("on"), cold=d.get("cold_s"),
                     warm=d.get("warm_times_s", []), median=d.get("warm_median_s"),
                     lo=d.get("warm_min_s"), digests=digests, plddt=plddts,
                     git=d.get("tt_bio_git", "")[:8], ttnn=d.get("ttnn_version"),
                     dev=d.get("visible_devices"), machine=d.get("machine")))

if rows:
    r0 = rows[0]
    print(f"host {r0['machine']} card {r0['dev']} ttnn {r0['ttnn']} tree {r0['git']}\n")
print(f"{'leg':6} {'trace':6} {'cold':>7} {'warm folds':>18} {'median':>8} {'min':>8} {'pred/hr/srv':>12}")
for r in rows:
    w = ",".join(f"{t:.2f}" for t in r["warm"])
    print(f"{r['name']:6} {str(r['trace']):6} {r['cold']:7.2f} {w:>18} "
          f"{r['median']:8.3f} {r['lo']:8.3f} {pps(r['median']):12.1f}")

# --- 1. parity -------------------------------------------------------------
print("\n=== PARITY (decides the lever before any wall) ===")
state = {}
for r in rows:
    print(f"  {r['name']:6} cif sha256 {r['digests']}  plddt {r['plddt']}")
    state[r["name"]] = (tuple(r["digests"]), tuple(r["plddt"]))
uniq = set(state.values())
if len(uniq) == 1 and rows:
    print("  VERDICT: bit-identical. Every leg, traced and untraced, returns the same CIF digest\n"
          "           and the same plDDT. The trace passes parity at 512 aa.")
else:
    offs = {k: v for k, v in state.items() if k.startswith("off")}
    ons = {k: v for k, v in state.items() if k.startswith("on")}
    print(f"  VERDICT: NOT bit-identical -- {len(uniq)} distinct (digest, plddt) states.")
    print(f"    trace OFF: {set(offs.values())}")
    print(f"    trace ON : {set(ons.values())}")

# --- 2. cross-process A/A --------------------------------------------------
print("\n=== CROSS-PROCESS A/A FLOOR (section 10 lists this as owed) ===")
med = {r["name"]: r["median"] for r in rows}
aa = []
for a, b in (("offA", "offB"), ("onA", "onB")):
    if a in med and b in med:
        aa.append(abs(med[a] - med[b]))
        print(f"  {a} {med[a]:.3f} vs {b} {med[b]:.3f}  ->  |A/A| = {abs(med[a]-med[b]):.3f} s")
floor = max(aa) if aa else None
if floor is not None:
    print(f"  A/A floor = {floor:.3f} s. A delta below this is not separable from process noise.")

# --- 3. the delta ----------------------------------------------------------
print("\n=== THE DIFFUSION TRACE AT 512 aa ===")
offs = [r["median"] for r in rows if not r["trace"]]
ons = [r["median"] for r in rows if r["trace"]]
if offs and ons:
    mo, mn = statistics.median(offs), statistics.median(ons)
    delta = mn - mo
    print(f"  trace OFF {mo:.3f} s   ({pps(mo):.1f} pred/hr/server)")
    print(f"  trace ON  {mn:.3f} s   ({pps(mn):.1f} pred/hr/server)")
    print(f"  delta = {delta:+.3f} s ({delta/mo*100:+.2f} %)")
    if floor is not None and abs(delta) <= floor:
        print(f"  NOT SEPARABLE: |{delta:.3f}| <= the {floor:.3f} s cross-process A/A floor.")
        print(f"  The trace is worth less than the process noise at this size. The prediction\n"
              f"  carried into this pass was -0.4 to -0.8 s, derived from -0.9 % measured on the\n"
              f"  smaller 7ROA at commit 25258f91. Report the bound, not a win.")
    else:
        print(f"  Clears the {floor:.3f} s A/A floor." if floor else "  (no A/A floor)")

    best = min(mo, mn)
    print("\n=== POSITION vs the DGX H200 per server ===")
    print(f"  H200 {H200:.3f} s/fold x 8 GPUs   = {3600/H200*8:7.1f} predictions/hour/server")
    print(f"  best TT arm {best:.3f} s x 32 AIP = {pps(best):7.1f} predictions/hour/server "
          f"= {pps(best)/(3600/H200*8)*100:.1f} %")
    print(f"  target <= {TARGET:.2f} s; short by {best - TARGET:+.3f} s")
    print(f"  vs pass-2 origin/main anchor {ANCHOR:.3f} s: {best - ANCHOR:+.3f} s")
    print(f"  vs pass-2 glue arm           {GLUE:.3f} s: {best - GLUE:+.3f} s")

# --- 4. pooled statistics --------------------------------------------------
# --repeat 2 makes `warm_median_s` the MAX of two folds, so one transient excursion
# (offB fold 2 at 85.56 s, the only warm fold outside 84.58-84.77) sets a whole leg's
# median. Pool the four warm folds per arm and report the min alongside, which is the
# estimator least contaminated by a transient.
print("\n=== POOLED OVER THE FOUR WARM FOLDS PER ARM ===")
pool = {True: [], False: []}
for r in rows:
    pool[r["trace"]].extend(r["warm"])
for tr in (False, True):
    v = sorted(pool[tr])
    if v:
        print(f"  trace {'ON ' if tr else 'OFF'}: {['%.3f' % x for x in v]}  "
              f"min {v[0]:.3f}  median {statistics.median(v):.3f}")
if pool[True] and pool[False]:
    dmin = min(pool[True]) - min(pool[False])
    dmed = statistics.median(pool[True]) - statistics.median(pool[False])
    print(f"  trace delta on min    {dmin:+.3f} s ({dmin/min(pool[False])*100:+.2f} %)")
    print(f"  trace delta on median {dmed:+.3f} s ({dmed/statistics.median(pool[False])*100:+.2f} %)")
    print(f"  cross-process A/A on min: OFF {abs(rows[0]['lo']-rows[2]['lo']):.3f} s, "
          f"ON {abs(rows[1]['lo']-rows[3]['lo']):.3f} s")
    b = min(pool[False])          # the fastest arm that PASSES parity
    print(f"\n  fastest parity-passing arm (trace OFF) {b:.3f} s -> {pps(b):.1f} pred/hr/server "
          f"= {pps(b)/(3600/H200*8)*100:.2f} % of the DGX H200, short of the target by {b-TARGET:+.3f} s")
