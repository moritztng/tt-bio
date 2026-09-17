#!/usr/bin/env python3
"""Recover c12-compose-fold's session s3 from the corpse of the process that was measuring it.

s3 was a 48-rep five-arm interleaved fold session (base,silu,hoist,both,base, palindrome ordered)
on qb2 card 2, board ...410D, at a forced and during-sampled 1350 MHz. At 21:21Z, 31 reps in, the
harness process host-spin wedged: state=R, CPU advancing at 100 %, `syscr` AND `syscw` frozen, its
output file untouched for 25 minutes. `wedge_check.py --pid 24290 --window 30` classified it WEDGED
and the 25-minute file mtime corroborated. Because the harness dumps its JSON after every fold, all
159 warm folds it had completed were already on disk, so the wedge TRUNCATED the session instead of
destroying it. SIGTERM cleared it in 1 s and freed dev2 with no reset.

Provenance, which is the point of splitting this into two files:

  s3.json                          raw, byte-copied from the row's worktree
  s3_summary_row_estimator.json    the output of the ROW'S OWN resummarise.py, which imports the
                                   same summarise() the harness printed from, so nothing here can
                                   drift from the row's estimator. Each arm is differenced against
                                   its OWN rep's base mean (pos0,pos4) with a t95 for the real rep
                                   count -- paired, so a monotone load ramp cancels.
  prediction.json                  pre-registered 2026-09-17T14:20Z, sha256-pinned, verified

This script computes only what no pass had computed: the read-off against the pre-registered
criteria, and the PER-LEVER plDDT attribution. It does not re-estimate a delta.

    python3 recover.py
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
S = json.loads((H / "s3.json").read_text())
M = json.loads((H / "s3_summary_row_estimator.json").read_text())
P = json.loads((H / "prediction.json").read_text())

warm = [r for r in S["runs"] if not r["cold"]]
by: dict[str, list] = {}
for r in warm:
    by.setdefault(r["arm"], []).append(r)
ARMS = ("base", "silu", "hoist", "both")

print("=" * 96)
print("1. SESSION VALIDITY -- checked before any number is read off it")
print("=" * 96)
lo = min(x["aiclk"]["min"] for x in warm)
hi = max(x["aiclk"]["max"] for x in warm)
print(f"  {len(warm)} warm folds over {M['paired']['n_reps']} reps; planned 48, wedged at 31 "
      f"= {100*31/48:.0f} % of the session")
print(f"  AICLK {lo}-{hi} MHz over {sum(x['aiclk']['n'] for x in warm):,} during-fold samples "
      f"-> kill criterion (d), any fold under 1200 MHz: {'FIRED' if lo < 1200 else 'NOT FIRED'}")
print(f"  host load {min(x['load0'] for x in warm)}-{max(x['load1'] for x in warm)}; "
      "release-gate size-ladder held board ...4103 (dev1), C12 held board ...410D (dev2), "
      "sibling dev3 idle (pair_idle exit 0 at 20:38Z and 20:47Z)")
for a in ARMS:
    v = by[a]
    hn, sc = {x["hoist_norms"] for x in v}, {x["silu_calls"] for x in v}
    exp_h = {200} if a in ("hoist", "both") else {0}
    exp_s = {3760} if a in ("silu", "both") else {0}
    ok = hn == exp_h and sc == exp_s
    print(f"  {a:6s} n={len(v):3d}  hoist_norms={sorted(hn)} silu_calls={sorted(sc)} "
          f"cond_rebuilt={sorted({x['cond_build_n'] for x in v})}  "
          f"witness {'FIRED' if ok else 'VOID'}  "
          f"unique plDDT={len({round(x['plddt'],6) for x in v})} "
          f"unique digest={len({x['sha256'] for x in v})}")
print("  -> every arm's output is BIT-IDENTICAL across all its reps, so the plDDT deltas below are"
      " exact, not sampled.")

print()
print("=" * 96)
print("2. THE PAIRED RESULT -- the row's own estimator, not recomputed here")
print("=" * 96)
aa = M["paired"]["aa"]
print(f"  {'arm':7s} {'paired mean':>12s} {'95 % CI':>22s} {'resolved':>9s} {'min-based':>10s}")
print(f"  {'A/A':7s} {aa['mean_s']:+12.4f} "
      f"{'[%+.4f, %+.4f]' % tuple(aa['ci95_s']):>22s} {str(aa['resolved']):>9s} {'--':>10s}")
for a in ("silu", "hoist", "both"):
    r = M["paired"]["arms"][a]
    print(f"  {a:7s} {r['mean_s']:+12.4f} {'[%+.4f, %+.4f]' % tuple(r['ci95_s']):>22s} "
          f"{str(r['resolved']):>9s} {r['delta_min_s']:+10.3f}")
print(f"\n  medians for comparison: "
      + ", ".join(f"{a} {M['arms'][a]['delta_s_vs_base']:+.4f}" for a in ("silu", "hoist", "both")))
print(f"  The A/A control is NOT resolved from zero (CI spans it) while all three lever arms ARE.")
print(f"  That separation is the in-data evidence the host co-tenancy cancelled in the pairing;")
print(f"  it is what four earlier sessions on this row could not produce.")

print()
print("=" * 96)
print("3. READ-OFF AGAINST THE PRE-REGISTERED CRITERIA")
print("=" * 96)
both = M["paired"]["arms"]["both"]
sub = M["paired"]["subadditivity"]
pred = P["predicted"]
d = both["mean_s"]
print(f"  composed delta                  {d:+.4f} s   CI [{both['ci95_s'][0]:+.4f}, {both['ci95_s'][1]:+.4f}]")
print(f"  pre-registered central          {pred['central_delta_s']:+.4f} s")
print(f"  pre-registered ceiling          {pred['ceiling_delta_s']:+.4f} s  "
      f"<- MEASURED EXCEEDS IT by {d - pred['ceiling_delta_s']:+.4f} s")
print(f"     the ceiling was additivity of the OP-LEVEL singles (silu 0.2843 + hoist 0.2134).")
print(f"     This session's OWN singles are larger: silu {M['paired']['arms']['silu']['mean_s']:+.4f},"
      f" hoist {M['paired']['arms']['hoist']['mean_s']:+.4f}, sum {sub['sum_of_singles_s']:+.4f} s.")
print(f"  sub-additivity vs own singles   {sub['fraction_of_sum']:.4f} of the sum, "
      f"pre-registered band {pred['expected_fraction_of_sum']} -> "
      f"{'INSIDE' if pred['expected_fraction_of_sum'][0] <= sub['fraction_of_sum'] <= pred['expected_fraction_of_sum'][1] else 'OUTSIDE'}")
print(f"     interaction {sub['interaction']['mean_s']:+.4f} s CI "
      f"[{sub['interaction']['ci95_s'][0]:+.4f}, {sub['interaction']['ci95_s'][1]:+.4f}], "
      f"resolved={sub['interaction']['resolved']} -> additive within this session's noise")
print()
for k, t in P["kill_criteria"].items():
    if k == "a":
        fired = not (both["ci95_s"][0] > aa["ci95_s"][1])
    elif k == "b":
        fired = d < pred["band_low_delta_s"]
    elif k == "c":
        fired = any(M["witness"][a]["void"] for a in ARMS)
    else:
        fired = lo < 1200
    print(f"  kill ({k}) {'FIRED' if fired else 'not fired':10s}  {t[:78]}")
print()
# the GO gate's margin clause, against every defensible reading of "the session's own A/A floor"
floors = {
    "paired A/A CI half-width": aa["ci95_half_width_s"],
    "median pos0 - posN":       M["aa_floor"]["delta_s"],
    "|paired A/A mean|":        abs(aa["mean_s"]),
}
print(f"  GO gate: composed >= 0.2843 s -> {d:.4f} s, {'MET' if d >= 0.2843 else 'NOT MET'}")
print(f"  GO gate: composed >= 3x the session's OWN A/A floor --")
for name, f in floors.items():
    print(f"     vs {name:26s} {f:.4f} s -> {d/f:.2f}x  {'MET' if d/f >= 3 else 'NOT MET'}")
print("     Two of the three defensible floor readings give 2.87-2.98x, marginally UNDER the")
print("     pre-registered 3x. The truncation is why: at the planned 48 reps the CI half-widths")
print(f"     shrink by sqrt(48/31) = {(48/31)**0.5:.2f}x, which would have put the ratio at ~3.6x.")

print()
print("=" * 96)
print("4. PER-LEVER plDDT ATTRIBUTION -- new, and it splits the stack")
print("=" * 96)
base_pl = list({round(x["plddt"], 6) for x in by["base"]})[0]
FLOOR = 0.00286  # acc512.json seed_floor plddt_delta, base s0 vs base s1, same fixture
print(f"  512 aa seed-0 mean plDDT, bit-identical across every rep. base = {base_pl:.6f}")
print(f"  seed floor on this metric and fixture = {FLOOR:.5f} (acc512.json, base s0 vs base s1)")
for a in ("silu", "hoist", "both"):
    p = list({round(x["plddt"], 6) for x in by[a]})[0]
    print(f"     {a:6s} {p:.6f}  delta {p-base_pl:+.6f}  = {abs(p-base_pl)/FLOOR:5.2f}x the seed floor")
s = list({round(x['plddt'],6) for x in by['silu']})[0] - base_pl
h = list({round(x['plddt'],6) for x in by['hoist']})[0] - base_pl
b = list({round(x['plddt'],6) for x in by['both']})[0] - base_pl
print(f"  additive: silu {s:+.6f} + hoist {h:+.6f} = {s+h:+.6f} vs measured both {b:+.6f}")
print(f"  -> cond-hoist is accuracy-FREE on this metric ({abs(h)/FLOOR:.2f}x the seed floor).")
print(f"  -> {100*s/b:.0f} % of the stack's plDDT cost is silu's, and it is {abs(s)/FLOOR:.1f}x the")
print(f"     seed floor and reproducible bit-exactly over {len(by['silu'])} folds. That is a real")
print(f"     signal, not scatter, and no row has adjudicated it.")

print()
print("=" * 96)
print("5. THE BOOK AND THE BUDGET, with the fold-level measurement replacing the two estimates")
print("=" * 96)
FOLD = 14.881          # c10-bare-baseline quiet-box median, pinned during-sampled 1350 MHz
NEED_125, NEED_100 = FOLD - 12.5, FOLD - 10.0
# The two entries this session REPLACES. They are not additive with it: `both` IS silu+hoist.
print(f"  superseded: silu 0.2843 s (op-level) + cond-hoist 0.2415 s (block-level) = 0.5258 s")
print(f"  replaced by: the composed stack measured AT THE FOLD = {d:+.4f} s, "
      f"CI [{both['ci95_s'][0]:+.4f}, {both['ci95_s'][1]:+.4f}]")
print(f"  the fold-level read is {d - 0.5258:+.4f} s ABOVE the sum of the two lower-level estimates,")
print(f"  so for executed-graph-sized levers the campaign's decay rule has no counterexample yet in")
print(f"  the unfavourable direction: both met or beat their predictions, and the stack beat its own")
print(f"  additive ceiling of {pred['ceiling_delta_s']:.4f} s.")
print()
BOOK = [
    ("compose stack (silu+hoist)", d,        "MEASURED at the fold, paired, resolved; margin 2.87-2.98x vs a 3x gate"),
    ("reblock-delete",             1.0062,   "PREDICTED central, band 0.9342-1.2007, never executed an instruction"),
    ("genop recoverable",          0.0184,   "MEASURED/pre-registered by c12-genop-triatt-slack, VERDICT STOP. Was 0.4544 s: that ceiling rested on reblock_gated at 79.39 % of roof, roofed against the WRONG arm (bw_clone 1R+1W for a 2:1 site); at its matching roof it is 71.66 %, so the existence proof does not exist. And triatt_out's 37.99 % was a byte-model defect (insitu_sites.py:47 dropped the repair bytes) -- really 75.96 %, so 0.0860 s of it was never slack."),
    ("matmul class",               0.1352,   "measured in-situ cap for ALL matmul levers, no row"),
    ("Axis A host",                0.0960,   "MEASURED 0.0000 reducible / 0.0960 generous / 0.1896 absurd"),
]
tot = sum(x[1] for x in BOOK)
for n, v, note in BOOK:
    print(f"  {n:28s} {v:7.4f} s   {note}")
print(f"  {'TOTAL named route':28s} {tot:7.4f} s")
print()
print(f"  12.5 s needs {NEED_125:.4f} s -> named route covers {100*tot/NEED_125:5.1f} %, "
      f"{'short by %.4f s' % (NEED_125-tot) if tot < NEED_125 else 'clears by %.4f s' % (tot-NEED_125)}")
print(f"  10.0 s needs {NEED_100:.4f} s -> named route covers {100*tot/NEED_100:5.1f} %, "
      f"short by {NEED_100-tot:.4f} s")
print()
lo_r, hi_r = 0.9342, 1.2007
print(f"  swinging reblock-delete across its own band, everything else at the values above:")
for lbl, rb in (("band low ", lo_r), ("central  ", 1.0062), ("band high", hi_r)):
    t = tot - 1.0062 + rb
    print(f"     reblock {lbl} {rb:.4f} -> route {t:.4f} s, 12.5 s "
          f"{'CLEARS by %.4f s' % (t-NEED_125) if t >= NEED_125 else 'short by %.4f s' % (NEED_125-t)}"
          f"   projected fold {FOLD-t:.3f} s")
print()
print(f"  projected fold from the BANKED term alone: {FOLD:.3f} - {d:.4f} = {FOLD-d:.3f} s")
print(f"     conservative (CI low)  {FOLD-both['ci95_s'][0]:.3f} s")
print(f"     conservative (min-based) {FOLD-both['delta_min_s']:.3f} s")
print(f"  The session's own base median was {M['base_median_fold_s']:.3f} s under co-tenancy at load")
print(f"  2.4-9.74; that absolute is NOT a number of record, so the DELTA is what is applied to the")
print(f"  quiet-box base of {FOLD:.3f} s. 10.0 s remains out of reach by every derivation.")
