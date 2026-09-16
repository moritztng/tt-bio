"""Known-answer controls for what this row ADDS to c10-bare-baseline's harness.

The baseline's 13 controls (timer units, holder census, geometry, route counters, host CPU witness)
are closed and are not redone here. What is new is: clock coverage scored per arm instead of
against a hardcoded 1350 MHz, the inverse-clock fit and its uncertainty, the 10.0 s demand
arithmetic, and the claim that this tree is comment-only different from the tree the 14.8813 s
number of record was measured on.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from clockarm import coverage, fit_inverse_clock, propagate_two_clock
from reduce import demand
from astsame import equivalent, code_fingerprint

PASSED, FAILED = [], []


def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    (PASSED if ok else FAILED).append(f"{name}: got {got!r} want {want!r}" + (f" +-{tol}" if tol else ""))


def interval(t0=0, t1=1_000_000_000):
    return {"start_monotonic_ns": t0, "end_monotonic_ns": t1}


def samples(mhz, n=200, t0=0, t1=1_000_000_000, watts=40.0):
    """Dense enough to clear the 10 ms gap limit over the interval, like the real 1 kHz sampler."""
    step = (t1 - t0) // (n + 1)
    return [{"read_start_ns": t0 + step * (i + 1), "read_end_ns": t0 + step * (i + 1) + 1000,
             "MHz": mhz if not isinstance(mhz, list) else mhz[i % len(mhz)], "W": watts, "C": 50.0}
            for i in range(n)]


# --- 1. coverage is scored against the fold's OWN target, both directions ---
s800 = samples(800)
check("800 samples pass at target 800", coverage(s800, interval(), 800)["pass"], True)
check("800 samples REJECT at target 1350", coverage(s800, interval(), 1350)["pass"], False)
s1350 = samples(1350)
check("1350 samples pass at target 1350", coverage(s1350, interval(), 1350)["pass"], True)
check("1350 samples REJECT at target 800", coverage(s1350, interval(), 800)["pass"], False)
for t in (1000, 1200):
    check(f"{t} samples pass at their own target", coverage(samples(t), interval(), t)["pass"], True)

# --- 2. a clock that moved mid-fold is rejected even though the target appears ---
mixed = coverage(samples([1350, 1350, 1200]), interval(), 1350)
check("mid-fold clock move rejects", mixed["pass"], False)
check("mid-fold move is visible as min != max", mixed["min_MHz"] != mixed["max_MHz"], True)

# --- 3. coverage gaps, read errors and thin sampling reject ---
gappy = samples(800, n=4, t0=0, t1=1_000_000_000)
check("a 200 ms sampling gap rejects", coverage(gappy, interval(), 800)["pass"], False)
erroring = samples(800) + [{"read_start_ns": 500_000, "read_end_ns": 501_000, "error": "OSError()"}]
check("a read error rejects", coverage(erroring, interval(), 800)["pass"], False)
check("two samples reject (minimum is three)",
      coverage(samples(800, n=2, t1=3_000_000), interval(0, 3_000_000), 800)["pass"], False)
check("samples outside the fold interval are not counted",
      coverage(samples(800, t0=2_000_000_000, t1=3_000_000_000), interval(), 800)["samples"], 0)
check("power is carried through coverage", coverage(s800, interval(), 800)["mean_W"], 40.0, 1e-9)

# --- 4. the fit recovers an exact known answer with zero residual ---
F0, C0 = 3.0, 16000.0
exact = [(f, F0 + C0 / f) for f in (1350, 1200, 1000, 800)]
fit = fit_inverse_clock(exact)
check("exact fit recovers F", fit["F_s"], F0, 1e-9)
check("exact fit recovers C", fit["C_Mcycles"], C0, 1e-6)
check("exact fit residual is zero", fit["max_abs_residual_s"], 0.0, 1e-9)
check("exact fit SE is zero, not None, with >2 clocks", fit["se_F_s"] is not None and fit["se_F_s"] < 1e-6, True)

# --- 5. two distinct clocks give no uncertainty, and say so instead of printing zero ---
two = fit_inverse_clock([(1350, F0 + C0 / 1350), (800, F0 + C0 / 800)])
check("two-clock fit recovers F", two["F_s"], F0, 1e-9)
check("two-clock fit refuses a fake SE", two["se_F_s"], None)
check("two-clock fit says why", "carries no information" in two.get("se_note", ""), True)
try:
    fit_inverse_clock([(1350, 14.0), (1350, 14.1)])
    FAILED.append("one distinct clock should raise")
except ValueError:
    PASSED.append("one distinct clock raises: a fixed term needs two clocks")

# --- 6. closed-form two-clock propagation matches the fit and has the analytic gains ---
p = propagate_two_clock(1350, F0 + C0 / 1350, 0.02, 800, F0 + C0 / 800, 0.02)
check("closed form matches the fit on F", p["F_s"], F0, 1e-9)
check("closed form matches the fit on C", p["C_Mcycles"], C0, 1e-6)
check("dF/dT at 1350 is the analytic 2.45455", p["dF_dt_gain"][0], 2.454545454545, 1e-9)
check("dF/dT at 800 is the analytic -1.45455", p["dF_dt_gain"][1], -1.454545454545, 1e-9)
check("F uncertainty is amplified, not averaged", p["se_F_s"], 0.02 * (2.454545454545 ** 2 + 1.454545454545 ** 2) ** .5, 1e-12)

# --- 7. a fit whose points do not lie on 1/f leaves a residual larger than the A/A floor ---
bent = [(1350, F0 + C0 / 1350), (1200, F0 + C0 / 1200 + 0.9), (1000, F0 + C0 / 1000), (800, F0 + C0 / 800)]
check("a bent sweep leaves a visible residual", fit_inverse_clock(bent)["max_abs_residual_s"] > 0.05, True)

# --- 8. the 10.0 s demand arithmetic, by hand ---
d = demand(3.5, 15350.0, 1350)
check("demand reproduces the fold", d["seconds_now"], 3.5 + 15350 / 1350, 1e-12)
check("F untouched allows 1350*(10-F) Mcycles", d["F_untouched"]["cycles_allowed_Mcycles"], 1350 * 6.5, 1e-9)
check("F untouched cut", d["F_untouched"]["cycle_cut_Mcycles"], 15350 - 8775, 1e-9)
check("F untouched cut pct", d["F_untouched"]["cycle_cut_pct"], 100 * (15350 - 8775) / 15350, 1e-9)
check("F cut to 1 s allows 12150 Mcycles", d["F_cut_to_1s"]["cycles_allowed_Mcycles"], 12150.0, 1e-9)
check("F cut to 1 s cut", d["F_cut_to_1s"]["cycle_cut_Mcycles"], 15350 - 12150, 1e-9)
check("cutting F alone does not reach the target here", d["F_alone_at_current_cycles_s"] > 10.0, True)
big = demand(11.0, 5000.0, 1350)
check("an F above the target is called impossible, not a negative cycle budget",
      "impossible" in big["F_untouched"], True)
check("an F above the target is flagged unreachable", big["reachable_with_F_untouched"], False)
small = demand(1.0, 12000.0, 1350)
check("a cycle count already under the allowance is reported as met", small["F_cut_to_1s"]["already_met"], True)

# --- 9. comment-only equivalence, with negative controls ---
base = "X = 1024\n\ndef f(a):\n    # keep a\n    return a * X\n"
check("a comment rewrite is equivalent", equivalent(base, base.replace("# keep a", "# retain a")), True)
check("a docstring rewrite is equivalent",
      equivalent('"""old prose."""\nY = 2\n', '"""new prose entirely."""\nY = 2\n'), True)
check("a dropped comment is equivalent", equivalent(base, base.replace("    # keep a\n", "")), True)
check("a one-token constant change is NOT equivalent", equivalent(base, base.replace("1024", "2048")), False)
check("an operator change is NOT equivalent", equivalent(base, base.replace("a * X", "a + X")), False)
check("a dropped statement is NOT equivalent", equivalent(base, base.replace("X = 1024\n", "")), False)
check("a docstring-only module still yields a fingerprint", bool(code_fingerprint('"""only prose."""\n')), True)

# --- 10. the criterion and prediction files say what the brief requires before any fold ---
crit = json.loads((HERE / "criterion.json").read_text())
pred = json.loads((HERE / "prediction.json").read_text())
check("four authorised arms", crit["clock"]["arms_MHz"], [1350, 1200, 1000, 800])
check("at least 6 accepted folds per cell required", crit["minimum_folds_per_cell"] >= 6, True)
check("predicted model cycles saved is zero", pred["model_cycles_saved_predicted"], 0)
check("the 800 MHz 512 aa prediction is recorded", pred["predicted_800_MHz_512_aa_s"], 22.7)
check("refutation criteria recorded before the run", len(crit["refutation"]) >= 3, True)
check("bars are not transferred between sizes", crit["accuracy"]["bars_A"], {"512": 0.6, "298": 0.35})

print("\n".join(f"PASS  {x}" for x in PASSED))
print("\n".join(f"FAIL  {x}" for x in FAILED))
print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
