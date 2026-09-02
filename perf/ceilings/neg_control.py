"""Negative control: mutate one row at a time and require the guard to catch it."""
import dataclasses, importlib, sys
import pytest
from tt_bio import size_limits as sl
import tests.test_size_limits as T

CASES = {
  "memory-bound row with NO failing size": dict(fail_at=None),
  "cap at or above the failing size":      dict(residues=576),
  "pass_at above the published cap":       dict(pass_at=9999),
  "evidence stripped":                     dict(evidence="short"),
  "bogus binds":                           dict(binds="probably_fine"),
  "bogus mechanism":                       dict(mechanism="vibes"),
  "LADDER_TOP claimed with a failing size":dict(binds=sl.LADDER_TOP),
}
base = sl.CEILINGS["opendde"]["wormhole_b0"]
fails = 0
for name, patch in CASES.items():
    bad = dataclasses.replace(base, **patch)
    sl.CEILINGS["opendde"]["wormhole_b0"] = bad
    try:
        T.test_row_is_internally_consistent("opendde", "wormhole_b0", bad)
        T.test_ladder_top_publishes_the_size_it_proved()
        print(f"  NOT CAUGHT: {name}")
    except AssertionError as e:
        fails += 1
        print(f"  caught: {name}")
    finally:
        sl.CEILINGS["opendde"]["wormhole_b0"] = base

# coverage guard: drop a row and require the shipped-model test to notice
drop = sl.CEILINGS.pop("opendde")
try:
    T.test_every_shipped_model_has_a_row(); print("  NOT CAUGHT: missing row for a shipped model")
except AssertionError: fails += 1; print("  caught: missing row for a shipped model")
finally: sl.CEILINGS["opendde"] = drop

# and a stale row for a model the CLI no longer has
sl.CEILINGS["ghost-model"] = {"wormhole_b0": base}
try:
    T.test_no_row_for_a_model_that_is_not_shipped(); print("  NOT CAUGHT: stale row")
except AssertionError: fails += 1; print("  caught: stale row for a retired model")
finally: del sl.CEILINGS["ghost-model"]

# a fabricated Blackhole row
sl.CEILINGS["opendde"]["blackhole"] = base
try:
    T.test_no_blackhole_rows_are_asserted(); print("  NOT CAUGHT: fabricated blackhole row")
except AssertionError: fails += 1; print("  caught: fabricated blackhole row")
finally: del sl.CEILINGS["opendde"]["blackhole"]

print(f"\n{fails}/10 invariants demonstrated to fail when violated")
sys.exit(0 if fails == 10 else 1)
