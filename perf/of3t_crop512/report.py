#!/usr/bin/env python3
"""Render this row's figures out of its own artifacts, so none is typed by hand.

Every number in `state/of3t-crop512.md` comes from here. The instruments are
`of3t-crop640`'s, unchanged -- `split_trace.py` for the rung and `project.py` for the fit --
and this only formats what they wrote. `table-provenance-must-be-per-entry` is the reason it
exists: a table assembled by hand from three JSONs is a transcription, and a transcription is
where a campaign's numbers quietly stop matching its artifacts.

    report.py --rung perf/of3t_crop512/out/split_512.json \
              --projection perf/of3t_crop512/out/PROJECTION.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GB = 1 << 30
OWNERS = ("recompute", "boundaries", "unattributed", "tape", "weights", "cotangents",
          "weight_grads", "other_leaf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=Path, required=True)
    ap.add_argument("--projection", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.rung.read_text())
    p = json.loads(a.projection.read_text())
    b, f = d["backward"], d["forward"]
    w = b["walk"]

    print("=== ENV ===")
    e = d["env"]
    print(f"host={e['host']} card={e['tt_visible_devices']} arch={e.get('arch')} "
          f"branch={e['branch']} commit={e['commit'][:9]}")
    print(e.get("aiclk_line"))
    print(f"backward ok={b.get('ok')} s={b.get('s')} forward s={f.get('s')}")

    print("\n=== PROFILE at the 512 backward peak ===")
    peak = b["dram_peak_b"]
    print(f"peak {peak:,} B = {peak/GB:.3f} GB at verb {b.get('at_verb')!r}, "
          f"{b.get('dram_live_allocs')} live DRAM allocations")
    print(f"walk gap to peak: {b.get('walk_gap_to_peak_b'):,} B "
          f"({b.get('walk_gap_to_peak_pct')} %), {b.get('walks')} walks, "
          f"step {b.get('walk_step_b_final',0)>>20} MB")
    print(f"walk census DRAM {w['census_dram_b']:,} B over {w['buffers']} buffers")
    tot = w["census_dram_b"]
    rows = [(o, w["by_owner"][o]["dram_b"], w["by_owner"][o]["n"]) for o in OWNERS
            if o != "unattributed"]
    rows.append(("unattributed", w["unattributed"]["dram_b"], w["unattributed"]["n"]))
    for o, v, n in sorted(rows, key=lambda r: -r[1]):
        print(f"| {o} | {v:,} | {v/GB:.3f} | {100*v/tot:.2f} % | {n} |")
    nb = peak - w["by_owner"]["boundaries"]["dram_b"]
    print(f"non-boundary term {nb:,} B; recompute is "
          f"{100*w['by_owner']['recompute']['dram_b']/nb:.2f} % of it")
    af = d["after_forward"]
    print(f"boundaries after forward (complete set): "
          f"{af['by_owner']['boundaries']['dram_b']:,} B "
          f"over {af['by_owner']['boundaries']['n']} pins; ckpt_pins={b.get('ckpt_pins')}")

    print("\n=== EXPONENTS (fitted on %s) ===" % p["exponents_fitted_on"])
    for o in OWNERS:
        print(f"  {o:14s} {p['owner_exponents'].get(o)}")
    cf = p["boundary_closed_form"]
    print(f"boundary closed form {cf['a_b_per_token2']}*N^2 + {cf['b_b_per_token']}*N built on "
          f"{cf['built_on_tokens']}: reproduces base rung to {cf['rel_err_pct']} %, "
          f"every rung exactly: {cf['reproduces_every_rung_after_forward']}")

    print("\n=== CONTROLS ===")
    for arm, c in p["controls"]["reproduces_base_rung"].items():
        print(f"  {arm:16s} reproduces the {p['base_tokens']} rung: {c['projected_b']:,} B vs "
              f"measured {c['measured_b']:,} B ({c['rel_err_pct']} %)")
    print(f"  clears the 640 observational floor ({p['controls']['640_floor_GB']} GB): "
          f"{p['controls']['clears_640_observational_floor']}")
    print(f"  walk gap to peak per rung (%): {p['controls']['walk_gap_to_peak_pct']}")
    print(f"  REFUTED: {p.get('REFUTED')}")

    print("\n=== PROJECTIONS ===")
    for n in ("640", "768"):
        for arm, r in p["projections"][n].items():
            print(f"{n} [{arm:16s}] peak {r['projected_peak_GB']:7.3f} GB  "
                  f"deficit {r['deficit_GB']:+7.3f}  "
                  f"spill->{r['lever_boundary_spill']['peak_after_GB']:7.3f}  "
                  f"halve->{r['lever_halve_recompute']['peak_after_GB']:7.3f}  "
                  f"BOTH->{r['both_levers']['peak_after_GB']:7.3f} GB  "
                  f"margin {r['both_levers']['margin_GB']:+7.3f}  fits={r['both_levers']['fits']}")
            if "third_lever" in r:
                print(f"     third lever must save {r['third_lever']['must_save_GB']} GB; "
                      f"largest remaining {r['third_lever']['largest_remaining_owners_GB']}")
    print("\n=== RUNG LADDER ===")
    for r in p["rungs"]:
        print(f"  {r['tokens']:4d}  peak {r['peak_b']:,} B = {r['peak_b']/GB:.3f} GB  "
              f"allocs {r['allocs']}  pins {r['boundary_pins']}  "
              f"walk gap {r['walk_gap_pct']} %  card {r['card']}  {r['branch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
