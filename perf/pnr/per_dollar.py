#!/usr/bin/env python3
"""Each published row's Galaxy Blackhole throughput per dollar against a DGX B200, from the page's own cost model.

    python3 perf/pnr/per_dollar.py [site/data/perf-512aa.json] [--rows mol1,bindcraft2,rfd4]

Same formulas as site/benchmarks/index.html: per hour = 3600 / s x accelerators x scaling efficiency;
purchase = per hour / list price; ownership = per hour / (price / (years x 8766 h) + kW x $/kWh). The purchase
ratio is also the BC4/RFD4/MOL1 list-price convention (32 x $562,154 / (8 x $160,000) = 14.054 x t_B200 / t_TT).
Rows below 1.0x on purchase are flagged HOLD: PNR does not publish them (state/pnr/PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "site" / "data" / "perf-512aa.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", nargs="?", type=Path, default=DATA)
    ap.add_argument("--rows", help="comma-separated row ids, default every published row")
    a = ap.parse_args()
    d = json.loads(a.data.read_text())
    P, cm = {p["id"]: p for p in d["platforms"]}, d["cost_model"]
    hours = cm["amortisation_years"] * cm["hours_per_year"]

    def per_hour(s, p):
        return 3600.0 / s * p["accelerators"] * p.get("scaling_efficiency", 1.0)

    def usd_h(p):
        return p["price_usd"] / hours + p["power_kw"] * cm["electricity_usd_per_kwh"]

    gx, nv = P["galaxy_bh"], P["dgx_b200"]
    rows = d["models"] + [r for k in ("design", "affinity") for r in d.get(k, {}).get("models", [])]
    want = set(a.rows.split(",")) if a.rows else None
    held = 0
    print(f"{'row':<16}{'TT s':>10}{'B200 s':>10}{'purchase':>10}{'ownership':>11}")
    for r in rows:
        if (want and r["id"] not in want) or (not want and r.get("hidden")):
            continue
        tt, b2 = r["cells"].get("p150a", {}), r["cells"].get("b200", {})
        if tt.get("status") != "measured" or b2.get("status") != "measured":
            print(f"{r['id']:<16}  not measured on both sides")
            continue
        t, b = (c.get("s_per_fold", c.get("s_per_design")) for c in (tt, b2))
        capex = (per_hour(t, gx) / gx["price_usd"]) / (per_hour(b, nv) / nv["price_usd"])
        tco = (per_hour(t, gx) / usd_h(gx)) / (per_hour(b, nv) / usd_h(nv))
        flag = "  HOLD (< 1.0x)" if capex < 1.0 else ""
        held += bool(flag)
        print(f"{r['id']:<16}{t:>10.3f}{b:>10.3f}{capex:>9.2f}x{tco:>10.2f}x{flag}")
    return 1 if held else 0


if __name__ == "__main__":
    sys.exit(main())
