#!/usr/bin/env python3
"""PROTOCOL A45: the GRADIENTS-DRAWS artifact. The four state-free fields per draw from
DRAWS_PD384.json, unedited, and the mass field on the six-draw per-tensor mean from DRAWMASS.json.

    draws_a45.py -> DRAWS_A45.json
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
W = H.parents[2]
d = json.load(open(W / "perf/of3t_paedraws/DRAWS_PD384.json"))
m = json.load(open(H / "DRAWMASS.json"))
per = []
for r in d["per_draw"]:
    f = r["five"]
    four = (f["unread_n"] == 0 and f["placed_but_carried_nothing_n"] == 0
            and f["multi_placed"] == 0 and f["global_rel"] <= f["bf16_global_rel"])
    per.append({"k": r["k"], "four_hold": four, "global_rel": f["global_rel"],
                "bf16_global_rel": f["bf16_global_rel"],
                "mass_at_or_better_this_draw": f["mass_at_or_better_than_bf16"]})
for p in per:  # DRAWMASS reproduces every published per-draw figure, or it is not this instrument
    assert abs(round(p["mass_at_or_better_this_draw"], 4)
               - m["per_draw_mass_at_or_better"][str(p["k"])]) < 1e-9, p
standing = m["lost_on_5_or_6_draws_mass"]
out = {"arm": "PD384", "protocol": "A45", "draws": len(per),
       "four_hold_every_draw": all(p["four_hold"] for p in per),
       "mean_mass_at_or_better_than_bf16": m["mean_mass_at_or_better"],
       "standing_losers_mass": standing,
       "standing_losers": m["lost_on_5_or_6_draws_top"][:3],
       "per_draw": per}
out["holds"] = out["four_hold_every_draw"] and out["mean_mass_at_or_better_than_bf16"] >= 0.95
json.dump(out, open(H / "DRAWS_A45.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_draw"}))
