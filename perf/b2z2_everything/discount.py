#!/usr/bin/env python3
"""The additivity discount, from the committed folds rather than from prose.

Two constructions, because no single session timed every arm:

  A  in-session singles: SAMP x TRUNK x MSA x HOSTONLY, all paired against one interleaved base
     in `singles_everything_wh_c8.json`, against the union measured in `fold_everything_wh_c8.json`.
  B  the host trio's MARGINAL inside one session: ALL / BX in `fold_everything_wh_c8.json`,
     against HOSTONLY standing alone. This one crosses no session boundary on the numerator.

Host only, reads the JSONs, invents nothing.
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
fold = json.loads((H / "fold_everything_wh_c8.json").read_text())
sing = json.loads((H / "singles_everything_wh_c8.json").read_text())

F, S = fold["paired_ratio"], sing["paired_ratio"]
out = {
    "fold_session": {"floor": fold["aa_floor_paired"], "ratios": F,
                     "loadavg": [fold["env"]["loadavg"][0], fold["env"]["loadavg_end"][0]]},
    "singles_session": {"floor": sing["aa_floor_paired"], "ratios": S,
                        "loadavg": [sing["env"]["loadavg"][0], sing["env"]["loadavg_end"][0]]},
}

prod = 1.0
for a in ("SAMP", "TRUNK", "MSA", "HOSTONLY"):
    prod *= S[a]
out["A_in_session_singles"] = {
    "product": round(prod, 5), "measured_union": F["ALL"],
    "discount_pct": round(100 * (1 - F["ALL"] / prod), 3)}

bx_prod = S["SAMP"] * S["TRUNK"] * S["MSA"]
out["A_bitexact_subset"] = {
    "product": round(bx_prod, 5), "measured": F["BX"],
    "discount_pct": round(100 * (1 - F["BX"] / bx_prod), 3)}

marg = F["ALL"] / F["BX"]
out["B_host_marginal"] = {
    "marginal_on_top_of_BX": round(marg, 5), "alone_same_box": S["HOSTONLY"],
    "alone_published": 1.06495,
    "discount_vs_alone_pct": round(100 * (1 - marg / S["HOSTONLY"]), 3),
    "discount_vs_published_pct": round(100 * (1 - marg / 1.06495), 3)}

# readability: a lever whose fold contribution is under the session's own A/A floor is not an
# individually readable fold lever, whatever it is worth on its own instrument.
out["readable_vs_floor"] = {
    a: {"ratio": S[a], "floor": sing["aa_floor_paired"],
        "multiple_of_floor": round((S[a] - 1) / (sing["aa_floor_paired"] - 1), 2),
        "readable": S[a] - 1 > 2 * (sing["aa_floor_paired"] - 1)}
    for a in ("SAMP", "TRUNK", "MSA", "HOSTONLY")}

out["falsifier"] = {
    "bar": "union more than 5 % BELOW the product of the surviving singles",
    "worst_discount_pct": max(out["A_in_session_singles"]["discount_pct"],
                              out["A_bitexact_subset"]["discount_pct"],
                              out["B_host_marginal"]["discount_vs_alone_pct"]),
    "fires": max(out["A_in_session_singles"]["discount_pct"],
                 out["B_host_marginal"]["discount_vs_alone_pct"]) > 5.0}

(H / "discount.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
