"""CONTROL for the mask audit: recompute a reading that is KNOWN to be masked, and the
deliberately unmasked variant of the same reading, from the same tensors.

A recomputation that moves a masked reading is wrong about the reading. A mask that is not
applied leaves the two variants equal. Both directions are needed, so both are measured here
against the numbers `of3t-orchestrator` published in `revision/arm_score_realtokens.json`
(masked) and `revision/arm_score_padded.json` (padded) at pass 176.

CPU only, float64 accumulation, no device.
"""
import json
import sys

import torch

D = "/tmp/of3t/revarm/"
L = lambda p: torch.load(D + p, map_location="cpu", weights_only=False)

b = L("trunk_entry_043.pt")
tm = b["single_mask"][0].bool()            # [384], 56 True
pm = tm[:, None] & tm[None, :]

a43, a50, a50off, abf = (L(f) for f in ("out_043_fp32.pt", "out_050_fp32.pt",
                                        "out_050_tboff.pt", "out_043_bf16.pt"))


def rel(e, t, k, masked):
    e, t = e[k][0], t[k][0]
    if masked:
        e, t = (e[tm], t[tm]) if k == "s" else (e[pm], t[pm])
    e, t = e.double(), t.double()
    return float((e - t).norm() / t.norm())


PUBLISHED = {
    ("REVISION_050_vs_043", "s", True): 0.03214910057635795,
    ("REVISION_050_vs_043", "z", True): 0.21040157247002225,
    ("REVISION_050_vs_043", "s", False): 0.0387806352311332,
    ("REVISION_050_vs_043", "z", False): 0.005374144660129897,
    ("UPSTREAM_OWN_BF16", "s", True): 0.012878133443183962,
    ("UPSTREAM_OWN_BF16", "z", True): 0.024409521001988132,
    ("UPSTREAM_OWN_BF16", "s", False): 0.01496833860003941,
    ("UPSTREAM_OWN_BF16", "z", False): 0.0196067979177397,
}
ARMS = {"REVISION_050_vs_043": (a50, a43),
        "UPSTREAM_OWN_BF16": (abf, a43),
        "CONTROL_050_tboff_vs_043": (a50off, a43)}

out = {
    "what": "a known-masked reading recomputed (must not move) and its deliberately unmasked "
            "variant (must move), from the tensors of3t-orchestrator scored at pass 176",
    "boundary": "real 0.4.3 trunk entry, 5nw3, 56 real tokens padded to 384",
    "real_tokens": int(tm.sum()), "total_tokens": int(tm.numel()),
    "arms": {},
}
for arm, (e, t) in ARMS.items():
    row = {}
    for k in ("s", "z"):
        m = rel(e, t, k, True)
        u = rel(e, t, k, False)
        row[k] = {"masked": m, "padded": u,
                  "dilution_padded_over_masked": (m / u) if u > 0 else None}
    out["arms"][arm] = row

# The audit bounds a padded reading it cannot recompute by rel_padded / sqrt(q), with q the
# REFERENCE's mass share inside the real block. That bound is exercised here on the one reading
# whose dilution is 39x, which is the hardest case the campaign has.
def q_of(t, k):
    t = t[k][0].double()
    r = (t[tm] if k == "s" else t[pm])
    return float(r.norm() ** 2 / t.norm() ** 2)


out["bound_check"] = {}
for k in ("s", "z"):
    q = q_of(a43, k)
    a = out["arms"]["REVISION_050_vs_043"][k]
    out["bound_check"][k] = {
        "q_ref_mass_inside_real_block": q, "one_over_sqrt_q": 1.0 / q ** 0.5,
        "padded": a["padded"], "bound": a["padded"] / q ** 0.5, "measured_masked": a["masked"],
        "bound_holds": a["padded"] / q ** 0.5 >= a["masked"],
        "slack_x": (a["padded"] / q ** 0.5) / a["masked"]}

checks = []
for (arm, k, masked), want in PUBLISHED.items():
    got = out["arms"][arm][k]["masked" if masked else "padded"]
    checks.append({"arm": arm, "track": k, "scope": "masked" if masked else "padded",
                   "published": want, "recomputed": got, "abs_diff": abs(got - want),
                   "identical": got == want})
out["reproduction_of_published"] = checks
out["control_masked_recomputes_unchanged"] = all(
    c["identical"] for c in checks if c["scope"] == "masked")
out["control_unmasked_moves"] = {
    k: {"masked": out["arms"]["REVISION_050_vs_043"][k]["masked"],
        "padded": out["arms"]["REVISION_050_vs_043"][k]["padded"],
        "moved_by_x": out["arms"]["REVISION_050_vs_043"][k]["masked"] /
                      out["arms"]["REVISION_050_vs_043"][k]["padded"]}
    for k in ("s", "z")}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "CONTROL_REVISION_ARM.json", "w"),
          indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
