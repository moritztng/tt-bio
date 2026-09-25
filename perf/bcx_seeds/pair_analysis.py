#!/usr/bin/env python3
"""bcx-seeds: grade the five matched pairs off the arms' own logs.

Every number this row reports about the pairs comes from here. Nothing is transcribed by hand:
the device arms are parsed from perf/bcx_seeds/seed*_stages.txt, the reference arms from
perf/bcx_seeds/reference/*, and seed 100's device arm from bcx-predictor's banked post_seed100
line, which is the one arm this row did not run itself and is marked as such.

Stage bars are BindCraft 2's own, read from settings/core/default.json on the pinned checkout:
screen 0.60, refine 0.60, anneal 0.65, harden 0.65. A stage "gate" is a stage BOTH arms reached
and were graded at; the pair agrees at a gate when both arms return the same PASS/REJECT.
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BARS = {"screen": 0.60, "refine": 0.60, "anneal": 0.65, "harden": 0.65}
STAGES = ["screen", "refine", "anneal", "harden", "mutate"]

LINE = re.compile(
    r"\s*(passed|rejected at)\s+(\w+)\s+design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")
TRAJ = re.compile(r"=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+) ===")


def parse(text):
    """Return [{traj, design, accepted, stages:[...]}] for every trajectory in a run log."""
    out = []
    for raw in text.splitlines():
        m = TRAJ.search(raw)
        if m:
            out.append({"traj": int(m.group(1)), "design": m.group(2),
                        "accepted": int(m.group(3)), "stages": []})
            continue
        m = LINE.match(raw)
        if m and out:
            out[-1]["stages"].append({
                "stage": m.group(2),
                "verdict": "PASS" if m.group(1) == "passed" else "REJECT",
                "i_ptm": float(m.group(3)), "plddt": float(m.group(4)),
                "filters": [f.strip() for f in (m.group(5) or "").split(",") if f.strip()],
            })
    return out


def first_traj(path):
    t = parse(Path(path).read_text())
    return t[0] if t else None


# --- the arms -------------------------------------------------------------------------------
# Seed 100's device arm is bcx-predictor's `post_seed100` on qb2 card 3, not re-run here. Its
# line is reproduced in the same log shape so it parses identically; the host difference is
# carried in the report rather than hidden.
POST_SEED100 = """=== trajectory 1 | pdl1_denovo_l71_746e618341b8c7e8 | accepted 0/10 ===
  passed screen design stage  i_pTM=0.80    pLDDT=0.78
  passed refine design stage  i_pTM=0.78    pLDDT=0.76
  passed anneal design stage  i_pTM=0.77    pLDDT=0.73
  passed harden design stage  i_pTM=0.75    pLDDT=0.68
  rejected at mutate design stage  i_pTM=0.10    pLDDT=0.38   due to [i_pTM, pLDDT]
"""

device = {
    0: first_traj(ROOT / "seed0_stages.txt"),
    100: parse(POST_SEED100)[0],
    200: first_traj(ROOT / "seed200_stages.txt"),
    300: first_traj(ROOT / "seed300_stages.txt"),
    400: first_traj(ROOT / "seed400_stages.txt"),
}
REF = ROOT / "reference"
reference_all = {
    0: parse((REF / "reference_seed0_qb2.txt").read_text()),
    100: parse((REF / "reference_seed100.txt").read_text()),
    200: parse((REF / "reference_seed200.txt").read_text()),
    300: parse((REF / "reference_seed300.txt").read_text()),
    400: parse((REF / "reference_seed400.txt").read_text()),
}
reference = {s: t[0] for s, t in reference_all.items()}

DEVICE_HOST = {0: "qb1 card 3", 100: "qb2 card 3 (bcx-predictor's post_seed100)",
               200: "qb1 card 3", 300: "qb1 card 3", 400: "qb1 card 3"}

# --- grade ----------------------------------------------------------------------------------
report = {"pairs": {}, "gates": {"shared": 0, "agree": 0, "disagree": []}, "deltas": [],
          "accepted": {"device": 0, "reference_traj1": 0, "reference_all_completed": 0}}

for seed in sorted(device):
    d, r = device[seed], reference[seed]
    dm = {s["stage"]: s for s in d["stages"]}
    rm = {s["stage"]: s for s in r["stages"]}
    rows, shared, agree = [], 0, 0
    for st in STAGES:
        ds, rs = dm.get(st), rm.get(st)
        row = {"stage": st, "bar_plddt": BARS.get(st)}
        row["device"] = (f"{ds['plddt']:.2f} / {ds['i_ptm']:.2f} {ds['verdict']}"
                         + (f" {ds['filters']}" if ds and ds["filters"] else "")) if ds else "—"
        row["reference"] = (f"{rs['plddt']:.2f} / {rs['i_ptm']:.2f} {rs['verdict']}"
                            + (f" {rs['filters']}" if rs and rs["filters"] else "")) if rs else "—"
        if ds and rs:
            shared += 1
            same = ds["verdict"] == rs["verdict"] and sorted(ds["filters"]) == sorted(rs["filters"])
            row["gate"] = "AGREE" if same else "DISAGREE"
            if same:
                agree += 1
            else:
                report["gates"]["disagree"].append(
                    {"seed": seed, "stage": st, "bar": BARS.get(st),
                     "device": row["device"], "reference": row["reference"],
                     "ref_margin_to_bar": (round(rs["plddt"] - BARS[st], 3)
                                           if st in BARS else None)})
            report["deltas"].append({"seed": seed, "stage": st,
                                     "d_plddt": round(ds["plddt"] - rs["plddt"], 3),
                                     "d_iptm": round(ds["i_ptm"] - rs["i_ptm"], 3)})
        else:
            row["gate"] = "—"
        rows.append(row)
    dlast, rlast = d["stages"][-1], r["stages"][-1]
    report["pairs"][seed] = {
        "design_device": d["design"], "design_reference": r["design"],
        "device_host": DEVICE_HOST[seed], "rows": rows,
        "shared_gates": shared, "agreeing_gates": agree,
        "outcome_device": f"{dlast['verdict']} at {dlast['stage']} {dlast['filters']}",
        "outcome_reference": f"{rlast['verdict']} at {rlast['stage']} {rlast['filters']}",
        "same_terminal_stage": dlast["stage"] == rlast["stage"],
        "accepted_device": d["accepted"], "accepted_reference": r["accepted"],
    }
    report["gates"]["shared"] += shared
    report["gates"]["agree"] += agree
    report["accepted"]["device"] += d["accepted"]
    report["accepted"]["reference_traj1"] += r["accepted"]

# The reference's own across-draw spread: every reference trajectory that reached a stage, so a
# device-vs-reference gap on a matched draw can be read against the movement BC2's own JAX shows
# between draws of the same seed.
spread = {}
completed_ref = 0
for seed, trajs in reference_all.items():
    for t in trajs:
        if t["stages"] and (t["stages"][-1]["verdict"] == "REJECT"):
            completed_ref += 1
        for s in t["stages"]:
            spread.setdefault(s["stage"], []).append(s["plddt"])
report["accepted"]["reference_all_completed"] = completed_ref
report["reference_own_spread"] = {
    st: {"n": len(v), "min": min(v), "max": max(v), "range": round(max(v) - min(v), 3)}
    for st, v in spread.items()}

dp = [x["d_plddt"] for x in report["deltas"]]
report["delta_summary"] = {
    "n_matched_stages": len(dp), "mean_d_plddt": round(statistics.mean(dp), 4),
    "min": min(dp), "max": max(dp),
    "log_resolution": 0.01,
}
print(json.dumps(report, indent=1))
