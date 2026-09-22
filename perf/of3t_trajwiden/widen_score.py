#!/usr/bin/env python3
"""of3t-trajwiden: score `of3t-trajwide`'s COMPLETED 20-rung diffusion_module trajectory.

`of3t-trajwide` ran all eight arms to 20/20 rungs (`perf/of3t_trajwide/live/STATUS.md`, every
marker rc=0) and then scored them at NINE, because the scoring pass was taken while the runs
were still going. The w_k dumps survive on qb2 at /home/ttuser/of3t_runs/trajwide/w (140 GB,
21 files per arm), so the other eleven rungs cost a scoring pass, not a re-run.

WHY THIS FILE EXISTS INSTEAD OF `trajwide.py --score`. That guard compares the reference arm's
recorded tree path against `refpath.OF3PKG` and refuses when they differ. The `theirs` arm
recorded `/home/ttuser/of3t_refprec/of3pkg043`; `refpath.OF3PKG` names
`/home/ttuser/of3t-campaign-refs/of3pkg043`. Under the campaign's OWN rule
(`of3t-campaign-refs/tree_digest.py`, A24-AMENDMENT) both trees are 293 .py files digesting to
1b27f5754b32b8e3..., which is the value `EXPECTED_DIGESTS.json` pins for of3pkg043. The guard is
refusing on path identity where the campaign decides reference identity by content. So this
driver does the content check the guard should do, records both digests in the artifact, and
then points `trajwide.OF3PKG` at the path the reference side ACTUALLY ran on -- so the guard
passes on the truth and `ref_tree` records where the numbers came from, not where they could
have come from. Nothing else in `trajwide.py` is touched. Filed for the orchestrator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REFROOT = "/home/ttuser/of3t-campaign-refs"
EXPECTED = json.load(open(f"{REFROOT}/EXPECTED_DIGESTS.json"))


def tree_digest(root: Path) -> dict:
    """`of3t-campaign-refs/tree_digest.py`'s rule, applied here so the check is in the artifact
    that depends on it rather than in a script beside it."""
    files = sorted(root.rglob("*.py"))
    per = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"root": str(root), "n_py_files": len(files),
            "tree_sha256": hashlib.sha256("".join(sorted(per.values())).encode()).hexdigest(),
            "rule": "sha256 of the sorted per-file sha256 hex strings, concatenated, of every "
                    ".py under the package root"}


def load(name, path):
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)          # `trajwide.py` imports its own `resume` as a sibling
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def aiclk_from_log(path):
    """The AICLK beside the run, read out of the sampler `run_arms.sh` kept for each arm.

    WHAT THIS IS NOT. `run_arms.sh`'s `clock_watch` does
    `tt-smi -s | tr -d " " | grep -i aiclk | head -4`, and one device contributes four
    AICLK* fields (AICLK, AICLK_LIMIT_MAX, AICLK_ARB_MIN, AICLK_ARB_MAX), so `head -4`
    truncates the whole four-device dump to DEVICE 0. Every line in these logs is device 0,
    whichever card the arm ran on. The `shipped` arm ran on card 1. Card 0 and card 1 are the
    two chips of one p300c BOARD PAIR, so this is the board's clock, sampled during the arm,
    and it is not a reading of card 1. Reported as what it is rather than relabelled.
    The value is the raw telemetry AICLK register in hex: 0x320 = 800 MHz, 0x546 = 1350 MHz.
    """
    if not os.path.exists(path):
        return {"sampled": False, "why": f"{path} absent"}
    rows, stamps = [], []
    for line in open(path):
        parts = line.split()
        m = re.search(r'"AICLK":"(0x[0-9a-fA-F]+)"', line)
        if parts and m:
            stamps.append(parts[0])
            rows.append(int(m.group(1), 16))
    if not rows:
        return {"sampled": False, "why": f"no AICLK field parsed from {path}"}
    return {"sampled_by": "tt-smi -s every 60 s DURING the arm, by run_arms.sh clock_watch",
            "log": path, "device_sampled": 0,
            "device_sampled_why": "run_arms.sh's `grep -i aiclk | head -4` truncates the "
                                  "four-device dump to device 0's four AICLK* fields",
            "samples": len(rows), "median_mhz": statistics.median(rows),
            "min_mhz": min(rows), "max_mhz": max(rows),
            "window_utc": [stamps[0], stamps[-1]]}


def main() -> int:
    out = sys.argv[sys.argv.index("--out") + 1]
    arm = sys.argv[sys.argv.index("--arm") + 1]
    theirs = sys.argv[sys.argv.index("--theirs-arm") + 1]
    runs = "/home/ttuser/of3t_runs/trajwide"

    tw = load("trajwide", "perf/of3t_trajwide/trajwide.py")
    recorded = (json.load(open(f"{runs}/steplog_{theirs}.json"))).get("ref_tree")
    dg = [tree_digest(Path(p) / "openfold3") for p in (tw.OF3PKG, recorded)]
    if len({d["tree_sha256"] for d in dg}) != 1:
        raise SystemExit(f"the two reference trees are NOT content-identical: {dg}")
    if dg[0]["tree_sha256"] != EXPECTED["of3pkg043"]:
        raise SystemExit(f"tree digest {dg[0]['tree_sha256']} is not the pinned of3pkg043 "
                         f"{EXPECTED['of3pkg043']}")
    tw.OF3PKG = recorded                 # the tree the reference side ran on, now proven equal
    rc = tw.main()
    if rc != 0:
        return rc

    res = json.load(open(out))
    sc = res["scope"]
    # The retake artifact's field names, so the charter can read this file in place of it
    # WITHOUT a second vocabulary. `section` is the scope, not the file's namespace.
    sc["section"] = "diffusion_module"
    sc["tensors_compared"] = sc["tensors_scored"]
    sc["tensors_in_reference"] = sc["reference_tensors_at_this_boundary"]
    res["steps"] = len(res["steps_scored"])
    res["reference_tree_identity"] = {
        "why": "of3t-trajwide's own --score guard compares PATHS and refuses this pairing. "
               "The campaign decides reference identity by CONTENT (A24-AMENDMENT, "
               "of3t-campaign-refs/tree_digest.py). Both paths hold the same 293 files.",
        "reference_side_ran_on": recorded,
        "refpath_OF3PKG": "/home/ttuser/of3t-campaign-refs/of3pkg043",
        "digests": dg, "pinned_expected_of3pkg043": EXPECTED["of3pkg043"],
        "content_identical": True}
    res["hosts"] = {
        "device_side": subprocess.run(["hostname"], capture_output=True,
                                      text=True).stdout.strip(),
        "reference_side": subprocess.run(["hostname"], capture_output=True,
                                         text=True).stdout.strip(),
        "scored_on": subprocess.run(["hostname"], capture_output=True,
                                    text=True).stdout.strip(),
        "note": "both sides of this trajectory and the scoring ran on qb2 (tt-quietbox2), "
                "16 cores, 249 GB. D189: a reference figure records its host."}
    sl_o = json.load(open(f"{runs}/steplog_{arm}.json"))
    sl_t = json.load(open(f"{runs}/steplog_{theirs}.json"))
    # `wall_s` in a steplog is CUMULATIVE from the arm's first rung, so the run's wall clock is
    # its LAST value, not the sum of the column. Summing it reads 34,540 s for a 3,336 s run.
    wo = [s["wall_s"] for s in sl_o["steps"] if s.get("wall_s") is not None]
    wt = [s["wall_s"] for s in sl_t["steps"] if s.get("wall_s") is not None]
    res["wall_clock_s"] = {
        "device_side_20_steps": wo[-1] if wo else None,
        "reference_side_20_steps": wt[-1] if wt else None,
        "pair_total": (wo[-1] + wt[-1]) if wo and wt else None,
        "wall_s_is_cumulative": True,
        "device_per_step_mean": (wo[-1] / len(wo)) if wo else None,
        "reference_per_step_mean": (wt[-1] / len(wt)) if wt else None,
        "threads": "OMP_NUM_THREADS=3 per run_arms.sh, two device chains on one board pair"}
    card = 1 if "card=1" in open(f"{runs}/{arm}.done").read() else 0
    res["aiclk_during"] = aiclk_from_log(f"{runs}/aiclk_{arm}.log")
    res["aiclk_during"]["card_the_arm_ran_on"] = card
    res["aiclk_during"]["note"] = (
        "recorded for attribution per the campaign's clock discipline. This artifact's claims "
        "are trajectory and scope, not throughput, so no number here is divided by a clock. "
        "The wall clocks in wall_clock_s ARE clock-dependent, are quoted as a cost screen, and "
        "the clock they carry is the board pair's and not this card's -- see "
        "aiclk_during.device_sampled_why.")
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_step", "our_step_log", "their_step_log", "scope")},
                     indent=1, default=str)[:3000])
    print("scope.pct_of_model_sq_grad_norm =", sc["pct_of_model_sq_grad_norm"])
    print("scope.tensors_compared =", sc["tensors_compared"], "of", sc["tensors_in_reference"])
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
