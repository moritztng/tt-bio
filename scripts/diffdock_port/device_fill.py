"""Phase 2 of the DiffDock accuracy verdict: the device column of `state/dfd/QUALITY.json`.

    device_fill.py --dir ~/scratch/dfd/eval/device --chip "qb2 p300 chip 0"

`--dir` holds `poses-gemm/` and `poses-tile/` as the device arm wrote them (each with its
`clock.json`) and `acc-gemm/`, `acc-tile/` as `pose_eval.py grade` wrote them. The host arm is
the frozen slice of the 428-set grade. Both bars are the ones pre-registered in
`state/dfd/EVAL.md` before any device pose existed:

  1. card faithful:    device gemm vs host torch
  2. tile may default: device tile vs device gemm

Each bar passes when the exact two-sided sign test over the discordant pairs has p > 0.05 and
the paired 95 % bootstrap CI of the top-1 difference contains 0. A bar over fewer than the full
subset is recorded as partial and is never a pass.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_eval  # noqa: E402
from quality_fill import CO, QJ, ROW  # noqa: E402

SUBSET = CO / "state/dfd/evidence/device-subset-80.json"
HOST = Path.home() / "scratch/dfd/eval/acc/per_complex-host85.json"
EQUIV = CO / "state/dfd/evidence/device-equivariance.json"
MIN_CLOCK_MHZ = 1200


def bar(a_path, b_path, out_path, n_subset):
    """compare() names its two sides tt/upstream; here they are a (the arm under test) and b."""
    c = pose_eval.compare(str(a_path), str(b_path), str(out_path))
    t, ci = c["top1"], c["top1"]["ci95_paired_bootstrap"]
    complete = c["n_common_graded"] == n_subset
    ok = bool(t["mcnemar_p"] > 0.05 and ci and ci[0] is not None and ci[0] <= 0 <= ci[1])
    return {
        "n_paired": c["n_common_graded"], "n_subset": n_subset, "complete": complete,
        "top1_a": t["tt"], "top1_b": t["upstream"], "delta_top1": t["delta"], "ci95_paired": ci,
        "discordant": {"a_only": t["mcnemar_discordant"]["tt_only"],
                       "b_only": t["mcnemar_discordant"]["upstream_only"]},
        "sign_test_p": t["mcnemar_p"],
        "median_rmsd_a": c["rmsd_distribution"]["median_tt"],
        "median_rmsd_b": c["rmsd_distribution"]["median_upstream"],
        "ks_p": c["rmsd_distribution"]["ks_p"],
        "outliers": c.get("outliers", {}),
        "verdict": ("PASS" if ok else "FAIL") if complete else "PARTIAL",
    }


def arm(d: Path, form: str):
    s = json.load(open(d / f"acc-{form}" / "summary.json"))
    clk_path = d / f"poses-{form}" / "clock.json"
    clk = json.load(open(clk_path)) if clk_path.exists() else None
    lo = (clk or {}).get("aiclk_min")
    return {
        "network": f"device, edge_tp={form}" + (" (tor_bond_conv on gemm)" if form == "tile" else ""),
        "n_requested": s["n_requested"], "n_scored": s["n_graded"],
        "top1_rmsd_lt2": s["top1_success_lt2A"], "top1_ci95_wilson": s["top1_success_ci95"],
        "median_rmsd": s["median_rmsd_top1"],
        "posebusters_validity": s["posebusters_validity_rate"],
        "nan_or_collapse": s["nonfinite_or_collapse"],
        "aiclk_during": clk,
        "clock_ok": None if lo is None else bool(lo >= MIN_CLOCK_MHZ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--chip", required=True, help="host, board and chip the two arms ran on")
    ap.add_argument("--tree", default=None, help="sha of the tree the arms docked with")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    n = len(json.load(open(SUBSET))["ids"])
    host = json.load(open(HOST))
    assert len(host) == n, f"host slice has {len(host)} complexes, subset has {n}"

    dev = {
        "subset": str(SUBSET), "n_subset": n, "seed": 42, "chip": a.chip, "tree": a.tree,
        "recipe": "upstream default: 20-step schedule, 19 actual steps, 10 samples, confidence on",
        "arms": {
            "host": {"network": "host torch HostScoreNetwork (slice of the 428-set grade)",
                     "n_scored": sum(1 for r in host if r["rmsd"][0] is not None),
                     "top1_rmsd_lt2": round(sum(r["rmsd"][0] < 2.0 for r in host) / n, 4)},
            "gemm": arm(d, "gemm"), "tile": arm(d, "tile"),
        },
        "bar1_card_faithful_gemm_vs_host": bar(
            d / "acc-gemm/per_complex.json", HOST, d / "compare-gemm-vs-host.json", n),
        "bar2_tile_may_default_tile_vs_gemm": bar(
            d / "acc-tile/per_complex.json", d / "acc-gemm/per_complex.json",
            d / "compare-tile-vs-gemm.json", n),
        "paths": {"dir": str(d), "script": "scripts/diffdock_port/device_fill.py"},
    }
    b1, b2 = dev["bar1_card_faithful_gemm_vs_host"], dev["bar2_tile_may_default_tile_vs_gemm"]
    dev["verdict"] = {"card_faithful": b1["verdict"], "tile_may_default": b2["verdict"]}
    print(json.dumps(dev, indent=1))
    if a.dry_run:
        return 0
    if not (b1["complete"] and b2["complete"]):
        sys.exit("refusing to write: an arm is short of the subset. A partial column is not a verdict.")

    q = json.load(open(QJ))
    q["device"] = dev
    if EQUIV.exists():
        q["equivariance"]["device_path"] = json.load(open(EQUIV))
    clocks = {f: dev["arms"][f]["aiclk_during"] for f in ("gemm", "tile")}
    q["clock"]["aiclk_mhz_during_run"] = clocks
    json.dump(q, open(QJ, "w"), indent=2)
    r = json.load(open(ROW))
    r.update(status="done", device=dev["verdict"],
             phase="host column and device column both graded and filed")
    r.pop("next", None)
    json.dump(r, open(ROW, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
