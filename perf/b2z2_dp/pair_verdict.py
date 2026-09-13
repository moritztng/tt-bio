#!/usr/bin/env python3
"""Fold ship_chain.sh's paired arms into the table the threshold is chosen from.

policy_verdict.py compares a passive scaling table against a separate directory of active control
arms, which is the shape the verify row measured in. ship_chain.sh writes both policies of a width
side by side in one directory instead, so the pairing is per width and in one session. This reads
that shape, and nothing else: the arms themselves come from dp_width.py unchanged.

Per width it reports the ratio the decision turns on (passive folds per hour / active folds per
hour), the digest equality that makes the ratio quotable at all, and the per-worker host thread
budget `runtime.host_thread_cap` would hand a worker at that width, which is what the shipped
threshold actually tests.

    pair_verdict.py --dir perf/b2z2_dp/ship --host-threads 64 --out ship_pairs.json
"""
import argparse
import json
from pathlib import Path

KEYS = ("n_timed_folds", "window_s", "folds_per_hour", "median_fold_s", "median_cores_per_fold",
        "total_cores", "median_step_ms", "cif_digests", "digest_unanimous")


def arm(p: Path):
    a = json.loads(p.read_text())
    if not a.get("all_children_ok"):
        return None
    r = a["result"]
    rec = {k: r[k] for k in KEYS}
    rec["total_host_cores"] = rec.pop("total_cores")
    rec |= {"artifact": p.name, "started_utc": a["started_utc"],
            "loadavg_at_start": a["loadavg_at_start"],
            # What the arm ACTUALLY ran at. A fixed-cap arm is not at the share the width
            # implies, and reporting the implied one made the re-verification pair read cap 2
            # when it ran at 8 -- the single number that whole comparison turns on.
            "threads_per_worker": int(a["host_thread_cap"]["cap_env"]["OMP_NUM_THREADS"]),
            "cap_fixed": a["host_thread_cap"]["fixed"]}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--host-threads", type=int, required=True,
                    help="the box's os.cpu_count(), the budget host_thread_cap splits")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    widths = sorted({int(p.name[1:].split("_")[0]) for p in args.dir.glob("w*_*.json")})
    pairs, digests = {}, set()
    for w in widths:
        row = {}
        for pol in ("active", "passive"):
            a = arm(args.dir / f"w{w}_{pol}.json") if (args.dir / f"w{w}_{pol}.json").exists() else None
            if a:
                row[pol] = a
                digests |= set(a["cif_digests"])
        if "active" in row and "passive" in row:
            row["folds_per_hour_ratio"] = round(
                row["passive"]["folds_per_hour"] / row["active"]["folds_per_hour"], 5)
            row["median_fold_ratio"] = round(
                row["passive"]["median_fold_s"] / row["active"]["median_fold_s"], 5)
            row["bit_exact"] = (row["active"]["cif_digests"] == row["passive"]["cif_digests"]
                                and row["active"]["digest_unanimous"]
                                and row["passive"]["digest_unanimous"])
        row["host_thread_cap_implied"] = max(1, args.host_threads // w)
        row["threads_per_worker"] = next(
            (row[p]["threads_per_worker"] for p in ("active", "passive") if p in row), None)
        pairs[str(w)] = row

    out = {"dir": str(args.dir), "host_threads": args.host_threads, "widths": pairs,
           "cif_digests_all_arms": sorted(digests),
           "bit_exact_across_everything": len(digests) == 1}
    text = json.dumps(out, indent=1)
    print(text if not args.out else "")
    if args.out:
        args.out.write_text(text + "\n")
        for w, r in pairs.items():
            if "folds_per_hour_ratio" in r:
                print(f"w={w:>3} cap={r['threads_per_worker']:>2}"
                      f"{'(fixed)' if r['active']['cap_fixed'] else '       '} "
                      f"active={r['active']['folds_per_hour']:>8.2f} "
                      f"passive={r['passive']['folds_per_hour']:>8.2f} "
                      f"ratio={r['folds_per_hour_ratio']:.4f} bit_exact={r['bit_exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
