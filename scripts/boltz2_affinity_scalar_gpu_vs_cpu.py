#!/usr/bin/env python3
"""Three-backend triangulation for the Boltz-2 affinity SCALAR
(affinity_pred_value = MW-corrected Δlog10(IC50), and affinity_probability_binary).

Computes the GPU-reference-vs-CPU-reference cross-backend distance on the SCALAR
and compares it to each backend's own self-floor. If the two pinned-boltz-2.2.1
bf16-mixed references (only execution device differs: x86 CPU vs CUDA GPU,
--no_kernels so the same torch-einsum triangle kernel) DISAGREE on the scalar by
the same magnitude as device-vs-either, the scalar GAP is a portable
bf16-backend-floor property (same class as pocket-lDDT). If the two references
AGREE tightly and only the device diverges, that is a real closable device-side
defect.

No device compute. Reuses the noise_floor_verdict core so R/D/X are directly
comparable to scripts/boltz2_affinity_parity.py.

Each --ref-dir is a harness-layout seed dir with affinity_<tid>.json.
Pass two groups: --a-dirs (backend A, e.g. CPU) and --b-dirs (backend B, e.g. GPU).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boltz2_affinity_parity import AFFINITY_KEYS  # noqa: E402
from pharma_parity import noise_floor_verdict  # noqa: E402


def _extract(ref_dir: Path, target_id: str) -> dict:
    cand = list(ref_dir.rglob(f"affinity_{target_id}.json"))
    if not cand:
        raise FileNotFoundError(f"no affinity_{target_id}.json under {ref_dir}")
    d = json.loads(cand[0].read_text())
    return {k: float(d[k]) for k in AFFINITY_KEYS if k in d}


def _floor(vals, key):
    r = [v[key] for v in vals if key in v]
    return [abs(a - b) for a, b in itertools.combinations(r, 2)], r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-dirs", nargs="+", required=True, help="backend A seed dirs (e.g. CPU)")
    ap.add_argument("--b-dirs", nargs="+", required=True, help="backend B seed dirs (e.g. GPU)")
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    a_dirs = [Path(d) for d in args.a_dirs]
    b_dirs = [Path(d) for d in args.b_dirs]
    a_vals = [_extract(d, args.target_id) for d in a_dirs]
    b_vals = [_extract(d, args.target_id) for d in b_dirs]

    print(f"### Triangulation: {args.a_label} vs {args.b_label}  ({args.target_id})\n")
    print(f"{args.a_label} seeds: {len(a_dirs)}   {args.b_label} seeds: {len(b_dirs)}\n")
    print("Per-seed affinity_pred_value:")
    print(f"| seed | {args.a_label} | {args.b_label} |")
    print("|---|---|---|")
    for i in range(max(len(a_vals), len(b_vals))):
        av = a_vals[i].get("affinity_pred_value", float("nan")) if i < len(a_vals) else float("nan")
        bv = b_vals[i].get("affinity_pred_value", float("nan")) if i < len(b_vals) else float("nan")
        print(f"| {i} | {av:.4f} | {bv:.4f} |")
    print()

    report = {"target": args.target_id, "a_label": args.a_label, "b_label": args.b_label,
              "n_a": len(a_dirs), "n_b": len(b_dirs), "metrics": {}}
    print("| metric | A self-floor (R_A) | B self-floor (R_B) | A-vs-B cross (X) | floor=max(R_A,R_B) | X/floor | within floor |")
    print("|---|---|---|---|---|---|---|")
    for key in AFFINITY_KEYS:
        a_floor, a_r = _floor(a_vals, key)
        b_floor, b_r = _floor(b_vals, key)
        if not a_r or not b_r:
            continue
        cross = [abs(a_v[key] - b_v[key]) for a_v in a_vals for b_v in b_vals]
        v = noise_floor_verdict(cross, a_floor, b_floor, key)
        floor = max(v["ref_floor"]["mean"], v["dev_floor"]["mean"])
        xmean = v["cross"]["mean"]
        within = xmean <= floor + max(v["ref_floor"]["std"], v["dev_floor"]["std"])
        report["metrics"][key] = {
            "R_A": v["ref_floor"], "R_B": v["dev_floor"], "X_AB": v["cross"],
            "floor": floor, "X_over_floor": (xmean / floor) if floor else float("inf"),
            "within_floor": within,
        }
        print(f"| {key} | {v['ref_floor']['mean']:.4f} | {v['dev_floor']['mean']:.4f} "
              f"| {xmean:.4f} | {floor:.4f} | {(xmean/floor) if floor else float('nan'):.2f} "
              f"| {'YES' if within else 'NO'} |")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
