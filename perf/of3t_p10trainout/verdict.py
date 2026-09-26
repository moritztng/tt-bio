#!/usr/bin/env python3
"""The verdict, computed from the artifacts rather than read off them by hand.

The bar is `perf/of3t_p10trainout/PREREGISTRATION.md` as amended by
`AMENDMENT-tolerance.md`: arm A (the device-native trunk, what ships) passes if its held-out
mean loss after N steps is within the REPLICATE floor of arm B's (the same run with the trunk's
softmax and layer norm exact), and if the arms moved the metric further than that floor in the
first place. Two arms that never moved agree perfectly.

The gap is also reported as a fraction of the movement, because "further apart than re-running
the same arm" and "far enough apart to matter" are different statements and the verdict owes
both.

    verdict.py --arm-a out/arm_A2s0c.json --arm-a-replicate out/arm_A2rep.json \
               --arm-a-seed out/arm_A2seed1.json --arm-b out/arm_B2trunk.json \
               --out out/VERDICT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict:
    d = json.loads(Path(p).read_text())
    if not d.get("ok"):
        raise SystemExit(f"{p} did not finish: {d.get('error', 'no error recorded')}")
    for k in ("eval_before", "eval_after"):
        if not d.get(k):
            raise SystemExit(f"{p} has no {k}; it cannot be graded")
    return d


def _clock(d: dict) -> str:
    c = (d.get("aiclk") or {}).get("0") or (d.get("aiclk") or {}).get(0) or {}
    if not c:
        return "NOT SAMPLED"
    return (f"{c['median']} MHz median sampled DURING, n={c['n']} "
            f"(min {c['min']}, max {c['max']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True, type=Path)
    ap.add_argument("--arm-a-replicate", required=True, type=Path,
                    help="arm A again at the SAME seed: the floor the tolerance is")
    ap.add_argument("--arm-a-seed", type=Path,
                    help="arm A at a different seed: reported, not the tolerance")
    ap.add_argument("--arm-b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    A, Ar, B = _load(a.arm_a), _load(a.arm_a_replicate), _load(a.arm_b)
    start = A["eval_before"]["mean_loss"]
    for name, d in (("arm A replicate", Ar), ("arm B", B)):
        if abs(d["eval_before"]["mean_loss"] - start) > 0:
            raise SystemExit(
                f"{name} scored the START checkpoint at {d['eval_before']['mean_loss']!r} where "
                f"arm A scored {start!r}. The two arms are not being read by one instrument, "
                f"and the comparison below would be a property of the instrument")

    a_after, b_after = A["eval_after"]["mean_loss"], B["eval_after"]["mean_loss"]
    replicate_floor = abs(a_after - Ar["eval_after"]["mean_loss"])
    movement = abs(a_after - start)
    gap = abs(a_after - b_after)

    moved = movement > replicate_floor
    within = gap <= replicate_floor
    verdict = "GO" if (moved and within) else ("NO-GO" if moved else "PARTIAL")

    out = {
        "verdict": verdict,
        "start_checkpoint_mean_loss": start,
        "arm_a_mean_loss": a_after,
        "arm_b_mean_loss": b_after,
        "gap": gap,
        "replicate_floor": replicate_floor,
        "tolerance": replicate_floor,
        "movement": movement,
        "movement_control_cleared": moved,
        "movement_over_floor": (movement / replicate_floor) if replicate_floor else None,
        "gap_over_floor": (gap / replicate_floor) if replicate_floor else None,
        "gap_over_movement": (gap / movement) if movement else None,
        "steps": A.get("argv") and B.get("argv"),
        "arm_a": {"file": str(a.arm_a), "exact_ops": A.get("exact_ops"),
                  "exact_scope": A.get("exact_scope"), "aiclk": _clock(A),
                  "per_target": {t["pdb_id"]: t["loss"] for t in A["eval_after"]["targets"]}},
        "arm_b": {"file": str(a.arm_b), "exact_ops": B.get("exact_ops"),
                  "exact_scope": B.get("exact_scope"), "aiclk": _clock(B),
                  "per_target": {t["pdb_id"]: t["loss"] for t in B["eval_after"]["targets"]}},
        "commit": A.get("precondition_slot_fix", {}).get("head_short"),
        "slot_fix_is_ancestor": A.get("precondition_slot_fix", {}).get("is_ancestor_of_head"),
    }
    if a.arm_a_seed:
        S = _load(a.arm_a_seed)
        out["seed_floor"] = abs(a_after - S["eval_after"]["mean_loss"])
        out["seed_floor_note"] = ("reported, NOT the tolerance: the seed sets the batch order, "
                                  "which the A/B design holds fixed")

    per = {k: (out["arm_a"]["per_target"][k], out["arm_b"]["per_target"][k])
           for k in out["arm_a"]["per_target"]}
    out["per_target_gap"] = {k: abs(x - y) for k, (x, y) in per.items()}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"VERDICT: {verdict}")
    print(f"  start {start:.6f} -> arm A {a_after:.6f}, arm B {b_after:.6f}")
    print(f"  gap {gap:.6e}  tolerance (replicate floor) {replicate_floor:.6e}  "
          f"= {out['gap_over_floor']:.2f}x the floor")
    print(f"  movement {movement:.6f} = {out['movement_over_floor']:.0f}x the floor; "
          f"gap is {out['gap_over_movement']:.4%} of it")
    for k, v in sorted(out["per_target_gap"].items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
