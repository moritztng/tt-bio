#!/usr/bin/env python3
"""Score fused-SDPA ragged-pad fold arms against a baseline, A/A control FIRST.

Distances only (lower = closer to the baseline arm). The A/A control is the baseline arm folded
twice: on a bit-exact instrument it is 0.0000 A, and any arm inside it is not a result.
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
sys.path.insert(0, str(WT / "scripts"))
from boltz2_fast_parity import CONF_KEYS, compare_structure, load_results  # noqa: E402


def cif(arm_dir: Path, model: str, tid: str) -> Path:
    hits = sorted(arm_dir.glob(f"*_results_{tid}/structures/{tid}.cif"))
    if not hits:
        hits = sorted(arm_dir.glob(f"**/structures/{tid}.cif"))
    if not hits:
        raise SystemExit(f"no cif under {arm_dir} for {tid}")
    return hits[0]


def res(arm_dir: Path, tid: str):
    """The one results row for `tid`. `load_results` keys a whole run dir by id."""
    hits = sorted(arm_dir.glob(f"*_results_{tid}/results.json"))
    if not hits:
        return {}
    rows = load_results(hits[0].parent)
    return rows.get(tid) or (next(iter(rows.values())) if len(rows) == 1 else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tid", required=True)
    ap.add_argument("--base", required=True, help="baseline arm name")
    ap.add_argument("--control", required=True, help="baseline-repeat arm name (the A/A)")
    ap.add_argument("--arms", required=True, help="comma-separated lever arm names")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    base = cif(root / a.base, a.model, a.tid)
    rows = []
    order = [("A/A control", a.control)] + [("lever", x) for x in a.arms.split(",") if x]
    for kind, arm in order:
        d = root / arm
        if not (d.exists() and list(d.glob("**/structures/*.cif"))):
            rows.append({"arm": arm, "kind": kind, "status": "MISSING"})
            continue
        s = compare_structure(base, cif(d, a.model, a.tid))
        rb, ra = res(root / a.base, a.tid), res(d, a.tid)
        conf = {k: (ra.get(k) - rb.get(k)) for k in CONF_KEYS
                if isinstance(ra.get(k), (int, float)) and isinstance(rb.get(k), (int, float))}
        rows.append({"arm": arm, "kind": kind, "status": "ok",
                     "kabsch_rmsd": s["kabsch_rmsd"], "coord_pcc": s.get("coord_pcc"),
                     "tm_score": s.get("tm_score"), "lddt": s.get("lddt"),
                     "n_matched": s.get("n_matched"), "conf_delta": conf})

    aa = next((r for r in rows if r["kind"] == "A/A control" and r["status"] == "ok"), None)
    floor = aa["kabsch_rmsd"] if aa else None
    print(f"model={a.model} tid={a.tid} baseline={a.base}")
    print(f"{'arm':26s} {'kind':12s} {'RMSD(A)':>10s} {'X/floor':>9s} {'lddt':>8s} {'pcc':>10s}  verdict")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['arm']:26s} {r['kind']:12s} {'MISSING':>10s}")
            continue
        x = r["kabsch_rmsd"]
        ratio = (x / floor) if floor else float("inf") if x > 0 else 0.0
        if r["kind"] == "A/A control":
            v = "floor"
        elif floor == 0.0:
            v = "OUTSIDE floor (real)" if x > 0 else "bit-exact, inside floor"
        else:
            v = "OUTSIDE floor" if ratio > 1.0 else "inside floor"
        print(f"{r['arm']:26s} {r['kind']:12s} {x:10.4f} {ratio:9.3f} "
              f"{r['lddt'] or 0:8.4f} {r['coord_pcc'] or 0:10.6f}  {v}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "tid": a.tid, "baseline": a.base, "aa_floor": floor, "rows": rows},
            indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
