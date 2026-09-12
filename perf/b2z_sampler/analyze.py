#!/usr/bin/env python3
"""Turn the step/recycle grid into a quality curve with a floor under it.

Primary metric is superposition-free CA lDDT against the target's own (200,3,seed 0) fold.
Kabsch RMSD is reported too, but it is the secondary: `cdk2x2_512` is two pseudo-domains on a
floppy hinge, and a few degrees of hinge rotation saturates a global RMSD while every domain
stays identical (`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). A distance-matrix
metric does not care about the hinge, so one instrument reads every panel member honestly.

The floor is the point. Each target also folds (200,3) at seeds 1 and 2 — same setting, different
noise draws — and the spread of those two against seed 0 is what "no change at all" looks like on
this target. An arm inside that band has not been shown to be worse than production; an arm
outside it has. Reporting a step-count RMSD without the floor next to it reports sampler chaos as
a quality loss.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import read_atoms  # noqa: E402

LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
INCLUSION_A = 15.0


def kabsch(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((a @ r.T) - b) ** 2).sum(1).mean()))


def compare(ref: Path, arm: Path) -> dict:
    ka, xa = read_atoms(ref)
    kb, xb = read_atoms(arm)
    ia = {k: i for i, k in enumerate(ka)}
    common = [k for k in kb if k in ia]
    ib = {k: i for i, k in enumerate(kb)}
    A = np.asarray([xa[ia[k]] for k in common])
    B = np.asarray([xb[ib[k]] for k in common])
    cai = [i for i, k in enumerate(common) if k[2] == "CA"]
    Ac, Bc = A[cai], B[cai]
    n = len(Ac)
    da = np.linalg.norm(Ac[:, None] - Ac[None], axis=-1)
    db = np.linalg.norm(Bc[:, None] - Bc[None], axis=-1)
    m = (da < INCLUSION_A) & ~np.eye(n, dtype=bool)
    diff = np.abs(da - db)[m]
    h = n // 2
    return {
        "n_atoms": len(A), "n_ca": n,
        "lddt_ca": round(float(np.mean([(diff <= t).mean() for t in LDDT_THRESHOLDS]) * 100), 3),
        "rmsd_allatom_A": round(kabsch(A, B), 4),
        "rmsd_ca_A": round(kabsch(Ac, Bc), 4),
        # Each half superposed on its own: on a hinged fixture this is the number that means
        # something, and on a real monomer it is just the global number twice.
        "rmsd_ca_half1_A": round(kabsch(Ac[:h], Bc[:h]), 4),
        "rmsd_ca_half2_A": round(kabsch(Ac[h:], Bc[h:]), 4),
        "identical_bytes": ref.read_bytes() == arm.read_bytes(),
    }


def best_cif(rec: dict) -> Path | None:
    """The winning sample's CIF: the one whose name has no ``_model_`` rank suffix."""
    for name, meta in (rec.get("cifs") or {}).items():
        if "_model_" not in name:
            return Path(meta["path"])
    return None


def analyse_target(path: Path) -> dict | None:
    d = json.loads(path.read_text())
    folds = [f for f in d.get("folds", []) if "error" not in f and f.get("cifs")]
    if not folds:
        return None
    ref = next((f for f in folds if f["role"] == "reference"), None)
    if ref is None:
        return None
    ref_cif = best_cif(ref)
    rows = []
    for f in folds:
        cif = best_cif(f)
        if cif is None or not cif.exists():
            continue
        row = {"steps": f["steps"], "recycles": f["recycles"], "seed": f["seed"],
               "role": f["role"], "fold_s": f["fold_s"], "plddt": f["plddt"],
               "denoise": (f.get("counts") or {}).get("denoise"),
               "trunk_recycles": (f.get("counts") or {}).get("trunk_recycles")}
        row.update(compare(ref_cif, cif) if cif != ref_cif else
                   {"lddt_ca": 100.0, "rmsd_allatom_A": 0.0, "rmsd_ca_A": 0.0,
                    "rmsd_ca_half1_A": 0.0, "rmsd_ca_half2_A": 0.0, "identical_bytes": True,
                    "n_atoms": None, "n_ca": None})
        rows.append(row)

    floor = [r for r in rows if r["role"] == "floor"]
    out = {
        "target": d["env"]["target"], "kind": d["env"]["target_spec"].get("kind"),
        "note": d["env"]["target_spec"].get("note"),
        "n_tokens": ref.get("n_tokens"), "n_residues": ref.get("n_residues"),
        "msa_depth": ref.get("msa_depth"), "card": d["env"].get("card"),
        "hardware": d["env"].get("hardware"), "rows": rows,
        "floor": {
            "n": len(floor),
            "lddt_ca_min": round(min((r["lddt_ca"] for r in floor), default=float("nan")), 3),
            "rmsd_ca_max_A": round(max((r["rmsd_ca_A"] for r in floor), default=float("nan")), 4),
            "rmsd_allatom_max_A": round(max((r["rmsd_allatom_A"] for r in floor),
                                            default=float("nan")), 4),
            "plddt_spread": round(max((r["plddt"] for r in floor), default=0)
                                  - min((r["plddt"] for r in floor), default=0), 5)
            if floor else None,
        },
    }
    # "Inside the floor" is the verdict that matters: this arm is not distinguishable from
    # re-running production with a different seed.
    fl = out["floor"]
    for r in rows:
        if r["role"] in ("reference", "floor") or fl["n"] == 0:
            r["inside_floor"] = None
        else:
            r["inside_floor"] = bool(r["lddt_ca"] >= fl["lddt_ca_min"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    targets = []
    for p in sorted(a.results.glob("*.json")):
        try:
            t = analyse_target(p)
        except Exception as e:                       # a target that failed must not hide the rest
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
            continue
        if t:
            targets.append(t)

    # Panel view: for every (steps, recycles) setting, how many targets stayed inside their own
    # seed floor, and what the worst lDDT across the panel was. The worst member decides, not the
    # mean -- a production default has to hold on the hardest target, not on average.
    settings: dict[tuple[int, int], dict] = {}
    for t in targets:
        for r in t["rows"]:
            if r["role"] == "floor" or r["seed"] != 0:
                continue
            k = (r["steps"], r["recycles"])
            s = settings.setdefault(k, {"steps": k[0], "recycles": k[1], "lddt": [], "plddt_d": [],
                                        "rmsd_ca": [], "inside": 0, "n": 0, "fold_s": []})
            s["n"] += 1
            s["lddt"].append(r["lddt_ca"])
            s["rmsd_ca"].append(r["rmsd_ca_A"])
            s["fold_s"].append(r["fold_s"])
            ref_p = next((x["plddt"] for x in t["rows"] if x["role"] == "reference"), None)
            if ref_p is not None and r["plddt"] is not None:
                s["plddt_d"].append(round(r["plddt"] - ref_p, 5))
            if r["inside_floor"]:
                s["inside"] += 1

    panel = []
    for k, s in sorted(settings.items(), key=lambda kv: (-kv[0][1], -kv[0][0])):
        panel.append({
            "steps": s["steps"], "recycles": s["recycles"], "n_targets": s["n"],
            "inside_floor": s["inside"],
            "lddt_worst": round(min(s["lddt"]), 3), "lddt_median": round(st.median(s["lddt"]), 3),
            "rmsd_ca_worst_A": round(max(s["rmsd_ca"]), 4),
            "plddt_delta_median": round(st.median(s["plddt_d"]), 5) if s["plddt_d"] else None,
            "plddt_delta_worst": round(min(s["plddt_d"]), 5) if s["plddt_d"] else None,
            "fold_s_median_wh_contended": round(st.median(s["fold_s"]), 2),
        })

    a.out.write_text(json.dumps({"panel": panel, "targets": targets}, indent=1))

    print(f"{'steps':>6} {'rec':>4} {'n':>3} {'inside':>7} {'lddt_worst':>11} "
          f"{'lddt_med':>9} {'rmsd_ca_worst':>14} {'dplddt_med':>11} {'dplddt_worst':>13}")
    for p in panel:
        print(f"{p['steps']:>6} {p['recycles']:>4} {p['n_targets']:>3} "
              f"{p['inside_floor']:>3}/{p['n_targets']:<3} {p['lddt_worst']:>11.2f} "
              f"{p['lddt_median']:>9.2f} {p['rmsd_ca_worst_A']:>14.3f} "
              f"{str(p['plddt_delta_median']):>11} {str(p['plddt_delta_worst']):>13}")
    print()
    for t in targets:
        print(f"{t['target']:>14} ({t['kind']}, {t['n_tokens']} tok) floor: "
              f"lDDT>={t['floor']['lddt_ca_min']:.2f} CA-RMSD<={t['floor']['rmsd_ca_max_A']:.3f} A "
              f"(n={t['floor']['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
