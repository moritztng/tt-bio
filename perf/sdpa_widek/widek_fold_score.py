#!/usr/bin/env python3
"""Score a wide-k fold A/B against the two controls `docs/boltz2-fast-parity.md` defines.

The lever changes the online-softmax reduction order, so it is not bit-exact and a nonzero
deviation is expected. It PASSES only if its deviation sits inside the band the fold's own
nondeterminism already spans:

    determinism floor   off vs off, same seed, rerun
    seed spread         off seed i vs off seed j
    the lever           on vs off, same seed

Reuses `compare_structure` from `scripts/boltz2_fast_parity.py` (per-chain Kabsch RMSD, coord PCC,
TM, lDDT) rather than restating the geometry.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_b2fp", ROOT / "scripts" / "boltz2_fast_parity.py")
_b2fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b2fp)

CONF = ["plddt", "ptm", "iptm", "confidence_score", "complex_plddt"]


def leg_dir(workdir: Path, tag: str) -> Path:
    """The one results dir a predict leg wrote, whatever the model named it."""
    cands = [p.parent for p in (workdir / tag).rglob("results.json")]
    if len(cands) != 1:
        raise SystemExit(f"{tag}: expected 1 results.json, found {len(cands)}")
    return cands[0]


def conf_of(d: Path) -> dict:
    return {r["id"]: r for r in json.loads((d / "results.json").read_text())}


def cmp_legs(workdir: Path, a: str, b: str) -> dict:
    da, db = leg_dir(workdir, a), leg_dir(workdir, b)
    ra, rb = conf_of(da), conf_of(db)
    ids = [i for i in ra if i in rb]
    out = {}
    for i in ids:
        g = _b2fp.compare_structure(da / "structures" / f"{i}.cif",
                                    db / "structures" / f"{i}.cif")
        out[i] = {
            "kabsch_rmsd": round(g["kabsch_rmsd"], 4), "coord_pcc": round(g["coord_pcc"], 6),
            "tm_score": round(g["tm_score"], 6), "lddt": round(g["lddt"], 6),
            "n_matched": g["n_matched"],
            "per_chain": {c: round(v["rmsd"], 4) for c, v in g["per_chain"].items()},
            "conf_delta": {k: round(rb[i][k] - ra[i][k], 6)
                           for k in CONF if k in ra[i] and k in rb[i]},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--ab", type=Path, required=True, help="the fold_ab json, for the picks/walls")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--repeat-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    ab = json.loads(a.ab.read_text())

    rec = {"workdir": str(a.workdir), "ab": str(a.ab), "seeds": seeds, "controls": {}, "lever": {}}

    rpt = f"off_s{a.repeat_seed}_repeat"
    if (a.workdir / rpt).exists():
        rec["controls"]["determinism"] = {
            f"off_s{a.repeat_seed} vs {rpt}": cmp_legs(a.workdir, f"off_s{a.repeat_seed}", rpt)}
    rec["controls"]["seed_spread"] = {
        f"off_s{i} vs off_s{j}": cmp_legs(a.workdir, f"off_s{i}", f"off_s{j}")
        for i, j in itertools.combinations(seeds, 2)}
    rec["lever"] = {f"off_s{s} vs on_s{s}": cmp_legs(a.workdir, f"off_s{s}", f"on_s{s}")
                    for s in seeds}

    def rmsds(block):
        return [v["kabsch_rmsd"] for cmps in block.values() for v in cmps.values()]

    def dconf(block, key):
        return [abs(v["conf_delta"].get(key, 0.0))
                for cmps in block.values() for v in cmps.values()]

    det = rmsds(rec["controls"].get("determinism", {}))
    spread = rmsds(rec["controls"]["seed_spread"])
    lev = rmsds(rec["lever"])
    ceiling = max(spread + det) if (spread or det) else float("inf")
    rec["summary"] = {
        "det_floor_rmsd": det, "seed_spread_rmsd": spread, "lever_rmsd": lev,
        "control_ceiling_rmsd": round(ceiling, 4),
        "lever_max_rmsd": round(max(lev), 4) if lev else None,
        "plddt_control_max": round(max(dconf(rec["controls"].get("determinism", {}), "plddt")
                                       + dconf(rec["controls"]["seed_spread"], "plddt")), 6),
        "plddt_lever_max": round(max(dconf(rec["lever"], "plddt")), 6),
    }
    s = rec["summary"]
    rec["summary"]["geometry_pass"] = bool(lev) and max(lev) <= ceiling
    rec["summary"]["plddt_pass"] = s["plddt_lever_max"] <= s["plddt_control_max"]
    rec["summary"]["arm_took"] = ab.get("arm_took")
    rec["summary"]["arm_off_clean"] = ab.get("arm_off_clean")
    rec["summary"]["verdict"] = (
        "GATE PASS" if (rec["summary"]["geometry_pass"] and rec["summary"]["plddt_pass"]
                        and ab.get("arm_took") and ab.get("arm_off_clean")) else "GATE FAIL")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["summary"], indent=1))
    for name, block in (("determinism", rec["controls"].get("determinism", {})),
                        ("seed spread", rec["controls"]["seed_spread"]),
                        ("LEVER", rec["lever"])):
        for k, cmps in block.items():
            for i, v in cmps.items():
                print(f"{name:12s} {k:28s} {i}: rmsd {v['kabsch_rmsd']:.4f} "
                      f"pcc {v['coord_pcc']:.6f} lddt {v['lddt']:.4f} "
                      f"dplddt {v['conf_delta'].get('plddt', 0):+.6f}", flush=True)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
