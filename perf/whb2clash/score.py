#!/usr/bin/env python3
"""Score one arm directory into a single JSON row, with the arm read off the probe.

Metrics come from `perf/wh-correctness/check_structure.py` -- its `clashes()` and
`chain_geometry()` are imported rather than reimplemented, so this scores exactly what the
sweep and the section 6.3 gate scored. Two things are added that the pre-registration needs
and the checker does not expose: the clash fraction as a number, and the clashing pairs
themselves, so two arms can be compared as *sets* and a reshuffle told apart from a worsening.

The arm is taken from `probe/*.json`, never from a command line: a run whose probe is missing,
or disagrees with the corner it was supposed to be, is not scoreable.

Usage: score.py RUNDIR --input target.yaml [--arm base] [--target P22303]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wh-correctness"))
import gemmi  # noqa: E402
import check_structure as CS  # noqa: E402


def clash_pairs(st):
    """The pairs behind CS.clashes()'s count, under identical filters."""
    model = st[0]
    ns = gemmi.NeighborSearch(st, 5.0).populate()
    seen = {}
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.element == gemmi.Element("H") or CS.VIRTUAL_ATOM.match(atom.name):
                    continue
                for m in ns.find_atoms(atom.pos, "\0", radius=CS.CLASH_DIST):
                    cra = m.to_cra(model)
                    if (cra.atom.element == gemmi.Element("H")
                            or CS.VIRTUAL_ATOM.match(cra.atom.name)):
                        continue
                    if (cra.chain.name == chain.name
                            and abs(cra.residue.seqid.num - res.seqid.num) < 2):
                        continue
                    dist = cra.atom.pos.dist(atom.pos)
                    if (dist < CS.DISULFIDE_MAX and atom.name == "SG" == cra.atom.name
                            and res.name == "CYS" == cra.residue.name):
                        continue
                    if dist < CS.CLASH_DIST and cra.atom.serial != atom.serial:
                        a = "%s/%d%s/%s" % (chain.name, res.seqid.num, res.name, atom.name)
                        b = "%s/%d%s/%s" % (cra.chain.name, cra.residue.seqid.num,
                                            cra.residue.name, cra.atom.name)
                        seen[tuple(sorted((a, b)))] = round(dist, 3)
    return sorted([[a, b, d] for (a, b), d in seen.items()], key=lambda r: r[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cifs = sorted(glob.glob(str(a.rundir / "**" / "*.cif"), recursive=True))
    probes = sorted(glob.glob(str(a.rundir / "probe" / "*.json")))
    row = {"target": a.target or a.input.stem, "arm": a.arm, "rundir": str(a.rundir),
           "n_cif": len(cifs), "n_probe": len(probes)}
    if not probes:
        row["error"] = "no probe -- run is not scoreable"
        print(json.dumps(row))
        return 1
    # The fold happens in a spawned child; take the probe that saw the real device grid.
    pr = [json.load(open(p)) for p in probes]
    row["probe"] = pr[-1]
    for k in ("SEQ_LEN_MORE_CHUNKING", "SDPA_DIV_K", "is_small_grid", "grid",
              "k_chunk_640", "k_chunk_768", "forced_slmc"):
        row[k] = pr[-1].get(k)
    if not cifs:
        row["error"] = "no structure produced"
        print(json.dumps(row))
        return 1

    rows = []
    for cif in cifs:
        st = gemmi.read_structure(cif)
        st.remove_alternative_conformations()
        st.setup_entities()
        n_clash, n_heavy, worst = CS.clashes(st)
        geo, fails, warns = CS.chain_geometry(st)
        digest = hashlib.sha256(Path(cif).read_bytes()).hexdigest()[:16]
        # chain_geometry only judges the backbone. The clash budget is applied in
        # check_structure's main(), so it has to be applied here too or a structure over
        # budget prints PASS -- which is the one verdict this gate must never get wrong.
        budget = max(CS.CLASH_MAX_ABS, CS.CLASH_MAX_FRAC * n_heavy)
        if n_heavy and n_clash > budget:
            fails = fails + ["%d heavy-atom clashes < %s A (%.2f%% of atoms, worst %s A)"
                             % (n_clash, CS.CLASH_DIST, 100 * n_clash / n_heavy, worst)]
        elif n_clash:
            warns = warns + ["%d marginal contacts < %s A (worst %s A)"
                             % (n_clash, CS.CLASH_DIST, worst)]
        rows.append({
            "cif": os.path.relpath(cif, a.rundir),
            "digest": digest,
            "n_clash": n_clash, "n_heavy": n_heavy,
            "clash_frac": (n_clash / n_heavy) if n_heavy else None,
            "worst_dist": worst,
            "clash_budget": budget,
            "backbone_gaps": sum(g["breaks"] for g in geo),
            "in_band_frac": min([g["in_band_frac"] for g in geo] or [None]),
            "n_res": sum(g["n_res"] for g in geo),
            "fails": fails, "warns": warns,
            "pairs": clash_pairs(st),
        })
    row["structures"] = rows
    for cand in sorted(glob.glob(str(a.rundir / "**" / "results.json"), recursive=True)):
        d = json.load(open(cand))
        d = d[0] if isinstance(d, list) and d else d
        if isinstance(d, dict):
            row["plddt"] = d.get("complex_plddt")
            row["ptm"] = d.get("ptm")
            row["runtime_s"] = d.get("runtime_s")
        break

    out = a.out or (a.rundir / "score.json")
    out.write_text(json.dumps(row, indent=1))
    s = rows[0]
    verdict = "FAIL" if s["fails"] else ("WARN" if s["warns"] else "PASS")
    print("%-10s %-6s slmc=%-5s k3=%d clash=%3d/%d (%.3f%%) budget=%.1f worst=%s "
          "gaps=%s inband=%s plddt=%s %s"
          % (row["target"], row["arm"], row["SEQ_LEN_MORE_CHUNKING"],
             int(bool(row["SDPA_DIV_K"])), s["n_clash"], s["n_heavy"],
             (s["clash_frac"] or 0) * 100, s["clash_budget"], s["worst_dist"],
             s["backbone_gaps"], s["in_band_frac"], row.get("plddt"), verdict))
    if len(rows) > 1:
        for r in rows[1:]:
            print("           %-6s  sample %s clash=%3d (%.3f%%) gaps=%s"
                  % (row["arm"], r["cif"], r["n_clash"], (r["clash_frac"] or 0) * 100,
                     r["backbone_gaps"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
