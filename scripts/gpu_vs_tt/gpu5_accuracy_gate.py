#!/usr/bin/env python3
"""Per-run accuracy gate for the five-model GPU benchmark.

Rule 6 of the task contract: a rung that fails its own accuracy check is not a speed
data point. Five models write five different confidence conventions, so the gate is
split in two:

  HARD  geometry. Finite coordinates, the residue count the fixture asks for, a median
        consecutive CA-CA distance in the protein band, and a radius of gyration that is
        neither a collapsed ball nor an exploded chain. A broken install, a truncated
        input or a NaN-producing kernel fails here. Failing this means the run is not a
        speed data point.
  SOFT  confidence. Mean plDDT off the CA B-factor column, normalised to 0-1. Reported,
        never a pass/fail on its own: the 512 aa fixture is a tandem-duplicated CDK2 and
        a single-sequence model is entitled to a low score on it without being broken.
        What IS a red flag is the same model+fixture landing far from a value already on
        record, so `--expect-plddt` prints a labelled delta rather than failing.

Pure stdlib on purpose: it has to run inside four different model venvs on a rented box
where gemmi/biotite may or may not be installed.

    python3 gpu5_accuracy_gate.py pred.cif --expect-residues 512 [--expect-plddt 0.8286]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Protein CA-CA virtual bond is 3.80 A for trans peptide bonds. The band is wide enough
# to absorb the handful of cis-proline (~2.9 A) and chain-break outliers a median shrugs
# off, and tight enough that a scrambled or half-scaled structure misses it.
CA_CA_LO, CA_CA_HI = 3.6, 4.0
# Rg of a compact globular protein scales as 2.2*N^0.38 A (Flory exponent fitted to the
# PDB); N=512 gives 24.0 A. Half that is a collapsed ball, 2.5x is a loosely packed
# multi-domain chain -- the 512 aa fixture is a tandem repeat and may well sit high in
# the band. Derived from N rather than hard-coded so the gate stays honest at other sizes.
def _rg_band(n: int) -> tuple[float, float]:
    expected = 2.2 * n ** 0.38
    return 0.5 * expected, 2.5 * expected
# A chain break is a consecutive pair beyond 5 A. Diffusion models leave a few; a
# structure that is mostly breaks is not a fold.
BREAK_CUT, BREAK_FRAC_MAX = 5.0, 0.05


def _parse_cif(text: str):
    """CA rows out of an mmCIF atom_site loop -> [(chain, resnum, x, y, z, bfac)]."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j, cols = i + 1, []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            cols.append(lines[j].strip())
            j += 1
        if not cols or not cols[0].startswith("_atom_site."):
            i = j
            continue
        idx = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}
        need = ("label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z")
        if any(n not in idx for n in need):
            raise SystemExit(f"atom_site loop is missing one of {need}")
        ch = idx.get("auth_asym_id", idx.get("label_asym_id"))
        rn = idx.get("auth_seq_id", idx.get("label_seq_id"))
        bf = idx.get("B_iso_or_equiv")
        while j < len(lines):
            s = lines[j].strip()
            if not s or s.startswith("#") or s == "loop_" or s.startswith("_"):
                break
            f = s.split()
            if len(f) >= len(cols) and f[idx["label_atom_id"]].strip('"') == "CA":
                out.append((
                    f[ch] if ch is not None else "A",
                    f[rn] if rn is not None else str(len(out)),
                    float(f[idx["Cartn_x"]]), float(f[idx["Cartn_y"]]), float(f[idx["Cartn_z"]]),
                    float(f[bf]) if bf is not None and f[bf] not in (".", "?") else float("nan"),
                ))
            j += 1
        return out
    return out


def _parse_pdb(text: str):
    out = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
            continue
        try:
            b = float(line[60:66])
        except ValueError:
            b = float("nan")
        out.append((line[21], line[22:27].strip(),
                    float(line[30:38]), float(line[38:46]), float(line[46:54]), b))
    return out


def gate(path: Path, expect_residues: int | None, expect_plddt: float | None) -> dict:
    text = path.read_text()
    cas = _parse_cif(text) if path.suffix.lower() in (".cif", ".mmcif") else _parse_pdb(text)
    r: dict = {"file": str(path), "n_ca": len(cas), "checks": {}, "fail": []}
    if not cas:
        r["fail"].append("no CA atoms parsed")
        r["pass"] = False
        return r

    xyz = [(x, y, z) for _, _, x, y, z, _ in cas]
    r["checks"]["all_finite"] = all(math.isfinite(v) for p in xyz for v in p)
    if not r["checks"]["all_finite"]:
        r["fail"].append("non-finite coordinate")

    if expect_residues is not None:
        r["checks"]["residue_count_ok"] = len(cas) == expect_residues
        if not r["checks"]["residue_count_ok"]:
            r["fail"].append(f"{len(cas)} CA atoms, fixture asks for {expect_residues}")

    d = [math.dist(xyz[k], xyz[k + 1]) for k in range(len(xyz) - 1)
         if cas[k][0] == cas[k + 1][0]]
    if d:
        med = sorted(d)[len(d) // 2]
        brk = sum(1 for v in d if v > BREAK_CUT) / len(d)
        r["ca_ca_median_A"], r["chain_break_frac"] = round(med, 4), round(brk, 4)
        r["checks"]["ca_ca_median_in_band"] = CA_CA_LO <= med <= CA_CA_HI
        r["checks"]["chain_breaks_ok"] = brk <= BREAK_FRAC_MAX
        if not r["checks"]["ca_ca_median_in_band"]:
            r["fail"].append(f"median CA-CA {med:.3f} A outside [{CA_CA_LO}, {CA_CA_HI}]")
        if not r["checks"]["chain_breaks_ok"]:
            r["fail"].append(f"{brk:.1%} of consecutive CA pairs exceed {BREAK_CUT} A")

    cx = [sum(p[k] for p in xyz) / len(xyz) for k in range(3)]
    rg = math.sqrt(sum(math.dist(p, cx) ** 2 for p in xyz) / len(xyz))
    rg_lo, rg_hi = _rg_band(len(xyz))
    r["radius_of_gyration_A"] = round(rg, 3)
    r["rg_band_A"] = [round(rg_lo, 1), round(rg_hi, 1)]
    r["checks"]["rg_in_band"] = rg_lo <= rg <= rg_hi
    if not r["checks"]["rg_in_band"]:
        r["fail"].append(f"radius of gyration {rg:.1f} A outside [{rg_lo:.1f}, {rg_hi:.1f}]")

    b = [v for *_, v in cas if math.isfinite(v)]
    if b:
        mean_b = sum(b) / len(b)
        # Every model in this benchmark writes plDDT into the CA B-factor, but Boltz-2 and
        # Protenix write 0-1 while the AF-lineage writers use 0-100. Scale off the value.
        r["plddt_scale"] = "0-100" if mean_b > 1.5 else "0-1"
        r["plddt_mean"] = round(mean_b / 100.0 if mean_b > 1.5 else mean_b, 6)
        if expect_plddt is not None:
            r["plddt_expected"] = expect_plddt
            r["plddt_delta"] = round(r["plddt_mean"] - expect_plddt, 6)

    r["pass"] = not r["fail"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", type=Path)
    ap.add_argument("--expect-residues", type=int, default=None)
    ap.add_argument("--expect-plddt", type=float, default=None,
                    help="reference plDDT for the same model+fixture; prints a delta, never fails")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    r = gate(a.structure, a.expect_residues, a.expect_plddt)
    print(json.dumps(r, indent=2))
    if a.json_out:
        a.json_out.write_text(json.dumps(r, indent=2) + "\n")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
