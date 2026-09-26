"""Build one af2ig rung at a chosen token count, from chains that already exist in the tree.

`scripts/af2_port/complex_input.py` already does this for a target CIF plus the 119-residue
anchor binder, and that caps the binder at 119 and the target at whatever `laczc_*.cif` holds --
1008 residues, so its largest reachable rung is 1127 tokens. The Blackhole bar is 1536, so this
takes the binder chain from a CIF too and reuses that file's readers for everything else.

Both chains are real contiguous protein chains and they have never seen each other, so the
interface is nonsense. That is the same deliberate choice complex_input.py documents: a rung
built here answers "does this token count fit and run", never "is this structure right".

    python3 perf/af2igcov/fixture.py --target-cif perf/pxdesign/targets/laczc_1008.cif \
        --target-residues 1008 --binder-cif perf/bhdesign/targets/big_1831.cif \
        --binder-asym B --binder-residues 528 --out-dir perf/af2igcov/work/t1008_b528
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "af2_port"))

from complex_input import _atom_line, cif_residues, pdb_residues  # noqa: E402

#: The featuriser's own table, so an unknown code becomes X here exactly as it does there.
from tt_bio._vendor.esm.utils.residue_constants import restype_3to1  # noqa: E402


def _residues(path: Path, chain: str):
    return cif_residues(path, chain) if path.suffix == ".cif" else pdb_residues(path, chain)


def build(out_dir: Path, target_cif: Path, target_residues: int, target_asym: str,
          binder_cif: Path, binder_residues: int, binder_asym: str) -> dict:
    target = _residues(target_cif, target_asym)
    binder = _residues(binder_cif, binder_asym)
    for what, have, want in (("target", len(target), target_residues),
                             ("binder", len(binder), binder_residues)):
        if have < want:
            raise ValueError(f"{what} source has {have} residues, need {want}")
    target, binder = target[:target_residues], binder[:binder_residues]

    lines, serial = [], 1
    for chain, residues in (("A", target), ("B", binder)):
        for position, (res_name, atoms) in enumerate(residues, start=1):
            for atom_name, xyz in atoms:
                lines.append(_atom_line(serial, atom_name, res_name, chain, position, xyz))
                serial += 1
        lines.append("TER")
    lines.append("END")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = out_dir / "complex.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    seq = "".join(restype_3to1.get(r, "X") for r, _ in binder)
    yaml = out_dir / "input.yaml"
    # Absolute: `tt-bio predict` copies the submission into a scratch dir before reading it,
    # so a path relative to this file resolves against a directory that does not have it.
    yaml.write_text(f"target:\n  file: {pdb.resolve()}\n  chain: A\n"
                    f"binder:\n  sequence: {seq}\n  chain: B\n")
    return {"yaml": str(yaml), "pdb": str(pdb), "atoms": serial - 1,
            "target_residues": len(target), "binder_residues": len(binder),
            "tokens": len(target) + len(binder), "binder_sequence_len": len(seq),
            "unknown_binder_residues": seq.count("X"),
            "target_source": f"{target_cif.name}:{target_asym}",
            "binder_source": f"{binder_cif.name}:{binder_asym}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cif", type=Path, required=True)
    ap.add_argument("--target-residues", type=int, required=True)
    ap.add_argument("--target-asym", default="A")
    ap.add_argument("--binder-cif", type=Path, required=True)
    ap.add_argument("--binder-residues", type=int, required=True)
    ap.add_argument("--binder-asym", default="B")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.out_dir, a.target_cif, a.target_residues, a.target_asym,
                           a.binder_cif, a.binder_residues, a.binder_asym), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
