"""Build a two-chain PDB at a chosen size, so the AF2 arms can be timed above the fixture.

The port's only committed complex input is the 208-token anchor
(`parity_artifacts/designpop_bg119/binder_complex.pdb`, 88 target + 119 binder). The cell that
carries a verdict is 848 tokens: `perf/pxdesign/targets/laczc_768.cif`'s 768-residue target plus
an 80-residue binder. This slices a target out of that CIF and a binder out of the anchor and
writes the two as one PDB, which is what `af2_data.complex_features` reads.

Real coordinates on both chains, and the two chains have never seen each other -- the interface is
nonsense. That is deliberate and it is safe here: every leg that consumes this output is a TIMING
leg, and AF2's cost depends on the token count and nothing else. No accuracy claim may be made on
an input from this file.

    python3 scripts/af2_port/complex_input.py --target-residues 768 --binder-residues 80 \\
        --out /tmp/laczc768_binder80.pdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CIF = ROOT / "perf/pxdesign/targets/laczc_768.cif"
DEFAULT_BINDER = ROOT / "scripts/af2_port/parity_artifacts/designpop_bg119/binder_complex.pdb"

#: `_atom_site` field offsets in the targets' CIF, read off its own loop header.
CIF_ATOM, CIF_RES, CIF_ASYM, CIF_SEQ, CIF_XYZ = 2, 4, 5, 7, (13, 14, 15)


def _atom_line(serial: int, name: str, res_name: str, chain: str, res_seq: int,
               xyz: tuple[float, float, float]) -> str:
    """One `ATOM` record. Only the fields `af2_data.parse_pdb_chain` reads are filled."""
    field = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
    return (f"ATOM  {serial:5d} {field}{res_name:>4} {chain}{res_seq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00")


def cif_residues(path: Path, asym: str) -> list[tuple[str, list[tuple[str, tuple]]]]:
    """`(res_name, [(atom_name, xyz)])` per residue of one chain, in file order."""
    out: list[tuple[str, list]] = []
    index: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        if not raw.startswith("ATOM"):
            continue
        f = raw.split()
        if f[CIF_ASYM] != asym:
            continue
        key = f[CIF_SEQ]
        if key not in index:
            index[key] = len(out)
            out.append((f[CIF_RES], []))
        out[index[key]][1].append((f[CIF_ATOM], tuple(float(f[c]) for c in CIF_XYZ)))
    return out


def pdb_residues(path: Path, chain: str) -> list[tuple[str, list[tuple[str, tuple]]]]:
    """The same shape, from a PDB. Insertion codes are rejected, as the featuriser does."""
    out: list[tuple[str, list]] = []
    index: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        if raw[:4] != "ATOM" or raw[21:22] != chain:
            continue
        if raw[26:27].strip():
            raise ValueError(f"insertion code at {chain} {raw[22:27]!r}")
        key = raw[22:26].strip()
        if key not in index:
            index[key] = len(out)
            out.append((raw[17:20].strip(), []))
        out[index[key]][1].append(
            (raw[12:16].strip(), tuple(float(raw[30 + 8 * k:38 + 8 * k]) for k in range(3))))
    return out


def build(out: Path, target_residues: int, binder_residues: int,
          cif: Path = DEFAULT_CIF, binder_pdb: Path = DEFAULT_BINDER,
          cif_asym: str = "A", binder_chain: str = "B") -> dict:
    target = cif_residues(cif, cif_asym)
    binder = pdb_residues(binder_pdb, binder_chain)
    for name, have, want in (("target", len(target), target_residues),
                             ("binder", len(binder), binder_residues)):
        if have < want:
            raise ValueError(f"{name} source has {have} residues, need {want}")
    lines, serial = [], 1
    for chain, residues in (("A", target[:target_residues]), ("B", binder[:binder_residues])):
        for position, (res_name, atoms) in enumerate(residues, start=1):
            for atom_name, xyz in atoms:
                lines.append(_atom_line(serial, atom_name, res_name, chain, position, xyz))
                serial += 1
        lines.append("TER")
    lines.append("END")
    out.write_text("\n".join(lines) + "\n")
    return {"out": str(out), "target_residues": target_residues,
            "binder_residues": binder_residues, "tokens": target_residues + binder_residues,
            "atoms": serial - 1, "target_source": str(cif), "binder_source": str(binder_pdb)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-residues", type=int, required=True)
    ap.add_argument("--binder-residues", type=int, default=80)
    ap.add_argument("--cif", type=Path, default=DEFAULT_CIF)
    ap.add_argument("--binder-pdb", type=Path, default=DEFAULT_BINDER)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import json
    print(json.dumps(build(args.out, args.target_residues, args.binder_residues,
                           args.cif, args.binder_pdb), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
