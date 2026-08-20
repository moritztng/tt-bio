"""The AF2-IG parity fixture: a target crop plus a synthetic binder, built the same way twice.

The reference capture (`capture_ref_features.py`, run in an external JAX env) and the tt-bio
featurizer are only comparable if they are fed byte-identical inputs, so the fixture geometry
lives here and nowhere else. One target crop from `perf/pxdesign/targets/laczc_*.cif` becomes
chain A; its first `binder` residues, translated clear of the target, become chain B. That is the
same synthetic binder the host-CPU measurement used, so the parity fixture and the perf
denominator describe the same shape.

`perf/pxdesign/af2ig_cpu_bench.py` (on `wk/pxdesign-af2ig-decision`) carries its own copy of
these three functions. Point it here when that branch merges.
"""
from __future__ import annotations

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Chain B is translated this far along x. Far enough that no target atom is within any contact
# cutoff, so the initial guess is an unbound pose and the fixture is not accidentally a hit.
BINDER_SHIFT_A = 60.0


def read_crop_cif(path: str) -> list[dict]:
    """Parse a `perf/pxdesign/targets/laczc_*.cif` crop into ordered residues.

    `make_targets.py` writes a fixed `_atom_site` header, one chain, renumbered 1..N. Columns are
    resolved by header name, never by index.
    """
    cols: list[str] = []
    residues: dict[int, dict] = {}
    order: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("_atom_site."):
                cols.append(line.split(".", 1)[1].strip())
                continue
            if not line.startswith("ATOM"):
                continue
            rec = dict(zip(cols, line.split()))
            key = int(rec["label_seq_id"])
            if key not in residues:
                residues[key] = {"comp": rec["label_comp_id"], "atoms": []}
                order.append(key)
            residues[key]["atoms"].append(
                (rec["label_atom_id"], rec["type_symbol"],
                 float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])))
    return [residues[k] for k in order]


def write_complex_pdb(path: str, target_res: list[dict], binder_res: list[dict],
                      shift: float = BINDER_SHIFT_A) -> int:
    """Target as chain A, binder as chain B translated by `shift`. Returns the atom count."""
    serial = 0
    with open(path, "w") as fh:
        for chain, res_list, dx in (("A", target_res, 0.0), ("B", binder_res, shift)):
            for i, res in enumerate(res_list, start=1):
                for (name, elem, x, y, z) in res["atoms"]:
                    serial += 1
                    aname = name if len(name) >= 4 else " %-3s" % name
                    fh.write("ATOM  %5d %s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                             % (serial, aname, res["comp"], chain, i,
                                x + dx, y, z, elem.rjust(2)))
            fh.write("TER\n")
        fh.write("END\n")
    return serial


def seq_of(res_list: list[dict]) -> str:
    return "".join(AA3TO1.get(r["comp"], "A") for r in res_list)


def build_fixture(cif_path: str, pdb_path: str, binder: int = 80) -> dict:
    """Write the complex PDB and return everything the two arms need to agree on."""
    target_res = read_crop_cif(cif_path)
    binder_res = target_res[:binder]
    atoms = write_complex_pdb(pdb_path, target_res, binder_res)
    return {
        "pdb": pdb_path,
        "target_residues": len(target_res),
        "binder_residues": len(binder_res),
        "n_tokens": len(target_res) + len(binder_res),
        "complex_atoms": atoms,
        "binder_seq": seq_of(binder_res),
        "target_seq": seq_of(target_res),
        "binder_shift_a": BINDER_SHIFT_A,
    }
