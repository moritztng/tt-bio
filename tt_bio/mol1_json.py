"""PDB / mmCIF -> Protenix / Mol-1 input JSON, upstream's own converter (`mol json`, vendored in tt_bio/_vendor/mol1).

The converter reads the wwPDB Chemical Component Dictionary for residue types, one-letter codes and bonds; tt-bio
fetches Mol-1's copy (weights row ``mol1-components``) on first use.
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def _use_components(path: Path) -> None:
    """Point the vendored CCD reader at ``path`` (upstream reads $MOL_ROOT_DIR/common/components.cif)."""
    from tt_bio._vendor.mol1 import ccd

    if ccd.COMPONENTS_FILE != str(path):
        ccd.COMPONENTS_FILE = str(path)
        for f in (ccd.biotite_load_ccd_cif, ccd.get_component_atom_array, ccd.get_one_letter_code, ccd.get_mol_type):
            f.cache_clear()


def structures(path: Path) -> list[Path]:
    """The .pdb / .cif files under ``path`` (a file, or a directory searched recursively), as upstream picks them."""
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    files = [p for p in files if p.name.endswith((".pdb", ".cif", ".pdb.gz", ".cif.gz"))]
    if not files:
        raise ValueError(f"no .pdb or .cif file in {path}")
    return files


def convert(src: Path, out_dir: Path, components: Path, altloc: str = "first", assembly_id: str | None = None,
            include_discont_poly_poly_bonds: bool = False) -> list[Path]:
    """Write ``<out_dir>/<stem>.json`` for every structure in ``src``; returns the paths written."""
    _use_components(components)
    from tt_bio._vendor.mol1.data_utils import pdb_to_cif
    from tt_bio._vendor.mol1.json_maker import cif_to_input_json

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in structures(src):
        stem = f.name.split(".")[0]
        dest = out_dir / f"{stem[:20]}.json"
        kw = dict(assembly_id=assembly_id, altloc=altloc, output_json=str(dest),
                  include_discont_poly_poly_bonds=include_discont_poly_poly_bonds)
        if ".pdb" in f.suffixes:
            with tempfile.TemporaryDirectory() as td:
                pdb = f
                if f.suffix == ".gz":
                    import gzip
                    pdb = Path(td) / f"{stem}.pdb"
                    pdb.write_bytes(gzip.decompress(f.read_bytes()))
                cif = Path(td) / "structure.cif"
                pdb_to_cif(str(pdb), str(cif))
                cif_to_input_json(str(cif), sample_name=stem[:20], **kw)
        else:
            cif_to_input_json(str(f), **kw)
        written.append(dest)
    return written
