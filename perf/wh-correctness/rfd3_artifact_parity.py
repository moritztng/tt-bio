"""Parity: the fix must change the artifact, not the chemistry.

Write the same features+coords with the pre-fix writer (recovered from git) and
the current one, then assert every surviving atom's coordinate is bit-identical
and that the only rows dropped are the is_virtual ones."""
import importlib.util, json, pathlib, subprocess, sys, tempfile
import numpy as np, torch
from tt_bio.rfd3.featurize import featurize
from tt_bio.rfd3.input import InputSpecification

old_src = subprocess.check_output(
    ["git", "show", "b776e4167:tt_bio/rfd3/design.py"], text=True)
old_path = pathlib.Path("/tmp/rfd3chk/design_old.py")
old_path.write_text(old_src)
# Load only _write_cif and its helpers: the module's device imports are
# irrelevant here and would drag in the whole engine.
import types
ns = {}
src_lines = old_src.splitlines()
start = next(i for i, l in enumerate(src_lines) if l.startswith("def _write_cif"))
end = next(i for i, l in enumerate(src_lines) if l.startswith("def run_design"))
exec("from pathlib import Path\n" + "\n".join(src_lines[start:end]), ns)
old = types.SimpleNamespace(**ns)
from tt_bio.rfd3 import design as new

def rows(p):
    lines = pathlib.Path(p).read_text().splitlines()
    hdr = [l.strip() for l in lines if l.strip().startswith("_atom_site.")]
    c = {h.split(".")[1]: i for i, h in enumerate(hdr)}
    out = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            f = l.split()
            out.append((f[c["label_atom_id"]], f[c["type_symbol"]], f[c["label_comp_id"]],
                        f[c["Cartn_x"]], f[c["Cartn_y"]], f[c["Cartn_z"]]))
    return out

PAY = pathlib.Path("perf/wh-correctness/results/payloads")
for cell in ["des_rfd3_binder", "des_rfd3_na", "des_rfd3_scaffold"]:
    pay = json.loads((PAY / f"{cell}.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td); s = td / "in.pdb"; s.write_text(pay["structure"])
        f = featurize(str(s), InputSpecification.from_dict(
            {"input": str(s), "contig": pay["contig"]}))
        n = f["atom_to_token_map"].shape[0]
        g = torch.Generator().manual_seed(0)
        coords = torch.randn(n, 3, generator=g) * 10.0
        po, pn = td / "old.cif", td / "new.cif"
        old._write_cif(coords, f, po); new._write_cif(coords, f, pn)
        ro, rn = rows(po), rows(pn)
        iv = f["is_virtual"].numpy().astype(bool)
        # rows the old writer emitted for non-virtual atoms, in order
        kept_old = [r for r, v in zip(ro, iv) if not v]
        coords_match = [r[3:] for r in kept_old] == [r[3:] for r in rn]
        print(f"{cell}: old={len(ro)} new={len(rn)} dropped={len(ro)-len(rn)} "
              f"is_virtual={int(iv.sum())} | coords bit-identical: {coords_match}")
        assert len(ro) - len(rn) == int(iv.sum()), "dropped rows != is_virtual count"
        assert coords_match, "COORDINATE CHANGE — not a parity-preserving fix"
print("PARITY OK: only is_virtual rows dropped; every surviving coordinate bit-identical")
