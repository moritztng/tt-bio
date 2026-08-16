"""Host-only check of tt_bio.rfd3.design._write_cif: featurize the sweep's own
RFD3 payloads, write a CIF with placeholder coordinates, and report what the
writer put in the file. No device, no sampler -- the fix is in the writer."""
import collections, json, pathlib, re, sys, tempfile
import torch
from tt_bio.rfd3.design import _write_cif
from tt_bio.rfd3.featurize import featurize
from tt_bio.rfd3.input import InputSpecification

PAY = pathlib.Path("perf/wh-correctness/results/payloads")

def report(tag, cif_path):
    lines = pathlib.Path(cif_path).read_text().splitlines()
    hdr = [l.strip() for l in lines if l.strip().startswith("_atom_site.")]
    cols = {h.split(".")[1]: i for i, h in enumerate(hdr)}
    rows = [l.split() for l in lines if l.startswith(("ATOM", "HETATM"))]
    g = lambda r, n: r[cols[n]]
    names = collections.Counter(g(r, "label_atom_id") for r in rows)
    elems = collections.Counter(g(r, "type_symbol") for r in rows)
    resn = collections.Counter(g(r, "label_comp_id") for r in rows)
    nv = sum(v for k, v in names.items() if re.fullmatch(r"V\d", k))
    print(f"--- {tag}: {len(rows)} atoms, V0-V8 = {nv}")
    print("    elements:", dict(elems))
    print("    resnames:", dict(resn.most_common(10)), "n_distinct =", len(resn))
    # element vs atom name agreement, the thing that was broken
    bad = [(g(r,"label_atom_id"), g(r,"type_symbol")) for r in rows
           if g(r,"label_atom_id").strip('"')[0].upper() != g(r,"type_symbol").upper()]
    print("    name/element disagreements:", len(bad), bad[:5])
    return len(rows), nv, dict(elems), dict(resn)

for cell in ["des_rfd3_binder", "des_rfd3_na", "des_rfd3_scaffold"]:
    pay = json.loads((PAY / f"{cell}.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        struct = td / "input.pdb"
        struct.write_text(pay["structure"])
        spec = InputSpecification.from_dict(
            {"input": str(struct), "contig": pay["contig"]})
        f = featurize(str(struct), spec)
        n_atom = f["atom_to_token_map"].shape[0]
        coords = torch.arange(n_atom * 3, dtype=torch.float32).reshape(n_atom, 3) * 0.01
        out = td / "out.cif"
        _write_cif(coords, f, out)
        iv = f["is_virtual"]
        print(f"### {cell}: featurized {n_atom} atoms, is_virtual True for {int(iv.sum())}")
        report(cell, out)
