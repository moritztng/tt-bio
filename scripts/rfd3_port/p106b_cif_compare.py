#!/usr/bin/env python3
"""p106b -- compare two RFD3 design outputs on what the model actually delivers.

p106's all-atom comparison failed its own shape check, and the failure was the instrument's, not
the arm's: both arms designed 685 residues, but they did not design the same residues everywhere,
and different amino acids have different sidechain atom counts (5126 atoms against 5124). An
all-atom array cannot be subtracted across a sequence change.

What RFD3 delivers is a backbone and a sequence, so that is what gets compared:

  backbone   N, CA, C, O keyed on (chain, residue number, atom name), which every residue has
             regardless of identity -- so this is a real per-atom disagreement, and since both
             arms ran the same seed on the same target down the same trajectory it needs no
             superposition and an RMSD here is not an alignment artifact
  sequence   the fraction of residues both arms gave the same three-letter code

Takes two output directories. No device, no fold: it reads CIFs that already exist.
"""
import json
import pathlib
import sys

import torch

A = pathlib.Path(sys.argv[1])
B = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "perf/p106/cif_compare.json")
BACKBONE = ("N", "CA", "C", "O")


def parse(cif):
    """{(chain, resnum, atom): (x,y,z)} and {(chain, resnum): resname}, from the _atom_site loop."""
    lines = cif.read_text().splitlines()
    atoms, seq = {}, {}
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        tags, j = [], i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            tags.append(lines[j].strip())
            j += 1
        if not any(t.startswith("_atom_site.") for t in tags):
            i = j
            continue
        col = {t: n for n, t in enumerate(tags)}

        def c(name, alts=()):
            for k in (name,) + alts:
                if "_atom_site." + k in col:
                    return col["_atom_site." + k]
            raise SystemExit("no _atom_site.%s in %s (have %s)" % (name, cif, tags))

        i_atom = c("label_atom_id", ("auth_atom_id",))
        i_res = c("label_comp_id", ("auth_comp_id",))
        i_ch = c("label_asym_id", ("auth_asym_id",))
        i_num = c("label_seq_id", ("auth_seq_id",))
        ix, iy, iz = (c("Cartn_" + a) for a in "xyz")
        while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
            f = lines[j].split()
            if len(f) >= len(tags):
                key = (f[i_ch], f[i_num])
                atoms[key + (f[i_atom],)] = (float(f[ix]), float(f[iy]), float(f[iz]))
                seq[key] = f[i_res]
            j += 1
        return atoms, seq
    raise SystemExit("no _atom_site loop in %s" % cif)


def main():
    ca = sorted(A.glob("*.cif"))
    cb = sorted(B.glob("*.cif"))
    if not ca or not cb:
        raise SystemExit("need a CIF in both %s and %s" % (A, B))
    at_a, seq_a = parse(ca[0])
    at_b, seq_b = parse(cb[0])

    res = sorted(set(seq_a) & set(seq_b))
    same = sum(1 for k in res if seq_a[k] == seq_b[k])
    keys = [k + (nm,) for k in res for nm in BACKBONE if k + (nm,) in at_a and k + (nm,) in at_b]
    xa = torch.tensor([at_a[k] for k in keys])
    xb = torch.tensor([at_b[k] for k in keys])
    per_atom = (xa - xb).norm(dim=-1)

    out = dict(
        a=str(ca[0]), b=str(cb[0]),
        n_atoms_a=len(at_a), n_atoms_b=len(at_b),
        n_residues_a=len(seq_a), n_residues_b=len(seq_b), n_residues_shared=len(res),
        sequence_identity=round(same / len(res), 5) if res else None,
        n_residues_differing=len(res) - same,
        n_backbone_compared=len(keys),
        backbone_rmsd=round(float((per_atom ** 2).mean().sqrt()), 4),
        backbone_median_shift=round(float(per_atom.median()), 4),
        backbone_max_shift=round(float(per_atom.max()), 4),
        backbone_p99_shift=round(float(per_atom.quantile(0.99)), 4),
    )
    print("residues %d shared, sequence identity %.4f (%d differ)"
          % (out["n_residues_shared"], out["sequence_identity"], out["n_residues_differing"]))
    print("atoms %d vs %d -- the difference is the sequence, not a missing residue"
          % (out["n_atoms_a"], out["n_atoms_b"]))
    print("backbone over %d atoms: RMSD %.4f A, median %.4f, p99 %.4f, max %.4f"
          % (out["n_backbone_compared"], out["backbone_rmsd"], out["backbone_median_shift"],
             out["backbone_p99_shift"], out["backbone_max_shift"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
