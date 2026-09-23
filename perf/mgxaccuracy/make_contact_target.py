#!/usr/bin/env python3
"""Cut a large design target out of a real assembly, and REFUSE one whose chains do not touch.

`perf/bhdesign/make_big_target.py` reaches a big token count by translating real chains 150 A
apart along +x, and says so: "It is a two-chain target, which is what a real complex is". For
the capacity question that is sound -- the chains keep their deposited geometry and the token
axis is real. For the QUALITY question it is not, and the difference is measurable rather than
a matter of taste:

    perf/bhdesign/targets/big_1831.cif   chain A-B centroid 246.9 A, min atom distance >150 A
                                         1536-residue crop: Rg 123.3 A, max radius 201.2 A
    1GPB biological assembly             chain A-A_2 centroid  52.9 A, min atom distance 1.88 A
                                         1646 residues:      Rg  38.4 A, max radius  64.6 A

A real complex has an INTERFACE. Two chains a quarter of a micron apart do not, and a designer
conditioned on a 64-bin distogram clamped at 22 A (`tt_bio/pxdesign/featurize.py:40`) receives
no information at all about their relative placement -- every inter-chain pair sits in the
saturated top bin. Measured consequence: pxdesign's own `fit_rmsd` reads 95.18 A on the
1536-residue crop of `big_1831.cif` against 0.0765 A at 512, and the binder it delivers is
14-41 A off the target with zero atom pairs within 5 A.

So this script builds the other kind of target and will not write one that is not in contact.

    python3 perf/mgxaccuracy/make_contact_target.py \\
        --cif /tmp/1gpb_asm.cif --chains A,A-2 --name gpb_dimer_1646 \\
        --out-dir perf/mgxaccuracy/targets

Source: `https://files.rcsb.org/download/1gpb-assembly1.cif` — the BIOLOGICAL ASSEMBLY, not the
asymmetric unit. 1GPB's deposited ASU carries one 823-residue chain and its dimer is generated
by symmetry, which is why `perf/ceilrfd3/targets/gpb_823.cif` is a single chain and why the
ladder had nothing above 1011 residues to crop in the first place.
"""
import argparse
import collections
import pathlib
import sys

# The column order `perf/bhdesign/make_big_target.py` writes, so the two targets are the same
# shape of file and `perf/bhdesign/ladder.py:crop_cif` treats them identically.
COLS = ["group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
        "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id",
        "auth_comp_id", "auth_asym_id", "auth_atom_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num", "id"]

CONTACT_A = 5.0     # an atom pair this close is an interface contact


def read_atoms(path: pathlib.Path):
    """(column names, ATOM rows). The loop_ header is read, never assumed."""
    cols, rows = [], []
    for line in path.read_text().splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
        elif st.startswith("ATOM"):        # polymer only: HETATM drops ligands and waters
            f = st.split()
            if len(f) == len(cols):
                rows.append(f)
    return cols, rows


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif", required=True, type=pathlib.Path)
    ap.add_argument("--chains", required=True,
                    help="comma list of source chain ids, in the order they should be written")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--contact-a", type=float, default=CONTACT_A)
    a = ap.parse_args()

    cols, rows = read_atoms(a.cif)
    ix = {n: cols.index(n) for n in cols}
    ch = ix.get("label_asym_id", ix.get("auth_asym_id"))
    sq = ix.get("label_seq_id", ix.get("auth_seq_id"))
    xi = ix["Cartn_x"]
    want = [c.strip() for c in a.chains.split(",") if c.strip()]

    src = collections.defaultdict(list)
    for r in rows:
        if r[ch] in want:
            src[r[ch]].append(r)
    missing = [c for c in want if c not in src]
    if missing:
        raise SystemExit(f"{a.cif}: no ATOM rows for chain(s) {missing}; "
                         f"file carries {sorted({r[ch] for r in rows})}")

    # --- the refusal. Every chain must touch at least one other chain. ---
    xyz = {c: np.array([[float(r[xi]), float(r[xi + 1]), float(r[xi + 2])] for r in v])
           for c, v in src.items()}
    touched = {c: False for c in want}
    for i in range(len(want)):
        for j in range(i + 1, len(want)):
            A, B = xyz[want[i]], xyz[want[j]]
            m = min(float(np.linalg.norm(A[k:k + 400][:, None, :] - B[None, :, :],
                                         axis=-1).min())
                    for k in range(0, len(A), 400))
            print(f"{want[i]}-{want[j]}: min atom distance {m:.2f} A, "
                  f"centroid {np.linalg.norm(A.mean(0) - B.mean(0)):.1f} A")
            if m <= a.contact_a:
                touched[want[i]] = touched[want[j]] = True
    stranded = [c for c, t in touched.items() if not t] if len(want) > 1 else []
    if stranded:
        raise SystemExit(
            f"REFUSED: chain(s) {stranded} have no atom within {a.contact_a} A of another "
            f"chain. A target whose chains do not touch is not a design problem -- it is two "
            f"targets in one file, and a distogram-conditioned designer cannot place them "
            f"relative to each other. See this script's docstring for the measurement.")

    out_rows, ids, total = [], "ABCDEFGH", 0
    for i, c in enumerate(want):
        seen = {}
        for r in src[c]:
            if r[sq] not in seen:
                seen[r[sq]] = len(seen) + 1          # renumber 1..N by sequential index
            out_rows.append({
                "group_PDB": "ATOM", "type_symbol": r[ix["type_symbol"]],
                "label_atom_id": r[ix["label_atom_id"]], "label_alt_id": ".",
                "label_comp_id": r[ix["label_comp_id"]],
                "label_asym_id": ids[i], "label_entity_id": str(i + 1),
                "label_seq_id": str(seen[r[sq]]), "pdbx_PDB_ins_code": ".",
                "auth_seq_id": str(seen[r[sq]]), "auth_comp_id": r[ix["label_comp_id"]],
                "auth_asym_id": ids[i], "auth_atom_id": r[ix["label_atom_id"]],
                "Cartn_x": "%.3f" % float(r[xi]), "Cartn_y": "%.3f" % float(r[xi + 1]),
                "Cartn_z": "%.3f" % float(r[xi + 2]),
                "pdbx_PDB_model_num": "1", "id": str(len(out_rows) + 1)})
        total += len(seen)
        print(f"{c} -> chain {ids[i]}: {len(seen)} residues, {len(src[c])} atoms")

    # Coordinates are NOT translated. The interface is the point; moving a chain destroys it.
    allxyz = np.vstack([xyz[c] for c in want])
    cen = allxyz.mean(0)
    print(f"{total} residues, {len(out_rows)} atoms, "
          f"Rg {np.sqrt(((allxyz - cen) ** 2).sum(1).mean()):.1f} A, "
          f"max radius {np.linalg.norm(allxyz - cen, axis=1).max():.1f} A")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.name}.cif"
    body = [f"data_{a.name}", "#", "loop_"] + [f"_atom_site.{c}" for c in COLS]
    body += [" ".join(r[c] for c in COLS) for r in out_rows] + ["#"]
    out.write_text("\n".join(body) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
