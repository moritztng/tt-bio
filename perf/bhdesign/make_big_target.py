"""Build a design target bigger than any single chain on hand, by putting real chains side by side.

    python3 perf/bhdesign/make_big_target.py \
        --parts perf/ceilrfd3/targets/laczc_1008.cif:A perf/ceilrfd3/targets/gpb_823.cif:A \
        --out perf/bhdesign/targets/big_1831.cif

The design models' capacity axis is TARGET residues, and the largest target this repo carries is
1DP0 chain A at 1011 -- which is why `capacity_gate.EXEMPT` says pxdesign's own fixture source
cannot reach a 1536 bar. Stacking two real chains reaches it without inventing coordinates: each
chain keeps its deposited geometry, and the second is translated clear of the first along +x so
nothing clashes. It is a two-chain target, which is what a real complex is, and not one chain with
a 150 A jump in its backbone -- that would read as a break to anything that scores geometry.
"""
import argparse
import pathlib

COLS = ["group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
        "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "auth_seq_id",
        "auth_comp_id", "auth_asym_id", "auth_atom_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num", "id"]


def read_atoms(path):
    """(column names, ATOM rows). The loop_ header is read, never assumed -- the two fixture
    families in this repo use different column orders."""
    cols, rows = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
        elif st.startswith(("ATOM", "HETATM")):
            f = st.split()
            if len(f) == len(cols):
                rows.append(f)
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, metavar="CIF:CHAIN")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--gap", type=float, default=150.0,
                    help="Angstroms between consecutive chains along +x")
    a = ap.parse_args()

    out_rows, chain_ids, shift, total_res = [], "ABCDEFGH", 0.0, 0
    for i, part in enumerate(a.parts):
        cif, _, chain = part.partition(":")
        cols, rows = read_atoms(cif)
        idx = {n: cols.index(n) for n in cols}
        ch = idx.get("label_asym_id", idx.get("auth_asym_id"))
        sq = idx.get("label_seq_id", idx.get("auth_seq_id"))
        keep = [r for r in rows if r[ch] == (chain or r[ch])]
        if not keep:
            raise SystemExit(f"{cif}: no atoms in chain {chain}")
        xs = [float(r[idx["Cartn_x"]]) for r in keep]
        dx = shift - min(xs)
        seen, nres = {}, 0
        for r in keep:
            if r[sq] not in seen:
                nres += 1
                seen[r[sq]] = nres
            out_rows.append({
                "group_PDB": "ATOM", "type_symbol": r[idx["type_symbol"]],
                "label_atom_id": r[idx["label_atom_id"]], "label_alt_id": ".",
                "label_comp_id": r[idx["label_comp_id"]],
                "label_asym_id": chain_ids[i], "label_entity_id": str(i + 1),
                "label_seq_id": str(seen[r[sq]]), "pdbx_PDB_ins_code": ".",
                "auth_seq_id": str(seen[r[sq]]), "auth_comp_id": r[idx["label_comp_id"]],
                "auth_asym_id": chain_ids[i], "auth_atom_id": r[idx["label_atom_id"]],
                "Cartn_x": "%.3f" % (float(r[idx["Cartn_x"]]) + dx),
                "Cartn_y": r[idx["Cartn_y"]], "Cartn_z": r[idx["Cartn_z"]],
                "pdbx_PDB_model_num": "1", "id": str(len(out_rows) + 1)})
        shift = max(xs) + dx + a.gap
        total_res += nres
        print(f"{cif} chain {chain}: {nres} residues, {len(keep)} atoms -> chain {chain_ids[i]}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    body = ["data_" + a.out.stem, "#", "loop_"]
    body += ["_atom_site." + c for c in COLS]
    body += [" ".join(r[c] for c in COLS) for r in out_rows]
    body += ["#"]
    a.out.write_text("\n".join(body) + "\n")
    print(f"{a.out}: {total_res} residues, {len(out_rows)} atoms, {len(a.parts)} chains")


if __name__ == "__main__":
    main()
