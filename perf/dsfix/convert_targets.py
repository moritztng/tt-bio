"""Convert one chain of each ladder target CIF to a single-chain PDB for RFD3 --from_pdb.

Regenerate: env python3 perf/dsfix/convert_targets.py  (gemmi 0.7.5, in the tt-bio env)
"""
import hashlib, pathlib, sys
import gemmi

SRC = pathlib.Path("examples/ground_truth_structures")
DST = pathlib.Path("perf/dsfix/targets")
LADDER = [("R0", "prot.cif", "A"), ("R1", "9dsg.cif", "A"), ("R2", "9loe.cif", "A"),
          ("R3", "9ma0.cif", "A"), ("R4", "9q6y.cif", "A")]

for rung, fn, chain in LADDER:
    st = gemmi.read_structure(str(SRC / fn))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    model = st[0]
    # MSE (selenomethionine) is a crystallography artifact, not a sequence modification.
    # RFD3's featuriser only tokenises the standard 20, so an MSE leaves a hole in the
    # contig ("contig indexes A17 not present in input structure"). Standard PDB
    # sanitisation, not a chimera. Only prot.cif carries any (2 of 117).
    for ch_ in model:
        for res in ch_:
            if res.name == "MSE":
                res.name = "MET"
                res.het_flag = "A"
                for at in res:
                    if at.name == "SE":
                        at.name = "SD"
                        at.element = gemmi.Element("S")
    keep = [ch.name for ch in model if ch.name != chain]
    for name in keep:
        model.remove_chain(name)
    ch = model[chain]
    # Drop anything the featuriser still would not tokenise, so the contig has no holes.
    drop = [r.seqid.num for r in ch
            if gemmi.find_tabulated_residue(r.name) is None
            or not gemmi.find_tabulated_residue(r.name).is_amino_acid()]
    if drop:
        for num in drop:
            for i, r in enumerate(ch):
                if r.seqid.num == num:
                    del ch[i]
                    break
    nres = len(ch)
    natom = sum(len(r) for r in ch)
    # RFD3 contigs index residues positionally from 1; renumber so A1-<nres> is the whole chain.
    for i, res in enumerate(ch, start=1):
        res.seqid.num = i
        res.label_seq = i
    st.setup_entities()
    out = DST / f"{rung}_{fn[:-4]}_{chain}.pdb"
    st.write_pdb(str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    src_sha = hashlib.sha256((SRC / fn).read_bytes()).hexdigest()
    seq = gemmi.one_letter_code([r.name for r in ch])
    print(f"{rung}\t{fn}:{chain}\tres={nres}\tatoms={natom}\tdropped={len(drop)}\t{out}")
    print(f"    sha256={sha}  src_sha256={src_sha}")
    print(f"    seq={seq}")
