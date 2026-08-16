"""List the clashing pairs each tree produces, to test whether whcut's extra contact is an
addition to whpre's set or a different set entirely. Mirrors check_structure.clashes()'s
rules: heavy atoms, >=2 residues apart or different chains, under CLASH_DIST."""
import sys, math, re, gemmi
CLASH_DIST = 2.0
VIRTUAL = re.compile(r"^(OXT$|.*V\d|X)")

def pairs(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]
    ns = gemmi.NeighborSearch(st, 5.0).populate()
    seen = set()
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.element == gemmi.Element("H") or VIRTUAL.match(atom.name):
                    continue
                for m in ns.find_atoms(atom.pos, "\0", radius=CLASH_DIST):
                    cra = m.to_cra(model)
                    if cra.atom.element == gemmi.Element("H") or VIRTUAL.match(cra.atom.name):
                        continue
                    if cra.chain.name == chain.name and abs(cra.residue.seqid.num - res.seqid.num) < 2:
                        continue
                    d = cra.atom.pos.dist(atom.pos)
                    if d >= CLASH_DIST or d == 0:
                        continue
                    a = f"{res.seqid.num}{res.name}.{atom.name}"
                    b = f"{cra.residue.seqid.num}{cra.residue.name}.{cra.atom.name}"
                    seen.add((tuple(sorted((a, b))), round(d, 3)))
    return {k: v for k, v in seen}

A, B = pairs(sys.argv[1]), pairs(sys.argv[2])
ka, kb = set(A), set(B)
print(f"whcut {len(ka)} pairs, whpre {len(kb)} pairs")
print(f"shared: {len(ka & kb)}")
print("only in whcut:")
for k in sorted(ka - kb): print("   ", k[0], round(A[k], 3))
print("only in whpre:")
for k in sorted(kb - ka): print("   ", k[0], round(B[k], 3))
