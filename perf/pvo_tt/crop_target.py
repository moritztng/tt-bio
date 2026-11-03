"""Epitope-centred crop of a deposited complex's antigen for RFD4 (state/pvo/INPUTS.md, RFD4 crop rule).

    python crop_target.py REF.cif ANTIGEN BINDER OUT_PREFIX [--hot 5] [--crop 12]

Hotspots = antigen residues with any heavy atom within --hot A of the deposited binder's heavy atoms. Crop = antigen
residues with any heavy atom within --crop A of a hotspot residue. Writes OUT_PREFIX.cif (the crop, original
numbering) and prints the residue segments, hotspot list and sizes.
"""
import sys
import gemmi

ref, ag, bd, out = sys.argv[1:5]
hot_r = float(sys.argv[sys.argv.index("--hot") + 1]) if "--hot" in sys.argv else 5.0
crop_r = float(sys.argv[sys.argv.index("--crop") + 1]) if "--crop" in sys.argv else 12.0
st = gemmi.read_structure(ref)
st.setup_entities()
st.remove_hydrogens()
st.remove_ligands_and_waters()
m = st[0]
A, B = m[ag], m[bd]
ns = gemmi.NeighborSearch(m, st.cell, 6).populate()


def near(res, chain, r):
    for a in res:
        for mark in ns.find_atoms(a.pos, "\0", radius=r):
            cra = mark.to_cra(m)
            if cra.chain.name == chain:
                return True
    return False


hot = [r for r in A if near(r, bd, hot_r)]
hotset = {r.seqid.num for r in hot}
ns2 = gemmi.NeighborSearch(m, st.cell, 13).populate()
keep = set()
for r in A:
    for a in r:
        for mark in ns2.find_atoms(a.pos, "\0", radius=crop_r):
            cra = mark.to_cra(m)
            if cra.chain.name == ag and cra.residue.seqid.num in hotset:
                keep.add(r.seqid.num)
                break
        if r.seqid.num in keep:
            break
nums = sorted(keep)
segs, s = [], [nums[0], nums[0]]
for n in nums[1:]:
    if n == s[1] + 1:
        s[1] = n
    else:
        segs.append(tuple(s)); s = [n, n]
segs.append(tuple(s))
for ch in list(m):
    if ch.name != ag:
        m.remove_chain(ch.name)
ch = m[ag]
for i in range(len(ch) - 1, -1, -1):
    if ch[i].seqid.num not in keep:
        del ch[i]
st.setup_entities()
doc = st.make_mmcif_document()
doc.write_file(out + ".cif")
print("antigen", ag, "residues", len(A), "binder", bd, "residues", len(B))
print("hotspots", len(hot), sorted(hotset))
print("crop", len(nums), "segments", segs)
