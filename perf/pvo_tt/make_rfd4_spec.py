"""Build the PVO RFD4 input: the deposited antigen as-is, a de-novo binder of PopVax's length, hotspots from the
deposited interface. Shape copied from state/rfd4/inputs/pdl1-224.yaml.

    python make_rfd4_spec.py REF.cif ANTIGEN BINDER OUTDIR [--binder 120] [--hot-n 3]

Writes OUTDIR/<name>.cif (antigen chain only, original numbering) and OUTDIR/<name>.yaml. The hotspots are the
--hot-n antigen residues with the most heavy-atom contacts under 5 A to the deposited binder, which is how a user
picks them from a known complex; the binder itself is not given to the model.
"""
import sys
from pathlib import Path
import gemmi

ref, ag, bd, outdir = sys.argv[1:5]
a = sys.argv
blen = int(a[a.index("--binder") + 1]) if "--binder" in a else 120
hot_n = int(a[a.index("--hot-n") + 1]) if "--hot-n" in a else 3
name = f"{Path(ref).stem}-{ag}-b{blen}"
out = Path(outdir)
out.mkdir(parents=True, exist_ok=True)

st = gemmi.read_structure(ref)
st.setup_entities()
st.remove_hydrogens()
st.remove_ligands_and_waters()
m = st[0]
ns = gemmi.NeighborSearch(m, st.cell, 6).populate()
contacts = {}
for res in m[ag]:
    n = 0
    for at in res:
        for mark in ns.find_atoms(at.pos, "\0", radius=5.0):
            if mark.to_cra(m).chain.name == bd:
                n += 1
    if n:
        contacts[res.seqid.num] = n
hot = sorted(contacts, key=lambda r: (-contacts[r], r))[:hot_n]
nums = [r.seqid.num for r in m[ag]]

for ch in list(m):
    if ch.name != ag:
        m.remove_chain(ch.name)
st.setup_entities()
st.make_mmcif_document().write_file(str(out / f"{name}.cif"))

sel = "\n".join(f'        - select: "sel(\'{ag}/*/{r}/*\')"' for r in hot)
(out / f"{name}.yaml").write_text(f"""# PVO RFD4 input: a {blen}-residue de-novo binder against {Path(ref).stem} chain {ag}
# ({len(nums)} modelled residues, {len(nums) + blen} total tokens), hotspots from the deposited interface.
# Same shape as state/rfd4/inputs/pdl1-224.yaml; the target runs at its deposited size (state/pvo-tt.md).
{name.replace('-', '_')}:
  input: {name}.cif
  contig: "{blen}-{blen}, /0, {ag}{nums[0]}-{nums[-1]}"
  conditions:
    C_CRDGRP:
      1:
        - select: "sel('{ag}/*/*/[N,CA,C,O]') and not has_nan_coord()"
    C_SEQ:
      true:
        - select: "sel('{ag}')"
    C_HOT:
      true:
{sel}
""")
print(name, "antigen residues", len(nums), "range", nums[0], nums[-1], "total tokens", len(nums) + blen)
print("hotspots", [(r, contacts[r]) for r in hot])
