#!/usr/bin/env python3
"""Does a structure carry a POLYMER-LIGAND covalent bond -- the predicate OF3's `bond` loss needs?

`of3t-auxheads` established that §6's last uncovered loss term is a polymer-ligand bond loss:
`core/loss/diffusion.py:208-210` builds `bond_mask = token_bonds * (is_polymer[..., None, :] *
is_ligand[..., None])`. All 8 of 8 corpus targets carry inter-token bonds and NONE carries a
polymer-ligand one, so the term cannot fire on the data the campaign holds, and the row left it
NOT COVERED with the predicate named rather than synthesising a bond to make a check pass.

This finds targets that satisfy it. The test is on the mmCIF's own annotation: a `_struct_conn`
row with `conn_type_id == covale` whose two partners sit in entities of different KIND, one
`polymer` and one `non-polymer` or `branched`.

Run it before trusting a reputation. Two structures everyone would name as covalent-ligand
complexes FAIL this test, for two different reasons, and both are recorded in the JSON beside it.

    ./check_predicate.py 4g5j 6vxx 6lu7        # ids are fetched from RCSB if not already local
"""
import json, pathlib, subprocess, sys

LIGAND_KINDS = {"non-polymer", "branched"}


def fetch(pdb_id, into):
    p = into / f"{pdb_id.lower()}.cif"
    if p.is_file() and p.stat().st_size > 5000:
        return p
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    subprocess.run(["curl", "-sS", "--max-time", "90", "-o", str(p), url], check=True)
    if p.stat().st_size < 5000:                      # an error page, not a structure
        p.unlink(missing_ok=True)
        return None
    return p


def polymer_ligand_covale(path):
    """Every covale link whose two partners are in entities of different kind."""
    import gemmi
    block = gemmi.cif.read(str(path)).sole_block()
    etype = {r[0]: r[1] for r in block.find("_entity.", ["id", "type"])}
    asym2ent = {r[0]: r[1] for r in block.find("_struct_asym.", ["id", "entity_id"])}
    out, covale = [], 0
    for r in block.find("_struct_conn.", ["conn_type_id", "ptnr1_label_asym_id",
                                          "ptnr1_label_comp_id", "ptnr2_label_asym_id",
                                          "ptnr2_label_comp_id"]):
        if r[0].strip('"\'') != "covale":
            continue
        covale += 1
        k1 = etype.get(asym2ent.get(r[1]), "?")
        k2 = etype.get(asym2ent.get(r[3]), "?")
        if (k1 == "polymer" and k2 in LIGAND_KINDS) or (k2 == "polymer" and k1 in LIGAND_KINDS):
            out.append({"a": r[2].strip('"\''), "a_kind": k1,
                        "b": r[4].strip('"\''), "b_kind": k2})
    return covale, out


if __name__ == "__main__":
    here = pathlib.Path(sys.argv[0]).resolve().parent
    ids = sys.argv[1:] or ["4g5j", "4byh", "5t3x", "6vxx", "7kj2", "6lu7", "1hzh", "1oxr",
                           "3pte", "6o0k"]
    res = {}
    for i in ids:
        p = fetch(i, here)
        if p is None:
            res[i] = {"error": "download failed"}
            continue
        covale, hits = polymer_ligand_covale(p)
        res[i] = {"covale_links": covale, "polymer_ligand": len(hits),
                  "satisfies_predicate": bool(hits),
                  "examples": sorted({f"{h['a']}[{h['a_kind']}]--{h['b']}[{h['b_kind']}]"
                                      for h in hits})[:4]}
    json.dump(res, sys.stdout, indent=1, sort_keys=True)
    print()
