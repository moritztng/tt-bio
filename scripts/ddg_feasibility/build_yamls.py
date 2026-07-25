import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_index, index_to_one

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

def chain_seq(struct, chain_id):
    chain = struct[0][chain_id]
    residues = []
    for res in chain:
        if res.id[0] != ' ':
            continue  # skip hetero/water
        resname = res.resname
        if resname not in AA3TO1:
            continue
        resnum = res.id[1]  # author numbering, ignoring icode for AB-Bind (single-digit muts)
        residues.append((resnum, AA3TO1[resname]))
    return residues

def seq_str(residues):
    return ''.join(a for _, a in residues)

def apply_mutation(residues, resnum, wt_aa, mut_aa):
    out = []
    found = False
    for num, aa in residues:
        if num == resnum:
            assert aa == wt_aa, f"expected {wt_aa} at {resnum}, found {aa}"
            out.append((num, mut_aa))
            found = True
        else:
            out.append((num, aa))
    assert found, f"resnum {resnum} not found"
    return out

COMPLEXES = {
    "1vfb": {
        "pdb": "/tmp/abbind/pdb/1vfb.pdb",
        "chain_map": {"L": "A", "H": "B", "C": "C"},  # AB-Bind label -> PDB chain
        "roles": {"A": "L", "B": "H", "C": "antigen"},
        "muts": ["H:G31E:-0.51", "C:D18A:0.3", "C:S24A:0.8", "H:D54A:1.0",
                 "L:Y32A:1.7", "C:Q121A:2.9", "L:W92A:3.3", "L:L46D:8.0"],
    },
    "3hfm": {
        "pdb": "/tmp/abbind/pdb/3hfm.pdb",
        "chain_map": {"L": "L", "H": "H", "Y": "Y"},
        "roles": {"H": "H", "L": "L", "Y": "antigen"},
        "muts": ["H:D32N:0.17", "L:Q53A:0.95", "L:N31D:1.34", "L:Y50F:2.36",
                 "H:C95A:5.52", "Y:K97A:6.18", "H:Y50A:8.0"],
    },
}

OUT_DIR = "/tmp/abbind/yaml"
os.makedirs(OUT_DIR, exist_ok=True)
parser = PDBParser(QUIET=True)
manifest = []

for cx_name, cx in COMPLEXES.items():
    struct = parser.get_structure(cx_name, cx["pdb"])
    base_seqs = {ab_label: chain_seq(struct, pdb_chain) for ab_label, pdb_chain in cx["chain_map"].items()}

    def write_yaml(tag, seqs):
        path = f"{OUT_DIR}/{cx_name}_{tag}.yaml"
        with open(path, "w") as f:
            f.write("version: 1\n")
            f.write(f"# AB-Bind {cx_name.upper()} variant={tag}\n")
            f.write("sequences:\n")
            for ab_label, residues in seqs.items():
                role = cx["roles"][cx["chain_map"][ab_label]]
                cid = ab_label
                f.write(f"  - protein:\n      id: {cid}\n      sequence: {seq_str(residues)}\n")
        return path

    wt_path = write_yaml("wt", base_seqs)
    manifest.append({"complex": cx_name, "tag": "wt", "mutation": None, "ddG": 0.0, "yaml": wt_path})

    for m in cx["muts"]:
        chain_label, mutstr, ddg = m.split(":")
        wt_aa, resnum, mut_aa = mutstr[0], int(mutstr[1:-1]), mutstr[-1]
        seqs = dict(base_seqs)
        seqs[chain_label] = apply_mutation(base_seqs[chain_label], resnum, wt_aa, mut_aa)
        tag = f"{chain_label}{wt_aa}{resnum}{mut_aa}"
        path = write_yaml(tag, seqs)
        manifest.append({"complex": cx_name, "tag": tag, "mutation": mutstr, "chain": chain_label,
                          "ddG": float(ddg), "yaml": path})
        print(f"{cx_name} {tag} ddG={ddg} -> {path}")

import json
json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
print(f"total variants: {len(manifest)}")
