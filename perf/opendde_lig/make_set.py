"""Build the ligand / glycan / modified-residue accuracy set into one bench.py data dir (needs network, no device).

    python perf/opendde_lig/make_set.py --data ~/odfp-lig-data

The protein-only 11-set (perf/pfm_accuracy/set.tsv) cannot see a change to how ligands, glycans or modified
residues are featurized. These seven deposited structures can: each input is the entry's SEQRES protein chains
(modified residues declared under `modifications:`) plus one bound molecule by CCD code, and the reference keeps
just those chains, so grade.py-style pocket scoring reads it directly. Writes:
  <data>/inputs/<name>.yaml   bench.py inputs (fold with --data <data>)
  <data>/ref/<name>.cif       the deposited model reduced to the folded chains
  <data>/msa/                 ColabFold-API MSAs in tt-bio's cache layout (unpaired per chain, paired per complex)
  <data>/set.tsv              name, ligand subchain, what it exercises
"""
import argparse, urllib.request
from pathlib import Path

import gemmi
import yaml

from tt_bio.cache import seq_hash
from tt_bio.main import _generate_esmfold2_a3m, _generate_paired_a3m

# name: (protein label_asym ids, ligand label_asym id, why)
SET = {
    "3PTB": ("A", "C", "trypsin + benzamidine: a small CCD ligand in a deep pocket"),
    "1STP": ("A", "B", "streptavidin + biotin"),
    "1HSG": ("AB", "C", "HIV-1 protease homodimer + indinavir (MK1): two chains around one ligand"),
    "1A6M": ("A", "D", "myoglobin + heme at 1.0 A"),
    "1ATP": ("AB", "E", "PKA with phospho-Ser/Thr (SEP, TPO) + PKI peptide + ATP: modified residues"),
    "7PJ5": ("A", "B", "LytB + chitotetraose (NAG x4): a multi-residue glycan ligand"),
    "7KMJ": ("A", "B", "Siglec + sialyl Lewis C (NAG-GAL-SIA): a branched glycan ligand"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("~/odfp-lig-data").expanduser())
ap.add_argument("--msa_server_url", default="https://api.colabfold.com")
a = ap.parse_args()
for d in ("inputs", "ref", "msa", "pdb"):
    (a.data / d).mkdir(parents=True, exist_ok=True)

rows = ["name\tligand\twhy"]
for name, (prot, lig, why) in SET.items():
    raw = a.data / "pdb" / f"{name}.cif"
    if not raw.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{name}.cif", raw)
    st = gemmi.read_structure(str(raw))
    st.setup_entities()
    sub = {sc.subchain_id(): sc for sc in st[0].subchains()}
    seqs, chains = {}, []
    for i, sid in enumerate(prot):
        ent = st.get_entity_of(sub[sid])
        mons = [gemmi.Entity.first_mon(m) for m in ent.full_sequence]
        letters, mods = [], []
        for pos, m in enumerate(mons, 1):
            info = gemmi.find_tabulated_residue(m)  # a modified residue carries its parent's letter (SEP -> s)
            if info is None or not info.is_standard():
                mods.append({"position": pos, "ccd": m})
            letters.append(info.one_letter_code.upper() if info and info.one_letter_code.strip() else "X")
        s = "".join(letters)
        seqs[seq_hash(s)] = s
        c = {"id": chr(65 + i), "sequence": s}
        if mods:
            c["modifications"] = mods
        chains.append({"protein": c})
    codes = [r.name for r in sub[lig]]
    chains.append({"ligand": {"id": chr(65 + len(prot)), "ccd": codes[0] if len(codes) == 1 else codes}})
    (a.data / "inputs" / f"{name}.yaml").write_text(
        f"version: 1\n# {name}: {why}\n" + yaml.safe_dump({"sequences": chains}, sort_keys=False))

    keep = set(prot) | {lig}
    ref = gemmi.read_structure(str(raw))
    ref.remove_alternative_conformations()
    ref.remove_hydrogens()
    for ch in ref[0]:
        for k in reversed(range(len(ch))):
            if ch[k].subchain not in keep:
                del ch[k]
    ref.remove_empty_chains()
    ref.setup_entities()
    ref.make_mmcif_document().write_file(str(a.data / "ref" / f"{name}.cif"))

    todo = {h: s for h, s in seqs.items() if not (a.data / "msa" / f"{h}.a3m").exists()}
    if todo:
        _generate_esmfold2_a3m(todo, name, a.data / "msa", None, False, a.msa_server_url, "greedy",
                               None, None, None)
    if len(seqs) > 1:
        _generate_paired_a3m(seqs, name, a.data / "msa", a.msa_server_url, "greedy", None, None, None)
    rows.append(f"{name}\t{lig}\t{why}")
    print(name, [len(s) for s in seqs.values()], codes, flush=True)
(a.data / "set.tsv").write_text("\n".join(rows) + "\n")
