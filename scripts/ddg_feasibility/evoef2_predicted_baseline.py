"""Run EvoEF2 ComputeBinding directly on tt-bio opendde-abag PREDICTED structures
(WT-folded and mutant-folded independently -- no BuildMutant, the mutation is already
baked into the folded sequence). RepairStructure first to normalize side chains.
Also pulls each fold's own ipTM (the raw-confidence baseline) from the shared results.json.
Requires the directory-mode fold campaign to have produced
/tmp/abbind/folds_v2/opendde_results_yaml/{structures/,results.json}."""
import json, os, re, subprocess
import gemmi

EVOEF2 = "/tmp/EvoEF2/EvoEF2"
MANIFEST = json.load(open("/tmp/abbind/yaml/manifest.json"))
STRUCT_DIR = "/tmp/abbind/folds_v2/opendde_results_yaml/structures"
RESULTS_JSON = "/tmp/abbind/folds_v2/opendde_results_yaml/results.json"
SPLIT = {"1vfb": "C,AB", "3hfm": "C,AB"}  # opendde-abag output ALWAYS renumbers chains sequentially by
# YAML listing order (tt_bio/main.py _chain_label, asym_id-based) -- NOT the custom "id:" strings in the
# input YAML. Both 1vfb and 3hfm YAMLs list the antigen 3rd, so it always lands on output chain "C";
# the two antibody chains (whatever their custom id was) always land on "A"+"B". Verified against a real
# opendde-abag CIF from a prior campaign before writing this.
WORKDIR = "/tmp/abbind/evoef2_pred"
os.makedirs(WORKDIR, exist_ok=True)


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.stdout + p.stderr


def cif_to_pdb(cif_path, pdb_path):
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.write_pdb(pdb_path)


def compute_binding(pdb, split, cwd):
    out = run([EVOEF2, "--command=ComputeBinding", f"--split_chains={split}", f"--pdb={pdb}"], cwd)
    m = re.search(r"Total\s*=\s*(-?\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"no Total found for {pdb} in {cwd}:\n{out[-1500:]}")
    return float(m.group(1))


def binding_energy_for_tag(tag, cx):
    cif = f"{STRUCT_DIR}/{tag}.cif"
    if not os.path.exists(cif):
        return None
    pdb_path = f"{WORKDIR}/{tag}.pdb"
    cif_to_pdb(cif, pdb_path)
    run([EVOEF2, "--command=RepairStructure", f"--pdb={tag}.pdb"], WORKDIR)
    repaired = f"{tag}_Repair.pdb"
    if not os.path.exists(f"{WORKDIR}/{repaired}"):
        return None
    return compute_binding(repaired, SPLIT[cx], WORKDIR)


def load_iptm_by_id():
    results = json.load(open(RESULTS_JSON))
    rows = results if isinstance(results, list) else results.get("rows", results)
    by_id = {}
    for r in rows:
        rid = r.get("id") or r.get("name")
        if rid:
            by_id[rid] = r
    return by_id


def main():
    iptm_by_id = load_iptm_by_id()
    wt_energy = {}
    wt_iptm = {}
    for m in MANIFEST:
        if m["mutation"] is None:
            cx, tag = m["complex"], m["tag"]
            full_tag = f"{cx}_{tag}"
            wt_energy[cx] = binding_energy_for_tag(full_tag, cx)
            row = iptm_by_id.get(full_tag)
            wt_iptm[cx] = row.get("iptm") if row else None
            print(f"{cx} WT: binding_energy={wt_energy[cx]} iptm={wt_iptm[cx]}")

    rows_out = []
    for m in MANIFEST:
        if m["mutation"] is None:
            continue
        cx, tag = m["complex"], m["tag"]
        full_tag = f"{cx}_{tag}"
        mut_energy = binding_energy_for_tag(full_tag, cx)
        row = iptm_by_id.get(full_tag)
        mut_iptm = row.get("iptm") if row else None
        rec = {"complex": cx, "tag": tag, "mutation": m["mutation"], "chain": m["chain"],
               "ddG_experimental": m["ddG"]}
        if mut_energy is not None and wt_energy.get(cx) is not None:
            rec["ddG_evoef2_predstruct"] = round(mut_energy - wt_energy[cx], 3)
        if mut_iptm is not None and wt_iptm.get(cx) is not None:
            rec["delta_iptm"] = round(wt_iptm[cx] - mut_iptm, 5)
        rec["mut_binding_energy"] = mut_energy
        rec["mut_iptm"] = mut_iptm
        rows_out.append(rec)
        print(f"{cx} {tag}: ddG_exp={m['ddG']:.2f} ddG_evoef2(pred)={rec.get('ddG_evoef2_predstruct')} "
              f"delta_iptm={rec.get('delta_iptm')}")

    json.dump(rows_out, open("/tmp/abbind/evoef2_predstruct_results.json", "w"), indent=2)
    print(f"\nwrote {len(rows_out)} records")


if __name__ == "__main__":
    main()
