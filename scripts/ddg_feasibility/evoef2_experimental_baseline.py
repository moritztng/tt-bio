import json, os, re, subprocess

EVOEF2 = "/tmp/EvoEF2/EvoEF2"
MANIFEST = json.load(open("/tmp/abbind/yaml/manifest.json"))
SPLIT = {"1vfb": "C,AB", "3hfm": "Y,HL"}
WORKDIR = {"1vfb": "/tmp/abbind/evoef2/1vfb", "3hfm": "/tmp/abbind/evoef2/3hfm"}
REPAIR_PDB = {"1vfb": "1vfb_Repair.pdb", "3hfm": "3hfm_Repair.pdb"}
# AB-Bind mutation-label chain -> actual PDB chain ID (differs for 1vfb only)
CHAIN_MAP = {"1vfb": {"L": "A", "H": "B", "C": "C"}, "3hfm": {"H": "H", "L": "L", "Y": "Y"}}


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.stdout + p.stderr


def compute_binding(pdb, split, cwd):
    out = run([EVOEF2, "--command=ComputeBinding", f"--split_chains={split}", f"--pdb={pdb}"], cwd)
    m = re.search(r"Total\s*=\s*(-?\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"no Total found in output for {pdb}:\n{out[-2000:]}")
    return float(m.group(1))


results = []
wt_energy = {}
for cx in ("1vfb", "3hfm"):
    wt_energy[cx] = compute_binding(REPAIR_PDB[cx], SPLIT[cx], WORKDIR[cx])
    print(f"{cx} WT binding energy = {wt_energy[cx]:.2f}")

for m in MANIFEST:
    if m["mutation"] is None:
        continue
    cx = m["complex"]
    chain = m["chain"]
    mutstr = m["mutation"]  # e.g. D18A
    wt_aa, resnum, mut_aa = mutstr[0], mutstr[1:-1], mutstr[-1]
    pdb_chain = CHAIN_MAP[cx][chain]
    indiv = f"{wt_aa}{pdb_chain}{resnum}{mut_aa};"
    cwd = WORKDIR[cx]
    with open(f"{cwd}/individual_list.txt", "w") as f:
        f.write(indiv + "\n")
    run([EVOEF2, "--command=BuildMutant", f"--pdb={REPAIR_PDB[cx]}",
         "--mutant_file=individual_list.txt"], cwd)
    model_pdb = REPAIR_PDB[cx].replace(".pdb", "_Model_0001.pdb")
    mut_energy = compute_binding(model_pdb, SPLIT[cx], cwd)
    ddg_evoef2 = mut_energy - wt_energy[cx]
    rec = {"complex": cx, "tag": m["tag"], "mutation": mutstr, "chain": chain,
           "ddG_experimental": m["ddG"], "wt_binding_energy": wt_energy[cx],
           "mut_binding_energy": mut_energy, "ddG_evoef2_expstruct": round(ddg_evoef2, 3)}
    results.append(rec)
    print(f"{cx} {m['tag']}: ddG_exp={m['ddG']:.2f}  ddG_evoef2(expstruct)={ddg_evoef2:.2f}")
    os.remove(f"{cwd}/{model_pdb}")

json.dump(results, open("/tmp/abbind/evoef2_expstruct_results.json", "w"), indent=2)
print(f"\nwrote {len(results)} records to /tmp/abbind/evoef2_expstruct_results.json")
