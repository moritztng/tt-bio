"""Pick the MOL1 quality case list from FoldBench + upstream examples + the fleet's pfm 11-set. Deterministic.

    python3 select.py FOLDBENCH_TARGETS_DIR > state/mol1/quality/CASES.json
Order inside a size bin is sha1(pdb id), so the pick is reproducible and blind to any result.
"""
import csv, hashlib, json, sys
from pathlib import Path

T = Path(sys.argv[1]); HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "sizes.json")); F = json.load(open(HERE / "feats.json"))
SEEDS = [42, 66, 101, 2024, 8888]  # FoldBench's five seeds
CAP = 1024  # TT-paired sets stay at sizes users run today; the GPU-only headline set has no cap
BINS = [(0, 384), (384, 640), (640, CAP + 1)]
h = lambda p: hashlib.sha1(p.encode()).hexdigest()
tok = lambda p: S[p]["polymer_monomer_count"]

def rows(task):
    return list(csv.DictReader(open(T / f"{task}.csv")))

def ids(task):
    return sorted({r["pdb_id"].split("-")[0] for r in rows(task)} & set(S))

def pick(task, n, taken):
    pool = [p for p in ids(task) if tok(p) <= CAP and p not in taken]
    bins = [sorted((p for p in pool if lo <= tok(p) < hi), key=h) for lo, hi in BINS]
    out = []
    while len(out) < n and any(bins):
        for b in bins:
            if b and len(out) < n:
                out.append(b.pop(0))
    return out

def feats(p):
    f = F.get(p, {})
    t = []
    if f.get("glycan"): t.append("ligand_multi_ccd")
    if f.get("glycan"): t.append("covalent_bond")
    if f.get("modres"): t.append("modified_residue")
    if f.get("ions"): t.append("ion")
    if f.get("ligs"): t.append("ligand_ccd")
    if any("deoxyribo" in x for x in f.get("poly", [])): t.append("dna")
    if any(x.startswith("polyribo") for x in f.get("poly", [])): t.append("rna")
    return t

PLAN = [  # (set, FoldBench task, n TT-paired assemblies, metric)
    ("ab", "interface_antibody_antigen", 32, "dockq"),
    ("pl", "interface_protein_ligand", 32, "lrmsd_lddt_pli"),
    ("pp", "interface_protein_protein", 16, "dockq"),
    ("pdna", "interface_protein_dna", 12, "dockq"),
    ("prna", "interface_protein_rna", 12, "dockq"),
    ("pep", "interface_protein_peptide", 8, "dockq"),
    ("mono", "monomer_protein", 12, "lddt"),
    ("rnamono", "monomer_rna", 4, "lddt"),
]
cases, taken = [], set()
for name, task, n, metric in PLAN:
    for p in pick(task, n, taken):
        taken.add(p)
        cases.append(dict(id=f"{p}-assembly1", set=name, source=f"FoldBench/{task}", truth=f"rcsb:{p}-assembly1",
                          tokens_est=tok(p), metric=metric, features=(["msa"] if name == "rnamono" else ["protein", "msa"]) + feats(p),
                          recipe="default"))

# Targeted feature cover: at least 4 TT-paired cases per structure-changing class FoldBench can supply.
for feat in ("ligand_multi_ccd", "modified_residue", "ion"):
    have = sum(feat in c["features"] for c in cases)
    pool = sorted((p for p in ids("interface_protein_ligand") + ids("interface_protein_protein")
                   if p not in taken and tok(p) <= CAP and feat in feats(p)), key=h)
    for p in pool[:max(0, 4 - have)]:
        taken.add(p)
        task = "interface_protein_ligand" if p in ids("interface_protein_ligand") else "interface_protein_protein"
        cases.append(dict(id=f"{p}-assembly1", set="feat", source=f"FoldBench/{task}", truth=f"rcsb:{p}-assembly1",
                          tokens_est=tok(p), metric="lrmsd_lddt_pli" if "ligand" in task else "dockq",
                          features=["protein", "msa"] + feats(p), recipe="default"))

# Recipe variants on cases already in the list (same truth, different input path).
pl = [c for c in cases if c["set"] == "pl"]
ab = [c for c in cases if c["set"] == "ab"]
for c in pl[:8]:
    cases.append(dict(c, id=c["id"] + ":smiles", set="feat", features=["protein", "msa", "ligand_smiles"],
                      recipe="ligand_as_smiles", base=c["id"]))
for c in pl[8:12]:
    cases.append(dict(c, id=c["id"] + ":file", set="feat", features=["protein", "msa", "ligand_file"],
                      recipe="ligand_as_sdf_file", base=c["id"]))
for c in ab[:4] + pl[12:16]:
    cases.append(dict(c, id=c["id"] + ":nomsa", set="feat", features=["protein", "msa_free"] + c["features"][2:],
                      recipe="use_msa_false", base=c["id"]))

# Upstream's own feature examples (precomputed MSA/template files ship with them).
UP = [("examples_with_template/example_9fm7.json", ["template"], "rcsb:9fm7-assembly1"),
      ("examples_with_template/example_mgyp004658859411.json", ["template"], None),
      ("example_with_json_template/demo_ab.json", ["template_json_inline"], None),
      ("examples_with_rna_msa/example_9gmw_2.json", ["rna", "rna_msa"], "rcsb:9gmw-assembly1"),
      ("example.json#7r6r", ["dna", "msa"], "rcsb:7r6r-assembly1"),
      ("example.json#7pzb", ["dna", "ligand_ccd", "msa"], "rcsb:7pzb-assembly1"),
      ("example.json#7wux", ["ligand_ccd", "msa"], "rcsb:7wux-assembly1"),
      ("example_without_msa.json#8cnh-assembly1", ["ligand_ccd", "msa_free"], "rcsb:8cnh-assembly1")]
for path, f, truth in UP:
    cases.append(dict(id="upstream:" + path, set="upstream", source="Mol-1@e6e64866 examples/" + path, truth=truth,
                      metric="pose_vs_gpu" if truth is None else "auto", features=["protein"] + f,
                      recipe="use_template_true" if "template" in f[0] else
                             "use_rna_msa_true" if "rna_msa" in f else
                             "use_msa_false" if "msa_free" in f else "default"))

# The fleet's own Protenix/OF3 accuracy set (perf/pfm_accuracy/set.tsv), so Mol-1 reads beside the SPD numbers.
for pdb in "9TH6 9W89 9W8A 9TY2 9W3L 9LV4 9LLG 28VJ 9HL2 9DBP 9PCQ".split():
    cases.append(dict(id=f"pfm:{pdb}", set="pfm11", source="tt-bio perf/pfm_accuracy/set.tsv", truth=f"rcsb:{pdb.lower()}",
                      metric="dockq", features=["protein", "msa"], recipe="default"))

headline = [dict(id=f"{p}-assembly1", tokens_est=tok(p)) for p in ids("interface_antibody_antigen")]
json.dump(dict(
    version=1, written="2026-10-31", by="mol1-quality",
    seeds=SEEDS, seeds_self_test=[7, 13, 77, 1234, 9999],
    seeds_note="FoldBench's five-seed budget (algorithms/Protenix/postprocess.py); self-test = upstream vs upstream on other seeds",
    recipe_default=dict(model="mol-1-base", N_cycle=20, N_step=200, N_sample=5, use_msa=True, use_template=False,
                        dtype="bf16 (upstream default)"),
    headline=dict(what="FoldBench-AB, all 172 interfaces / %d assemblies, GPU only: reproduce Molence's 75.0 %%"
                       " (top-ranked DockQ>=0.23, five-seed budget)" % len(headline), seeds=SEEDS,
                  cases=headline),
    paired=cases), sys.stdout, indent=1)
