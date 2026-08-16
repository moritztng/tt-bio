#!/usr/bin/env python3
"""The JapanFold correctness matrix: every cell, its expected outcome, and a resumable runner.

One cell is one submission to the live public API with a stated expectation. A `reject`
cell passes only on a 400 that names the limit or the missing capability; an `ok` cell
passes only if the job succeeds AND every structure it returns survives
`check_structure.py`, including the composition check against the submitted input.

Composition fixtures are deliberately small (<= 80 residues). This axis asks whether a
model accepts a chemistry and folds it sanely, not how it scales; size lives on its own
axis and on its own fixtures.

    matrix.py --list                       # the cell table, with expectations
    matrix.py --run --group composition    # run one group, 3 at a time, resumable
    matrix.py --report                     # pass/fail per cell from the JSONL

Resume is by cell name against the JSONL, so a killed run costs at most the cells that
were in flight.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIX = HERE / "fixtures"

FOLD_MODELS = ["boltz2", "esmfold2", "esmfold2-fast", "protenix-v2", "opendde", "opendde-abag"]

# 64 aa, protein G B1 domain doubled to a length that folds fast and is not a stub.
P64 = ("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTEKPEVIDAS")
# A different 60 aa chain, for the heterodimer cell.
Q60 = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNI")

# The repo's own 1ahw antibody-antigen fixture (examples/1ahw_abag.yaml), comments and
# the version key stripped: the API takes the sequences block.
ABAG_1AHW = 'sequences:\n  - protein:\n      id: A\n      sequence: TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG\n  - protein:\n      id: H\n      sequence: EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGLIDPENGNTIYDPKFQGKASITADTSSNTAYLQLSSLTSEDTAVYYCARDNSYYFDYWGQGTTLTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKI\n  - protein:\n      id: L\n      sequence: DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPWKSPKTLIYYATSLADGVPSRFSGSGSGQDYSLTISSLESDDTATYYCLQHGESPYTFGGGTKLEINRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC\n'



# CDK2 (PDB 1HCL) apo, 298 aa, the fleet's own size fixture. `perf/size512/fixtures/cdk2x2_<N>`
# is this sequence tiled and truncated to N, so building sizes the same way keeps this axis
# comparable with what wh-perf-* measures on the same box.
CDK2_298 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
            "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
            "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
            "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")


def cdk2(n: int) -> str:
    """The fleet fixture's sequence at length n: tile the 298 aa domain, truncate."""
    return (CDK2_298 * (n // len(CDK2_298) + 1))[:n]


def cdk2_rot(n: int, k: int) -> str:
    """`cdk2(n)` with the domain rotated k residues before tiling, so a set of these is a
    set of genuinely different sequences over a real protein alphabet rather than copies."""
    rotated = CDK2_298[k % len(CDK2_298):] + CDK2_298[:k % len(CDK2_298)]
    return (rotated * (n // len(rotated) + 1))[:n]


# Size x model. Sizes are chosen against known boundaries, not round numbers, and every
# model is run at its own advertised cap and one past it -- the brief's "at the limit it
# must work, past it it must fail cleanly".
#
# Depth differs per model on purpose:
#   esmfold2 / esmfold2-fast  a full ladder, because 0.6/0.8 measured esmfold2 running out
#                             of DRAM at both 124 and 620 aa while 64 aa folds. The two ids
#                             are the same target on the same ladder, so the pair is its own
#                             A/B for whether --fast is what saves it.
#   boltz2                    the control: the only model that failed nothing on composition.
#                             640/641 is where three L1 gates go dark (memory
#                             `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`).
#   protenix-v2               coarse. 193-384 aa is a known defect owned by
#                             wh-perf-protenix-v2; 256 is in it, and prod runs the unpatched
#                             tree, so this measures what a user hits rather than re-deriving
#                             someone else's root cause.
#   opendde / opendde-abag    two points plus the cap. Composition already showed zero good
#                             structures from six cells, so depth here buys little.
SIZE_LADDER = {
    "boltz2":        [128, 256, 512, 640, 641, 1024],
    "esmfold2":      [128, 192, 256, 384, 512, 640, 1024],
    "esmfold2-fast": [128, 192, 256, 384, 512, 640, 1024],
    "protenix-v2":   [128, 256, 512, 640, 980],
    # 544/576/608/640 are the cap ladder: the catalog publishes 788/779, both of which 500
    # on this deployment, and the honest number is the last size that folds HERE. 608 is in
    # because wh-perf-opendde §8.8 puts the caught-throw fallback's limit "above 608 tokens"
    # — if the crossing is a token count rather than a residue count, that is where it shows.
    "opendde":       [128, 512, 544, 576, 608, 640, 788],
    "opendde-abag":  [128, 512, 544, 576, 608, 640, 779],
}


def yaml_cell(body: str) -> str:
    return body


# name -> (yaml input, which caps the input needs). The cap names match
# tt_bio/platform/limits.py, which is what decides accept vs reject per model.
COMPOSITIONS = {
    "single":      (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n", set()),
    "homodimer":   (f"sequences:\n  - protein: {{id: [A, B], sequence: {P64}}}\n", {"multichain"}),
    "heterodimer": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    f"  - protein: {{id: B, sequence: {Q60}}}\n", {"multichain"}),
    "ligand_smiles": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                      "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n", {"ligands"}),
    "ligand_ccd":  (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, ccd: ATP}\n", {"ligands"}),
    "dna_duplex":  ("sequences:\n  - dna: {id: A, sequence: ATGCATGCATGCATGCATGC}\n"
                    "  - dna: {id: B, sequence: GCATGCATGCATGCATGCAT}\n", {"nucleic"}),
    "rna":         ("sequences:\n  - rna: {id: A, sequence: GGCUAGCUAGCUAGCUAGCC}\n", {"nucleic"}),
    "protein_dna": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - dna: {id: B, sequence: ATGCATGCATGCATGCATGC}\n"
                    "  - dna: {id: C, sequence: GCATGCATGCATGCATGCAT}\n", {"nucleic"}),
    "affinity":    (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
                    "properties:\n  - affinity: {binder: L}\n", {"affinity"}),
    "constraints": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
                    "constraints:\n  - pocket: {binder: L, contacts: [[A, 10], [A, 14]]}\n",
                    {"constraints"}),
    # 1ahw: tissue-factor antigen + Fab 5G9 heavy/light, the repo's own antibody-antigen
    # fixture. It is the only composition here that is in OpenDDE's distribution, so it is
    # what separates "OpenDDE is broken" from "OpenDDE was handed a target it is not for".
    "abag":        (ABAG_1AHW, {"multichain"}),
    # The gap read out of limits.py: `modifications` is advertised for boltz2 and the two
    # esmfold2 ids only, and `_check_model_caps` has no branch for it, so a modified residue
    # sent to protenix-v2 or either opendde is neither served nor refused by contract.
    "modres":      (f"sequences:\n  - protein: {{id: A, sequence: {P64}, modifications: "
                    "[{position: 12, ccd: SEP}]}\n", {"modifications"}),
    # Position 11 is THR and TPO is phosphothreonine, so this is the modification a user
    # would actually ask for. It exists to remove the one objection to the `modres` cell,
    # whose L->SEP is a chemically odd substitution: if a model drops this one too, the
    # drop is about the model and not about the request.
    "modres_tpo":  (f"sequences:\n  - protein: {{id: A, sequence: {P64}, modifications: "
                    "[{position: 11, ccd: TPO}]}\n", {"modifications"}),
}

# What each model advertises, as of 2026-08-16 on the live catalog. Every expectation in
# the matrix is derived from this, and it is asserted against the live /v1/models on every
# run, so a catalog change breaks the matrix instead of silently re-baselining it.
MODEL_CAPS = {
    "boltz2":        {"msa", "ligands", "nucleic", "affinity", "constraints", "multichain",
                      "modifications", "potentials", "pae"},
    "esmfold2":      {"msa", "multichain", "modifications"},
    "esmfold2-fast": {"multichain", "modifications"},
    "protenix-v2":   {"msa", "ligands", "nucleic", "multichain", "pae"},
    "opendde":       {"msa", "multichain"},
    "opendde-abag":  {"msa", "multichain"},
}
MAX_RESIDUES = {"boltz2": 1024, "esmfold2": 1024, "esmfold2-fast": 1024,
                "protenix-v2": 980, "opendde": 788, "opendde-abag": 779}


def assert_catalog_unchanged() -> None:
    """The matrix's expectations are only as true as the catalog they were read from."""
    r = subprocess.run(["curl", "-s", "-m", "30", "https://api.japanfold.com/v1/models"],
                       capture_output=True, text=True)
    live = json.loads(r.stdout)
    got = {m["id"]: set(m.get("caps", [])) for m in live["models"]}
    if got != MODEL_CAPS:
        raise SystemExit(f"catalog moved: live {got} != recorded {MODEL_CAPS}")
    gotmax = {m["id"]: m["max_residues"] for m in live["models"]}
    if gotmax != MAX_RESIDUES:
        raise SystemExit(f"max_residues moved: live {gotmax} != recorded {MAX_RESIDUES}")

# Malformed and hostile inputs. Each carries what the cell is actually asking.
HOSTILE = {
    "bad_constraint_chain": (
        f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
        "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
        "constraints:\n  - pocket: {binder: L, contacts: [[Z, 10]]}\n",
        "constraint names chain Z, which does not exist"),
    "constraint_oob_index": (
        f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
        "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
        "constraints:\n  - pocket: {binder: L, contacts: [[A, 9999]]}\n",
        "residue index past the end of the chain"),
    "ligand_only": ("sequences:\n  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n",
                    "no polymer at all"),
    "one_residue": ("sequences:\n  - protein: {id: A, sequence: M}\n",
                    "a chain too short to have a backbone"),
    "ambiguous_letters": ("sequences:\n  - protein: {id: A, sequence: "
                          "MBJOUXZMBJOUXZMBJOUXZMBJOUXZMBJOUXZMBJOUXZ}\n",
                          "B/J/O/U/X/Z must map to UNK and keep the length"),
    "lowercase": (f"sequences:\n  - protein: {{id: A, sequence: {P64.lower()}}}\n",
                  "must not be read as masked or gapped"),
    "all_x": ("sequences:\n  - protein: {id: A, sequence: " + "X" * 256 + "}\n",
              "the garbage canary: a constant confidence field must be caught"),
}

# One malformed target among ten valid ones. The brief names it and the other hostile
# cells do not reach it: they are all single-target, so none of them can tell whether a
# batch is validated as a whole or dispatched target by target. Pass = one 400 at submit
# naming the bad target; fail = 202 followed by nine structures and a silent tenth hole.
MIXED_BATCH = [f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"] * 9 + [
    "sequences:\n  - protein: {id: A, sequence: ''}\n"]

# --- variants: the served knobs nothing has ever tested -----------------------
# `fast`, `use_msa_server`, `output_format`, `diffusion_samples`, `recycling_steps` and
# `sampling_steps` all reach the engine through `params`. Every cell here is the 64 aa
# fixture unless stated, so the axis is cheap; what it is looking for is a knob that is
# silently ignored, which returns a successful-looking wrong answer.
VARIANTS = {
    "var_fast_off_boltz2":         ("boltz2", {"fast": False}, "ok"),
    "var_fast_off_esmfold2-fast":  ("esmfold2-fast", {"fast": False}, "unknown"),
    "var_msa_off_boltz2":          ("boltz2", {"use_msa_server": False}, "ok"),
    "var_msa_on_esmfold2-fast":    ("esmfold2-fast", {"use_msa_server": True}, "unknown"),
    "var_msa_off_protenix-v2":     ("protenix-v2", {"use_msa_server": False}, "ok"),
    "var_pdb_boltz2":              ("boltz2", {"output_format": "pdb"}, "ok"),
    "var_pdb_esmfold2-fast":       ("esmfold2-fast", {"output_format": "pdb"}, "ok"),
    # Added after esmfold2-fast's PDB came back with every B-factor 0.00 while boltz2's
    # carried pLDDT: a third model decides whether the dropped confidence is one model's
    # writer or the PDB path in general.
    "var_pdb_protenix-v2":         ("protenix-v2", {"output_format": "pdb"}, "ok"),
    "var_samples5_boltz2":         ("boltz2", {"diffusion_samples": 5}, "ok"),
    "var_samples5_protenix-v2":    ("protenix-v2", {"diffusion_samples": 5}, "ok"),
    "var_recycle10_boltz2":        ("boltz2", {"recycling_steps": 10}, "ok"),
    "var_steps500_boltz2":         ("boltz2", {"sampling_steps": 500}, "ok"),
}

# --- design ------------------------------------------------------------------
# Shapes read off the deployed api_v1.py: BoltzGen takes {protocol, spec}, RFD3 takes
# {protocol, structure, contig}. Every payload below was dry-run against the deployed
# limits.py (validate_payloads.py, 16/16 reach the engine).
LYSOZYME = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCS"
            "ALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")
LAC_O1, LAC_O1_C = "AATTGTGAGCGGATAACAATT", "AATTGTTATCCGCTCACAATT"
HIV_TAR = "GGCAGAUCUGAGCCUGGGAGCUCUCUGGC"

# cell -> (protocol, spec, designed-chain range). The range is the assertion: a binder
# outside its own requested `NN..MM` is a silent wrong answer.
BG_DESIGNS = {
    "des_bg_protein": ("nanobody-anything", f"""entities:
  - protein:
      id: B
      sequence: 110..130
  - protein:
      id: A
      msa: empty
      sequence: {LYSOZYME}
""", ("B", 110, 130)),
    "des_bg_sm": ("protein-small_molecule", """entities:
  - protein:
      id: B
      sequence: 80..120
  - ligand:
      id: A
      smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
""", ("B", 80, 120)),
    "des_bg_dna": ("protein-anything", f"""entities:
  - protein:
      id: B
      sequence: 60..90
  - dna:
      id: A
      sequence: {LAC_O1}
  - dna:
      id: C
      sequence: {LAC_O1_C}
""", ("B", 60, 90)),
    "des_bg_rna": ("protein-anything", f"""entities:
  - protein:
      id: B
      sequence: 60..90
  - rna:
      id: A
      sequence: {HIV_TAR}
""", ("B", 60, 90)),
}

# cell -> (protocol, fixture, contig, total-residue range). RFD3 does not name the
# designed chain the way a BoltzGen spec does, so the assertion is on the whole polymer:
# a contig is an exact statement of how many residues come out. `A1-150,60-80` is 150
# kept plus 60..80 designed = 210..230; `A1-10,20,A31-40` is 10+20+10 = 40 exactly;
# `A1-12,B1-12,60-80` is 24 nucleotides plus 60..80 designed = 84..104.
RFD3_DESIGNS = {
    "des_rfd3_binder":   ("rfd3-binder", "iai_protein.pdb", "A1-150,60-80", (210, 230)),
    "des_rfd3_scaffold": ("rfd3-scaffold", "iai_protein.pdb", "A1-10,20,A31-40", (40, 40)),
    # B13-24, not B1-12: 1BNA numbers its second strand 13-24 continuing from the first,
    # so B1 does not exist. The engine said so exactly ("contig indexes B1 not present in
    # input structure") -- after the job had been dispatched, because limits.check_rfd3_design
    # parses the contig without holding it against the structure that was submitted with it.
    "des_rfd3_na":       ("rfd3-na-binder", "1bna_dna.pdb", "A1-12,B13-24,60-80", (84, 104)),
}

# --- embed -------------------------------------------------------------------
# Three genuinely different real chains, so "no two pooled vectors are equal" is a real
# question and not an artefact of three copies of one sequence.
EMB3 = {"p64": P64, "q60": Q60, "cdk2": CDK2_298}
EMB_CAPS = {"esmc-300m": 2000, "esmc-600m": 2000, "esmc-6b": 1968,
            "saprot-650m": 2000, "saprot-1.3b": 2000}


def cells(group: str) -> list[dict]:
    out = []
    if group in ("composition", "all"):
        for cname, (yml, need) in COMPOSITIONS.items():
            for model in FOLD_MODELS:
                served = need <= MODEL_CAPS[model]
                # `modifications` has no branch in `_check_model_caps`, so a modified residue
                # sent to a model that does not advertise it is neither served by contract nor
                # refused. Measured 2026-08-16: protenix-v2, opendde and opendde-abag all
                # return 202. The contract is undefined, so the cell observes and does not grade.
                expect = "ok" if served else "reject"
                if cname.startswith("modres") and not served:
                    expect = "unknown"
                out.append({"cell": f"comp_{cname}_{model}", "kind": "predict",
                            "expect": expect,
                            "payload": {"model": model, "name": cname, "input": yml},
                            "yaml": yml, "group": "composition"})
    if group in ("size", "all"):
        for model, sizes in SIZE_LADDER.items():
            for n in sizes:
                out.append({"cell": f"size_{n}_{model}", "kind": "predict", "expect": "ok",
                            "payload": {"model": model, "name": f"cdk2_{n}",
                                        "input": f"sequences:\n  - protein: {{id: A, "
                                                 f"sequence: {cdk2(n)}}}\n"},
                            "yaml": f"sequences:\n  - protein: {{id: A, sequence: {cdk2(n)}}}\n",
                            "group": "size"})
            # One past this model's own cap must be a 400 that names the limit.
            over = MAX_RESIDUES[model] + 1
            yml = f"sequences:\n  - protein: {{id: A, sequence: {cdk2(over)}}}\n"
            out.append({"cell": f"size_over_{over}_{model}", "kind": "predict", "expect": "reject",
                        "payload": {"model": model, "name": f"cdk2_{over}", "input": yml},
                        "yaml": yml, "group": "size"})
    if group in ("hostile", "all"):
        for hname, (yml, why) in HOSTILE.items():
            out.append({"cell": f"hostile_{hname}", "kind": "predict", "expect": "unknown",
                        "payload": {"model": "boltz2", "name": hname, "input": yml},
                        "yaml": yml, "why": why, "group": "hostile"})
        out.append({"cell": "hostile_mixed_batch", "kind": "predict", "expect": "reject",
                    "payload": {"model": "boltz2", "name": "mixed_batch",
                                "targets": MIXED_BATCH},
                    "yaml": MIXED_BATCH[-1], "group": "hostile",
                    "why": "one empty sequence among ten targets must refuse the whole "
                           "submission, not dispatch nine and drop one"})
    if group in ("variants", "all"):
        yml = f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
        for cell, (model, params, expect) in VARIANTS.items():
            out.append({"cell": cell, "kind": "predict", "expect": expect,
                        "payload": {"model": model, "name": cell, "input": yml,
                                    "params": params},
                        "yaml": yml, "group": "variants"})
    if group in ("design", "all"):
        for cell, (protocol, spec, rng) in BG_DESIGNS.items():
            out.append({"cell": cell, "kind": "design", "expect": "ok",
                        "payload": {"model": "boltzgen", "protocol": protocol, "name": cell,
                                    "spec": spec,
                                    "params": {"num_designs": 10, "budget": 10}},
                        "yaml": spec, "group": "design",
                        "design_chain": rng[0], "design_min": rng[1], "design_max": rng[2]})
        for cell, (protocol, fixture, contig, rng) in RFD3_DESIGNS.items():
            struct = (FIX / fixture).read_text()
            out.append({"cell": cell, "kind": "design", "expect": "ok",
                        "payload": {"model": "rfd3", "protocol": protocol, "name": cell,
                                    "structure": struct, "contig": contig,
                                    "params": {"num_designs": 5, "num_timesteps": 200}},
                        "yaml": contig, "group": "design",
                        "design_min": rng[0], "design_max": rng[1]})
    if group in ("embed", "all"):
        for model, n in EMB_CAPS.items():
            # 50 DIFFERENT sequences, not 50 copies of one. The first run of this cell
            # tiled the same 2000-mer 50 times and got 50 bit-identical vectors back,
            # which proves the artifacts are keyed and returned but not that the service
            # can compute 50 distinct ones -- a dedup would serve it from a single
            # forward pass and the cell could not tell. Rotating the tiling start by i
            # gives 50 genuine 2000-residue sequences over the same real alphabet.
            seqs = {f"s{i}": cdk2_rot(n, i) for i in range(50)}
            out.append({"cell": f"emb_cap_{model}", "kind": "embed", "expect": "ok",
                        "payload": {"model": model, "name": f"cap_{model}",
                                    "sequences": [{"id": k, "sequence": v}
                                                  for k, v in seqs.items()]},
                        "json": seqs, "group": "embed"})
        over = {"s0": cdk2(EMB_CAPS["esmc-6b"] + 1)}
        out.append({"cell": "emb_over_esmc-6b", "kind": "embed", "expect": "reject",
                    "payload": {"model": "esmc-6b", "name": "over",
                                "sequences": [{"id": "s0", "sequence": over["s0"]}]},
                    "json": over, "group": "embed"})
        three = [{"id": k, "sequence": v} for k, v in EMB3.items()]
        # `distinct` and `determinism` submit the SAME payload twice on purpose: the
        # first asks whether three different sequences give three different vectors,
        # the second whether the served path is deterministic. analyze_embed.py holds
        # the two artifact sets against each other.
        for cell in ("emb_distinct_esmc-600m", "emb_determinism_esmc-600m"):
            out.append({"cell": cell, "kind": "embed", "expect": "ok",
                        "payload": {"model": "esmc-600m", "name": cell, "sequences": three},
                        "json": EMB3, "group": "embed"})
        out.append({"cell": "emb_pool_cls_esmc-600m", "kind": "embed", "expect": "ok",
                    "payload": {"model": "esmc-600m", "name": "pool_cls",
                                "sequences": three, "params": {"pool": "cls"}},
                    "json": EMB3, "pool": "cls", "group": "embed"})
        out.append({"cell": "emb_parquet_esmc-600m", "kind": "embed", "expect": "ok",
                    "payload": {"model": "esmc-600m", "name": "parquet",
                                "sequences": three, "params": {"format": "parquet"}},
                    "json": EMB3, "group": "embed"})
    return out


def run_cell(c: dict, outjs: Path, deadline: int) -> int:
    d = RESULTS / "payloads"
    d.mkdir(parents=True, exist_ok=True)
    pf = d / f"{c['cell']}.json"
    pf.write_text(json.dumps(c["payload"]))
    # An embed cell's "input" is the {id: sequence} map the checker holds the returned
    # vector against; everything else submits a YAML target and checks composition.
    if "json" in c:
        inf = d / f"{c['cell']}.input.json"
        inf.write_text(json.dumps(c["json"]))
    else:
        inf = d / f"{c['cell']}.input.yaml"
        inf.write_text(c["yaml"])
    # "unknown" cells are recorded, not graded: the point is to observe what the service
    # does with an input nobody has decided the contract for yet.
    expect = "ok" if c["expect"] == "unknown" else c["expect"]
    cmd = [sys.executable, str(HERE / "jf_cell.py"), "--cell", c["cell"], "--kind", c["kind"],
           "--expect", expect, "--payload", str(pf), "--input", str(inf),
           "--out", str(outjs), "--artifacts", str(RESULTS / "artifacts"),
           "--deadline", str(deadline)]
    if "pool" in c:
        cmd += ["--pool", c["pool"]]
    for k, flag in (("design_chain", "--design-chain"), ("design_min", "--design-min"),
                    ("design_max", "--design-max")):
        if k in c:
            cmd += [flag, str(c[k])]
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--markdown", action="store_true",
                    help="the composition x model table, generated from the JSONL so the "
                         "state doc is never hand-transcribed")
    ap.add_argument("--group", default="all")
    ap.add_argument("--only", help="comma-separated cell-name substrings")
    ap.add_argument("--expect-only", choices=("ok", "reject", "unknown"),
                    help="run one expectation class; reject cells are instant, so they buy "
                         "most of the matrix before a single fold is spent")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="the per-IP cap is 8 active jobs; stay well under it")
    ap.add_argument("--deadline", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=RESULTS / "matrix.jsonl")
    a = ap.parse_args()

    todo = cells(a.group)
    if a.only:
        keys = [k.strip() for k in a.only.split(",")]
        todo = [c for c in todo if any(k in c["cell"] for k in keys)]
    if a.expect_only:
        todo = [c for c in todo if c["expect"] == a.expect_only]

    if a.list:
        for c in todo:
            print(f"{c['cell']:44s} {c['expect']}")
        print(f"{len(todo)} cells")
        return 0

    if a.markdown:
        rows = {}
        if a.out.exists():
            for line in a.out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["cell"]] = r
        if a.group == "size":
            allsizes = sorted({s for v in SIZE_LADDER.values() for s in v})
            print("| model | " + " | ".join(str(s) for s in allsizes) + " | cap+1 |")
            print("|---" * (len(allsizes) + 2) + "|")
            notes = []
            for model, sizes in SIZE_LADDER.items():
                out_cells = []
                for s in allsizes:
                    if s not in sizes:
                        out_cells.append("")
                        continue
                    r = rows.get(f"size_{s}_{model}")
                    if r is None:
                        out_cells.append("-")
                    elif r.get("pass"):
                        out_cells.append("fold")
                    else:
                        out_cells.append("**FAIL**")
                        notes.append(f"- `size_{s}_{model}`: {r.get('status') or 'submit ' + str(r.get('submit_status'))}"
                                     f" -- {r.get('why', '')[:200]}")
                over = rows.get(f"size_over_{MAX_RESIDUES[model] + 1}_{model}")
                out_cells.append("-" if over is None else ("400" if over.get("pass") else "**FAIL**"))
                print(f"| {model} | " + " | ".join(out_cells) + " |")
            if notes:
                print("\nFailures:\n")
                print("\n".join(notes))
            return 0

        # A cell is one of: PASS, FAIL (with the reason), 400 (refused as the catalog says),
        # or "-" for not yet run. The reason column is the point, so it goes underneath.
        print("| input | " + " | ".join(FOLD_MODELS) + " |")
        print("|---" * (len(FOLD_MODELS) + 1) + "|")
        notes = []
        for cname in COMPOSITIONS:
            cells_out = []
            for model in FOLD_MODELS:
                r = rows.get(f"comp_{cname}_{model}")
                if r is None:
                    cells_out.append("-")
                elif r.get("submit_status") == 400:
                    cells_out.append("400" if r.get("pass") else "**400 unexpected**")
                elif r.get("pass"):
                    cells_out.append("fold")
                else:
                    cells_out.append("**FAIL**")
                    what = r.get("status") or f"submit {r.get('submit_status')}"
                    notes.append(f"- `comp_{cname}_{model}`: {what} -- {r.get('why', '')[:220]}")
            print(f"| {cname} | " + " | ".join(cells_out) + " |")
        if notes:
            print("\nFailures:\n")
            print("\n".join(notes))
        return 0

    if a.report:
        rows = {}
        if a.out.exists():
            for line in a.out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["cell"]] = r
        for c in todo:
            r = rows.get(c["cell"])
            v = "-" if r is None else ("PASS" if r.get("pass") else "FAIL")
            print(f"{c['cell']:44s} {c['expect']:8s} {v:5s} {r.get('why','') if r else ''}"[:160])
        print(f"{sum(1 for c in todo if c['cell'] in rows)}/{len(todo)} run")
        return 0

    assert_catalog_unchanged()
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cell"])
    todo = [c for c in todo if c["cell"] not in done]
    print(f"{len(todo)} cells to run ({len(done)} already recorded)", flush=True)
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(lambda c: run_cell(c, a.out, a.deadline), todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
