#!/usr/bin/env python3
"""Every Nesso-1 input capability, one fixture each, featurized and predicted once.

The port's capability table was written from ``nesso/data/yaml_input.py`` and then only
ever exercised on one shape: one protein chain, one SMILES ligand. Six rows of that table
had no fixture at all, and one of them (``esm:``) turned out to be silently ignored --
``prepare`` collected the user's path and dropped it, so the 650M encoder recomputed the
embedding instead. A capability with no fixture is a claim, not a feature.

Each row asserts what only that row can show: chain and entity counts for the multi-chain
and ``id: [A, B]`` shapes, the binder landing on the right ligand when there are two, and
for ``esm:`` that a supplied embedding reproduces the inline one bit for bit.

Assets that cannot be committed as text (an SDF with 3D coordinates, an RDKit conformer
pickle, a 1.6 MB ESM-2 tensor) are generated here, deterministically, into the scratch
dir; the YAMLs are written next to them so every path in a YAML resolves.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... <env>/bin/python \
        scripts/nesso1_port/capability_matrix.py --out perf/nesso1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.nesso1_input import (  # noqa: E402
    CLI_PREDICT_ARGS,
    collate,
    find_ccd,
    prepare,
)

FEAT_SEED = 20260820

# CDK2 (PDB 1HCL) apo, first 128 aa: the fleet fixture, short enough that seven rows fit in
# one device context. Verbatim from perf/nesso1/make_inputs.py:CDK2_298.
PROT_A = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
          "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRV")
# A second, different protein for the multi-chain row: the same domain's C-terminal 96 aa.
PROT_B = "IFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
# The upstream README ligand (22 heavy atoms) and the tutorial ligand (tyrosine, 13).
LIG_README = "Fc1ccc(cc1)C(=O)Nc1ccc(cc1)S(=O)(=O)N"
LIG_TUTORIAL = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
CCD_CODE = "ATP"

SCALARS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
    "entropy_crop_pp",
    "entropy_crop_pl",
    "entropy_crop_ll",
)


def protein(pid: str | list[str], seq: str, esm: str | None = None) -> str:
    out = "  - protein:\n      id: %s\n      sequence: %s\n" % (
        pid if isinstance(pid, str) else "[%s]" % ", ".join(pid), seq)
    if esm:
        out += "      esm: %s\n" % esm
    return out


def ligand(lid: str, **kw) -> str:
    out = "  - ligand:\n      id: %s\n" % lid
    for k, v in kw.items():
        out += "      %s: '%s'\n" % (k, v)
    return out


def write_yaml(path: Path, body: str, binder: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sequences:\n%sproperties:\n  - affinity:\n      binder: %s\n" % (body, binder))
    return path


def make_sdf(path: Path, smiles: str) -> Path:
    """An SDF with 3D coordinates, which is the whole point of the sdf: input."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = FEAT_SEED
    AllChem.EmbedMolecule(mol, params)
    mol = Chem.RemoveHs(mol)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return path


def make_conformer_pkl(path: Path, smiles: str) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = FEAT_SEED + 1
    AllChem.EmbedMolecule(mol, params)
    mol = Chem.RemoveHs(mol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(mol))
    return path


def featurize(yaml_path: Path, scratch: Path, ccd: Path, esm_cache: Path | None) -> dict:
    ds, manifest, failed = prepare(
        yaml_path, scratch, ccd_pkl=ccd, num_workers=0, esm_cache=esm_cache
    )
    if failed:
        raise SystemExit("preprocessing failed for %s" % failed)
    torch.manual_seed(FEAT_SEED)  # center_random_augmentation draws off the global RNG
    item = ds[0]
    if item.get("exception"):
        raise SystemExit("featurizer raised on %s" % yaml_path.name)
    return collate(item)


def shape_of(feats: dict) -> dict:
    tok = feats["token_pad_mask"][0].bool()
    atom = feats["atom_pad_mask"][0].bool()
    asym = feats["asym_id"][0][tok]
    ent = feats["entity_id"][0][tok]
    row = {
        "n_tokens": int(tok.numel()),
        "n_real_tokens": int(tok.sum()),
        "n_real_atoms": int(atom.sum()),
        "n_chains": int(torch.unique(asym).numel()),
        "n_entities": int(torch.unique(ent).numel()),
    }
    if "affinity_token_mask" in feats:
        m = feats["affinity_token_mask"][0].bool() & tok
        row["n_binder_tokens"] = int(m.sum())
        row["binder_chains"] = sorted(int(a) for a in torch.unique(asym[m[tok]]))
    return row


def scalars_of(pred: dict) -> dict:
    return {k: float(pred[k].reshape(-1)[0]) for k in SCALARS if k in pred}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path,
                    default=Path("~/scratch/nesso1/capability").expanduser())
    ap.add_argument("--weights", default="recursionpharma/nesso")
    ap.add_argument("--esm-cache", type=Path, default=None)
    ap.add_argument("--trunk", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--out", type=Path, default=REPO / "perf/nesso1")
    args = ap.parse_args()

    ccd = find_ccd()
    yml = args.scratch / "yaml"
    assets = args.scratch / "assets"
    sdf = make_sdf(assets / "readme22.sdf", LIG_README)
    conf = make_conformer_pkl(assets / "readme22_conformer.pkl", LIG_README)

    rows: list[tuple[str, Path, dict]] = [
        ("baseline", write_yaml(yml / "baseline.yaml",
                                protein("A", PROT_A) + ligand("B", smiles=LIG_README), "B"),
         {"n_chains": 2, "n_entities": 2, "binder_chains": [1]}),
        ("multi_ligand", write_yaml(yml / "multi_ligand.yaml",
                                    protein("A", PROT_A) + ligand("B", smiles=LIG_README)
                                    + ligand("C", smiles=LIG_TUTORIAL), "B"),
         {"n_chains": 3, "n_entities": 3, "binder_chains": [1]}),
        ("multi_ligand_binder_c", write_yaml(yml / "multi_ligand_binder_c.yaml",
                                             protein("A", PROT_A) + ligand("B", smiles=LIG_README)
                                             + ligand("C", smiles=LIG_TUTORIAL), "C"),
         {"n_chains": 3, "n_entities": 3, "binder_chains": [2]}),
        ("multi_chain", write_yaml(yml / "multi_chain.yaml",
                                   protein("A", PROT_A) + protein("B", PROT_B)
                                   + ligand("C", smiles=LIG_README), "C"),
         {"n_chains": 3, "n_entities": 3, "binder_chains": [2]}),
        ("id_list", write_yaml(yml / "id_list.yaml",
                               protein(["A", "B"], PROT_A) + ligand("C", smiles=LIG_README), "C"),
         {"n_chains": 3, "n_entities": 2, "binder_chains": [2]}),
        ("ccd", write_yaml(yml / "ccd.yaml",
                           protein("A", PROT_A) + ligand("B", ccd=CCD_CODE), "B"),
         {"n_chains": 2, "n_entities": 2, "binder_chains": [1]}),
        ("sdf", write_yaml(yml / "sdf.yaml",
                           protein("A", PROT_A) + ligand("B", sdf=sdf), "B"),
         {"n_chains": 2, "n_entities": 2, "binder_chains": [1]}),
        ("conformer_pkl", write_yaml(yml / "conformer_pkl.yaml",
                                     protein("A", PROT_A)
                                     + ligand("B", smiles=LIG_README, conformer=conf), "B"),
         {"n_chains": 2, "n_entities": 2, "binder_chains": [1]}),
    ]

    from tt_bio.nesso1 import Nesso1

    model = Nesso1.from_pretrained(
        args.weights, use_tenstorrent=True,
        trunk_fp32=args.trunk == "fp32", affinity_fp32=True,
    )
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)

    report = {
        "gate": "nesso1_capability_matrix",
        "trunk": args.trunk,
        "affinity": "fp32",
        "ccd_pkl": str(ccd),
        "rows": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "capability_matrix.json"

    baseline_esm_src = None
    for name, path, expect in rows:
        t0 = time.perf_counter()
        feats = featurize(path, args.scratch / name, ccd, args.esm_cache)
        feat_s = time.perf_counter() - t0
        shape = shape_of(feats)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(feats)
        row = {"row": name, "yaml": path.name, "featurize_s": feat_s,
               "predict_s": time.perf_counter() - t0, **shape,
               "scalars": scalars_of(pred), "expected": expect}
        bad = {k: (shape.get(k), v) for k, v in expect.items() if shape.get(k) != v}
        row["shape_ok"] = not bad
        row["shape_mismatch"] = bad
        row["finite"] = all(
            torch.isfinite(torch.tensor(v)).item() for v in row["scalars"].values())
        report["rows"].append(row)
        print("  %-22s %4d tok %2d chain %2d ent  binder%s  affinity %.6f  %s"
              % (name, shape["n_real_tokens"], shape["n_chains"], shape["n_entities"],
                 shape.get("binder_chains"), row["scalars"]["affinity_pred_value"],
                 "OK" if row["shape_ok"] and row["finite"] else "MISMATCH %s" % bad),
              flush=True)
        if name == "baseline":
            src = sorted((args.scratch / name / "processed/esm_embeddings").glob("*.safetensors"))
            baseline_esm_src = src[0] if src else None
        out_path.write_text(json.dumps(report, indent=2) + "\n")

    # esm: -- the one row that needs a prior run to have produced an embedding file. It is
    # scored against the conformer_pkl row rather than the baseline, and pins the same
    # conformer, because a SMILES ligand is re-embedded per row and ETKDG draws its seed off
    # the process RNG state: comparing against a re-embedded ligand measures the conformer,
    # not the embedding. With the conformer pinned the supplied embedding is the only
    # difference left, so a used one has to reproduce the computed one exactly.
    esm_row = {"row": "esm_precomputed", "src": str(baseline_esm_src)}
    if baseline_esm_src is None:
        esm_row["error"] = "baseline produced no esm embedding to reuse"
    else:
        path = write_yaml(yml / "esm.yaml",
                          protein("A", PROT_A, esm=str(baseline_esm_src))
                          + ligand("B", smiles=LIG_README, conformer=conf), "B")
        scratch = args.scratch / "esm_precomputed"
        feats = featurize(path, scratch, ccd, args.esm_cache)
        with torch.no_grad():
            pred = model.predict(feats)
        esm_row.update(shape_of(feats))
        esm_row["scalars"] = scalars_of(pred)
        ref_row = next(r for r in report["rows"] if r["row"] == "conformer_pkl")
        base = ref_row["scalars"]
        worst = max(abs(esm_row["scalars"][k] - base[k]) for k in base)
        esm_row["scored_against"] = ref_row["row"]
        esm_row["max_delta_vs_baseline"] = worst
        esm_row["bit_exact_vs_baseline"] = worst == 0.0
        linked = scratch / "processed/esm_embeddings" / baseline_esm_src.name
        esm_row["linked_file_is_symlink"] = linked.is_symlink()
        esm_row["linked_target"] = str(linked.resolve()) if linked.exists() else None
        # a path that does not exist must raise instead of silently recomputing
        bogus = write_yaml(yml / "esm_bogus.yaml",
                           protein("A", PROT_A, esm="/nonexistent/nope.safetensors")
                           + ligand("B", smiles=LIG_README, conformer=conf), "B")
        try:
            featurize(bogus, args.scratch / "esm_bogus", ccd, args.esm_cache)
            esm_row["missing_path_raises"] = False
        except FileNotFoundError:
            esm_row["missing_path_raises"] = True
        print("  %-22s delta vs conformer_pkl %.3e  symlink=%s  missing-raises=%s"
              % ("esm_precomputed", worst, esm_row["linked_file_is_symlink"],
                 esm_row["missing_path_raises"]), flush=True)
    report["rows"].append(esm_row)

    report["all_ok"] = all(
        r.get("shape_ok", True) and r.get("finite", True) and "error" not in r
        for r in report["rows"]
    ) and report["rows"][-1].get("bit_exact_vs_baseline", False)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nwrote %s  all_ok=%s" % (out_path, report["all_ok"]))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
