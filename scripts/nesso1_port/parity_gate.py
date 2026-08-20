#!/usr/bin/env python3
"""Nesso-1 host-pipeline parity gate: card-free, upstream-free, CPU only.

Scores the vendored host pipeline (``tt_bio._vendor.nesso.data``) against a
COMMITTED upstream capture. Two legs:

  featurizer  every tensor ``NessoFeaturizer.process`` produces, compared
              bit-exact against ``ref_feats.pt``. Runs from the committed
              ``processed/`` directory, so no upstream install, no RDKit
              conformer generation, no device.

  conformer   re-derives the ligand conformer from the fixture YAML with the
              installed RDKit and compares it to the committed one. This leg is
              EXPECTED to be version-sensitive and it is reported separately:
              RDKit 2025.09.6 and 2026.03.5 give the same 13 atoms in the same
              order but coordinates ~1.5 A apart for the tutorial ligand. A
              mismatch here does not mean the port is wrong, it means an
              end-to-end comparison against upstream numbers would be comparing
              two different inputs and must feed the committed conformer instead.

Both legs must be reproducible without the ``nesso`` package installed. Refresh
the capture with ``scripts/nesso1_port/capture_ref.py`` in a venv that has it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "scripts" / "nesso1_port" / "parity_artifacts"
FEAT_SEED = 20260820


def featurizer_parity(fixture: Path) -> dict:
    import torch

    sys.path.insert(0, str(REPO))
    from tt_bio._vendor.nesso.data.featurizer import NessoFeaturizer
    from tt_bio._vendor.nesso.data.inference import InferenceDataset
    from tt_bio._vendor.nesso.data.types import Manifest

    processed = fixture / "processed"
    ref_path = fixture / "ref_feats.pt"
    for p in (processed, ref_path, fixture / "standard_aa_mols.pkl"):
        if not p.exists():
            return {"leg": "featurizer", "verdict": "ERROR",
                    "error": f"missing committed artifact {p}"}

    meta = json.loads((fixture / "meta.json").read_text())
    ds = InferenceDataset(
        manifest=Manifest.load(processed / "manifest.json"),
        target_dir=processed,
        ligand_dir=processed / "rdkit_conformers",
        ccd_pkl=fixture / "standard_aa_mols.pkl",
        use_esm_all_layers=False,
        num_dist_bins=64,
        min_dist=2.0,
        max_dist=meta.get("max_dist", 22.0),
        atoms_per_window_queries=32,
        featurizer=NessoFeaturizer(
            esm_emb_dir=processed / "esm_embeddings",
            esm_emb_dim=1280,
            esm_num_layers=33,
        ),
    )
    # share the augmentation draws with the capture; see capture_ref.py
    torch.manual_seed(int(meta.get("feat_seed", FEAT_SEED)))
    got = ds[0]
    if got.get("exception"):
        return {"leg": "featurizer", "verdict": "ERROR",
                "error": "vendored featurizer raised on the committed fixture"}

    ref = torch.load(ref_path, weights_only=True)
    mismatches, checked = [], []
    for key in sorted(ref):
        a, b = ref[key], got.get(key)
        if b is None:
            mismatches.append({"key": key, "why": "absent from vendored output"})
            continue
        checked.append(key)
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatches.append({"key": key, "why": "shape/dtype",
                               "ref": [list(a.shape), str(a.dtype)],
                               "got": [list(b.shape), str(b.dtype)]})
        elif not torch.equal(a, b):
            d = (a.float() - b.float()).abs()
            mismatches.append({"key": key, "why": "value",
                               "max_abs": float(d.max()),
                               "n_diff": int((d > 0).sum())})
    extra = sorted(set(k for k, v in got.items() if isinstance(v, torch.Tensor)) - set(ref))
    return {
        "leg": "featurizer",
        "verdict": "PASS" if not mismatches else "FAIL",
        "keys_total": len(ref),
        "keys_bitexact": len(checked) - len(mismatches),
        "extra_keys_in_port": extra,
        "mismatches": mismatches,
        "n_tokens": int(ref["token_pad_mask"].shape[-1]),
    }


def conformer_parity(fixture: Path) -> dict:
    """Non-blocking: does the installed RDKit reproduce the committed conformer?"""
    import pickle

    import numpy as np
    import rdkit
    import yaml

    sys.path.insert(0, str(REPO))
    from tt_bio._vendor.nesso.data.yaml_input import _parse_entity, _chain_data_for_entity

    schema = yaml.safe_load((next(fixture.glob("*.yaml"))).read_text())
    ligs = [item["ligand"] for item in schema["sequences"] if "ligand" in item]
    committed = sorted((fixture / "processed" / "rdkit_conformers").glob("*.pkl"))
    if not ligs or not committed:
        return {"leg": "conformer", "verdict": "SKIP",
                "why": "fixture has no ligand or no committed conformer"}

    out = {"leg": "conformer", "rdkit": rdkit.__version__, "ligands": []}
    worst = 0.0
    for block, pkl in zip(ligs, committed):
        data = _chain_data_for_entity(_parse_entity("ligand", block), None, [1])
        fresh = np.array([a["coord"] for a in data.residues[0]["atoms"]])
        with pkl.open("rb") as fh:
            mol = pickle.load(fh)
        conf = mol.GetConformer(0)
        ref = np.array([list(conf.GetAtomPosition(i))
                        for i in range(mol.GetNumAtoms())])
        if fresh.shape != ref.shape:
            out["ligands"].append({"pkl": pkl.name, "why": "atom count",
                                   "fresh": list(fresh.shape), "ref": list(ref.shape)})
            worst = float("inf")
            continue
        d = float(np.abs(fresh - ref).max())
        worst = max(worst, d)
        out["ligands"].append({"pkl": pkl.name, "n_atoms": int(ref.shape[0]),
                               "max_abs_coord_delta": d})
    out["max_abs_coord_delta"] = worst
    # 1e-5 A separates float32 round-trip noise (the committed pickle stores doubles,
    # the featurizer reads float32) from real version drift, measured at 1.85 A
    # between RDKit 2025.09.6 and 2026.03.5 on this ligand.
    ok = worst < 1e-5
    out["verdict"] = "MATCH" if ok else "VERSION-DRIFT"
    out["note"] = ("committed conformer reproduced to float32 precision" if ok else
                   "installed RDKit generates a different conformer; feed the "
                   "committed pickle (or an sdf:/conformer: input) before comparing "
                   "any end-to-end number against upstream")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=ARTIFACTS / "tyr48")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report = {
        "gate": "nesso1_host_pipeline",
        "fixture": args.fixture.name,
        "featurizer": featurizer_parity(args.fixture),
        "conformer": conformer_parity(args.fixture),
    }
    report["verdict"] = report["featurizer"]["verdict"]
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
