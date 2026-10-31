"""Capture Mol-1 upstream input features as goldens for tests/test_mol1_features.py.

Runs upstream's own featurizer, `InferenceDataset.process_one`, on every job in the given JSON files and pickles
the `input_feature_dict` (numpy) plus the atom table. Run it in the upstream venv, isolated, from outside the
upstream clone, with MOL_ROOT_DIR pointing at the release `common/` data:

    MOL_ROOT_DIR=/home/moritz/mol1-upstream/io-data \
      /home/moritz/mol1-upstream/venv/bin/python -I scripts/mol1_io/feature_goldens.py \
      --out ~/mol1-goldens/io_features.pkl cases/*.json [--use_rna_msa] [--no_msa]

Upstream augments ref_pos with a random rigid transform at inference; the seed below pins it so a rerun is
byte-identical, and the test compares ref_pos up to that transform anyway.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from mol.data.inference.infer_dataloader import InferenceDataset


PATH_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")


def absolutize(job, base):
    """Upstream resolves relative MSA/template paths against the working directory; the examples are written for
    the repository root. Resolve them against `base` instead, so nothing runs from inside the clone."""
    for s in job["sequences"]:
        body = next(iter(s.values()))
        for k in PATH_KEYS:
            if isinstance(body, dict) and body.get(k) and not Path(body[k]).is_absolute():
                body[k] = str((Path(base) / body[k]).resolve())
    return job


def featurize(job, use_msa, use_rna_msa, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa, ds.use_rna_msa, ds.msa_pair_as_unpair = use_msa, use_rna_msa, True
    ds.use_template, ds.online_template_featurizer = False, None
    data, aa, _ = ds.process_one(single_sample_dict=job)
    feats = {k: v.numpy() for k, v in data["input_feature_dict"].items() if torch.is_tensor(v)}
    atoms = {"atom_name": aa.atom_name.tolist(), "res_name": aa.res_name.tolist(),
             "res_id": aa.res_id.tolist(), "chain_id": aa.chain_id.tolist(), "element": aa.element.tolist()}
    return {"features": feats, "atoms": atoms, "job": job}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_msa", action="store_true")
    ap.add_argument("--use_rna_msa", action="store_true")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--base", default=".", help="directory relative MSA/template paths are resolved against")
    a = ap.parse_args()
    out = {}
    for p in a.inputs:
        for job in json.loads(Path(p).read_text()):
            job = absolutize(job, a.base)
            out[job["name"]] = featurize(job, not a.no_msa, a.use_rna_msa, a.seed)
            f = out[job["name"]]["features"]
            print(job["name"], "tokens", f["token_index"].shape[0], "atoms", f["atom_to_token_idx"].shape[0],
                  "msa", f["msa"].shape)
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(Path(a.out).expanduser(), "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
