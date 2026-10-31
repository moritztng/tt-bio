"""Capture Mol-1 upstream input features as goldens for tests/test_mol1_features.py.

Runs upstream's own featurizer, `InferenceDataset.process_one`, on every job in the given JSON files and saves
the `input_feature_dict` plus the atom table, one compressed `<out>/<job>.npz` per job (integer features narrowed
to the smallest dtype that holds them, values unchanged). Run it in the upstream venv, isolated, from outside the
upstream clone, with MOL_ROOT_DIR pointing at the release `common/` data:

    MOL_ROOT_DIR=/home/moritz/mol1-upstream/io-data \
      /home/moritz/mol1-upstream/venv/bin/python -I scripts/mol1_io/feature_goldens.py \
      --out tests/data/mol1_io/features/msa cases/*.json [--use_rna_msa] [--no_msa] [--use_template]

--use_template builds upstream's inference TemplateHitFeaturizer from MOL_ROOT_DIR (mmcif/, common/), without
remote fetches, so kalign must be on PATH and every hit's mmCIF present.

Upstream augments ref_pos with a random rigid transform at inference; the seed below pins it so a rerun is
byte-identical, and the test compares ref_pos up to that transform anyway.
"""
import argparse
import json
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


def template_featurizer():
    """Upstream's InferenceDataset template featurizer (mol/data/inference/infer_dataloader.py), fetch_remote off."""
    import os

    from mol.data.template.template_utils import TemplateHitFeaturizer
    root = Path(os.environ["MOL_ROOT_DIR"])
    return TemplateHitFeaturizer(
        mmcif_dir=str(root / "mmcif"), template_cache_dir="", max_hits=4, max_template_date="2021-09-30",
        release_dates_path=str(root / "common/release_date_cache.json"),
        obsolete_pdbs_path=str(root / "common/obsolete_to_successor.json"),
        _shuffle_top_k_prefiltered=None, _max_template_candidates_num=20, fetch_remote=False)


def featurize(job, use_msa, use_rna_msa, seed, use_template=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = InferenceDataset.__new__(InferenceDataset)
    ds.use_msa, ds.use_rna_msa, ds.msa_pair_as_unpair = use_msa, use_rna_msa, True
    ds.use_template, ds.online_template_featurizer = use_template, (template_featurizer() if use_template else None)
    data, aa, _ = ds.process_one(single_sample_dict=job)
    feats = {k: narrow(v.numpy()) for k, v in data["input_feature_dict"].items() if torch.is_tensor(v)}
    atoms = {f"atoms.{k}": np.asarray(getattr(aa, k)) for k in ("atom_name", "res_name", "res_id", "chain_id", "element")}
    return feats | atoms


def narrow(a):
    if a.dtype.kind in "iub" and a.size:
        for t in (np.int8, np.int16, np.int32):
            if np.iinfo(t).min <= a.min() and a.max() <= np.iinfo(t).max:
                return a.astype(t)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_msa", action="store_true")
    ap.add_argument("--use_rna_msa", action="store_true")
    ap.add_argument("--use_template", action="store_true")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--base", default=".", help="directory relative MSA/template paths are resolved against")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for p in a.inputs:
        for job in json.loads(Path(p).read_text()):
            f = featurize(absolutize(job, a.base), not a.no_msa, a.use_rna_msa, a.seed, a.use_template)
            np.savez_compressed(out / f"{job['name']}.npz", **f)
            print(job["name"], "tokens", f["token_index"].shape[0], "atoms", f["atom_to_token_idx"].shape[0],
                  "msa", f["msa"].shape, "KB", (out / f"{job['name']}.npz").stat().st_size // 1024)


if __name__ == "__main__":
    main()
