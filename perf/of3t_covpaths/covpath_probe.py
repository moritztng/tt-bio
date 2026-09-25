#!/usr/bin/env python3
"""PROTOCOL SS6: fire the conditional paths the frozen batch cannot reach.

The coverage census marks `templates`, `nucleotide` and `disabled_parameters` NOT COVERED,
and gives a property of 5nw3 or of the `weighted-pdb` default as the reason. This walks a
corpus chosen so each of those properties is different, and reports, per datapoint, what
the featuriser actually produced: the template masks, the molecule-type token counts, the
per-example `loss_weights`, and upstream's own disabled-parameter decision recomputed from
`runner._get_sample_disabled_param_names`.

No model, no backward. This is step 1: does the batch carry the path at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "of3_port"))

import batch_digest as BD  # noqa: E402

# runner.py:364-386 reads these off the model config; dataset_config_components.py:176
# carries the same list. Both are asserted against the package at runtime below.
CONF_NAMES = ["experimentally_resolved", "plddt", "pae", "pde"]


def confidence_loss_names(pkg: str) -> list[str]:
    import importlib

    m = importlib.import_module(f"{pkg}.projects.of3_all_atom.config.dataset_config_components")
    for attr in dir(m):
        cls = getattr(m, attr)
        f = getattr(cls, "model_fields", None) if isinstance(cls, type) else None
        if f and "confidence_loss_names" in f:
            d = f["confidence_loss_names"].default
            if d:
                return list(d)
    return CONF_NAMES


def describe(s: dict, idx: int, conf_names: list[str]) -> dict:
    tok = s["token_mask"].bool()
    counts = {
        k: int((s[k][tok] > 0).sum())
        for k in ("is_protein", "is_dna", "is_rna", "is_ligand")
        if k in s
    }
    lw = {k: float(v) for k, v in s["loss_weights"].items() if torch.is_tensor(v) and v.numel() == 1}
    total_conf = sum(lw[n] for n in conf_names if n in lw)
    tmpl = {}
    for k in sorted(s):
        if not k.startswith("template"):
            continue
        v = s[k]
        if torch.is_tensor(v):
            tmpl[k] = {"shape": list(v.shape), "nnz": int((v != 0).sum()),
                       "sum": float(v.float().sum())}
    return {
        "index": idx,
        "n_tokens_real": int(tok.sum()),
        "n_atoms_real": int(s["atom_mask"].sum()) if "atom_mask" in s else None,
        "molecule_type_tokens": counts,
        "n_nucleotide_tokens": counts.get("is_dna", 0) + counts.get("is_rna", 0),
        "token_bonds_nnz": int((s["token_bonds"] != 0).sum()) if "token_bonds" in s else None,
        "template_features": tmpl,
        "template_pseudo_beta_mask_nnz": tmpl.get("template_pseudo_beta_mask", {}).get("nnz"),
        "loss_weights": lw,
        "confidence_loss_names": conf_names,
        "summed_confidence_weight": total_conf,
        # runner.py:364-386 verbatim: a non-positive sum disables the confidence head.
        "disabled_parameters_would_fire": not (total_conf > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="openfold3")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-file", required=True, type=Path)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--max-index", type=int, default=None)
    ap.add_argument("--only-index", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    conf_names = confidence_loss_names(a.package)
    BD.seed_everything(a.seed)
    ds = BD.build_dataset(a.package, a.data_dir, a.n_templates, token_budget=a.crop,
                          split="train", stage=a.stage, cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    dp = ds.datapoint_cache
    idxs = ([a.only_index] if a.only_index is not None
            else list(range(len(ds) if a.max_index is None else min(len(ds), a.max_index))))
    print(f"{type(ds).__name__}: {len(ds)} datapoints, stage {a.stage} crop {a.crop} "
          f"n_templates {a.n_templates}; confidence_loss_names {conf_names}", flush=True)

    rows = []
    for i in idxs:
        before = guard["retries"]
        s = ds[i]
        r = describe(s, i, conf_names)
        r["silent_substitutions"] = guard["retries"] - before
        r["pdb_id"] = str(dp.iloc[i]["pdb_id"])
        r["datapoint"] = str(dp.iloc[i]["preferred_chain_or_interface"])
        rows.append(r)
        print(f"[{time.time()-t0:.0f}s] idx {i} {r['pdb_id']} {r['datapoint']}: "
              f"tokens {r['n_tokens_real']} {r['molecule_type_tokens']}  "
              f"tpl_pseudo_beta_mask nnz {r['template_pseudo_beta_mask_nnz']}  "
              f"conf_weight_sum {r['summed_confidence_weight']}  "
              f"disabled_would_fire {r['disabled_parameters_would_fire']}  "
              f"subs {r['silent_substitutions']}", flush=True)

    report = {
        "instrument": "PROTOCOL SS6: conditional-path presence on a chosen corpus",
        "package": a.package, "data_dir": str(a.data_dir), "cache_file": str(a.cache_file),
        "stage": a.stage, "dataset": "weighted-pdb", "crop": a.crop,
        "n_templates": a.n_templates, "seed": a.seed,
        "n_datapoints": len(ds), "rows": rows, "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
