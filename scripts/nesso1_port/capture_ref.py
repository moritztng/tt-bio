#!/usr/bin/env python3
"""Capture an upstream Nesso-1 reference for the port's parity gate.

Runs with the UPSTREAM ``nesso`` package importable (a separate venv -- never the
tt-bio env, installing into a venv a timed run is using corrupts that run). Writes,
for one committed fixture YAML:

  processed/            upstream preprocessing output, committed verbatim so the gate
                        needs neither an upstream install nor a matching RDKit
  ref_feats.pt          every tensor the featurizer produces, for bit-exact scoring
  ref_acts.pt           module activations captured by forward hook, for per-block parity
  ref_out.json          the affinity scalars the CLI writes
  meta.json             package versions + a sha256 per artifact

Why the preprocessed dir is committed rather than regenerated: RDKit ETKDG conformer
coordinates are version-dependent. 2025.09.6 (tt-bio) and 2026.03.5 (upstream today)
give the same 13 atoms in the same order but coordinates ~1.5 A apart for the tutorial
ligand. Regenerating the conformer at gate time would silently compare two different
inputs. Both versions are self-consistent run to run, so a committed conformer makes
the host leg exactly reproducible.

Usage:
  <ref_venv>/bin/python scripts/nesso1_port/capture_ref.py \
      --fixture scripts/nesso1_port/parity_artifacts/tyr48 [--recycling_steps 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
FEAT_SEED = 20260820


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_upstream_cli(fixture: Path, yaml_path: Path, recycling_steps: int) -> Path:
    """Run the upstream CLI so ``processed/`` and the affinity scalars are the real thing."""
    out_dir = fixture / "_upstream_out"
    cmd = [
        sys.executable.replace("/python", "/nesso"),
        "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--accelerator", "cpu",
        "--precision", "32",
        "--recycling_steps", str(recycling_steps),
        "--override",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def find_ccd() -> Path:
    """Locate the cached ccd.pkl the CLI downloaded (413 MB, never committed)."""
    base = Path(os.environ.get("NESSO_CACHE", ".cache")).expanduser()
    hits = sorted(base.rglob("models--recursionpharma--nesso/snapshots/*/ccd.pkl"))
    if not hits:
        raise SystemExit("could not locate ccd.pkl; pass --ccd")
    return hits[0]


def build_dataset(processed: Path, ccd_pkl: Path):
    from nesso.data.featurizer import NessoFeaturizer
    from nesso.data.inference import InferenceDataset
    from nesso.data.types import Manifest

    manifest = Manifest.load(processed / "manifest.json")
    featurizer = NessoFeaturizer(
        esm_emb_dir=processed / "esm_embeddings", esm_emb_dim=1280, esm_num_layers=33
    )
    return InferenceDataset(
        manifest=manifest,
        target_dir=processed,
        ligand_dir=processed / "rdkit_conformers",
        ccd_pkl=ccd_pkl,
        use_esm_all_layers=False,
        num_dist_bins=64,
        min_dist=2.0,
        max_dist=22.0,
        atoms_per_window_queries=32,
        featurizer=featurizer,
    )


HOOK_TARGETS = {
    "input_embedder": "s_inputs",
    "rel_pos": "relative_position_encoding",
    "esm_module": "esm_module_out",
    "pairformer_module": "pairformer_out",
    "distogram_head": "distogram_head_out",
    "affinity_module": "affinity_1",
    "affinity_module2": "affinity_2",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--recycling_steps", type=int, default=5)
    ap.add_argument("--checkpoint", default="recursionpharma/nesso")
    ap.add_argument("--ccd", type=Path, default=None,
                    help="path to ccd.pkl; auto-discovered under NESSO_CACHE if omitted")
    ap.add_argument("--skip-cli", action="store_true",
                    help="reuse an existing _upstream_out/ instead of re-running the CLI")
    args = ap.parse_args()

    fixture = args.fixture.resolve()
    yamls = sorted(fixture.glob("*.yaml"))
    if len(yamls) != 1:
        raise SystemExit(f"expected exactly one fixture yaml in {fixture}, found {len(yamls)}")
    yaml_path = yamls[0]

    out_dir = fixture / "_upstream_out"
    if not args.skip_cli:
        out_dir = run_upstream_cli(fixture, yaml_path, args.recycling_steps)
    processed = out_dir / "processed"
    if not processed.is_dir():
        raise SystemExit(f"missing {processed}; run without --skip-cli")

    # 1. the featurizer output, every tensor, for the bit-exact host leg.
    #    ccd.pkl is 413 MB and never committed; only the 21 standard-AA mols the
    #    featurizer actually reads are, so the gate stays self-contained.
    ccd_pkl = args.ccd or find_ccd()
    import pickle
    from nesso.data.inference import load_standard_aa_mols
    aa_mols = load_standard_aa_mols(ccd_pkl)
    with (fixture / "standard_aa_mols.pkl").open("wb") as fh:
        pickle.dump(aa_mols, fh)
    print(f"ccd.pkl {ccd_pkl} -> {len(aa_mols)} standard-AA mols")

    ds = build_dataset(processed, ccd_pkl)
    # process_atom_features applies center_random_augmentation, a random
    # roto-translation per conformer drawn from the GLOBAL torch RNG -- not from the
    # RandomState(idx) the featurizer is handed. Featurization is therefore a
    # sampling step, and reference and port must share the draws or the comparison
    # is meaningless. Seed immediately before the fetch, and record the seed.
    torch.manual_seed(FEAT_SEED)
    feats = ds[0]
    if feats.get("exception"):
        raise SystemExit("upstream featurizer raised; see traceback above")
    ref_feats = {k: v for k, v in feats.items() if isinstance(v, torch.Tensor)}
    print(f"captured {len(ref_feats)} feature tensors")

    # 2. module activations, for per-block parity as each module lands
    from nesso.model.models.nesso1 import Nesso1

    model = Nesso1.from_pretrained(args.checkpoint)
    model.eval()
    model.use_kernels = False

    acts: dict[str, torch.Tensor] = {}
    counters: dict[str, int] = {}
    handles = []

    def make_hook(tag: str):
        def hook(_mod, _inp, out):
            n = counters.get(tag, 0)
            counters[tag] = n + 1
            key = tag if n == 0 else f"{tag}#{n}"
            if isinstance(out, torch.Tensor):
                acts[key] = out.detach().clone()
            elif isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        acts[f"{key}.{k}"] = v.detach().clone()
        return hook

    for attr, tag in HOOK_TARGETS.items():
        mod = getattr(model, attr, None)
        if mod is not None:
            handles.append(mod.register_forward_hook(make_hook(tag)))

    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
             for k, v in feats.items()}
    with torch.no_grad():
        out = model(batch, recycling_steps=args.recycling_steps,
                    refine_protein_inference=True)
    for h in handles:
        h.remove()

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            acts[f"out.{k}"] = v.detach().clone()
    print(f"captured {len(acts)} activation tensors")

    torch.save(ref_feats, fixture / "ref_feats.pt")
    torch.save(acts, fixture / "ref_acts.pt")

    pred = sorted((out_dir / "predictions").glob("*/affinity.json"))
    scalars = json.loads(pred[0].read_text()) if pred else {}
    (fixture / "ref_out.json").write_text(json.dumps(scalars, indent=2) + "\n")

    import numpy, rdkit, transformers
    meta = {
        "fixture": yaml_path.name,
        "checkpoint": args.checkpoint,
        "ccd_pkl": str(ccd_pkl),
        "recycling_steps": args.recycling_steps,
        "feat_seed": FEAT_SEED,
        "n_tokens": int(ref_feats["token_pad_mask"].shape[-1]),
        "versions": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "rdkit": rdkit.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
        "feature_keys": sorted(ref_feats),
        "activation_keys": sorted(acts),
        "sha256": {
            p.name: sha256(p)
            for p in [fixture / "ref_feats.pt", fixture / "ref_acts.pt",
                      fixture / "ref_out.json",
                      fixture / "standard_aa_mols.pkl"]
        },
    }
    (fixture / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_tokens", "versions")}, indent=2))
    print(json.dumps(scalars, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
