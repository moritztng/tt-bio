#!/usr/bin/env python3
"""Nesso-1 torch-reference parity: our ``tt_bio.nesso1.Nesso1`` vs the upstream capture.

Card-free and upstream-free. By default it featurizes from the committed ``processed/``
directory, so the RDKit conformer is the captured one and never regenerated. With
``--from-yaml`` it instead runs the shipped host pipeline (``tt_bio.nesso1_input``) from
the fixture YAML, which puts the parse, the conformer embedding and the ESM-2 encoder
inside the gate -- the part the committed-``processed/`` path cannot see. That leg needs
``ccd.pkl`` (413 MB, next to the checkpoint) and downloads ESM-2 once. Measured bit-exact
on tyr48: 11/11 scalars, delta 0.0.

Legs:

  scalars   the affinity values and distogram entropies from the committed
            ``ref_scalars.json``, i.e. upstream's own ``predict_step`` under the SAME
            featurization seed the gate reseeds to. Bit-exact is the bar. This leg
            always runs and it is the blocking one.

  cli_draw  the same scalars as the upstream CLI wrote them (``ref_out.json``).
            NON-BLOCKING and expected to differ: ``center_random_augmentation``
            draws a random roto-translation per conformer off the global torch RNG,
            and the CLI's dataloader had its own RNG state. Measured 0.043 apart on
            ``affinity_pred_value1`` for tyr48 with every module activation bit-exact,
            which is the size of the draw effect, not of a port error.

  modules   per-module activations from ``ref_acts.pt``, matched by forward hook on the
            same seven attributes ``capture_ref.py`` hooked. Skipped when the capture
            is absent (it is 29 MB and gitignored); regenerate it with capture_ref.py.

Usage:
  <tt-bio env>/bin/python scripts/nesso1_port/model_parity.py \
      [--fixture scripts/nesso1_port/parity_artifacts/tyr48] [--weights <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ARTIFACTS = REPO / "scripts" / "nesso1_port" / "parity_artifacts"
FEAT_SEED = 20260820

# same attributes capture_ref.py hooks, same tags, so the keys line up
HOOK_TARGETS = {
    "input_embedder": "s_inputs",
    "rel_pos": "relative_position_encoding",
    "esm_module": "esm_module_out",
    "pairformer_module": "pairformer_out",
    "distogram_head": "distogram_head_out",
    "affinity_module": "affinity_1",
    "affinity_module2": "affinity_2",
}

# The upstream CLI runs with refine_protein_inference on and a 256-token budget.
CLI_PREDICT_ARGS = {
    "pose_protein_cutoff": 15.0,
    "affinity_protein_cutoff": 15.0,
    "refine_protein_inference": True,
    "refine_protein_cutoff": 22.0,
    "refine_protein_tokens_budget": 256,
}


def load_feats(fixture: Path) -> tuple[dict, dict]:
    from tt_bio._vendor.nesso.data.featurizer import NessoFeaturizer
    from tt_bio._vendor.nesso.data.inference import InferenceDataset
    from tt_bio._vendor.nesso.data.types import Manifest

    processed = fixture / "processed"
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
    torch.manual_seed(int(meta.get("feat_seed", FEAT_SEED)))
    item = ds[0]
    if item.get("exception"):
        raise SystemExit("vendored featurizer raised on the committed fixture")
    feats = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
        for k, v in item.items()
    }
    return feats, meta


def load_feats_from_yaml(fixture: Path, work: Path | None, ccd: Path | None):
    """Featurize through the shipped host pipeline, YAML in, batch out.

    num_workers=0 on purpose: RDKit takes its conformer embedding seed from the process
    RNG state, so a pool would produce a different ligand geometry than the committed
    capture and the leg would be scoring the parse topology rather than the pipeline.
    """
    import tempfile

    from tt_bio.nesso1_input import collate, prepare

    yamls = sorted(fixture.glob("*.yaml"))
    if len(yamls) != 1:
        raise SystemExit(f"expected one yaml in {fixture}, found {len(yamls)}")
    meta = json.loads((fixture / "meta.json").read_text())
    out = Path(work).expanduser() if work else Path(tempfile.mkdtemp(prefix="nesso1_gate_"))
    dataset, manifest, failed = prepare(yamls[0], out, ccd_pkl=ccd, num_workers=0)
    if failed:
        raise SystemExit(f"host pipeline failed to parse {failed}")
    torch.manual_seed(int(meta.get("feat_seed", FEAT_SEED)))
    item = dataset[0]
    if item.get("exception"):
        raise SystemExit("host pipeline featurizer raised")
    return collate(item), meta


def build_model(weights: str | Path, recycling_steps: int):
    from tt_bio.nesso1 import Nesso1

    model = Nesso1.from_pretrained(weights)
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)
    model.predict_args["recycling_steps"] = recycling_steps
    return model


def score_scalars(pred: dict, ref: dict, tol: float, leg: str = "scalars") -> dict:
    rows, worst = [], 0.0
    for key, want in sorted(ref.items()):
        got = pred.get(key)
        if got is None:
            rows.append({"key": key, "why": "absent from our output"})
            worst = float("inf")
            continue
        got = float(got.reshape(-1)[0])
        d = abs(got - float(want))
        worst = max(worst, d)
        rows.append({"key": key, "ref": float(want), "got": got, "abs_delta": d})
    return {
        "leg": leg,
        "verdict": "PASS" if worst <= tol else "FAIL",
        "tol": tol,
        "max_abs_delta": worst,
        "values": rows,
    }


def score_modules(acts: dict, ref_acts: dict) -> dict:
    rows, worst_rel = [], 0.0
    missing = []
    for key in sorted(ref_acts):
        if key.startswith("out."):
            continue  # covered by the scalar leg and by pdistogram below
        a = ref_acts[key]
        b = acts.get(key)
        if b is None:
            missing.append(key)
            continue
        if a.shape != b.shape:
            rows.append({"key": key, "why": "shape",
                         "ref": list(a.shape), "got": list(b.shape)})
            worst_rel = float("inf")
            continue
        a32, b32 = a.float(), b.float()
        denom = a32.abs().max().clamp(min=1e-12)
        rel = float((a32 - b32).abs().max() / denom)
        worst_rel = max(worst_rel, rel)
        rows.append({
            "key": key,
            "shape": list(a.shape),
            "max_abs": float((a32 - b32).abs().max()),
            "max_rel": rel,
            "bit_exact": bool(torch.equal(a, b)),
        })
    return {
        "leg": "modules",
        "verdict": "PASS" if worst_rel <= 1e-4 and not missing else "FAIL",
        "max_rel_delta": worst_rel,
        "n_compared": len(rows),
        "n_bit_exact": sum(1 for r in rows if r.get("bit_exact")),
        "missing_from_port": missing,
        "activations": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=ARTIFACTS / "tyr48")
    ap.add_argument("--weights", default="recursionpharma/nesso",
                    help="local snapshot dir or Hub repo id")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="scalar tolerance against the shared-draw reference; "
                         "0.0 because the torch assembly is bit-exact")
    ap.add_argument("--from-yaml", action="store_true",
                    help="featurize from the fixture YAML through tt_bio.nesso1_input "
                         "instead of the committed processed/ dir, so the parse, the "
                         "RDKit conformer and the ESM-2 encoder are inside the gate")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch dir for --from-yaml (default: a temp dir)")
    ap.add_argument("--ccd", type=Path, default=None,
                    help="ccd.pkl for --from-yaml; auto-discovered under NESSO_CACHE if omitted")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    fixture = args.fixture.resolve()
    if args.from_yaml:
        feats, meta = load_feats_from_yaml(fixture, args.work, args.ccd)
    else:
        feats, meta = load_feats(fixture)
    model = build_model(args.weights, meta.get("recycling_steps", 5))

    acts: dict[str, torch.Tensor] = {}
    counters: dict[str, int] = {}

    def make_hook(tag: str):
        def hook(_m, _i, out):
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

    handles = [getattr(model, a).register_forward_hook(make_hook(t))
               for a, t in HOOK_TARGETS.items() if hasattr(model, a)]
    with torch.no_grad():
        pred = model.predict(feats)
    for h in handles:
        h.remove()

    report = {
        "gate": "nesso1_model_parity",
        "fixture": fixture.name,
        "input_path": "yaml_via_nesso1_input" if args.from_yaml else "committed_processed",
        "n_tokens": int(feats["token_pad_mask"].shape[-1]),
        "recycling_steps": meta.get("recycling_steps", 5),
        "scalars": score_scalars(
            pred, json.loads((fixture / "ref_scalars.json").read_text()), args.tol
        ),
        "cli_draw": score_scalars(
            pred,
            json.loads((fixture / "ref_out.json").read_text()),
            float("inf"),
            leg="cli_draw",
        ),
    }
    ref_acts_path = fixture / "ref_acts.pt"
    if ref_acts_path.exists():
        report["modules"] = score_modules(
            acts, torch.load(ref_acts_path, weights_only=True)
        )
    else:
        report["modules"] = {"leg": "modules", "verdict": "SKIP",
                             "why": f"{ref_acts_path.name} absent (gitignored); "
                                    "regenerate with capture_ref.py"}

    report["cli_draw"]["note"] = (
        "non-blocking: the CLI featurized under a different "
        "center_random_augmentation draw; see the module leg for the real comparison"
    )
    legs = [report["scalars"]["verdict"], report["modules"]["verdict"]]
    report["verdict"] = "FAIL" if "FAIL" in legs else "PASS"
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
