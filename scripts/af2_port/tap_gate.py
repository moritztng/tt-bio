"""Score `tt_bio.af2_reference` against the captured JAX activation taps.

Card-free and upstream-free: it reads the two committed artifacts
(`ref_inputs.npz` for the inputs, `ref_taps.npz` for the reference activations) and the AF2
checkpoint. The inputs come from the capture rather than from `tt_bio.af2_data` on purpose --
the featurizer is already scored bit-exact by `parity_gate.py`, so driving the reference from
the same arrays JAX consumed keeps this gate measuring the model and nothing else.

Each tap is scored two ways, because either one alone is blind:

* Pearson correlation over the stored elements. Catches a wrong permutation, a wrong block, a
  swapped left/right, a missing residual.
* The full-array mean, standard deviation and sum of squares, relatively. Catches a uniform
  shift or a scale error, which a correlation cannot see.

    PYTHONPATH=. env/bin/python3 scripts/af2_port/tap_gate.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ARTIFACTS = Path(__file__).resolve().parent / "parity_artifacts" / "laczc128_b80"
DEFAULT_PARAMS = os.path.expanduser("~/pxd_tool_weights/af2/params_model_1_ptm.npz")

# The trunk's float32 boundaries. Inside the bf16 trunk the bar has to allow for bf16's ~3
# decimal digits accumulating over 48 blocks; the two ends of the trunk are float32 on both
# sides and get the tight bar.
PCC_BAR = 0.998
STATS_BAR = 0.02

INT_KEYS = ("aatype", "residue_index", "asym_id", "sym_id", "entity_id", "extra_msa",
            "template_aatype", "residx_atom14_to_atom37", "residx_atom37_to_atom14")


def load_inputs(path: Path) -> tuple[dict, dict]:
    """Split the captured input dict into the model features and the recycling state."""
    feats: dict[str, torch.Tensor] = {}
    prev: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            if key.startswith(("_meta/", "opt/", "seq/", "params/")) or key == "bias":
                continue
            value = np.asarray(npz[key])
            if value.dtype == np.bool_:
                tensor = torch.from_numpy(value.astype(np.bool_))
            elif value.dtype.kind in "iu":
                tensor = torch.from_numpy(value.astype(np.int64))
            else:
                tensor = torch.from_numpy(value.astype(np.float32))
            if key.startswith("prev/"):
                prev[key[len("prev/"):]] = tensor
            else:
                feats[key] = tensor
    return feats, prev


class Taps:
    """Collects the torch-side activations under the same names the JAX capture used."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def add(self, name: str, value: torch.Tensor) -> None:
        self.values[name] = value.detach()

    def install(self, model) -> list:
        handles = []

        def hook(name):
            def fn(_module, _args, output):
                self.add(name, output)
            return fn

        for jax_name, module in (
            ("linear/preprocess_1d#0/out", model.embed["preprocess_1d"]),
            ("linear/preprocess_msa#0/out", model.embed["preprocess_msa"]),
            ("linear/left_single#0/out", model.embed["left_single"]),
            ("linear/right_single#0/out", model.embed["right_single"]),
            ("linear/pair_activiations#0/out", model.embed["pair_activations"]),
            ("linear/extra_msa_activations#0/out", model.embed["extra_msa_activations"]),
            ("linear/prev_pos_linear#0/out", model.recycle["prev_pos_linear"]),
            ("norm/prev_msa_first_row_norm#0/out", model.recycle["prev_msa_norm"]),
            ("norm/prev_pair_norm#0/out", model.recycle["prev_pair_norm"]),
            ("linear/single_activations#0/out", model.single_activations),
        ):
            handles.append(module.register_forward_hook(hook(jax_name)))
        if model.template is not None:
            for jax_name, module in (
                ("linear/template_single_embedding#0/out", model.template.single_embedding),
                ("linear/template_projection#0/out", model.template.single_projection),
                ("template_embedding#0/out", model.template),
            ):
                handles.append(module.register_forward_hook(hook(jax_name)))
            # `TemplatePairStack` is the whole 2-block stack, so its output is the last block's.
            last = model.template.pair_stack[-1]
            handles.append(last.register_forward_hook(hook("template_pair_stack#0/out")))
        for stack, tag in ((model.extra_msa, "extra_msa_stack"),
                           (model.evoformer, "evoformer_iteration")):
            for i, block in enumerate(stack):
                def fn(_module, _args, output, i=i, tag=tag):
                    self.add(f"{tag}#{i}/msa", output[0])
                    self.add(f"{tag}#{i}/pair", output[1])
                handles.append(block.register_forward_hook(fn))
        return handles


def score_one(ref, base: str, arr: torch.Tensor) -> dict:
    flat = arr.reshape(-1).float().numpy().astype(np.float64)
    shape = tuple(int(x) for x in ref[f"{base}/shape"])
    out = {"tap": base, "shape_ref": shape, "shape_got": tuple(arr.shape)}
    if math.prod(shape) != flat.size:
        out["verdict"] = "SHAPE"
        return out
    if f"{base}/full" in ref:
        want, got = ref[f"{base}/full"].astype(np.float64), flat
        out["sampled"] = int(got.size)
    else:
        idx = ref[f"{base}/idx"]
        want, got = ref[f"{base}/val"].astype(np.float64), flat[idx]
        out["sampled"] = int(idx.size)
    if want.std() < 1e-12 and got.std() < 1e-12:
        pcc = 1.0 if abs(want.mean() - got.mean()) < 1e-6 else 0.0
    else:
        pcc = float(np.corrcoef(want, got)[0, 1])
    mean_r, std_r, lo_r, hi_r, sq_r = ref[f"{base}/stats"]
    scale = max(abs(float(std_r)), 1e-6)
    out |= {
        "pcc": pcc,
        "d_mean": abs(float(flat.mean()) - float(mean_r)) / scale,
        "d_std": abs(float(flat.std()) - float(std_r)) / scale,
        "d_sumsq": abs(float((flat * flat).sum()) - float(sq_r)) / max(abs(float(sq_r)), 1e-9),
    }
    out["verdict"] = ("PASS" if pcc >= PCC_BAR and out["d_mean"] <= STATS_BAR
                      and out["d_std"] <= STATS_BAR and out["d_sumsq"] <= STATS_BAR else "FAIL")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--taps", default=None)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--evoformer-blocks", type=int, default=None,
                    help="truncate the Evoformer for a smoke test; scoring skips missing taps")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tt_bio.af2_reference import NUM_EVOFORMER_BLOCKS, load_trunk
    from tt_bio.af2_weights import load_af2_state_dict

    suffix = "" if args.stage == "complex" else "_monomer"
    inputs = Path(args.inputs or ARTIFACTS / f"ref_inputs{suffix}.npz")
    taps_path = Path(args.taps or ARTIFACTS / f"ref_taps{suffix}.npz")
    feats, prev = load_inputs(inputs)
    blocks = args.evoformer_blocks or NUM_EVOFORMER_BLOCKS
    # The monomer stage is the model_3_ptm config: same checkpoint, template stack dropped.
    model = load_trunk(load_af2_state_dict(args.params), template=args.stage == "complex",
                       num_evoformer_blocks=blocks)

    taps = Taps()
    handles = taps.install(model)
    with torch.no_grad():
        out = model(feats, prev)
    for handle in handles:
        handle.remove()
    taps.add("evoformer#0/single", out["single"])
    taps.add("evoformer#0/pair", out["pair"])
    taps.add("evoformer#0/msa", out["msa"])
    taps.add("evoformer#0/msa_first_row", out["msa_first_row"])
    taps.add("predicted_aligned_error_head#0/logits", out["pae_logits"])

    if not taps_path.exists():
        print(json.dumps({"mode": "af2ig_taps", "stage": args.stage,
                          "verdict": "NO_REFERENCE",
                          "taps_produced": sorted(taps.values),
                          "finite": {k: bool(torch.isfinite(v).all())
                                     for k, v in taps.values.items()}}, indent=1))
        return 0

    with np.load(taps_path, allow_pickle=False) as npz:
        ref = {k: npz[k] for k in npz.files}
    bases = sorted({k.rsplit("/", 1)[0] for k in ref if k.endswith("/shape")})
    rows, skipped = [], []
    for base in bases:
        if base in taps.values:
            rows.append(score_one(ref, base, taps.values[base]))
        else:
            skipped.append(base)

    failed = [r for r in rows if r["verdict"] != "PASS"]
    report = {
        "mode": "af2ig_taps",
        "stage": args.stage,
        "verdict": "PASS" if rows and not failed else "FAIL",
        "taps_scored": len(rows),
        "taps_failed": len(failed),
        "pcc_min": min((r["pcc"] for r in rows if "pcc" in r), default=None),
        "not_in_reference": sorted(set(taps.values) - set(bases)),
        "not_implemented": skipped,
        "failures": failed[:12],
        "rows": rows,
    }
    print(json.dumps(report, indent=1, default=float))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
