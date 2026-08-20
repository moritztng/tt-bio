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

A tap the torch side never produced is a FAIL at full recycles, not a silent omission: the
reference names every tap the model owes, so a disappearing one has to be loud. `--recycles 0`
is a single-pass smoke test and only reports which taps it skipped.

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

# ColabDesign's production setting: four forward passes (`design.py:147-205`).
DEFAULT_RECYCLES = 3

# What PXDesign actually filters on, and how close the torch arm has to land. The thresholds
# ship against values rounded to two decimals, so these bars are 2-3x tighter than production.
SCALAR_BARS = {"plddt": 2e-3, "ptm": 2e-3, "i_ptm": 5e-3,
               "pae": 2e-3, "i_pae": 5e-3, "unscaled_i_pae": 0.155}


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
    """Collects the torch-side activations under the same names the JAX capture used.

    The call index is a flat counter per tag, exactly as in `capture_ref_jax.py`: the four
    recycles share it, so `linear/preprocess_1d#0` is the first pass and `structure_module#3`
    the fourth, with no special-casing anywhere.
    """

    def __init__(self, keep: set[str] | None = None) -> None:
        self.keep = keep
        self.values: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.produced = 0

    def emit(self, tag: str, payload: dict) -> None:
        index = self.counts.get(tag, 0)
        self.counts[tag] = index + 1
        for key, value in payload.items():
            name = f"{tag}#{index}/{key}"
            self.produced += 1
            if self.keep is None or name in self.keep:
                self.values[name] = value.detach()

    def install(self, model) -> list:
        """Hook the trunk's stage boundaries. The per-pass outputs are emitted by `collect`."""
        handles = []

        def hook(tag):
            def fn(_module, _args, output):
                self.emit(tag, {"out": output})
            return fn

        for tag, module in (
            ("linear/preprocess_1d", model.embed["preprocess_1d"]),
            ("linear/preprocess_msa", model.embed["preprocess_msa"]),
            ("linear/left_single", model.embed["left_single"]),
            ("linear/right_single", model.embed["right_single"]),
            ("linear/pair_activiations", model.embed["pair_activations"]),
            ("linear/extra_msa_activations", model.embed["extra_msa_activations"]),
            ("linear/prev_pos_linear", model.recycle["prev_pos_linear"]),
            ("norm/prev_msa_first_row_norm", model.recycle["prev_msa_norm"]),
            ("norm/prev_pair_norm", model.recycle["prev_pair_norm"]),
            ("linear/single_activations", model.single_activations),
        ):
            handles.append(module.register_forward_hook(hook(tag)))
        if model.template is not None:
            for tag, module in (
                ("linear/template_single_embedding", model.template.single_embedding),
                ("linear/template_projection", model.template.single_projection),
                ("template_embedding", model.template),
            ):
                handles.append(module.register_forward_hook(hook(tag)))
            # `TemplatePairStack` is the whole 2-block stack, so its output is the last block's.
            handles.append(model.template.pair_stack[-1].register_forward_hook(
                hook("template_pair_stack")))
        for stack, tag in ((model.extra_msa, "extra_msa_stack"),
                           (model.evoformer, "evoformer_iteration")):
            for block in stack:
                def fn(_module, _args, output, tag=tag):
                    self.emit(tag, {"msa": output[0], "pair": output[1]})
                handles.append(block.register_forward_hook(fn))
        return handles

    def collect(self, out: dict) -> None:
        """The taps that are one per recycle rather than one per module call."""
        self.emit("evoformer", {k: out[k] for k in ("single", "pair", "msa", "msa_first_row")})
        self.emit("predicted_aligned_error_head",
                  {"logits": out["pae_logits"], "breaks": out["pae_breaks"]})
        self.emit("structure_module", out["structure"])
        self.emit("predicted_lddt_head", {"logits": out["plddt_logits"]})


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
    ap.add_argument("--recycles", type=int, default=DEFAULT_RECYCLES,
                    help="0 runs a single pass; only the first recycle's taps are then scored")
    ap.add_argument("--drop-tap", default=None,
                    help="discard one tap by name, to prove the missing-tap branch really fails")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tt_bio.af2_confidence import PAE_MAX_ERROR_BIN, confidence_scalars
    from tt_bio.af2_reference import load_af2_model, run_recycles
    from tt_bio.af2_weights import load_af2_state_dict

    suffix = "" if args.stage == "complex" else "_monomer"
    inputs = Path(args.inputs or ARTIFACTS / f"ref_inputs{suffix}.npz")
    taps_path = Path(args.taps or ARTIFACTS / f"ref_taps{suffix}.npz")
    feats, prev = load_inputs(inputs)
    # The monomer stage is the model_3_ptm config: same checkpoint, template stack dropped.
    model = load_af2_model(load_af2_state_dict(args.params), template=args.stage == "complex")

    ref = None
    if taps_path.exists():
        with np.load(taps_path, allow_pickle=False) as npz:
            ref = {k: npz[k] for k in npz.files}
    bases = sorted({k.rsplit("/", 1)[0] for k in ref if k.endswith("/shape")}) if ref else []

    taps = Taps(keep=set(bases) if ref else None)
    handles = taps.install(model)
    for out in run_recycles(model, feats, prev, num_recycles=args.recycles):
        taps.collect(out)
        last = out
    for handle in handles:
        handle.remove()
    if args.drop_tap:
        taps.values.pop(args.drop_tap, None)

    if ref is None:
        print(json.dumps({"mode": "af2ig_taps", "stage": args.stage,
                          "verdict": "NO_REFERENCE",
                          "taps_produced": sorted(taps.values),
                          "finite": {k: bool(torch.isfinite(v).all())
                                     for k, v in taps.values.items()}}, indent=1))
        return 0

    rows, skipped = [], []
    for base in bases:
        if base in taps.values:
            rows.append(score_one(ref, base, taps.values[base]))
        else:
            skipped.append(base)

    # Truncated runs are smoke tests and are expected to leave taps unproduced. A full run is
    # not: every reference tap has an owner, so a missing one is a regression, not a gap.
    full = args.recycles == DEFAULT_RECYCLES
    failed = [r for r in rows if r["verdict"] != "PASS"]

    # The scalars come off the last recycle, so they only mean anything at full recycles.
    scalars, scalars_failed = [], []
    if full:
        meta = json.loads(bytes(ref["_meta/json"]).decode())
        binder = (meta["fixture"]["binder_residues"]
                  if meta["production"]["protocol"] == "binder" else None)
        got = confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                                 feats["seq_mask"], feats["asym_id"], binder_len=binder)
        want = dict(meta["log"])
        if "i_pae" in want:
            want["unscaled_i_pae"] = want["i_pae"] * PAE_MAX_ERROR_BIN
        for name, value in got.items():
            if name not in want:
                continue
            row = {"scalar": name, "got": value, "want": want[name],
                   "delta": abs(value - want[name]), "bar": SCALAR_BARS[name]}
            row["verdict"] = "PASS" if row["delta"] <= row["bar"] else "FAIL"
            scalars.append(row)
        scalars_failed = [r for r in scalars if r["verdict"] != "PASS"]
    report = {
        "mode": "af2ig_taps",
        "stage": args.stage,
        "recycles": args.recycles,
        "verdict": ("PASS" if rows and not failed and not scalars_failed
                    and not (full and skipped) else "FAIL"),
        "taps_scored": len(rows),
        "taps_failed": len(failed),
        "taps_produced": taps.produced,
        "pcc_min": min((r["pcc"] for r in rows if "pcc" in r), default=None),
        "scalars": scalars,
        "scalars_failed": len(scalars_failed),
        "not_implemented": skipped,
        "failures": failed[:12],
        "rows": rows,
    }
    print(json.dumps(report, indent=1, default=float))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
