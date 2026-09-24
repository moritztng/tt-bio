"""Score tt-bio's AF2-multimer_v3 trunk against AlphaFold's own JAX activations.

`multimer_ref_jax.py` runs `modules_multimer.EmbeddingsAndEvoformer` under haiku with the
checkpoint's parameters and stores the fixture it used together with every block's output. This
script feeds the same fixture to `tt_bio.af2_reference.AF2Model(multimer=True)` and compares,
component by component, then at the trunk output.

Two arms, and they answer different questions:

* `--dtype float64` asks whether the transform is RIGHT. A transcription error -- a swapped
  triangle-multiplication half, the outer product mean on the wrong side of the row attention,
  a relative-encoding bin off by one -- survives any amount of precision, so it has to be scored
  against a reference that is not itself an approximation. The JAX arm runs float32 and the torch
  arm float64, so what is left at the end is float32's own rounding and nothing else.
* `--dtype bfloat16` is what production runs, and its number is a precision reading, not a
  correctness one.

    PYTHONPATH=. env/bin/python3 scripts/af2_port/multimer_parity.py \
        --ref scripts/af2_port/parity_artifacts/multimer_trunk/ref_fp32.npz \
        --npz ~/pxd_tool_weights/af2/params_model_1_multimer_v3.npz --dtype float64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tt_bio.af2_data import ATOM_ORDER
from tt_bio.af2_reference import load_af2_model
from tt_bio.af2_weights import load_af2_state_dict

DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}
INTEGER_FEATURES = ("aatype", "residue_index", "asym_id", "entity_id", "sym_id", "extra_msa",
                    "template_aatype")


def load_reference(path: str) -> tuple[dict, dict, dict]:
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(bytes(arrays.pop("_meta/json")).decode())
    batch = {k[len("batch/"):]: v for k, v in arrays.items() if k.startswith("batch/")}
    taps = {k: v for k, v in arrays.items() if not k.startswith("batch/")}
    return batch, taps, meta


def to_feats(batch: dict, dtype: torch.dtype) -> dict:
    """The JAX fixture as the torch trunk's feature dict.

    `template_pseudo_beta` and its mask are what `modules.pseudo_beta_fn` computes inside the
    JAX template embedder and what `tt_bio.af2_data` precomputes on the torch side: glycine
    takes CA, everything else CB, and the mask follows the atom that was taken.
    """
    wide = torch.float64 if dtype == torch.float64 else torch.float32
    feats = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        feats[key] = tensor.long() if key in INTEGER_FEATURES else tensor.to(wide)
    positions = feats["template_all_atom_positions"]
    mask = feats["template_all_atom_mask"]
    is_gly = (feats["template_aatype"] == 7).unsqueeze(-1)
    feats["template_pseudo_beta"] = torch.where(
        is_gly, positions[..., ATOM_ORDER["CA"], :], positions[..., ATOM_ORDER["CB"], :])
    feats["template_pseudo_beta_mask"] = torch.where(
        is_gly.squeeze(-1), mask[..., ATOM_ORDER["CA"]], mask[..., ATOM_ORDER["CB"]])
    # multimer_v3 pads target_feat to 21 itself; the fixture stores the 20-wide one-hot.
    return feats


def report(name: str, got: torch.Tensor, want: np.ndarray) -> dict:
    a = got.detach().to(torch.float64).numpy().reshape(-1)
    b = np.asarray(want, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape, f"{name}: torch {a.shape} against jax {b.shape}"
    scale = max(float(np.abs(b).max()), 1e-30)
    abs_err = np.abs(a - b)
    denom = np.sqrt((a * a).mean()) * np.sqrt((b * b).mean())
    return {
        "tap": name,
        "shape": tuple(want.shape),
        "max_abs": float(abs_err.max()),
        "rel_max": float(abs_err.max() / scale),
        "rms": float(np.sqrt(((a - b) ** 2).mean())),
        "pcc": float(((a - a.mean()) * (b - b.mean())).mean()
                     / max(a.std() * b.std(), 1e-30)),
        "cos": float((a * b).mean() / max(denom, 1e-30)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dtype", default="float64", choices=sorted(DTYPES))
    ap.add_argument("--json", help="write the per-tap table here")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    batch, taps, meta = load_reference(args.ref)
    assert not meta["bfloat16"], "the reference arm must be the float32 one"
    feats = to_feats(batch, dtype)

    state = load_af2_state_dict(args.npz, multimer=True)
    model = load_af2_model(state, multimer=True, trunk_dtype=dtype,
                           num_evoformer_blocks=meta["blocks"],
                           num_extra_msa_blocks=meta["extra_blocks"])
    if dtype == torch.float64:
        model.double()

    captured: dict = {}

    def capture(tag):
        def hook(module, inputs, output):
            captured.setdefault(tag, []).append(output)
        return hook

    for index, block in enumerate(model.extra_msa):
        block.register_forward_hook(capture(f"extra_msa_stack#{index}"))
    for index, block in enumerate(model.evoformer):
        block.register_forward_hook(capture(f"evoformer_iteration#{index}"))
    for index, block in enumerate(model.template.pair_stack):
        block.register_forward_hook(capture(f"template_embedding_iteration#{index}"))
    model.template.register_forward_hook(capture("template_embedding#0"))

    prev = {"prev_pos": feats["prev_pos"], "prev_pair": feats["prev_pair"],
            "prev_msa_first_row": feats["prev_msa_first_row"]}
    with torch.no_grad():
        out = model(feats, prev)

    rows = []
    for index in range(meta["extra_blocks"]):
        tag = f"extra_msa_stack#{index}"
        rows.append(report(f"{tag}/pair", captured[tag][0][1], taps[f"{tag}/pair"]))
    for index in range(meta["blocks"]):
        tag = f"evoformer_iteration#{index}"
        msa, pair = captured[tag][0]
        rows.append(report(f"{tag}/msa", msa, taps[f"{tag}/msa"]))
        rows.append(report(f"{tag}/pair", pair, taps[f"{tag}/pair"]))
    for index in range(2):
        tag = f"template_embedding_iteration#{index}"
        rows.append(report(f"{tag}/act", captured[tag][0], taps[f"{tag}/out"]))
    rows.append(report("template_embedding/out", captured["template_embedding#0"][0],
                       taps["template_embedding#0/out"]))
    for key in ("single", "pair", "msa", "msa_first_row"):
        rows.append(report(f"trunk/{key}", out[key], taps[f"out/{key}"]))

    width = max(len(r["tap"]) for r in rows)
    print(f"{'tap':<{width}}  {'max_abs':>10} {'rel_max':>9} {'rms':>10} {'pcc':>10} {'cos':>10}")
    for r in rows:
        keep = (not r["tap"].startswith("evoformer_iteration")
                or int(r["tap"].split("#")[1].split("/")[0]) in (0, 1, 2, 23, 46, 47))
        if keep:
            print(f"{r['tap']:<{width}}  {r['max_abs']:10.3e} {r['rel_max']:9.2e} "
                  f"{r['rms']:10.3e} {r['pcc']:10.7f} {r['cos']:10.7f}")
    worst = max(rows, key=lambda r: r["rel_max"])
    print(f"\nworst relative error: {worst['tap']} {worst['rel_max']:.3e}")
    print(f"lowest pcc:           {min(rows, key=lambda r: r['pcc'])['tap']} "
          f"{min(r['pcc'] for r in rows):.9f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"dtype": args.dtype, "reference": Path(args.ref).name, "meta": meta, "taps": rows},
            indent=1) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
