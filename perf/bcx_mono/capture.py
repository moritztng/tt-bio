#!/usr/bin/env python3
"""Capture BindCraft 2's own Evoformer boundary for a one-chain and a two-chain fold.

`pos_control.json` says the device arm agrees to 1.6% on the 211-residue complex and reads
0.534 against 0.950 on the 115-residue target alone. The splice replaces exactly one thing --
the 48 Evoformer blocks -- so whatever is wrong is a function of what those blocks are handed.
This dumps the handoff: `(msa, pair, msa_mask, pair_mask)` in and `(msa, pair)` out, per
recycle, from BindCraft 2's unmodified JAX, for every case. No device, no tt-bio code.
"""
import json, os, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "perf" / "bcx_predictor"), str(ROOT)):
    sys.path.insert(0, p)

import jax
import bc2_state as B
from splice import find_evoformer_masks
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.af2 import campaign_length_bucket

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
OUT = HERE / "capture"
OUT.mkdir(exist_ok=True)

s = B.campaign_settings()
bucket = campaign_length_bucket(s)
_, states, _ = B.design_state(s)
CASES = {
    "complex": states,
    "target_alone": {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k != "binder"}},
    "binder_alone": {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k == "binder"}},
}


def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=4, trunk="jax")


def split_plddt(pred, state="hPDL1"):
    sp = pred[state]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    out, i = {}, 0
    for name in sorted(sp.protein_complex):
        n = len(sp.protein_complex[name])
        out[name] = round(float(plddt[i:i + n].mean()), 4)
        i += n
    out["_all"] = round(float(plddt.mean()), 4)
    out["_ptm"] = round(float(sp.metrics["ptm"]), 4)
    return out


import contextlib
from bindcraft.af.alphafold.model import layer_stack as LS
from bindcraft.af.alphafold.model import modules


@contextlib.contextmanager
def tap(store):
    real = LS.layer_stack

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if getattr(fn, "__name__", None) != "evoformer_fn":
                return made(fn)
            masks = find_evoformer_masks(fn)
            store["mask_keys"] = sorted(masks)
            stacked = made(fn)

            def record(tag):
                def cb(msa, pair, mmask, pmask):
                    store.setdefault(tag, []).append(
                        {"msa": np.asarray(msa), "pair": np.asarray(pair),
                         "msa_mask": np.asarray(mmask), "pair_mask": np.asarray(pmask)})
                return cb

            def tapped(x):
                act, key = x
                jax.debug.callback(record("in"), act["msa"], act["pair"],
                                   masks["msa"], masks["pair"])
                out, key2 = stacked((act, key))
                jax.debug.callback(record("out"), out["msa"], out["pair"],
                                   masks["msa"], masks["pair"])
                return out, key2
            return tapped
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        yield store
    finally:
        modules.layer_stack.layer_stack = real


summary = {"bucket": bucket, "cases": {}}
for name, st in CASES.items():
    store = {}
    t0 = time.time()
    with tap(store):
        pred = model().predict(st)
    dt = round(time.time() - t0, 1)
    plddt = split_plddt(pred)
    last_in, last_out = store["in"][-1], store["out"][-1]
    np.savez_compressed(OUT / f"{name}.npz",
                        in_msa=last_in["msa"], in_pair=last_in["pair"],
                        msa_mask=last_in["msa_mask"], pair_mask=last_in["pair_mask"],
                        out_msa=last_out["msa"], out_pair=last_out["pair"])
    summary["cases"][name] = {
        "plddt_jax": plddt, "seconds": dt, "recycles_taped": len(store["in"]),
        "mask_keys": store["mask_keys"],
        "shapes": {k: list(np.shape(v)) for k, v in last_in.items()},
        "msa_mask_rows_all_ones": [bool((last_in["msa_mask"][r] == 1).all())
                                   for r in range(last_in["msa_mask"].shape[0])],
        "msa_mask_zero_frac": float((last_in["msa_mask"] == 0).mean()),
        "pair_mask_all_ones": bool((last_in["pair_mask"] == 1).all()),
        "pair_mask_zero_frac": float((last_in["pair_mask"] == 0).mean()),
        "dtypes": {k: str(np.asarray(v).dtype) for k, v in last_in.items()},
    }
    print(name, json.dumps(summary["cases"][name], indent=1), flush=True)

(HERE / "capture.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
