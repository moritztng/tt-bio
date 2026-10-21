#!/usr/bin/env python3
"""Dump the structure module's real inputs from one BindCraft 2 forward on the host, for grading the card IPA.

Random single/pair make the layer ill-conditioned (NOTES "Pass 2"), so the card kernel is graded on
what the trunk actually hands the structure module: BindCraft 2's shipped PD-L1 draw, its own
`AlphaFoldDesignModel` on JAX (no card), `generate_monomer_rigids` wrapped to save single, pair and
seq_mask at every call (one per recycle) through `jax.debug.callback`.

    PYTHONPATH=.:<BindCraft 2 checkout> python perf/bc4_ipa/dump_sm_inputs.py --out sm_inputs.npz
"""
import argparse, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))


def main(a):
    import jax
    from bindcraft.af.alphafold.model import folding_multimer as fm
    saved = []
    orig = fm.generate_monomer_rigids

    def save(single, pair, mask):
        saved.append((np.asarray(single), np.asarray(pair), np.asarray(mask)))

    def wrapped(representations, batch, config, global_config, safe_key):
        jax.debug.callback(save, representations["single"], representations["pair"], batch["seq_mask"])
        return orig(representations, batch, config, global_config, safe_key)
    fm.generate_monomer_rigids = wrapped

    import test_bindcraft2_hw as hw
    from bindcraft.af2 import AlphaFoldDesignModel
    _, protein_states, _ = hw._pdl1_draw()
    model = AlphaFoldDesignModel(presets=a.preset, data_dir=a.params, max_cache_size=1,
                                 num_recycle=a.recycle, length_bucket_size=32, dropout=False)
    model.predict(protein_states)
    out = {}
    for i, (s, p, m) in enumerate(saved):
        out[f"single_{i}"], out[f"pair_{i}"], out[f"seq_mask_{i}"] = s, p, m
        print(i, s.shape, p.shape, float(m.sum()), flush=True)
    np.savez(a.out, **out)
    print("saved", a.out, len(saved), "calls")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--preset", default="model_1_multimer_v3")
    ap.add_argument("--recycle", type=int, default=1)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
