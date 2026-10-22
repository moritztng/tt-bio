"""tt-bio side of the pairing bisect: dump every seam before the first trunk pass.

Writes the preprocessed batch, the emit masks, the groups and ``x_0`` (the initial noise, drawn
with the global stream reseeded exactly where upstream's ``full_simulation`` reseeds it) to one
``.pt``. `pair_compare.py` diffs it against upstream's step-0 capture.

No model and no trunk: this seam is upstream of both.

    PYTHONPATH=<worktree>:<cli-lib> /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/pair_tt_dump.py --spec <spec.yaml> --out <dir>
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="/home/moritz/.coworker/state/rfd4/inputs/pdl1-128.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/moritz/rfd4-pair-runs/tt/seed0")
    ap.add_argument("--work", default="/home/moritz/rfd4-pair-runs/tt/inputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import sampler as S
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

    designs = rfd4.prepare_examples(a.spec, a.work)
    features, meta = rfd4.featurize(designs[0].cif, seed=a.seed)
    batch = E.preprocess(features)
    n_tokens = batch["padding_mask"].shape[1]

    spec = S.recycle_strategy("thinking_medium")
    emit_masks = compute_flow_emit_masks(batch)
    groups = E._groups(batch)
    rng = S.DesignRng(a.seed, 1, mode="upstream")
    torch.manual_seed(a.seed)
    with torch.no_grad():
        x_0 = S.sample_noise(spec, n_tokens, rng, emit_masks=emit_masks,
                             coord_cond_mask=batch.get("coord_cond_mask"), groups=groups,
                             training=False)
        depths = S.recycle_depths(spec, spec.nsteps)
        ts, gt = S.sample_schedule(spec, spec.nsteps)

    torch.save({"batch": {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)},
                "batch_keys": sorted(batch.keys()),
                "emit_masks": emit_masks, "groups": groups, "x_0": x_0,
                "data_modes": list(spec.modalities.keys()),
                "ts": ts, "gt": gt, "depths": list(depths), "n_tokens": n_tokens},
               f"{a.out}/seam0.pt")
    print(f"tokens={n_tokens} modes={list(spec.modalities)} depths[:5]={list(depths)[:5]}")
    for mode, v in x_0.items():
        print(f"x_0[{mode}] {tuple(v.shape)} sum={float(v.double().sum()):.6f} "
              f"absmean={float(v.double().abs().mean()):.6f}")


if __name__ == "__main__":
    main()
