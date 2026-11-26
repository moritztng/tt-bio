"""Build the committed RFD4 training fixture from upstream's own data pipeline.

Runs `rfproteina.datapipes.pipeline.build_design_train_pipeline` with the release pretraining
recipe's settings (crop 256, spatial crops, p_random_rotation 1.0, the three loss-weight
predicates, centroid conditioning) over upstream's published `test_dataset` (~50 PLINDER
systems, `scripts/download_assets.sh --with-test-dataset`), collates with upstream's own
`collate_batch` at `fixed_dimensions={I: 256}` and `num_replicates=exploration_k`, then keeps
only the arrays the objective and its parity harness read. The result is one ~MB-scale .npz
that `tt_bio.train.rfd4_data` loads with numpy alone, so the test suite needs neither upstream,
atomworks, nor the dataset.

    DATA_PATH=~/r4t/objective/data ~/rfd4-cpu-venv/bin/python perf/r4t/build_fixture.py \
        --out tt_bio/train/fixtures/rfd4_train_16.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.train.rfd4_data import FIXTURE_KEYS, LATENT_PLACEHOLDER  # noqa: E402

CROP = 256   # hero_pretrain.yaml `_max_tokens`; the tutorial experiment composes it

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16, help="examples in the fixture")
    ap.add_argument("--k", type=int, default=4, help="exploration_k replicates per example")
    ap.add_argument("--seed", type=int, default=145, help="hero_pretrain.yaml's own seed")
    a = ap.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from rfproteina.datapipes.collate import collate_batch

    repo = Path(os.environ["RFD4_REPO"]).expanduser()
    # Upstream's own tutorial experiment over its published test_dataset: it composes the
    # release pipeline defaults (crop 256, the loss-weight predicates, centroid conditioning)
    # and the plinder design-task preset, so nothing here restates a recipe number.
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="training_proteina",
                      overrides=["experiment=tutorial/finetune_test_dataset",
                                 "+single=true", "+nolog=true"])
    ds = instantiate(cfg.datasets.train.test_dataset.dataset)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    samples, used = [], []
    for i in range(len(ds)):
        if len(samples) == a.n:
            break
        try:
            s = ds[i]
        except Exception as e:                       # upstream's own pipeline rejects systems
            print(f"skip {i}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            continue
        samples.append(s)
        used.append(str(s["example_id"]))
    if len(samples) < a.n:
        raise SystemExit(f"only {len(samples)} of {a.n} examples survived the pipeline")

    batch = collate_batch(samples, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                          fixed_dimensions={"I": CROP}, keep_atom_array=False, num_replicates=a.k)
    feats = batch["feats"] if "feats" in batch else batch
    out = {}
    for key in FIXTURE_KEYS:
        v = feats.get(key)
        if v is None:
            raise SystemExit(f"upstream's collate produced no {key!r}; keys: {sorted(batch)}")
        out[key] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    out["example_ids"] = np.array(used)
    out["crop"] = np.int64(CROP)
    out["exploration_k"] = np.int64(a.k)
    out["latent_dim"] = np.int64(LATENT_PLACEHOLDER)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"{a.out}: {len(used)} examples x k={a.k} = {out['padding_mask'].shape[0]} rows, "
          f"{a.out.stat().st_size / 1e6:.2f} MB")
    print("tokens per row:", out["padding_mask"].sum(-1)[::a.k].tolist())


if __name__ == "__main__":
    main()
