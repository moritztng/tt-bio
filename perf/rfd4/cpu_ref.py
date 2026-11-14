"""Upstream RFD4 reference designs on CPU, at the released recipe.

    python3 perf/rfd4/cpu_ref.py --seed 0 --n 30 --spec <spec.yaml> --out <dir>

The quality grade is TT designs against upstream's, and both arms must come off the SAME recipe:
the harness self-test arms in `state/rfd4/QUALITY.json` ran `nsteps=20`, a shortened recipe chosen
so two arms fit on a shared CPU box, so a full-recipe TT arm graded against them would compare two
different recipes. This runs upstream's released `thinking_medium` (60 steps, 180 trunk passes),
the same recipe `tt-bio design --model rfd4` runs, on the same input the TT arm used.

Upstream's `run_generation` hardcodes `accelerator="gpu"`, so the Trainer arguments are patched and
nothing else (the approach rfd4-ref's capture.py and the self-test's cpu_designs.py both take).
A B200 reference arm is owed by rfd4-fair-gpu; CPU fp32 is upstream's own implementation at
upstream's own recipe, and is the reference this row grades against until that arm lands.
"""
import argparse
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def force_cpu_fp32_trainer() -> None:
    import lightning as L

    base = L.Trainer

    class CPUTrainer(base):
        def __init__(self, *args, **kwargs):
            kwargs.update(accelerator="cpu", devices=1, num_nodes=1, precision="32-true")
            super().__init__(*args, **kwargs)

    L.Trainer = CPUTrainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--threads", type=int, default=6, help="pc is shared; do not take it all")
    ap.add_argument("--nsteps", type=int, default=None,
                    help="shorten the recipe (SELF-TEST ONLY: a shortened arm is not gradeable "
                         "against a full-recipe one)")
    a = ap.parse_args(argv)

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(a.threads)
    seed_everything(a.seed)
    import rfd4_proteina as rfd4

    model = rfd4.Proteina.from_pretrained("nvidia/RFD4-Proteina")
    force_cpu_fp32_trainer()
    seed_everything(a.seed)
    overrides = [
        f"inputs={a.spec}", f"out_dir={a.out}", f"seed={a.seed}",
        f"num_replicates={a.n}", "batch_size=1", "compile_lengths=null",
        "num_workers=0", "~metrics",
    ]
    if a.nsteps is not None:
        overrides += [f"generation.args.nsteps={a.nsteps}",
                      f"generation.n_recycle_schedule=[[{a.nsteps},1]]"]
    print(f"overrides: {overrides}", flush=True)
    rfd4.generate(model, overrides=overrides)
    print("DESIGNS-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
