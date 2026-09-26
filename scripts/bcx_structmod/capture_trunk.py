"""Capture the structure module's REAL inputs out of a BindCraft 2 design step.

    /home/ttuser/bcx_e2e_venv/bin/python3 scripts/bcx_structmod/capture_trunk.py \
        --binder-length 146 --seed 100 --out perf/bcx_structmod/out/trunk_b146_s100.npz

Why this exists. `ref_float64.py`'s synthetic `single` and `pair` are standard normal. They have
the right scale -- both go straight into a LayerNorm, which removes it -- but not the channel
correlation of a trunk output, and the difference is not cosmetic: on the synthetic input the
IPA's three logit terms come out at rms 5587 (scalar), 68.7 (point) and 2.2 (pair), the scalar
term 80x the geometry the sqrt(1/3) is there to balance it against, and the attention lands at
median row max 0.97. A softmax that saturated grades tie-breaks, not arithmetic.

How. `folding_multimer.StructureModule.__call__` is wrapped with a `jax.debug.callback`, which
is the one way to get a CONCRETE array out of a jitted design step without disabling jit and
without touching BindCraft 2's checkout. The callback writes the npz and then ends the process:
the capture is the deliverable, the rest of the trajectory is not, and a campaign left running
to its own end would hold a card and an hour for nothing.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))

import bc2_state as B                                                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binder-length", type=int, default=146)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--bucket", type=int, default=0,
                    help="length_bucket_size; 0 leaves BindCraft 2's own default alone")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--skip", type=int, default=0,
                    help="capture the Nth structure-module call rather than the first; the "
                         "first is recycle 0 of the first design model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    project = out.parent / f"capture_{out.stem}"

    import jax
    from bindcraft import campaign
    from bindcraft.af.alphafold.model import folding_multimer

    seen = [0]
    original = folding_multimer.StructureModule.__call__

    def save(single, pair, aatype, seq_mask):
        seen[0] += 1
        if seen[0] <= args.skip:
            print(f"structure-module call {seen[0]}, skipping", flush=True)
            return
        single = np.asarray(single)
        pair = np.asarray(pair)
        # A design step folds five models. If BindCraft 2 vmaps them the callback sees the
        # batch axis, and one model's representation is what the structure module consumes.
        while single.ndim > 2:
            single, pair = single[0], pair[0]
            aatype, seq_mask = np.asarray(aatype)[0], np.asarray(seq_mask)[0]
        np.savez(out, single=single.astype(np.float64), pair=pair.astype(np.float64),
                 aatype=np.asarray(aatype).astype(np.int32),
                 seq_mask=np.asarray(seq_mask).astype(np.float64))
        n = single.shape[0]
        print(f"captured call {seen[0]}: n={n} n_real={int(np.sum(seq_mask))} "
              f"single{single.shape} pair{pair.shape} -> {out}", flush=True)
        print(f"  single rms={float(np.sqrt((single ** 2).mean())):.4f} "
              f"pair rms={float(np.sqrt((pair ** 2).mean())):.4f}", flush=True)
        sys.stdout.flush()
        # The capture is the deliverable. Nothing after this point is wanted, and unwinding a
        # jitted trajectory from inside a host callback is not a thing worth engineering.
        os._exit(0)

    def hooked(self, representations, batch, safe_key=None):
        print(f"  [trace] structure module reached, single={representations['single'].shape}",
              flush=True)
        jax.debug.callback(save, representations["single"], representations["pair"],
                           batch["aatype"], batch["seq_mask"])
        return original(self, representations, batch, safe_key)

    folding_multimer.StructureModule.__call__ = hooked

    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 f"project_folder={project}"]
    if args.binder_length:
        overrides.append(f"binder_lengths=[{args.binder_length}]")
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    settings = B.campaign_settings(args.settings, overrides)

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    print("running one design trajectory until the structure module is reached", flush=True)
    campaign.run_campaign(settings, str(project),
                          af2_weights=args.params, mpnn_weights=mpnn, max_trajectories=1)
    print("the trajectory finished without reaching the structure module", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
