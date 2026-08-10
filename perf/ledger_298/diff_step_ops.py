#!/usr/bin/env python3
"""Per-op device time + shapes for ONE real diffusion denoise step, measured inside a real fold.

W1's `pf_block_ops.py` could build a Pairformer block standalone because the block's inputs are
just (s, z). The diffusion denoise cannot: `cond` is the trunk's output, so the only way to get
production shapes and production memory configs is to run the fold and instrument a step in place.

This runs the same fold `scripts/gpu_vs_tt/tt_baseline.py` times (298 aa, 10 recycles, 200 sampling
steps, 1 sample, trace=False), and patches `DiffusionModule.denoise` so that inside the warm fold:

  * `--time-steps` steps are timed with `synchronize_device` on both sides and nothing patched ->
    the denoise-step wall, the denominator every per-op sum is checked against;
  * exactly ONE later step runs with W1's op wrapper armed -> the per-op record.

Everything else passes straight through, so the fold's own timing is undisturbed outside those
steps. `edm_sample` is wrapped too, giving the diffusion stage wall on this card in the same
process. Step 0 is never used: `_atom_cond` and the DiT per-block pair biases are computed on the
first denoise call of a fold and cached for the other 199, so step 0 is not steady state.

The instrumented fold's output structure is garbage and is discarded: the op wrapper re-runs every
op `--reps` times, and the diffusion path contains in-place ops. This is a measurement run only.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$PWD \
      python3 perf/ledger_298/diff_step_ops.py --model protenix-v2 \
        --target examples/prot300.yaml --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m \
        --out perf/ledger_298/diffops_protenix-v2_298aa.json
"""
import argparse
import importlib.util
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402

_spec = importlib.util.spec_from_file_location("pf_block_ops", Path(__file__).with_name("pf_block_ops.py"))
PB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PB)

# Host<->device transfers are a real part of a denoise step (the decoder's r_update comes back to
# host every step), so unlike W1's block pass they are wrapped and carry their own rows.
PB.OPS = PB.OPS + ["from_torch", "to_torch"]

STEP_WALLS = []
STAGE = {"s": 0.0, "n": 0}
CNT = {"c": 0}
MARKS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200, help="sampling steps per fold (denoise calls)")
    ap.add_argument("--fold", type=int, default=1, help="0 = cold fold, 1 = first warm fold")
    ap.add_argument("--time-steps", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    ap.add_argument("--op-step", type=int, default=40)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--small-us", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    import tt_bio.protenix as P
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    orig_denoise = P.DiffusionModule.denoise
    orig_edm = P.edm_sample
    time_steps = set(args.time_steps)

    def edm(*a, **kw):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = orig_edm(*a, **kw)
        ttnn.synchronize_device(dev)
        STAGE["s"] += time.perf_counter() - t0
        STAGE["n"] += 1
        return out

    def denoise(self, x_noisy, t_hat, cond):
        c = CNT["c"]
        CNT["c"] = c + 1
        fold, step = divmod(c, args.steps)
        if fold != args.fold:
            return orig_denoise(self, x_noisy, t_hat, cond)
        if step in time_steps:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = orig_denoise(self, x_noisy, t_hat, cond)
            ttnn.synchronize_device(dev)
            STEP_WALLS.append(time.perf_counter() - t0)
            return out
        if step == args.op_step:
            saved = PB.patch()
            PB.STATE.update(dev=dev, reps=args.reps, small_us=args.small_us, on=True)
            try:
                out = orig_denoise(self, x_noisy, t_hat, cond)
                ttnn.synchronize_device(dev)
            finally:
                PB.STATE["on"] = False
                for ns, nm, fn in saved:
                    setattr(ns, nm, fn)
            print(f"instrumented step {step}: {len(PB.RECORDS)} ops", flush=True)
            return out
        return orig_denoise(self, x_noisy, t_hat, cond)

    P.DiffusionModule.denoise = denoise
    P.edm_sample = edm

    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    one_fold, meta, state = tb.build_fold(args.model, msa_dir, args.target, args.msa_a3m)
    cold_s, _m = one_fold()
    MARKS["cold_fold_s"] = round(cold_s, 3)
    MARKS["stage_cold_s"] = round(STAGE["s"], 4)
    STAGE["s"] = 0.0
    warm_s, _m = one_fold()
    MARKS["warm_fold_s"] = round(warm_s, 3)
    MARKS["stage_warm_s"] = round(STAGE["s"], 4)

    P.DiffusionModule.denoise = orig_denoise
    P.edm_sample = orig_edm

    step_wall = st.median(STEP_WALLS) if STEP_WALLS else 0.0
    tot = sum(r["s"] for r in PB.RECORDS)
    print(f"step wall = {step_wall * 1e3:.3f} ms (n={len(STEP_WALLS)}, "
          f"spread {min(STEP_WALLS) * 1e3:.3f}-{max(STEP_WALLS) * 1e3:.3f})", flush=True)
    print(f"diffusion stage warm = {MARKS['stage_warm_s']:.3f} s over {args.steps} steps "
          f"= {MARKS['stage_warm_s'] / args.steps * 1e3:.3f} ms/step", flush=True)
    print(f"ops={len(PB.RECORDS)}  sum={tot * 1e3:.3f} ms  coverage of step wall="
          f"{100 * tot / step_wall if step_wall else 0:.1f}%", flush=True)

    # Same key names pf_block_ops.py emits, so ledger_from_ops.py consumes this unchanged:
    # "block" here is one denoise step and calls_per_fold is the sampling-step count.
    json.dump({"model": args.model, "n": 320, "c_z": 0, "block_wall_s": step_wall,
               "reps": args.reps, "n_ops": len(PB.RECORDS), "sum_s": tot, "fatal": None,
               "step_walls_s": STEP_WALLS, "marks": MARKS, "steps_per_fold": args.steps,
               "stage_warm_s": MARKS["stage_warm_s"], "records": PB.RECORDS},
              open(args.out, "w"), indent=1)
    state.reset()


if __name__ == "__main__":
    main()
