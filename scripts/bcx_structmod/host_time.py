"""Host seconds for the same module, measured on the same box under the same load.

The brief prices the structure module at 1.409 s of host per round at n=211 and roughly 2.4 s
at n=288, from `bcx-tmplseam`'s re-attribution. That number was taken on another host at
another load, and this row's device number is taken at loadavg ~21 on a dispatch-bound module,
so subtracting one from the other is the exact mistake `bcx-extrawire` made. This measures the
host arm HERE: BindCraft 2's own `folding_multimer.StructureModule`, jitted, float32, same
n=288 capture, same fold-boundary cotangent, same box, minutes apart from the device run.

It is still not the round A/B the brief asks for -- that needs the swapped program, both arms
interleaved in one process -- but it is a host-removed number that at least shares a host with
the device-added one.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="a --boundary fold reference dump")
    ap.add_argument("--bc2", default="/home/ttuser/bcx_e2e/bc2")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.bc2)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jax
    import jax.numpy as jnp
    import haiku as hk
    from ref_float64 import _haiku_params
    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer, prng

    ref = np.load(args.ref)
    single = np.asarray(ref["single"], np.float32)
    pair = np.asarray(ref["pair"], np.float32)
    aatype = np.asarray(ref["aatype"], np.int32)
    seq_mask = np.asarray(ref["seq_mask"], np.float32)
    ct_act = np.asarray(ref["ct_act"], np.float32)
    ct_traj = np.asarray(ref["ct_traj"], np.float32)
    ct_unnorm = np.asarray(ref["ct_unnormalized"], np.float32)
    n = single.shape[0]

    cfg = af_config.model_config("model_1_multimer_v3")
    sm_config, gconfig = cfg.model.heads.structure_module, cfg.model.global_config
    with gconfig.unlocked():
        gconfig.bfloat16 = False
        gconfig.bfloat16_output = False
    params = _haiku_params(args.params, np.float32)

    def forward(single, pair):
        module = folding_multimer.StructureModule(sm_config, gconfig)
        return module(representations={"single": single, "pair": pair},
                      batch={"aatype": jnp.asarray(aatype), "seq_mask": seq_mask,
                             "use_dropout": False},
                      safe_key=prng.SafeKey(jax.random.PRNGKey(0)))

    def scalar(single, pair):
        o = forward(single, pair)
        return (jnp.sum(o["act"] * ct_act) + jnp.sum(o["traj"] * ct_traj)
                + jnp.sum(o["sidechains"]["unnormalized_angles_sin_cos"] * ct_unnorm))

    fnet = hk.without_apply_rng(hk.transform(forward))
    gnet = hk.without_apply_rng(hk.transform(scalar))
    fwd = jax.jit(lambda s, z: fnet.apply(params, s, z))
    both = jax.jit(jax.value_and_grad(lambda s, z: gnet.apply(params, s, z), argnums=(0, 1)))

    s, z = jnp.asarray(single), jnp.asarray(pair)

    def timed(fn, reps):
        times, loads = [], []
        for _ in range(reps):
            t0 = time.time()
            out = fn()
            jax.block_until_ready(out)
            times.append(time.time() - t0)
            loads.append(os.getloadavg()[0])
        return times, loads

    timed(lambda: fwd(s, z), args.warm)
    timed(lambda: both(s, z), args.warm)
    f_times, f_loads = timed(lambda: fwd(s, z), args.reps)
    b_times, b_loads = timed(lambda: both(s, z), args.reps)

    report = {
        "n": n, "host": os.uname().nodename, "reps": args.reps, "dtype": "float32",
        "threads": os.environ.get("XLA_FLAGS", ""),
        "forward": {"median_s": statistics.median(f_times), "min_s": min(f_times),
                    "max_s": max(f_times), "times_s": [round(t, 4) for t in f_times],
                    "load1_median": statistics.median(f_loads)},
        "forward_and_backward": {"median_s": statistics.median(b_times), "min_s": min(b_times),
                                 "max_s": max(b_times),
                                 "times_s": [round(t, 4) for t in b_times],
                                 "load1_median": statistics.median(b_loads)},
    }
    report["backward_implied_s"] = (report["forward_and_backward"]["median_s"]
                                    - report["forward"]["median_s"])
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
