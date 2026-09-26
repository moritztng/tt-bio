"""AF2's multimer structure module, jitted, at the captured n=288, plus a timer.

Shared by `hostcores.py` (the configuration ladder) and `loadcurve.py` (the co-tenant ladder)
so both arms build the same program from the same capture.

The capture `ref_n288.npz` carries `ct_act` but not `ct_traj` or `ct_unnormalized`, so those
two cotangents are seeded from a fixed PRNG. Under jit the compiled program and its cost depend
on the SHAPES, not the cotangent values, and every arm uses the same seeded pair. Do not read
an accuracy number off anything built here.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

REF = "/home/moritz/bcx_hostcut_art/ref_n288.npz"
BC2 = "/home/moritz/bcx_shipped/bc2"
PARAMS = "/home/moritz/bcx_shipped/af2_params/params_model_1_multimer_v3.npz"


def build(ref_path=REF, bc2=BC2, params_path=PARAMS, n_slice=None):
    """`(fwd, both, single, pair, n)`, both jitted and not yet traced.

    `n_slice` takes the leading n residues of the capture, which keeps the aatype
    distribution and the pair statistics of a real trunk output rather than inventing
    them. It is a size ladder for the cost model, not a second configuration of the
    round: the round only ever runs the bucketed n.
    """
    sys.path.insert(0, bc2)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "bcx_structmod"))
    import jax
    import jax.numpy as jnp
    import haiku as hk
    from ref_float64 import _haiku_params
    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer, prng

    ref = np.load(ref_path)
    single = np.asarray(ref["single"], np.float32)
    pair = np.asarray(ref["pair"], np.float32)
    aatype = np.asarray(ref["aatype"], np.int32)
    seq_mask = np.asarray(ref["seq_mask"], np.float32)
    ct_act = np.asarray(ref["ct_act"], np.float32)
    if n_slice:
        single, pair = single[:n_slice], pair[:n_slice, :n_slice]
        aatype, seq_mask = aatype[:n_slice], seq_mask[:n_slice]
        ct_act = ct_act[:n_slice]
    n = single.shape[0]

    rng = np.random.default_rng(0)
    ct_traj = rng.standard_normal((8, n, 3, 4), dtype=np.float32)
    ct_unnorm = rng.standard_normal((8, n, 7, 2), dtype=np.float32)

    cfg = af_config.model_config("model_1_multimer_v3")
    sm_config, gconfig = cfg.model.heads.structure_module, cfg.model.global_config
    with gconfig.unlocked():
        gconfig.bfloat16 = False
        gconfig.bfloat16_output = False
    params = _haiku_params(params_path, np.float32)

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
    return fwd, both, jnp.asarray(single), jnp.asarray(pair), n


def timed(fn, reps):
    """One row per rep: wall, this process's CPU across all its threads, loadavg1."""
    import jax
    rows = []
    for _ in range(reps):
        c0, w0 = time.process_time(), time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        w = time.perf_counter() - w0
        rows.append({"wall_s": w, "cpu_s": time.process_time() - c0,
                     "cores": (time.process_time() - c0) / w, "load1": os.getloadavg()[0]})
    return rows


def summary(rows):
    return {"median_wall_s": round(statistics.median(r["wall_s"] for r in rows), 4),
            "min_wall_s": round(min(r["wall_s"] for r in rows), 4),
            "max_wall_s": round(max(r["wall_s"] for r in rows), 4),
            "median_cpu_s": round(statistics.median(r["cpu_s"] for r in rows), 4),
            "median_cores": round(statistics.median(r["cores"] for r in rows), 2),
            "median_load1": round(statistics.median(r["load1"] for r in rows), 2),
            "rows": [{k: round(v, 4) for k, v in r.items()} for r in rows]}
