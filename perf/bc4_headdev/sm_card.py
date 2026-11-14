#!/usr/bin/env python3
"""The whole structure module on card (tt_bio.af2_structure.Card, fp32 HiFi4), forward and taped
backward, graded against the same chain's float64 VJP in JAX (af2_structure.NP(jnp), itself 4e-12
from BindCraft 2, sm_check.py), on real BindCraft 2 trunk outputs (perf/bc4_ipa/dump_sm_inputs.py).
Cotangents are fixed random draws on what BindCraft 2 reads: the final act and the final frame.
Times eager forward, eager forward+backward and traced forward+backward, with in-run AICLK.

usage: sm_card.py --params <npz> --inputs <sm_inputs.npz> [--call 0] [--reps 10] [--out json]
"""
import argparse, json, os, sys, time

import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bc4_ipa"))


def host_card(S, base, ops):
    """`base` with the named unary ops run on the host in float64, forward and VJP (jax), for a bisect."""
    import jax, jax.numpy as jnp, ttnn, torch
    jo = S.NP(jnp)

    class HostCard(base):
        def _host(self, fn, x):
            xv = jnp.asarray(ttnn.to_torch(x.value).double().numpy())
            y, vjp = jax.vjp(fn, xv)
            out = self.raw(np.asarray(y))
            def make():
                def bw(g):
                    gx, = vjp(jnp.asarray(ttnn.to_torch(g).double().numpy()))
                    x.add_grad(self.raw(np.asarray(gx)))
                return bw
            return self.A._tape(out, [x], make)

    host = HostCard
    if "ln" in ops:
        host.layer_norm = lambda self, x, gb, eps=1e-5: self._host(
            lambda v: jo.layer_norm(v, tuple(jnp.asarray(ttnn.to_torch(t.value).double().numpy()) for t in gb), eps), x)
    if "softmax" in ops:
        host.softmax = lambda self, x: self._host(jo.softmax, x)
    if "sqrt" in ops:
        host.sqrt_eps = lambda self, x, eps: self._host(lambda v: jo.sqrt_eps(v, eps), x)
    if "recip" in ops:
        host.recip = lambda self, x: self._host(jo.recip, x)
    return host


def main(a):
    import jax, jax.numpy as jnp
    from tt_bio import af2_structure as S
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if k.startswith(S.ROOT)}
    k = S.pack(p)
    d = np.load(a.inputs)
    single, pair = (np.asarray(d[f"{nm}_{a.call}"], np.float64) for nm in ("single", "pair"))
    real = single.shape[0]
    n = -(-real // 32) * 32
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    single = np.pad(single, ((0, n - real), (0, 0)))
    pair = np.pad(pair, ((0, n - real), (0, n - real), (0, 0)))
    mu = single.mean(-1, keepdims=True); var = ((single - mu) ** 2).mean(-1, keepdims=True)
    act0 = ((single - mu) / np.sqrt(var + 1e-5) * p[S.ROOT + "single_layer_norm//scale"]
            + p[S.ROOT + "single_layer_norm//offset"]) @ p[S.ROOT + "initial_projection//weights"] \
        + p[S.ROOT + "initial_projection//bias"]
    rng = np.random.default_rng(7)
    seeds = [rng.standard_normal(s) * np.r_[np.ones(real), np.zeros(n - real)][:, None]
             for s in ((n, S.CS), (n, 9), (n, 3))]

    # float64 reference: the same chain, differentiated by JAX
    jo, kj = S.NP(jnp), {nm: (tuple(jnp.asarray(x) for x in v) if isinstance(v, tuple) else
                             jnp.asarray(v) if isinstance(v, np.ndarray) else v) for nm, v in k.items()}
    kb = jnp.asarray(S.key_bias(mask))
    def f(a0, z):
        act, traj = S.stack(jo, kj, a0, z, kb, layers=a.layers)
        return act, traj[-1][0], traj[-1][1]
    ref, vjp = jax.vjp(f, jnp.asarray(act0), jnp.asarray(pair))
    ref = [np.asarray(x) for x in ref]
    ref_g = [np.asarray(x) for x in vjp(tuple(jnp.asarray(s) for s in seeds))]

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as A, tenstorrent as T
    import ipa_fused_tt as IT
    dev = T.get_device(trace="diffusion")
    o = (host_card(S, S.Card, a.host_ops.split(",")) if a.host_ops else S.Card)(dev)
    kd = o.upload(k)
    pre = [o.raw(act0), o.raw(pair)]
    kbd, sd = o.put(S.key_bias(mask)), [o.raw(s) for s in seeds]

    def fwd():
        with A.no_grad():
            act, traj = S.stack(o, kd, A.Tensor(pre[0]), A.Tensor(pre[1]), kbd, layers=a.layers)
        return act, traj[-1][0], traj[-1][1]

    def step():
        ins = [A.Tensor(v, requires_grad=True) for v in pre]
        act, traj = S.stack(o, kd, *ins, kbd, layers=a.layers)
        A.backward([act, traj[-1][0], traj[-1][1]], seeds=sd)
        return [x.grad for x in ins]

    get = lambda x: ttnn.to_torch(x.value if hasattr(x, "value") else x).float().numpy().astype(np.float64)
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(x))
    out = [get(x) for x in fwd()]
    grads = [get(x) for x in step()]
    res = {"host_ops": a.host_ops, "layers": a.layers, "n": n, "real": real, "call": a.call,
           "fwd_rel_l2": {nm: rel(x[:real], y[:real]) for nm, x, y in zip(("act", "R", "t"), ref, out)},
           "ca_rmsd_A": float(np.sqrt(((10 * (ref[2] - out[2])[:real]) ** 2).sum(-1).mean())),
           "ca_max_A": float(np.abs(10 * (ref[2] - out[2])[:real]).max()),
           "bwd_rel_l2": {"d_act0": rel(ref_g[0][:real], grads[0][:real]),
                          "d_pair": rel(ref_g[1][:real, :real], grads[1][:real, :real])}}
    print(json.dumps(res), flush=True)
    if a.dump:
        np.savez(a.dump, ref_d_act0=ref_g[0], ref_d_pair=ref_g[1], d_act0=grads[0], d_pair=grads[1],
                 act0=act0, seed_act=seeds[0], seed_R=seeds[1], seed_t=seeds[2])
        if a.layers == 1:
            return

    def timed(fn, reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / reps * 1e3
    clock = IT.Clock().__enter__()
    res["eager_fwd_ms"] = timed(fwd, a.reps)
    res["eager_fwd_bwd_ms"] = timed(step, a.reps)
    for nm, fn in (("traced_fwd_ms", fwd), ("traced_fwd_bwd_ms", step)):
        try:
            fn(); ttnn.synchronize_device(dev)
            tid, _ = T.capture_trace(dev, fn)
            res[nm] = timed(lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False), a.reps)
            ttnn.release_trace(dev, tid)
        except Exception as e:                                       # noqa: BLE001
            res[nm] = f"capture failed: {str(e)[:300]}"
    clock.__exit__()
    res.update({"aiclk_median_by_chip": [clock.median(i) for i in range(len(clock.samples[0]))] if clock.samples else None,
                "loadavg": os.getloadavg(), "reps": a.reps, "visible": os.environ.get("TT_VISIBLE_DEVICES"),
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--host-ops", default="", help="bisect: comma list of ln,softmax,sqrt,recip run on host f64")
    ap.add_argument("--out")
    ap.add_argument("--dump", help="npz of both gradients, the input and the seeds")
    main(ap.parse_args())
