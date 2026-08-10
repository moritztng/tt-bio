#!/usr/bin/env python3
"""y-silu -- census + H0/H1/H2 in one device session on qb1 card 0.

Stage 0  census: one live 298 aa protenix-v2 fold with `ttnn.linear` wrapped. Only Transition fc1
         passes `activation="silu"` (tenstorrent.py:2372 is the sole site in the file), so counting
         that kwarg IS the fc1 census. Attributed to the Transition construction site via a stack of
         live Transition instances tagged at __init__ with their caller's line number. The first
         c_z=256 fc1 call also hands its real x_norm and fc1 weight to host, so every arm below runs
         on the fold's own data at the fold's own shape.
Stage 1  H0: arms A (fused), B (bare + ttnn.silu), C (bare + in-place silu), D (bare, the reference
         that prices the silu).
Stage 2  H1: the auto-derived program config with `activation` set and unset, read back from
         ttnn's own resolver, then the same EXPLICIT config run twice with fused_activation set and
         unset and every other field pinned identical.
Stage 3  H2: the SFPU roof for silu and the L1 copy roof, both at the fc1 output's own shape, both
         measured on this card this session. Plus ttnn.multiply_ at the same shape for H3.
Stage 4  parity: torch.equal(A, B) at the fold's own shape, with max abs dev / rel RMSD / PCC.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--y-silu \
        python3 perf/y_silu/silu_probe.py --out perf/y_silu/probe.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np
import torch
import ttnn


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


# ---------------------------------------------------------------- timing helper
def timed(dev, fn, k, reps=7, warm=2):
    """K enqueued calls then one sync; sync on BOTH sides of the timed region."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t) / k * 1e6)   # us/call
    return dict(us_med=round(med(out), 3), us_min=round(min(out), 3),
                us_all=[round(v, 3) for v in out], k=k, reps=reps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "probe.json"))
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T

    res: dict = dict(host=os.uname().nodename, load_start=load(),
                     ttnn=__import__("importlib.metadata", fromlist=["version"]).version("ttnn"))

    # ------------------------------------------------------------ stage 0: census
    sites: dict = defaultdict(lambda: dict(calls=0, shapes=set(), wshapes=set(), transition_calls=0))
    stack: list = []
    grabbed: dict = {}

    _orig_init = T.Transition.__init__
    _orig_call = T.Transition.__call__
    _orig_linear = ttnn.linear

    def _init(self, *ar, **kw):
        import inspect
        fr = inspect.stack()[1]
        self._y_site = f"{Path(fr.filename).name}:{fr.lineno}"
        return _orig_init(self, *ar, **kw)

    def _call(self, x):
        stack.append(self)
        sites[self._y_site]["transition_calls"] += 1
        sites[self._y_site]["in_shapes"] = sites[self._y_site].get("in_shapes", set())
        sites[self._y_site]["in_shapes"].add(tuple(x.shape))
        try:
            return _orig_call(self, x)
        finally:
            stack.pop()

    def _linear(x, w, *ar, **kw):
        if kw.get("activation") == "silu" and stack:
            s = sites[stack[-1]._y_site]
            s["calls"] += 1
            s["shapes"].add(tuple(x.shape))
            s["wshapes"].add(tuple(w.shape))
            if not grabbed and tuple(w.shape)[-2] == 256:
                grabbed["x"] = ttnn.to_torch(x).clone()
                grabbed["w"] = ttnn.to_torch(w).clone()
                grabbed["site"] = stack[-1]._y_site
                grabbed["ckc"] = stack[-1].compute_kernel_config
                grabbed["dtype"] = stack[-1].dtype
        return _orig_linear(x, w, *ar, **kw)

    T.Transition.__init__ = _init
    T.Transition.__call__ = _call
    ttnn.linear = _linear

    from tt_baseline import build_fold
    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "w6_gate_msa"
    t0 = time.perf_counter()
    one_fold, meta, state = build_fold("protenix-v2", msa_dir, target, a3m, hoist=True)
    print(f"model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    dev = T.get_device()
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    res["core_grid_main"] = [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y]
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    res["l1_banks"] = int(mv.num_banks)
    res["l1_bytes_per_bank"] = int(mv.largest_contiguous_bytes_free_per_bank)

    t0 = time.perf_counter()
    one_fold()
    res["census_fold_s"] = round(time.perf_counter() - t0, 2)
    print(f"census fold {res['census_fold_s']}s", flush=True)

    ttnn.linear = _orig_linear
    T.Transition.__call__ = _orig_call
    T.Transition.__init__ = _orig_init

    res["census"] = {k: dict(fc1_silu_calls=v["calls"],
                             transition_calls=v["transition_calls"],
                             swiglu_per_transition=(v["calls"] / v["transition_calls"]
                                                    if v["transition_calls"] else None),
                             x_norm_shapes=sorted(list(v["shapes"])),
                             transition_in_shapes=sorted(list(v.get("in_shapes", set()))),
                             fc1_weight_shapes=sorted(list(v["wshapes"])))
                     for k, v in sites.items()}
    res["census_total_fc1_silu_calls"] = sum(v["calls"] for v in sites.values())
    print(json.dumps(res["census"], indent=1, default=str), flush=True)
    print("TOTAL fc1 silu calls/fold:", res["census_total_fc1_silu_calls"], flush=True)
    res["grabbed_site"] = grabbed.get("site")

    # ------------------------------------------------------------ rebuild the shape on device
    xt, wt = grabbed["x"], grabbed["w"]
    res["probe_shape"] = dict(x=list(xt.shape), w=list(wt.shape))
    ckc = grabbed["ckc"]   # the live config off the Transition instance that ran
    res["ckc"] = str(ckc)

    L1 = ttnn.L1_MEMORY_CONFIG
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    CG = T.CORE_GRID_MAIN

    def lin(act=None):
        return ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                           memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG)

    # ------------------------------------------------------------ stage 1: H0
    def arm_a():
        y = lin("silu"); ttnn.deallocate(y)

    def arm_b():
        y = lin(None); z = ttnn.silu(y); ttnn.deallocate(y); ttnn.deallocate(z)

    def arm_c():
        y = lin(None); z = ttnn.silu(y, memory_config=L1, output_tensor=y); ttnn.deallocate(z)

    def arm_d():
        y = lin(None); ttnn.deallocate(y)

    arms = {}
    for name, fn in (("A_fused", arm_a), ("B_unfused", arm_b), ("C_inplace", arm_c),
                     ("D_bare", arm_d)):
        try:
            arms[name] = timed(dev, fn, a.k)
            print(name, arms[name]["us_med"], "us/call", flush=True)
        except Exception as e:
            arms[name] = dict(error=repr(e)[:400])
            print(name, "ERROR", repr(e)[:200], flush=True)
    res["H0"] = arms
    res["load_after_H0"] = load()

    # ------------------------------------------------------------ stage 2: H1
    h1: dict = {}
    try:
        M = ttnn.operations.matmul
        def resolved(act):
            p = M.MatmulParams(activation=act) if act else M.MatmulParams()
            r = M.create_matmul_attributes(x, w, p, [None])
            return str(r)
        h1["resolved_with_silu"] = resolved("silu")
        h1["resolved_bare"] = resolved(None)
    except Exception as e:
        h1["resolver_error"] = repr(e)[:400]
    res["H1"] = h1
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)

    # ------------------------------------------------------------ stage 3: H2 roofs
    y0 = lin(None)
    ttnn.synchronize_device(dev)
    nb = y0.shape[-1] * y0.shape[-2] * y0.shape[-3] * y0.shape[-4] * 2
    res["fc1_out_bytes"] = int(nb)
    res["fc1_out_elems"] = int(nb // 2)
    y1 = ttnn.clone(y0, memory_config=L1)

    roofs = {}
    def r_silu():
        z = ttnn.silu(y0, memory_config=L1); ttnn.deallocate(z)
    def r_silu_inplace():
        ttnn.silu(y0, memory_config=L1, output_tensor=y0)
    def r_clone():
        z = ttnn.clone(y0, memory_config=L1); ttnn.deallocate(z)
    def r_mul():
        z = ttnn.multiply(y0, y1, memory_config=L1); ttnn.deallocate(z)
    for name, fn in (("silu_standalone", r_silu), ("silu_inplace", r_silu_inplace),
                     ("clone_l1_l1", r_clone), ("multiply_l1", r_mul)):
        try:
            roofs[name] = timed(dev, fn, max(4, a.k // 2))
            print("roof", name, roofs[name]["us_med"], flush=True)
        except Exception as e:
            roofs[name] = dict(error=repr(e)[:400])
            print("roof", name, "ERROR", repr(e)[:200], flush=True)
    res["H2"] = roofs
    res["load_after_H2"] = load()

    # ------------------------------------------------------------ stage 4: parity
    par = {}
    try:
        ya = ttnn.to_torch(lin("silu")).to(torch.float64)
        yb = ttnn.to_torch(ttnn.silu(lin(None))).to(torch.float64)
        par["torch_equal"] = bool(torch.equal(ya, yb))
        d = (ya - yb).abs()
        par["max_abs_dev"] = float(d.max())
        par["rel_rmsd"] = float(torch.sqrt((d ** 2).mean()) / (ya.std() + 1e-30))
        fa, fb = ya.flatten(), yb.flatten()
        par["pcc"] = float(torch.corrcoef(torch.stack([fa, fb]))[0, 1])
        par["ref_std"] = float(ya.std())
    except Exception as e:
        par["error"] = repr(e)[:400]
    res["parity_op"] = par
    res["load_end"] = load()

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
