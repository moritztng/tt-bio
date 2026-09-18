#!/usr/bin/env python3
"""Gradcheck for the ATTACH POINT: the gradient of the shipped `tt_bio.ops` call sites.

`perf/hallgrad/gradcheck.py` checks `tt_bio.autograd`'s ops directly. What it cannot check
is what A1 actually builds -- that a call written the way `protenix.py` and `tenstorrent.py`
write it, carrying that site's compute kernel config, core grid, epsilon and fused
activation, produces a correct gradient once the tape is installed under it.

Evidence in the same order, and the bars, the case builders, the metrics and the
finite-difference reference check are all imported from that harness rather than restated:

1. The float64 reference is torch autograd, checked against central differences in float64
   before any device number is compared to it. Reference suspect -> the case fails.
2. The device gradient against that reference, inputs rounded to the device dtype first and
   the reference fed the rounded values, and backward seeded with a random tensor, not ones:
   a sum seed cannot see a wrong reduction axis.
3. Controls. `--break-silu` gates silu's backward on its output, which is legitimate for
   relu and wrong here, and must fail. And grad-off with the hook INSTALLED must be
   bit-exact to the shipped op, which is R1 inside a taped module.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("_gc", REPO / "perf" / "hallgrad" / "gradcheck.py")
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)

EPS_PROD = 1e-5          # every tt-bio layer norm on the inference path
FD_BAR = 2e-6            # gradcheck.py's own reference bar


def build(name, rng):
    if name == "layernorm":
        return GC.case_layernorm(rng)
    t = GC.case_linear(rng)
    if name == "linear_nobias":
        t.pop("b")
    return t


def ref_forward(name, t):
    if name == "layernorm":
        mu = t["x"].mean(-1, keepdim=True)
        xc = t["x"] - mu
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + EPS_PROD) * t["gamma"] + t["beta"]
    y = t["x"] @ t["w"]
    if "b" in t:
        y = y + t["b"]
    if name == "linear_silu":
        return y * torch.sigmoid(y)
    if name == "linear_relu":
        return torch.relu(y)
    if name == "linear_sigmoid":
        return torch.sigmoid(y)
    return y


def tt_forward(name, ops, t, ckc, core_grid):
    """The call written the way a shipped module writes it."""
    if name == "layernorm":
        return ops.layer_norm(t["x"], t["gamma"], t["beta"], epsilon=EPS_PROD,
                              compute_kernel_config=ckc)
    if name == "linear_nobias":
        return ops.linear(t["x"], t["w"], compute_kernel_config=ckc, core_grid=core_grid,
                          narrow_proj=True)
    act = name.split("_")[1] if "_" in name else None
    return ops.linear(t["x"], t["w"], t["b"], activation=act,
                      compute_kernel_config=ckc, core_grid=core_grid)


def run_case(name, ttnn, ag, ops, tt, dt, torch_dt, ckc, seed):
    rng = np.random.default_rng(seed)
    raw = build(name, rng)
    rounded = {k: torch.from_numpy(v).to(torch_dt).to(torch.float64) for k, v in raw.items()}
    ref = {k: v.clone().requires_grad_(True) for k, v in rounded.items()}
    wt = torch.from_numpy(rng.standard_normal(tuple(ref_forward(name, ref).shape)))

    def loss_fn(_r=ref):
        return (ref_forward(name, _r) * wt).sum()

    fd_worst, n_probed, n_elig = GC.fd_check(loss_fn, list(ref.values()), n_probe=24, seed=seed)
    res = {"case": name, "fd_worst": float(fd_worst), "fd_probed": n_probed,
           "fd_eligible": n_elig, "fd_ok": bool(fd_worst < FD_BAR), "grads": {}}
    if not res["fd_ok"]:
        res["pass"] = False
        return res
    ref_grads = {k: v.grad.detach().numpy().copy() for k, v in ref.items()}

    dev = {k: ag.Tensor(ttnn.from_torch(v.to(torch_dt), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                        device=tt.get_device()), requires_grad=True)
           for k, v in rounded.items()}
    ag.install()
    try:
        out = tt_forward(name, ops, dev, ckc, tt.CORE_GRID_MAIN)
        if not isinstance(out, ag.Tensor):
            res["pass"] = False
            res["error"] = "the hook declined a call whose operands are on the tape"
            return res
        out.backward(seed=ttnn.from_torch(wt.to(torch_dt), dtype=dt,
                                          layout=ttnn.TILE_LAYOUT, device=tt.get_device()))
        got = {k: ttnn.to_torch(v.grad).to(torch.float64).numpy() if v.grad is not None else None
               for k, v in dev.items()}
    finally:
        ag.uninstall()

    ok = True
    for k, r in ref_grads.items():
        if got[k] is None:
            res["grads"][k] = {"pass": False, "error": "no gradient reached this input"}
            ok = False
            continue
        m = GC.metrics(got[k].reshape(r.shape), r)
        m["pass"] = bool(m["rel_l2"] <= GC.REL_L2_BAR and m["cos"] >= GC.COS_BAR)
        ok = ok and m["pass"]
        res["grads"][k] = m
    res["pass"] = bool(ok)
    return res


def control_grad_off(ttnn, ag, ops, tt, dt, torch_dt, ckc, seed):
    """R1 inside a taped module. The hook is INSTALLED for all three, and all three must
    come back bit-identical to the shipped op: nothing on the tape, on the tape but frozen,
    and requiring a gradient but inside `no_grad`. `_GRAD_ENABLED` alone gives none of them
    -- it prunes the node after the caller has computed the forward."""
    rng = np.random.default_rng(seed)
    t = GC.case_linear(rng)
    raw = {k: ttnn.from_torch(torch.from_numpy(v).to(torch_dt), dtype=dt,
                              layout=ttnn.TILE_LAYOUT, device=tt.get_device())
           for k, v in t.items()}
    shipped = ttnn.to_torch(ops.shipped_linear(raw["x"], raw["w"], raw["b"],
                                               compute_kernel_config=ckc,
                                               core_grid=tt.CORE_GRID_MAIN))
    out = {}
    ag.install()
    try:
        a = ops.linear(raw["x"], raw["w"], raw["b"], compute_kernel_config=ckc,
                       core_grid=tt.CORE_GRID_MAIN)
        out["untaped_declined"] = not isinstance(a, ag.Tensor)
        out["untaped_bit_exact"] = bool(torch.equal(ttnn.to_torch(a), shipped))

        fz = {k: ag.Tensor(v, requires_grad=False) for k, v in raw.items()}
        b = ops.linear(fz["x"], fz["w"], fz["b"], compute_kernel_config=ckc,
                       core_grid=tt.CORE_GRID_MAIN)
        out["frozen_untaped"] = isinstance(b, ag.Tensor) and b.node is None
        out["frozen_bit_exact"] = bool(torch.equal(ttnn.to_torch(b.value), shipped))

        lv = {k: ag.Tensor(v, requires_grad=True) for k, v in raw.items()}
        with ag.no_grad():
            c = ops.linear(lv["x"], lv["w"], lv["b"], compute_kernel_config=ckc,
                           core_grid=tt.CORE_GRID_MAIN)
        out["nograd_untaped"] = c.node is None
        out["nograd_bit_exact"] = bool(torch.equal(ttnn.to_torch(c.value), shipped))

        d = ops.linear(lv["x"], lv["w"], lv["b"], compute_kernel_config=ckc,
                       core_grid=tt.CORE_GRID_MAIN)
        out["live_call_is_taped"] = d.node is not None
    finally:
        ag.uninstall()
    out["pass"] = all(bool(v) for v in out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="linear,linear_nobias,linear_silu,linear_relu,"
                                       "linear_sigmoid,layernorm")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fidelity", default="HiFi2", choices=["LoFi", "HiFi2", "HiFi4"])
    ap.add_argument("--break-silu", action="store_true",
                    help="negative control: silu's backward gated on its output, which is "
                         "legitimate for relu and wrong here. Must fail.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    from tt_bio import ops

    if args.break_silu:
        def broken(x):
            out_v = ttnn.multiply(x.value, ttnn.sigmoid(x.value))

            def mk():
                def bw(g):
                    x.add_grad(ttnn.multiply(g, ttnn.gtz(out_v)))
                return bw
            return ag._tape(out_v, [x], mk)
        ag._ACTIVATIONS["silu"] = broken

    dt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[args.dtype]
    torch_dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tt.get_device()
    # The kernel config a shipped module carries, not the tape's precise() default: the
    # point of the attach point is that the forward is the site's forward.
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, args.fidelity), math_approx_mode=False,
        fp32_dest_acc_en=(args.dtype == "float32"), packer_l1_acc=True)

    rows = [run_case(c, ttnn, ag, ops, tt, dt, torch_dt, ckc, args.seed)
            for c in args.cases.split(",") if c]
    ctrl = control_grad_off(ttnn, ag, ops, tt, dt, torch_dt, ckc, args.seed)
    rep = {"dtype": args.dtype, "fidelity": args.fidelity, "seed": args.seed,
           "break_silu": args.break_silu,
           "rel_l2_bar": GC.REL_L2_BAR, "cos_bar": GC.COS_BAR, "fd_bar": FD_BAR,
           "cases": rows, "control_grad_off": ctrl,
           "all_pass": bool(all(r["pass"] for r in rows) and ctrl["pass"])}
    print(json.dumps(rep, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
