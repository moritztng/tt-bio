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

# --------------------------------------------------------------------------------------
# Bars, per op class rather than one number for everything.
#
# `gradcheck.py` ships a single REL_L2_BAR = 1.0e-2 derived from the bf16 mantissa, and it
# is the wrong shape twice over.
#
# 1. It is a SMOOTH-op bar. relu's gradient is a step, so the backward gates on a sign test
#    of the forward and a coordinate that crosses the kink contributes its whole gradient
#    rather than a rounded one. Measured here: every smooth case lands at 0.0024-0.0049 and
#    relu alone lands at 0.0625, and raising the math fidelity to HiFi4 only pulls it to
#    0.0128, still over. Whatever sets relu's error, it is not the mantissa, so a
#    mantissa-derived bar cannot score it. Kinked ops get a DIRECTION bar instead: cosine
#    similarity, which is what an optimiser consumes, plus the agreement of the gate mask
#    itself. That is the criterion, not a looser rel_L2.
# 2. It ignores op class. `train-b2-abb3-port` measured, on qb1 at 1350 MHz, that a matmul on
#    FP32 operands keeps only ~11 mantissa bits whatever the kernel config says -- 1.25e-03
#    relative at HiFi4 with fp32_dest_acc_en, 7.05e-03 at HiFi2, 2.85e-02 at LoFi -- while
#    eltwise is fp32-exact at 3.0e-07. Its K ladder is flat (1.04e-03 at K=1, 1.48e-03 at
#    K=512) and K=1 accumulates nothing, so it is input rounding and no accumulator setting
#    fixes it. `ttnn.sum` carries 1.2e-02 over 96 fp32 terms, so reductions round like
#    matmuls. An fp32 arm therefore does NOT collapse a reduction-carrying gradient to
#    machine epsilon, and scoring one against 1.0e-2 while scoring a pure-eltwise gradient
#    against the same number wastes four orders of magnitude of signal on the second.
#
# Every bar below is the measured floor for its class times a headroom factor, stated.
# --------------------------------------------------------------------------------------
BARS = {
    # class      dtype        rel_l2     cos        floor it comes from
    ("eltwise", "float32"):   (3.0e-06,  0.9999990, "10x the measured fp32 eltwise floor 3.0e-07"),
    ("eltwise", "bfloat16"):  (1.0e-02,  0.9999,    "the bf16 mantissa, sqrt(2)*2^-9 = 2.76e-03"),
    ("reduction", "float32"): (2.5e-02,  0.9999,    "3.5x the measured fp32 HiFi2 matmul floor "
                                                    "7.05e-03; fp32 does not mean exact here"),
    ("reduction", "bfloat16"):(1.0e-02,  0.9999,    "the bf16 mantissa, sqrt(2)*2^-9 = 2.76e-03"),
    # A kinked op is not scored against the true reference at all. See KINK below.
    ("kink", "float32"):      (None,     None,      "not scored; split into MASK_BAR and the "
                                                    "device-gate arm, see KINK"),
    ("kink", "bfloat16"):     (None,     None,      "not scored; split into MASK_BAR and the "
                                                    "device-gate arm, see KINK"),
}

# KINK. A discontinuous gradient asks two questions and one number cannot answer both, so the
# criterion splits them and scores each.
#
#   1. Did the forward land on the SAME SIDE of the kink? -> MASK_BAR, the fraction of gate
#      coordinates on which the device and the float64 reference agree. This is the only part
#      that is genuinely about the kink, and it is a property of the FORWARD.
#   2. Is the backward's ARITHMETIC right? -> score the device gradient against a float64
#      reference recomputed with the DEVICE'S OWN gate, on the ordinary reduction bar. Not
#      circular: the gate itself is scored separately by (1), so a wrong gate cannot hide here
#      and a wrong backward cannot hide behind a wrong gate.
#
# Cosine was the obvious third option and it is not independent. For an error made of a few
# isolated large deviations, cos ~= 1 - rel_L2^2/2, and the measurement confirms it to three
# digits: rel_L2 0.0625 against cos 0.99805, and 1 - 0.0625^2/2 = 0.99805. A cos bar on a
# kinked op is a restatement of the rel_L2 bar it was meant to replace.
MASK_BAR = 0.99

# Which class each gradient falls in. The backward of a broadcast contains a reduction, so
# "eltwise" means eltwise ALL THE WAY DOWN, not an eltwise forward.
CLASS = {
    ("linear", "x"): "reduction", ("linear", "w"): "reduction", ("linear", "b"): "reduction",
    ("linear_nobias", "x"): "reduction", ("linear_nobias", "w"): "reduction",
    ("linear_silu", "x"): "reduction", ("linear_silu", "w"): "reduction",
    ("linear_silu", "b"): "reduction",
    ("linear_sigmoid", "x"): "reduction", ("linear_sigmoid", "w"): "reduction",
    ("linear_sigmoid", "b"): "reduction",
    ("linear_relu", "x"): "kink", ("linear_relu", "w"): "kink", ("linear_relu", "b"): "kink",
    ("layernorm", "x"): "reduction", ("layernorm", "gamma"): "reduction",
    ("layernorm", "beta"): "reduction",
    # Pure-eltwise backwards: dx = g*b and dy/dx from the retained output only. No sum, no
    # matmul, nothing to round. These are the cases the per-class bar exists to score.
    ("mul", "a"): "eltwise", ("mul", "b"): "eltwise",
    ("sigmoid", "x"): "eltwise", ("silu", "x"): "eltwise", ("add", "a"): "eltwise",
    ("add", "b"): "eltwise",
}
KINK_CASES = {"linear_relu"}


ELTWISE = ("mul", "add", "sigmoid", "silu")


def build(name, rng):
    if name == "layernorm":
        return GC.case_layernorm(rng)
    if name in ("mul", "add"):
        return {"a": rng.standard_normal((64, 128)), "b": rng.standard_normal((64, 128))}
    if name in ("sigmoid", "silu"):
        return {"x": rng.standard_normal((64, 128))}
    t = GC.case_linear(rng)
    if name == "linear_nobias":
        t.pop("b")
    return t


def ref_forward(name, t):
    if name == "mul":
        return t["a"] * t["b"]
    if name == "add":
        return t["a"] + t["b"]
    if name == "sigmoid":
        return torch.sigmoid(t["x"])
    if name == "silu":
        return t["x"] * torch.sigmoid(t["x"])
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


def tt_forward(name, ops, t, ckc, core_grid, ag):
    """The call written the way a shipped module writes it.

    The four eltwise cases are the tape's own ops rather than `ops.` calls, because that is
    how the attach point reaches them: the hook COMPOSES a taped activation on the output of
    a linear instead of fusing one into the packer. They are here to give the eltwise bar
    something to score, since every gradient of a linear or a layer norm carries a reduction.
    """
    if name == "mul":
        return ag.mul(t["a"], t["b"])
    if name == "add":
        return ag.add(t["a"], t["b"])
    if name == "sigmoid":
        return ag.sigmoid(t["x"])
    if name == "silu":
        return ag.silu(t["x"])
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
        out = tt_forward(name, ops, dev, ckc, tt.CORE_GRID_MAIN, ag)
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
    dts = "float32" if dt == ttnn.float32 else "bfloat16"
    for k, r in ref_grads.items():
        if got[k] is None:
            res["grads"][k] = {"pass": False, "error": "no gradient reached this input"}
            ok = False
            continue
        m = GC.metrics(got[k].reshape(r.shape), r)
        cls = CLASS[(name, k)]
        rel_bar, cos_bar, why = BARS[(cls, dts)]
        m["class"] = cls
        m["rel_l2_bar"] = rel_bar
        m["cos_bar"] = cos_bar
        m["bar_from"] = why
        m["pass"] = bool((cos_bar is None or m["cos"] >= cos_bar)
                         and (rel_bar is None or m["rel_l2"] <= rel_bar))
        ok = ok and m["pass"]
        res["grads"][k] = m

    if name in KINK_CASES:
        pre_dev = torch.from_numpy(
            ttnn.to_torch(ops.linear.shipped(
                dev["x"].value, dev["w"].value, dev["b"].value,
                compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN)).to(torch.float64).numpy())
        pre_ref = rounded["x"] @ rounded["w"] + rounded["b"]
        # (1) the forward's kink placement
        agree = float(((pre_dev > 0) == (pre_ref > 0)).to(torch.float64).mean())
        res["mask_agreement"] = agree
        res["mask_bar"] = MASK_BAR
        res["mask_pass"] = bool(agree >= MASK_BAR)
        ok = ok and res["mask_pass"]
        # (2) the backward's arithmetic, under the gate the device actually used
        gate = (pre_dev > 0).to(torch.float64)
        dg = {k: v.clone().requires_grad_(True) for k, v in rounded.items()}
        ((dg["x"] @ dg["w"] + dg["b"]) * gate * wt).sum().backward()
        rel_bar, cos_bar, why = BARS[("reduction", dts)]
        res["device_gate"] = {"bar_from": why, "rel_l2_bar": rel_bar, "cos_bar": cos_bar}
        for k, v in dg.items():
            m = GC.metrics(got[k].reshape(tuple(v.shape)), v.grad.detach().numpy())
            m["pass"] = bool(m["rel_l2"] <= rel_bar and m["cos"] >= cos_bar)
            ok = ok and m["pass"]
            res["device_gate"][k] = m

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
    shipped = ttnn.to_torch(ops.linear.shipped(raw["x"], raw["w"], raw["b"],
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
                                       "linear_sigmoid,layernorm,mul,add,sigmoid,silu")
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
           "break_silu": args.break_silu, "fd_bar": FD_BAR, "mask_bar": MASK_BAR,
           "bars": {f"{c}/{d}": {"rel_l2": r, "cos": co, "from": w}
                    for (c, d), (r, co, w) in BARS.items()},
           "cases": rows, "control_grad_off": ctrl,
           "all_pass": bool(all(r["pass"] for r in rows) and ctrl["pass"])}
    print(json.dumps(rep, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
