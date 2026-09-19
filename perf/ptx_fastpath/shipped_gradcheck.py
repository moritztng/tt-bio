#!/usr/bin/env python3
"""Gradients of the SHIPPED tt-bio modules, against a float64 reference.

`perf/hallgrad/gradcheck.py` proved the tape's own ops. This proves the shipped modules --
the tuned ones in `tt_bio/tenstorrent.py`, with their chunking, their L1 routings and their
program configs -- differentiate to the same answer, run under `tt_bio.autograd.tape()`
with no change to their source.

The evidence order is `gradcheck.py`'s and the reason is the same: a reference nobody
checked is how a campaign chases a confident wrong number.

1. The float64 reference is validated against float64 central finite differences first.
2. Inputs are rounded to the device dtype, the reference is fed those rounded values
   upcast to float64, and the device gradient is compared to it. So what is measured is
   the module's error, not the input quantisation.
3. A control: grad-off against grad-on. The tape computes its value by calling the
   shipped verb, so the seam itself must change nothing, and with `TT_BIO_UNFUSED_SILU=1`
   it measures 0.0. On the shipped default it does not, and the reason is known and
   singular: `ttnn.linear(activation="silu")` fuses the activation into the packer, the
   tape composes it instead because silu's output cannot be inverted back to its input,
   and that is one extra bf16 rounding of the pre-activation. So bit-exactness is reported
   rather than required, and the forward is held to the same accuracy bar as the
   gradients -- which is the standing rule, not a concession made here.

The bar is `gradcheck.py`'s, derived from the bf16 mantissa rather than from the result:
1.0e-2 relative L2 against a 2.8e-3 two-operand floor, and cosine above 0.9999 because
direction is what an optimiser consumes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallgrad.gradcheck import COS_BAR, REL_L2_BAR, fd_check, metrics   # noqa: E402


def clocks(stop, out):
    """AICLK sampled DURING the run. A perf number without one is not a measurement, and a
    correctness run still records it so the reading is attributable."""
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for dev in json.loads(r.stdout).get("device_info", []):
                c = dev.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(1.0)


# ------------------------------------------------------------------ the module under test

def transition_weights(rng, c, hidden):
    """`Transition`'s five tensors, in the (out, in) torch layout `torch_to_tt` transposes."""
    return {
        "norm.weight": torch.tensor(rng.standard_normal(c) * 0.3 + 1.0),
        "norm.bias": torch.tensor(rng.standard_normal(c) * 0.1),
        "fc1.weight": torch.tensor(rng.standard_normal((hidden, c)) / math.sqrt(c)),
        "fc2.weight": torch.tensor(rng.standard_normal((hidden, c)) / math.sqrt(c)),
        "fc3.weight": torch.tensor(rng.standard_normal((c, hidden)) / math.sqrt(hidden)),
    }


def transition_ref(x, w):
    """Transition in float64: layer norm, SwiGLU, out projection. tenstorrent.py:8299.

    Written from the shipped source rather than from Protenix's, so a disagreement is the
    tape's and not a port difference that was already signed off elsewhere.
    """
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    xn = (x - mean) / torch.sqrt(var + 1e-5) * w["norm.weight"] + w["norm.bias"]
    x1 = torch.nn.functional.silu(xn @ w["fc1.weight"].t())
    x2 = xn @ w["fc2.weight"].t()
    return (x1 * x2) @ w["fc3.weight"].t()


CASES = {"transition": (transition_weights, transition_ref)}


# ------------------------------------------------------------------ the run

def run(name, shape, hidden, seed, probes):
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    rng = np.random.default_rng(seed)
    c = shape[-1]
    make_w, ref_fn = CASES[name]
    w64 = {k: v.to(torch.float64) for k, v in make_w(rng, c, hidden).items()}
    x64 = torch.tensor(rng.standard_normal(shape) * 2.0 + 1.0, dtype=torch.float64)
    # A weighted sum, so the loss reads every output coordinate; a plain sum would be blind
    # to any error that cancels across a row, which for a norm is most of them.
    lw = torch.tensor(rng.standard_normal(shape[:-1] + (c,)), dtype=torch.float64)

    def bf(t):
        return t.to(torch.bfloat16).to(torch.float64)

    # -- 1. the reference, against finite differences, entirely on host in float64 --------
    params = [x64.clone().requires_grad_(True)] + \
             [v.clone().requires_grad_(True) for v in w64.values()]
    names = ["x"] + list(w64)
    pmap = dict(zip(names, params))

    def loss_fn():
        return (ref_fn(pmap["x"], {k: pmap[k] for k in w64}) * lw).sum()

    worst, probed, eligible = fd_check(loss_fn, params, n_probe=probes, seed=seed)
    print(f"  reference vs float64 central differences: worst {worst:.2e} "
          f"over {probed} of {eligible} eligible coordinates")
    if worst > 1e-6:
        print("  REFERENCE FAILED its own finite-difference check; nothing below is evidence")
        return False

    # -- 2. the same reference on the ROUNDED values, which is what the device sees -------
    rparams = [bf(x64).requires_grad_(True)] + [bf(v).requires_grad_(True) for v in w64.values()]
    rmap = dict(zip(names, rparams))
    rloss = (ref_fn(rmap["x"], {k: rmap[k] for k in w64}) * lw).sum()
    rloss.backward()
    ref_out = ref_fn(rmap["x"], {k: rmap[k] for k in w64}).detach()

    # -- 3. the shipped module, taped ----------------------------------------------------
    device = tt.get_device()

    def dev(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)

    sd = {k: v.detach().to(torch.float32) for k, v in
          zip(names[1:], [bf(v) for v in w64.values()])}
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if device.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    module = tt.Transition(sd, ckc)

    # The control first: grad OFF must be the shipped forward, byte for byte.
    xt = dev(x64)
    off = ttnn.to_torch(module(xt)).to(torch.float64)

    xa = ag.Tensor(dev(x64), requires_grad=True)
    params_tt = {}
    with ag.tape():
        out = module(xa)
        on = ttnn.to_torch(out.value).to(torch.float64)
    bitexact = torch.equal(off, on)
    print(f"  grad-off vs grad-on forward: {'BIT-EXACT' if bitexact else 'DIFFERS'}"
          f"  max_abs {float((off - on).abs().max()):.3e}")

    # The module's weights are device tensors it owns; retape them as leaves so their
    # gradients land somewhere. Done after the forward would be too late, so redo it.
    for attr in ("norm_weight", "norm_bias", "fc1_weight", "fc2_weight", "fc3_weight"):
        params_tt[attr] = ag.Tensor(getattr(module, attr), requires_grad=True)
        setattr(module, attr, params_tt[attr])
    xa = ag.Tensor(dev(x64), requires_grad=True)
    with ag.tape():
        out = module(xa)
    out.backward(seed=dev(lw))

    got = {"x": xa.grad}
    key = {"norm_weight": "norm.weight", "norm_bias": "norm.bias",
           "fc1_weight": "fc1.weight", "fc2_weight": "fc2.weight", "fc3_weight": "fc3.weight"}
    for attr, t in params_tt.items():
        got[key[attr]] = t.grad

    fwd = metrics(ttnn.to_torch(out.value).to(torch.float64).numpy(), ref_out.numpy())
    ok = fwd["rel_l2"] <= REL_L2_BAR and fwd["cos"] >= COS_BAR
    print(f"  forward            rel_l2 {fwd['rel_l2']:.3e}  cos {fwd['cos']:.6f}  "
          f"{'PASS' if ok else 'FAIL'}")
    for nm in names:
        g = got.get(nm)
        if g is None:
            print(f"  {nm:14s} NO GRADIENT")
            ok = False
            continue
        gd = ttnn.to_torch(g).to(torch.float64).numpy()
        # A weight arrives on device transposed to (in, out); the reference holds (out, in).
        r = rmap[nm].grad.detach().numpy()
        if gd.shape != r.shape and gd.shape == r.T.shape:
            r = r.T
        gd = gd.reshape(r.shape)
        m = metrics(gd, r)
        good = m["rel_l2"] <= REL_L2_BAR and m["cos"] >= COS_BAR
        ok = ok and good
        print(f"  d{nm:13s} rel_l2 {m['rel_l2']:.3e}  cos {m['cos']:.6f}  "
              f"max_abs {m['max_abs']:.3e}  {'PASS' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="transition")
    ap.add_argument("--shape", default="1,64,64,128")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=24)
    a = ap.parse_args()
    shape = tuple(int(v) for v in a.shape.split(","))

    samples, stop = [], threading.Event()
    th = threading.Thread(target=clocks, args=(stop, samples), daemon=True)
    th.start()
    print(f"{a.case}  shape {shape}  hidden {a.hidden}  seed {a.seed}")
    try:
        ok = run(a.case, shape, a.hidden, a.seed, a.probes)
    finally:
        stop.set()
        th.join(timeout=3)
    if samples:
        print(f"  aiclk during run: min {min(samples)} max {max(samples)} MHz "
              f"({len(samples)} samples)")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
