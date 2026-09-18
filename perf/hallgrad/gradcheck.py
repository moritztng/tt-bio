#!/usr/bin/env python3
"""Gradient correctness harness for tt_bio.autograd.

Three levels of evidence, in the order that makes each trustworthy:

1. The float64 reference is itself checked against central finite differences in float64
   before anything on device is compared to it. A reference nobody verified is how a
   campaign ends up chasing a confident wrong number.
2. The device gradient is compared to that reference. Inputs are rounded to the device
   dtype FIRST and the reference is fed the rounded values upcast to float64, so what is
   measured is the op's own error and not the input quantisation.
3. Controls. An fp32 arm must show the error collapse -- if a formula were wrong rather
   than imprecise, more precision would not fix it -- and a deliberately broken arm must
   fail, so we know the check can fail at all.

The bar is set from the bf16 mantissa before the run, not from the result. bfloat16 keeps
8 explicit mantissa bits, so its unit roundoff is 2^-9 = 1.95e-3. Two rounded operands
entering a product give sqrt(2) * u = 2.8e-3 of relative error, and fp32 destination
accumulation keeps the reduction from adding to it. So 2.8e-3 is the floor and the bar is
1.0e-2 relative L2, 3.6x above it. Direction is what an optimiser consumes, so cosine
similarity must also clear 0.9999.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

REL_L2_BAR = 1.0e-2
COS_BAR = 0.9999
BF16_FLOOR = math.sqrt(2.0) * 2.0 ** -9


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    diff = g - r
    rn = np.linalg.norm(r)
    big = np.abs(r) > 0.01 * np.abs(r).max() if r.size else np.zeros(0, bool)
    return {
        "rel_l2": float(np.linalg.norm(diff) / rn) if rn > 0 else float("nan"),
        "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
        "max_rel": float((np.abs(diff[big]) / np.abs(r[big])).max()) if big.any() else 0.0,
        "cos": float(g @ r / (np.linalg.norm(g) * rn)) if rn > 0 and np.linalg.norm(g) > 0 else 0.0,
    }


def fd_check(loss_fn, params, n_probe=40, h=1e-5, seed=0, mag_floor=1e-6):
    """Central differences in float64 against torch's float64 analytic gradient.

    This validates the REFERENCE, so it runs entirely in float64 and never touches a device.

    Only coordinates carrying real signal are probed. A central difference can resolve a
    component only if its contribution to the loss clears float64 roundoff on the loss
    itself: |dL/dx| * 2h has to beat eps * |L| ~= 2e-16 * |L|, so with h = 1e-5 anything
    below ~1e-11 * |L| is measuring roundoff rather than a gradient. Softmax is what forces
    this: a peaked row has components at 1e-12 and probing them returns pure cancellation
    noise. Coordinates under ``mag_floor`` times the largest analytic component are skipped
    and the eligible count is returned, so the skipping is visible in the output rather
    than buried in a tolerance.

    Returns (worst relative disagreement, coordinates probed, coordinates eligible).
    """
    rng = np.random.default_rng(seed)
    for p in params:
        if p.grad is not None:
            p.grad = None
    loss = loss_fn()
    loss.backward()
    # The resolution floor, derived rather than tuned: a central difference recovers
    # |dL/dx| * 2h, and the loss itself is only known to eps * |L|. Demand 1e4 of margin
    # over that so the probe reads gradient and not float64 roundoff.
    resolution_floor = 1.0e4 * 2.22e-16 * abs(loss.item()) / (2.0 * h)
    worst, probed, eligible = 0.0, 0, 0
    for p in params:
        flat = p.detach().reshape(-1)
        ana = p.grad.detach().reshape(-1).clone()
        scale = ana.abs().max().item()
        if scale == 0.0:
            continue
        floor = max(mag_floor * scale, resolution_floor)
        ok = (ana.abs() >= floor).nonzero().reshape(-1).numpy()
        eligible += int(ok.size)
        if ok.size == 0:
            continue
        idx = rng.choice(ok, size=min(n_probe, ok.size), replace=False)
        for i in idx:
            i = int(i)
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + h
            lp = loss_fn().item()
            with torch.no_grad():
                flat[i] = orig - h
            lm = loss_fn().item()
            with torch.no_grad():
                flat[i] = orig
            num = (lp - lm) / (2.0 * h)
            a = ana[i].item()
            # Scaled by the gradient's own magnitude, not by this element's. A per-element
            # ratio is dominated by whichever probed coordinate is smallest -- its finite
            # difference is a difference of two nearly equal float64 loss values, so its
            # relative noise blows up while its absolute contribution stays negligible.
            # The question being asked is whether the reference agrees with finite
            # differences to within a small fraction of the gradient it reports.
            worst = max(worst, abs(num - a) / scale)
            probed += 1
    return worst, probed, eligible


# ---------------------------------------------------------------- cases

def case_linear(rng, shape=(64, 128), n_out=96):
    """One linear, the base case: dX, dW, db each come from a different rule."""
    m, k = shape
    return {
        "x": rng.standard_normal((m, k)),
        "w": rng.standard_normal((k, n_out)) / math.sqrt(k),
        "b": rng.standard_normal((1, n_out)),
    }


def case_chain(rng, m=64, k=128, h=96, n=32):
    """Two linears. Proves the tape composes and that the middle gradient is routed."""
    return {
        "x": rng.standard_normal((m, k)),
        "w1": rng.standard_normal((k, h)) / math.sqrt(k),
        "w2": rng.standard_normal((h, n)) / math.sqrt(h),
    }


def case_fanin(rng, m=64, k=128, n=96):
    """x feeds two linears whose outputs are added. Proves add_grad SUMS on fan-in.

    This is the case a tape gets wrong by overwriting instead of accumulating, and the
    failure is invisible on any single-path graph.
    """
    return {
        "x": rng.standard_normal((m, k)),
        "wa": rng.standard_normal((k, n)) / math.sqrt(k),
        "wb": rng.standard_normal((k, n)) / math.sqrt(k),
    }


def case_layernorm(rng, m=64, k=128):
    """Layer norm: the first op where a wrong reduction axis silently half-works."""
    return {
        "x": rng.standard_normal((m, k)) * 3.0 + 7.0,
        "gamma": rng.standard_normal((1, k)) * 0.5 + 1.0,
        "beta": rng.standard_normal((1, k)) * 0.1,
    }


def case_softmax(rng, m=64, k=128):
    """Softmax: its Jacobian is rank-deficient, so a sum loss would read zero gradient.

    The harness weights the loss instead, which is why every case uses sum(out * W).
    """
    return {"x": rng.standard_normal((m, k)) * 2.0}


TRIATT = dict(B=8, H=2, N=64, d=32)
TRIATT_SCALE = TRIATT["d"] ** -0.5
# chunk the leading axis to 2 of 8 and the query axis to 16 of 64, so the check exercises
# 4 leading blocks x 4 query blocks = 16 recomputed score blocks and both accumulations.
TRIATT_CHUNK = dict(chunk=2, q_chunk=16)


def case_triatt(rng):
    """Triangle attention with the scores materialised: the maths, before the chunking."""
    c = TRIATT
    return {
        "q": rng.standard_normal((c["B"], c["H"], c["N"], c["d"])),
        "k": rng.standard_normal((c["B"], c["H"], c["N"], c["d"])),
        "v": rng.standard_normal((c["B"], c["H"], c["N"], c["d"])),
        "bias": rng.standard_normal((1, c["H"], c["N"], c["N"])) * 0.5,
    }


def case_triatt_chunked(rng):
    """The same maths with a chunked-recompute backward. Must agree with the above."""
    return case_triatt(rng)


def case_triatt_gated(rng):
    """Attention then the sigmoid gate, which is how the block actually ends.

    Composing the new op with `mul` and `sigmoid` from the tape is the point: if the gate
    had to be folded into the attention op, the tape would not be composing.
    """
    d = case_triatt(rng)
    c = TRIATT
    d["g"] = rng.standard_normal((c["B"], c["H"], c["N"], c["d"]))
    return d


def case_mm_tb(rng, m=64, k=96, n=128):
    """a @ b^T on NON-SQUARE operands. Square shapes hide a wrong re-association."""
    return {"a": rng.standard_normal((m, k)) , "b": rng.standard_normal((n, k))}


def case_mm_ta(rng, m=96, k=64, n=128):
    """a^T @ b, also non-square."""
    return {"a": rng.standard_normal((k, m)), "b": rng.standard_normal((k, n))}


def case_permute(rng, n=32, c=64):
    """A pair-shaped permute into channel-major and back."""
    return {"x": rng.standard_normal((n, n, c))}


def case_paircontract(rng, n=32, c=48):
    """TriangleMultiplication's outgoing contraction, composed from permute and matmul."""
    return {"a": rng.standard_normal((n, n, c)) * 0.3,
            "b": rng.standard_normal((n, n, c)) * 0.3}


def case_paircontract_in(rng, n=32, c=48):
    """The incoming variant, which contracts the OTHER index. A wrong axis here is the
    single most likely trimul defect and it is invisible on a symmetric input."""
    return {"a": rng.standard_normal((n, n, c)) * 0.3,
            "b": rng.standard_normal((n, n, c)) * 0.3}


# ------------------------------------------------------------------ LoRA (ptxft-build)
#
# rank and scaling are tt-train's SFT defaults (examples/lora_llama/train_lora_llama_sft.py:
# LORA_RANK 8, LORA_ALPHA 16), so scaling = alpha/rank = 2.0.
LORA = dict(rank=8, alpha=16.0)
LORA_SCALING = LORA["alpha"] / LORA["rank"]


def case_lora(rng, m=64, k=96, n=128):
    """A LoRA linear over a FROZEN base weight. NON-SQUARE in and out, on purpose.

    B is random here and NOT the zero it is initialised to, and that is the whole reason
    this case is trustworthy. At B = 0 the down-projection's gradient is
    dA = x^T @ (g @ B^T) = 0 identically, so a gradcheck at the initialisation would
    report dA correct while measuring nothing at all -- the same vacuous pass the harness
    avoids by weighting its loss instead of summing it. The B = 0 property is checked
    separately by --lora-init, where it is the thing under test rather than the setup.
    """
    return {
        "x": rng.standard_normal((m, k)),
        "w": rng.standard_normal((k, n)) / math.sqrt(k),
        "a": rng.uniform(-1.0 / math.sqrt(k), 1.0 / math.sqrt(k), (k, LORA["rank"])),
        "b": rng.standard_normal((LORA["rank"], n)) * 0.1,
    }


def case_lora_pair(rng, nres=32, cz=48, n=64):
    """The same adapter on a PAIR-shaped input [N, N, c], which is the real trunk shape.

    This is the weight gradient's own hard case and the reason it gets a case of its own:
    dA and dB must sum over EVERY leading dim, so a backward that leaves the leading axes
    in place gets one dW per row of the pair tensor instead of their sum. On a 2D input
    that defect does not exist; on [N, N, c] it is a silent factor-of-N error in the
    direction the optimiser then follows. The brief asks specifically whether the thin-K
    hand-rolled route survives the weight-gradient reduction axis, and this is the check.
    """
    return {
        "x": rng.standard_normal((nres, nres, cz)) * 0.5,
        "w": rng.standard_normal((cz, n)) / math.sqrt(cz),
        "a": rng.uniform(-1.0 / math.sqrt(cz), 1.0 / math.sqrt(cz), (cz, LORA["rank"])),
        "b": rng.standard_normal((LORA["rank"], n)) * 0.1,
    }


def case_lora_bias(rng, m=64, k=96, n=128):
    """With a base bias, which stays frozen too: five tensors, three of them trainable."""
    d = case_lora(rng, m, k, n)
    d["bias"] = rng.standard_normal((1, n)) * 0.1
    return d


def torch_forward(name, t):
    if name in ("lora", "lora_pair", "lora_bias"):
        base = t["x"] @ t["w"]
        if "bias" in t:
            base = base + t["bias"]
        return base + ((t["x"] @ t["a"]) @ t["b"]) * LORA_SCALING
    if name == "linear":
        return t["x"] @ t["w"] + t["b"]
    if name == "chain":
        return (t["x"] @ t["w1"]) @ t["w2"]
    if name == "fanin":
        return t["x"] @ t["wa"] + t["x"] @ t["wb"]
    if name == "layernorm":
        mu = t["x"].mean(-1, keepdim=True)
        xc = t["x"] - mu
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + 1e-6) * t["gamma"] + t["beta"]
    if name == "softmax":
        return torch.softmax(t["x"], dim=-1)
    if name in ("triatt", "triatt_chunked", "triatt_gated"):
        s = t["q"] @ t["k"].transpose(-2, -1) * TRIATT_SCALE + t["bias"]
        o = torch.softmax(s, dim=-1) @ t["v"]
        return o * torch.sigmoid(t["g"]) if name == "triatt_gated" else o
    if name == "mm_tb":
        return t["a"] @ t["b"].transpose(-2, -1)
    if name == "mm_ta":
        return t["a"].transpose(-2, -1) @ t["b"]
    if name == "permute":
        return t["x"].permute(2, 0, 1)
    if name == "paircontract":
        return torch.einsum("ikc,jkc->ijc", t["a"], t["b"])
    if name == "paircontract_in":
        return torch.einsum("kic,kjc->ijc", t["a"], t["b"])
    raise KeyError(name)


def tt_forward(name, ag, t):
    if name in ("lora", "lora_pair", "lora_bias"):
        from tt_bio import finetune as ft
        return ft.lora_linear(t["x"], t["w"], t["a"], t["b"], t.get("bias"),
                              scaling=LORA_SCALING)
    if name == "linear":
        return ag.linear(t["x"], t["w"], t["b"])
    if name == "chain":
        return ag.linear(ag.linear(t["x"], t["w1"]), t["w2"])
    if name == "fanin":
        return ag.add(ag.linear(t["x"], t["wa"]), ag.linear(t["x"], t["wb"]))
    if name == "layernorm":
        return ag.layer_norm(t["x"], t["gamma"], t["beta"])
    if name == "softmax":
        return ag.softmax(t["x"], dim=-1)
    if name in ("triatt", "triatt_chunked", "triatt_gated"):
        kw = TRIATT_CHUNK if name != "triatt" else {}
        o = ag.triangle_attention(t["q"], t["k"], t["v"], t["bias"],
                                  scale=TRIATT_SCALE, **kw)
        return ag.mul(o, ag.sigmoid(t["g"])) if name == "triatt_gated" else o
    if name == "mm_tb":
        return ag.matmul(t["a"], t["b"], transpose_b=True)
    if name == "mm_ta":
        return ag.matmul(t["a"], t["b"], transpose_a=True)
    if name == "permute":
        return ag.permute(t["x"], (2, 0, 1))
    if name == "paircontract":
        return ag.pair_contract(t["a"], t["b"])
    if name == "paircontract_in":
        return ag.pair_contract(t["a"], t["b"], incoming=True)
    raise KeyError(name)


CASES = {
    "linear": case_linear, "chain": case_chain, "fanin": case_fanin,
    "layernorm": case_layernorm, "softmax": case_softmax,
    "triatt": case_triatt, "triatt_chunked": case_triatt_chunked,
    "triatt_gated": case_triatt_gated,
    "mm_tb": case_mm_tb, "mm_ta": case_mm_ta, "permute": case_permute,
    "paircontract": case_paircontract, "paircontract_in": case_paircontract_in,
    "lora": case_lora, "lora_pair": case_lora_pair, "lora_bias": case_lora_bias,
}

# Tensors that must receive NO gradient. For LoRA the frozen base weight is not an
# accounting detail: if W takes a gradient you are full fine-tuning and calling it LoRA,
# and the failure is invisible in every error metric because the gradient is correct. The
# harness asserts absence, which is why these are declared rather than simply omitted.
FROZEN = {"lora": ("w",), "lora_pair": ("w",), "lora_bias": ("w", "bias")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases",
                    default="linear,chain,fanin,layernorm,softmax,triatt,triatt_chunked,"
                            "triatt_gated")
    ap.add_argument("--lora-init", action="store_true",
                    help="control: at B=0 the adapter is an exact no-op and dA is exactly 0")
    ap.add_argument("--chunk-invariance", action="store_true",
                    help="difference two chunkings of the attention backward against each other")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--break-layernorm-axis", action="store_true",
                    help="negative control: reduce layer norm over the wrong axis")
    ap.add_argument("--fd-probe", type=int, default=24)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    if args.break_layernorm_axis:
        _orig = ag.layer_norm

        def broken(x, gamma=None, beta=None, *, eps=1e-6, config=None):
            out = _orig(x, gamma, beta, eps=eps, config=config)
            node = out.node

            def bw(g):
                # the defect under test: reduce the backward over dim 0, not the last dim
                dn = ttnn.multiply(g, gamma.value) if gamma is not None else g
                m0 = ttnn.mean(dn, dim=0, keepdim=True)
                x.add_grad(ttnn.subtract(dn, m0))
            node.fn = bw
            return out
        ag.layer_norm = broken

    dt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[args.dtype]
    torch_dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = tt.get_device()

    print(f"# dtype={args.dtype} seed={args.seed} bars: rel_l2<={REL_L2_BAR:.1e} "
          f"cos>={COS_BAR} | bf16 quantisation floor {BF16_FLOOR:.2e}")
    print(f"{'case':<10} {'param':<7} {'rel_l2':>10} {'max_abs':>10} {'max_rel':>10} "
          f"{'cos':>10}  verdict")
    failures = []
    for name in args.cases.split(","):
        # Seeded per case so `--cases softmax` gives the same inputs as `--cases a,b,softmax`.
        # Sharing one generator across cases made every number depend on the case list.
        rng = np.random.default_rng([args.seed, abs(hash(name)) % (2 ** 31)])
        raw = CASES[name](rng)
        frozen = FROZEN.get(name, ())
        # Round to the device dtype FIRST, then upcast for the reference, so the comparison
        # isolates the op's error from the input quantisation.
        rounded = {k: torch.from_numpy(v).to(torch_dt).to(torch.float64)
                   for k, v in raw.items()}
        ref = {k: v.clone().requires_grad_(k not in frozen)
               for k, v in rounded.items()}
        out_shape = tuple(torch_forward(name, ref).shape)
        wnp = rng.standard_normal(out_shape)
        wt = torch.from_numpy(wnp).to(torch.float64)

        def loss_fn(_ref=ref, _name=name, _wt=wt):
            return (torch_forward(_name, _ref) * _wt).sum()

        fd_worst, n_probed, n_elig = fd_check(
            loss_fn, [v for k, v in ref.items() if k not in frozen],
            n_probe=args.fd_probe, seed=args.seed)
        ref_grads = {k: v.grad.detach().numpy().copy()
                     for k, v in ref.items() if k not in frozen}
        fd_ok = fd_worst < 2e-6
        print(f"{name:<10} {'[fd]':<7} {'':>10} {'':>10} {fd_worst:>10.2e} {'':>10}  "
              f"{'reference OK' if fd_ok else 'REFERENCE SUSPECT'} "
              f"({n_probed} of {n_elig} resolvable coords probed)")
        if not fd_ok:
            failures.append(f"{name}: float64 reference disagrees with finite differences "
                            f"by {fd_worst:.2e}")
            continue

        tt_t = {k: ag.Tensor(
            ttnn.from_torch(v.to(torch_dt), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device),
            requires_grad=(k not in frozen)) for k, v in rounded.items()}
        out = tt_forward(name, ag, tt_t)
        seed_t = ttnn.from_torch(wt.to(torch_dt), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                 device=device)
        out.backward(seed=seed_t)

        for k in raw:
            if k in frozen:
                leaked = tt_t[k].grad is not None
                if leaked:
                    failures.append(f"{name}/{k}: a FROZEN tensor received a gradient")
                print(f"{name:<10} {k:<7} {'':>10} {'':>10} {'':>10} {'':>10}  "
                      f"{'FAIL (frozen tensor got a gradient)' if leaked else 'FROZEN, no gradient'}")
                continue
            if tt_t[k].grad is None:
                failures.append(f"{name}/{k}: no gradient reached this input")
                print(f"{name:<10} {k:<7} {'':>10} {'':>10} {'':>10} {'':>10}  NO GRADIENT")
                continue
            got = ttnn.to_torch(tt_t[k].grad).to(torch.float64).numpy()
            m = metrics(got, ref_grads[k])
            ok = m["rel_l2"] <= REL_L2_BAR and m["cos"] >= COS_BAR
            if not ok:
                failures.append(f"{name}/{k}: rel_l2={m['rel_l2']:.3e} cos={m['cos']:.6f}")
            print(f"{name:<10} {k:<7} {m['rel_l2']:>10.2e} {m['max_abs']:>10.2e} "
                  f"{m['max_rel']:>10.2e} {m['cos']:>10.6f}  {'PASS' if ok else 'FAIL'}")

    if args.lora_init:
        from tt_bio import finetune as ft
        print()
        print("# LoRA initialisation control. tt-train initialises B to exactly zero")
        print("# (modules/lora.py:64-67), which has two consequences this arm measures")
        print("# rather than assumes: the adapted forward must be BIT-IDENTICAL to the")
        print("# base linear, so 'before' in a held-out comparison is the shipped model;")
        print("# and dA must be exactly 0, which is why case_lora uses a non-zero B.")
        rng3 = np.random.default_rng([args.seed, 4242])
        k_in, n_out = 96, 128
        xh = torch.from_numpy(rng3.standard_normal((64, k_in))).to(torch_dt)
        wh = torch.from_numpy(rng3.standard_normal((k_in, n_out))
                              / math.sqrt(k_in)).to(torch_dt)
        mk = lambda v, rg: ag.Tensor(ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                                     device=device), requires_grad=rg)
        cfg = ft.LoraConfig(rank=LORA["rank"], alpha=LORA["alpha"])
        xt, wt3 = mk(xh, True), mk(wh, False)
        a0, b0 = ft.lora_factors(k_in, n_out, cfg, device, dtype=dt,
                                 rng=np.random.default_rng(11))
        adapted = ft.lora_linear(xt, wt3, a0, b0, scaling=cfg.scaling)
        base_only = ag.linear(mk(xh, False), mk(wh, False))
        av = ttnn.to_torch(adapted.value).to(torch.float64).numpy()
        bv = ttnn.to_torch(base_only.value).to(torch.float64).numpy()
        exact = bool((av == bv).all())
        print(f"{'B init':<22} max|B| = {np.abs(ttnn.to_torch(b0.value).to(torch.float64).numpy()).max():.3e}")
        print(f"{'adapted == base':<22} {'BIT-IDENTICAL' if exact else 'DIFFERS'} "
              f"(max abs diff {np.abs(av - bv).max():.3e})")
        if not exact:
            failures.append("lora-init: the adapter at B=0 is not an exact no-op")
        seed3 = ttnn.from_torch(
            torch.from_numpy(rng3.standard_normal((64, n_out))).to(torch_dt),
            dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        adapted.backward(seed=seed3)
        da = (np.abs(ttnn.to_torch(a0.grad).to(torch.float64).numpy()).max()
              if a0.grad is not None else float("nan"))
        db = (np.abs(ttnn.to_torch(b0.grad).to(torch.float64).numpy()).max()
              if b0.grad is not None else float("nan"))
        print(f"{'dA at B=0':<22} max abs {da:.3e}  {'exactly zero' if da == 0.0 else 'NONZERO'}")
        print(f"{'dB at B=0':<22} max abs {db:.3e}  "
              f"{'nonzero, so B can leave 0' if db > 0 else 'ZERO -- B could never move'}")
        if da != 0.0:
            failures.append(f"lora-init: dA at B=0 is {da:.3e}, not exactly zero")
        if not (db > 0):
            failures.append("lora-init: dB at B=0 is zero, so the adapter can never train")

    if args.chunk_invariance:
        print()
        print("# chunk invariance: the SAME inputs, two chunkings of the attention backward.")
        print("# A chunked recompute that drops or double-counts a block shows up here and")
        print("# nowhere else, because the arms are compared to each other, not to a reference.")
        rng2 = np.random.default_rng([args.seed, 991])
        raw = case_triatt(rng2)
        rounded = {kk: torch.from_numpy(vv).to(torch_dt).to(torch.float64)
                   for kk, vv in raw.items()}
        wt2 = torch.from_numpy(rng2.standard_normal(
            (TRIATT["B"], TRIATT["H"], TRIATT["N"], TRIATT["d"]))).to(torch_dt)
        arms = {}
        for label, kw in (("whole", {}), ("chunked", TRIATT_CHUNK)):
            tt_t = {kk: ag.Tensor(
                ttnn.from_torch(vv.to(torch_dt), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                device=device), requires_grad=True)
                for kk, vv in rounded.items()}
            o = ag.triangle_attention(tt_t["q"], tt_t["k"], tt_t["v"], tt_t["bias"],
                                      scale=TRIATT_SCALE, **kw)
            o.backward(seed=ttnn.from_torch(wt2, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                            device=device))
            arms[label] = {kk: ttnn.to_torch(tt_t[kk].grad).to(torch.float64).numpy()
                           for kk in raw}
        print(f"{'param':<10} {'rel_l2':>10} {'max_abs':>10} {'cos':>10}  verdict")
        for kk in raw:
            m = metrics(arms["chunked"][kk], arms["whole"][kk])
            ok = m["rel_l2"] <= REL_L2_BAR and m["cos"] >= COS_BAR
            if not ok:
                failures.append(f"chunk-invariance/{kk}: rel_l2={m['rel_l2']:.3e}")
            print(f"{kk:<10} {m['rel_l2']:>10.2e} {m['max_abs']:>10.2e} "
                  f"{m['cos']:>10.6f}  {'PASS' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"GRADCHECK FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
    else:
        print("GRADCHECK PASS: every gradient within the pre-registered bar")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
