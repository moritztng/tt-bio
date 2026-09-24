#!/usr/bin/env python3
"""The AF2 backward on the shipped trunk: is it correct, and what does a real block cost?

`tt_bio.taped_ttnn.tape()` wrapped around `AF2DeviceModel`'s own device blocks (`tt_bio/af2.py`,
`model_1_ptm` weights, MSA depth 1, extra-MSA mask all zero as BindCraft 2 sets it). Nothing in
the model is re-implemented here: the device arm calls the shipped `AF2EvoformerBlock` and
`AF2PairBlock` objects exactly as `AF2DeviceModel.evoformer_stack` / `extra_msa_stack` do, and
the reference arm is `af2_reference.py`'s own blocks promoted to float64.

Subcommands, each one device open:

  vjp    per-block VJP against float64, teacher-forced block by block, plus the controls
         (zero seed, permuted cotangent) and the torch bf16/fp32 envelope on the same inputs
  stack  the whole 4 + 48 stack under one tape from sequence logits: the torch seam, both
         stack outputs seeded in ONE backward, the float64 whole-stack gradient, a swept
         directional finite difference, the tape-reach count, the branch-point census and the
         tape-node census
  time   forward and backward per block, n in {128, 256}, K in {1, 2, 4}, Evoformer and
         extra-MSA separately, fit a + b*K, every point stamped with the AICLK sampled at 1 s
         inside its own timed window
  fit    how many Evoformer blocks fit uncheckpointed at n, and the checkpointed per-block cost

The reference is float64 all the way through. `af2_reference.LayerNorm` hard-codes `.float()`,
which would quietly run every LayerNorm of a `.double()` model in float32; `_float64_layernorm`
below widens it for float64 inputs only and leaves the bf16/fp32 arms byte-identical.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_PARAMS = os.path.expanduser("~/.boltz/af2/params/params_model_1_ptm.npz")
OUT = ROOT / "perf" / "bcx_afgrad"
PROXY_B256 = 0.23204        # perf/hallgrad p2 `_block` fwd+bwd per block at n=256, P300c


# ------------------------------------------------------------------------------ reference


def _float64_layernorm():
    from tt_bio import af2_reference as R

    def forward(self, x):
        dtype = x.dtype
        y = x.double() if dtype == torch.float64 else x.float()
        mean = y.mean(-1, keepdim=True)
        if self.fast_variance:
            var = (y * y).mean(-1, keepdim=True) - mean * mean
        else:
            var = (y - mean).square().mean(-1, keepdim=True)
        out = (y - mean) * torch.rsqrt(var + self.eps) * self.weight.to(y.dtype) \
            + self.bias.to(y.dtype)
        return out.to(dtype)

    R.LayerNorm.forward = forward


def load_models(params, device_arm=True):
    """(device model or None, float64 reference, fp32 reference, bf16 reference)."""
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict
    _float64_layernorm()
    state = load_af2_state_dict(params)
    dm = None
    if device_arm:
        from tt_bio.af2 import load_af2_device_model
        dm = load_af2_device_model(state, template=False, trunk_dtype=torch.bfloat16)
    ref = {}
    for name, dt in (("f64", torch.float64), ("f32", torch.float32), ("bf16", torch.bfloat16)):
        m = load_af2_model(state, template=False, trunk_dtype=dt)
        # Parameters stay float32 in the bf16/fp32 arms, which is AF2's own convention (the
        # Linear casts the weight to the activation dtype per call). float64 promotes them.
        ref[name] = m.double() if dt == torch.float64 else m
        for p in ref[name].parameters():
            p.requires_grad_(False)
    return dm, ref


def embed(model, logits, residue_index):
    """Sequence logits -> (msa (1, n, 256), pair (n, n, 128)) through the model's own embedding.

    `AF2Model.forward` up to the stacks, with BindCraft's featurisation of a soft sequence
    (ColabDesign `_update_seq`: the softmax fills msa_feat 0:20 and the cluster profile 25:45,
    target_feat is the same 20 columns), one MSA row, first recycle (prev all zero), no
    template. Differentiable in `logits` by torch autograd: this is the seam's host side.
    """
    from tt_bio.af2_reference import (MAX_RELATIVE_FEATURE, RECYCLE_DGRAM,
                                      dgram_from_positions, pseudo_beta)
    dtype = model.trunk_dtype
    n = logits.shape[0]
    p = torch.softmax(logits.to(torch.float64 if dtype == torch.float64 else torch.float32), -1)
    target_feat = F.pad(p, (1, 1)).to(dtype)
    z5, z4 = p.new_zeros(n, 5), p.new_zeros(n, 4)
    msa_feat = torch.cat([p, z5, p, z4], dim=-1).to(dtype)[None]
    e = model.embed
    msa = e["preprocess_1d"](target_feat).unsqueeze(0) + e["preprocess_msa"](msa_feat)
    pair = e["left_single"](target_feat).unsqueeze(1) + e["right_single"](target_feat).unsqueeze(0)
    aatype = logits.detach().argmax(-1)
    prev_pos = torch.zeros(n, 37, 3)
    dgram = dgram_from_positions(pseudo_beta(aatype, prev_pos), *RECYCLE_DGRAM).to(dtype)
    pair = pair + model.recycle["prev_pos_linear"](dgram)
    msa = msa + model.recycle["prev_msa_norm"](torch.zeros(n, 256)).to(dtype)[None]
    pair = pair + model.recycle["prev_pair_norm"](torch.zeros(n, n, 128)).to(dtype)
    off = residue_index[:, None] - residue_index[None, :]
    rel = F.one_hot((off + MAX_RELATIVE_FEATURE).clamp(0, 2 * MAX_RELATIVE_FEATURE),
                    2 * MAX_RELATIVE_FEATURE + 1).to(dtype)
    pair = pair + e["pair_activations"](rel)
    return msa, pair


def ref_extra(model, i, pair):
    """Extra-MSA block i on the reference: one zero extra-MSA row under an all-zero mask."""
    n = pair.shape[0]
    dt = pair.dtype
    extra = torch.zeros(1, n, 64, dtype=dt)
    return model.extra_msa[i](extra, pair, torch.zeros(1, n, dtype=dt),
                              torch.ones(n, n, dtype=dt))[1]


def ref_evo(model, i, msa, pair):
    n = pair.shape[0]
    dt = pair.dtype
    return model.evoformer[i](msa, pair, torch.ones(msa.shape[:2], dtype=dt),
                              torch.ones(n, n, dtype=dt))


def ref_stack(model, msa, pair, k_extra, k_evo, keep=False):
    """The stack on the reference. `keep` returns every block boundary for teacher forcing."""
    bounds = []
    for i in range(k_extra):
        if keep:
            bounds.append(("extra", i, None, pair))
        pair = ref_extra(model, i, pair)
    for i in range(k_evo):
        if keep:
            bounds.append(("evo", i, msa, pair))
        msa, pair = ref_evo(model, i, msa, pair)
    return (msa, pair, bounds) if keep else (msa, pair)


# ------------------------------------------------------------------------------ device arm


class Dev:
    """The device side: the shipped blocks, under `taped_ttnn.tape()`."""

    def __init__(self, dm):
        import ttnn
        from tt_bio import autograd as ag
        from tt_bio import taped_ttnn
        self.ttnn, self.ag, self.tt = ttnn, ag, taped_ttnn
        self.dm = dm
        self.device = dm._device

    def up(self, t):
        return self.ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def down(self, t, shape):
        x = torch.Tensor(self.ttnn.to_torch(t)).float()
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        assert tuple(x.shape) == tuple(shape), (tuple(x.shape), shape)
        return x

    def leaf(self, t):
        return self.ag.Tensor(self.up(t), requires_grad=True)

    def sync(self):
        self.ttnn.synchronize_device(self.device)

    def extra(self, i, z):
        blk = self.dm.device_extra_msa[i]
        const = self.dm._up(self.dm.opm_constant[i].reshape(1, 1, -1))
        return blk(blk._residual(z, const))

    def evo(self, i, m, z):
        return self.dm.device_evoformer[i](m, z)

    def stack(self, m, z, k_extra, k_evo, extra_first=0, evo_first=0, ckpt=False):
        for i in range(extra_first, extra_first + k_extra):
            z = self.ag.checkpoint(lambda t, i=i: self.extra(i, t), z) if ckpt else self.extra(i, z)
        for i in range(evo_first, evo_first + k_evo):
            if ckpt:
                m, z = self.ag.checkpoint(lambda a, b, i=i: self.evo(i, a, b), m, z)
            else:
                m, z = self.evo(i, m, z)
        return m, z

    def seed(self, t, like):
        """A cotangent in the root's own device shape."""
        shape = [int(d) for d in like.value.shape]
        return self.ttnn.from_torch(t.detach().reshape(shape).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def grad(self, leaf, shape):
        return self.down(leaf.grad, shape) if leaf.grad is not None else torch.zeros(shape)


# ------------------------------------------------------------------------------ metrics


def rel_l2(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


def cmp(a, b):
    return {"rel_l2": rel_l2(a, b), "cos": cosine(a, b)}


def ref_vjp(fn, inputs, cotangents):
    xs = [x.detach().clone().requires_grad_(True) for x in inputs]
    outs = fn(*xs)
    outs = outs if isinstance(outs, tuple) else (outs,)
    gs = torch.autograd.grad(outs, xs, [c.to(o.dtype) for c, o in zip(cotangents, outs)],
                             allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, xs)], outs


def bf(t):
    return t.detach().to(torch.bfloat16).double()


# ------------------------------------------------------------------------------ stamps


def stamp(card):
    sysfs = pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{card}/device/subsystem_device")
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "tt_bio"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        sha, dirty = "?", "?"
    return {"host": socket.gethostname(), "card": card,
            "subsystem_device": sysfs.read_text().strip() if sysfs.exists() else None,
            "commit": sha, "tt_bio_dirty": bool(dirty), "torch_threads": torch.get_num_threads(),
            "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def clock_trace():
    from perf.hallgrad.e2e_distogram import ClockTrace
    return ClockTrace(period=1.0).start()


def window(trace, t0, t1):
    xs = sorted(c for t, c in trace.samples if t0 <= t <= t1)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {path}", flush=True)


# ------------------------------------------------------------------------------ censuses


LAYOUT_NODES = {"_identity_grad", "_sliced"}
SHAPE_NODES = LAYOUT_NODES | {"_v_reshape", "_v_permute", "_v_transpose", "_v_concat", "_v_pad",
                              "_v_concat_heads", "_v_create_qkv_heads", "reshape", "permute",
                              "concat", "narrow"}


def node_census(ag, roots):
    order = ag._reverse_topo(roots)
    kinds = collections.Counter()
    for t in order:
        if t.node is not None:
            kinds[t.node.fn.__qualname__.split(".<locals>")[0]] += 1
    total = sum(kinds.values())
    lay = sum(v for k, v in kinds.items() if k in LAYOUT_NODES)
    shp = sum(v for k, v in kinds.items() if k in SHAPE_NODES)
    return {"reach": len(order), "nodes": total, "kinds": dict(kinds.most_common()),
            "identity_plus_sliced": lay, "identity_plus_sliced_share": lay / max(total, 1),
            "all_shape_ops": shp, "all_shape_ops_share": shp / max(total, 1)}


class TapingCensus:
    """Count every `ops.taping()` call by the shipped call site that asked."""

    def __init__(self):
        from tt_bio import ops
        self.ops, self.real = ops, ops.taping
        self.counts = collections.Counter()

    def __enter__(self):
        real, counts = self.real, self.counts

        def taping():
            f = sys._getframe(1)
            while f is not None and f.f_code.co_name == "_taping":
                f = f.f_back
            site = f"{pathlib.Path(f.f_code.co_filename).name}:{f.f_lineno}"
            counts[site] += 1
            return real()

        self.ops.taping = taping
        return self

    def __exit__(self, *exc):
        self.ops.taping = self.real
        return False


# ------------------------------------------------------------------------------ vjp


def cmd_vjp(args):
    torch.manual_seed(args.seed)
    dm, ref = load_models(args.params)
    dev = Dev(dm)
    ag = dev.ag
    n = args.n
    logits = torch.randn(n, 20) * 2.0
    ridx = torch.arange(n)
    with torch.no_grad():
        msa0, pair0 = embed(ref["f64"], logits.double(), ridx)
        k_extra, k_evo = args.extra, args.evo
        # Reference forward on the f64 trunk, keeping every block's input, and the float64
        # cotangent at every block's output from one loss: a fixed random linear readout of
        # both stack outputs. That cotangent is what each device block is fed (teacher forcing).
    msa_t, pair_t = msa0.clone().requires_grad_(True), pair0.clone().requires_grad_(True)
    m_out, z_out, bounds = ref_stack(ref["f64"], msa_t, pair_t, k_extra, k_evo, keep=True)
    wm = torch.randn(m_out.shape, dtype=torch.float64) / m_out.numel() ** 0.5
    wz = torch.randn(z_out.shape, dtype=torch.float64) / z_out.numel() ** 0.5
    # Cotangent at each block OUTPUT: retain grads on each boundary input of the next block.
    for kind, i, m, z in bounds:
        for t in (m, z):
            if t is not None and t.requires_grad:
                t.retain_grad()
    loss = (wm * m_out).sum() + (wz * z_out).sum()
    loss.backward()
    # boundary j's input is block j-1's output, so block j's output cotangent is boundary
    # j+1's grad (or the readout for the last block).
    cot = []
    for j, (kind, i, m, z) in enumerate(bounds):
        if j + 1 < len(bounds):
            _, _, m1, z1 = bounds[j + 1]
            cot.append((m1.grad if kind == "evo" else None, z1.grad))
        else:
            cot.append((wm if kind == "evo" else None, wz))
    del loss
    rows = []
    blocks = [int(b) for b in args.blocks.split(",")] if args.blocks else None
    for j, (kind, i, m, z) in enumerate(bounds):
        tag = f"{kind}{i}"
        if blocks is not None and j not in blocks:
            continue
        gm, gz = cot[j]
        zin = bf(z)
        min_ = bf(m) if m is not None else None
        # --- float64 reference VJP on the SAME bf16-rounded inputs, and the envelope arms
        arms = {}
        for arm in ("f64", "f32", "bf16"):
            mod = ref[arm]
            dt = mod.trunk_dtype
            if kind == "extra":
                g, _ = ref_vjp(lambda a, i=i, mod=mod: ref_extra(mod, i, a),
                               [zin.to(dt)], [gz])
                arms[arm] = {"dz": g[0].double()}
            else:
                g, _ = ref_vjp(lambda a, b, i=i, mod=mod: ref_evo(mod, i, a, b),
                               [min_.to(dt), zin.to(dt)], [gm, gz])
                arms[arm] = {"dm": g[0].double(), "dz": g[1].double()}
        r64 = arms["f64"]

        def device_vjp(gm_, gz_):
            gc.collect()
            zl = dev.leaf(zin)
            ml = dev.leaf(min_) if min_ is not None else None
            with dev.tt.tape():
                if kind == "extra":
                    zo = dev.extra(i, zl)
                    roots, seeds = [zo], [dev.seed(gz_, zo)]
                else:
                    mo, zo = dev.evo(i, ml, zl)
                    roots, seeds = [mo, zo], [dev.seed(gm_, mo), dev.seed(gz_, zo)]
            census = node_census(ag, roots)
            ag.backward(roots, seeds)
            out = {"dz": dev.grad(zl, zin.shape)}
            if ml is not None:
                out["dm"] = dev.grad(ml, min_.shape)
            fwd = {"z": dev.down(zo.value, zin.shape)}
            if ml is not None:
                fwd["m"] = dev.down(mo.value, min_.shape)
            del roots, seeds, zo, zl, ml
            gc.collect()
            return out, fwd, census

        d, fwd, census = device_vjp(gm, gz)
        row = {"block": tag, "reach": census["reach"], "nodes": census["nodes"]}
        for key in d:
            row[key] = cmp(d[key], r64[key])
            row[key]["norm_ref"] = float(r64[key].norm())
            row[key + "_torch_bf16"] = cmp(arms["bf16"][key], r64[key])
            row[key + "_torch_f32"] = cmp(arms["f32"][key], r64[key])
        # forward on the same inputs, so a bad gradient can be told from a bad forward
        with torch.no_grad():
            if kind == "extra":
                fz = ref_extra(ref["f64"], i, zin)
                row["fwd_z"] = cmp(fwd["z"], fz)
            else:
                fm, fz = ref_evo(ref["f64"], i, min_, zin)
                row["fwd_m"], row["fwd_z"] = cmp(fwd["m"], fm), cmp(fwd["z"], fz)
        if j in (0, len(bounds) - 1) or args.controls_all:
            # Controls that must MOVE the reading.
            perm = lambda t: t.flatten()[torch.randperm(t.numel())].reshape(t.shape)
            dp, _, _ = device_vjp(perm(gm) if gm is not None else None, perm(gz))
            row["permuted"] = {k: cmp(dp[k], r64[k]) for k in dp}
            d0, _, _ = device_vjp(torch.zeros_like(gm) if gm is not None else None,
                                  torch.zeros_like(gz))
            row["zero_seed_max_abs"] = {k: float(v.abs().max()) for k, v in d0.items()}
            row["census"] = census
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "census"}), flush=True)
    save(f"vjp_n{n}.json", {"stamp": stamp(args.card), "n": n, "seed": args.seed,
                            "loss": "fixed random linear readout of (msa_out, pair_out)",
                            "rows": rows})


# ------------------------------------------------------------------------------ stack


def cmd_stack(args):
    torch.manual_seed(args.seed)
    dm, ref = load_models(args.params)
    dev = Dev(dm)
    ag = dev.ag
    n, ke, kv = args.n, args.extra, args.evo
    ridx = torch.arange(n)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5

    def loss64(lg):
        m, z = embed(ref["f64"], lg, ridx)
        m, z = ref_stack(ref["f64"], m, z, ke, kv)
        return (wm * m).sum() + (wz * z).sum()

    def device_grad(lg, zero=False, census=False, ckpt=False):
        """dL/dlogits: torch embedding (bf16, as the shipped model) -> device stack under one
        tape -> torch readout. Both stack outputs seeded in ONE backward."""
        gc.collect()
        lgt = lg.clone().float().requires_grad_(True)
        m0, z0 = embed(ref["bf16"], lgt, ridx)
        ml, zl = dev.leaf(m0), dev.leaf(z0)
        with TapingCensus() as tc:
            t0 = time.time()
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, ke, kv, ckpt=ckpt)
            dev.sync()
            t1 = time.time()
        mo_h = dev.down(mo.value, m0.shape).double()
        zo_h = dev.down(zo.value, z0.shape).double()
        val = float((wm * mo_h).sum() + (wz * zo_h).sum())
        cen = node_census(ag, [mo, zo]) if census else {"reach": len(ag._reverse_topo([mo, zo]))}
        gmo = torch.zeros_like(wm) if zero else wm
        gzo = torch.zeros_like(wz) if zero else wz
        t2 = time.time()
        ag.backward([mo, zo], [dev.seed(gmo, mo), dev.seed(gzo, zo)])
        dev.sync()
        t3 = time.time()
        gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
        torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
        if ckpt:
            ag.release_pins()
        del mo, zo, ml, zl
        gc.collect()
        return (lgt.grad.double(), val, cen, dict(tc.counts),
                {"fwd_s": t1 - t0, "bwd_s": t3 - t2})

    blob = {"stamp": stamp(args.card), "n": n, "k_extra": ke, "k_evo": kv, "seed": args.seed,
            "loss": "fixed random linear readout of (msa_out, pair_out), logits over all n"}
    g_dev, val_dev, cen, taping, tim = device_grad(logits, census=True, ckpt=args.ckpt)
    blob["device"] = {"loss": val_dev, "grad_norm": float(g_dev.norm()), "timing": tim,
                      "census": cen, "taping_sites": taping}
    print(json.dumps(blob["device"]["timing"]), cen["reach"], flush=True)
    # float64 whole-stack gradient by torch autograd
    lg64 = logits.double().clone().requires_grad_(True)
    t0 = time.time()
    L = loss64(lg64)
    L.backward()
    g64 = lg64.grad.detach().clone()
    blob["f64"] = {"loss": float(L), "grad_norm": float(g64.norm()), "seconds": time.time() - t0}
    blob["device_vs_f64"] = cmp(g_dev, g64)
    print("device vs f64", blob["device_vs_f64"], flush=True)
    # directional finite difference in float64 along the DEVICE gradient
    d = g_dev / g_dev.norm()
    fd = []
    with torch.no_grad():
        for eps in [float(e) for e in args.eps.split(",")]:
            lp, lm = float(loss64(logits.double() + eps * d)), float(loss64(logits.double() - eps * d))
            slope = (lp - lm) / (2 * eps)
            fd.append({"eps": eps, "L+": lp, "L-": lm, "fd_slope": slope,
                       "device_predicted": float(g_dev.norm()),
                       "f64_predicted": float(g64 @ d),
                       "ratio_fd_over_device": slope / float(g_dev.norm())})
            print(json.dumps(fd[-1]), flush=True)
    blob["fd_along_device"] = fd
    best = min(fd, key=lambda r: abs(r["fd_slope"] - r["f64_predicted"]))
    blob["fd_best"] = best
    # controls
    gz_, _, cz, _, _ = device_grad(logits, zero=True)
    blob["zero_seed_max_abs"] = float(gz_.abs().max())
    reach = [cz["reach"]]
    for s in range(2):
        _, _, c, _, _ = device_grad(logits + 0.1 * torch.randn_like(logits))
        reach.append(c["reach"])
    blob["reach_across_steps"] = [cen["reach"]] + reach
    perm = g_dev.flatten()[torch.randperm(g_dev.numel())].reshape(g_dev.shape)
    blob["permuted_device_grad_vs_f64"] = cmp(perm, g64)
    save(f"stack_n{n}_e{ke}_v{kv}{'_ckpt' if args.ckpt else ''}.json", blob)


# ------------------------------------------------------------------------------ reach vs K


def cmd_reach(args):
    """Tape reach and node census per K, for both stacks: must grow by a fixed amount a block."""
    torch.manual_seed(args.seed)
    dm, ref = load_models(args.params)
    dev = Dev(dm)
    ag = dev.ag
    n = args.n
    m0, z0 = embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
    out = {"stamp": stamp(args.card), "n": n, "evo": {}, "extra": {}}
    for stack_name in ("extra", "evo"):
        for k in (1, 2, 3, 4):
            gc.collect()
            ml, zl = dev.leaf(m0.detach()), dev.leaf(z0.detach())
            with TapingCensus() as tc:
                with dev.tt.tape():
                    mo, zo = dev.stack(ml, zl, k if stack_name == "extra" else 0,
                                       k if stack_name == "evo" else 0)
            roots = [zo] if stack_name == "extra" else [mo, zo]
            c = node_census(ag, roots)
            c["taping_sites"] = dict(tc.counts)
            out[stack_name][k] = c
            print(stack_name, k, c["reach"], c["nodes"], flush=True)
            del mo, zo, ml, zl, roots
            gc.collect()
    save(f"reach_n{n}.json", out)


# ------------------------------------------------------------------------------ time


def cmd_time(args):
    torch.manual_seed(args.seed)
    dm, ref = load_models(args.params, device_arm=True)
    dev = Dev(dm)
    ag = dev.ag
    trace = clock_trace()
    blob = {"stamp": stamp(args.card), "steps": args.steps, "warm_skip": args.warm, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0 = embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
        m0, z0 = m0.detach(), z0.detach()
        wm = torch.randn(m0.shape) / m0.numel() ** 0.5
        wz = torch.randn(z0.shape) / z0.numel() ** 0.5
        for stack_name in args.stacks.split(","):
            for k in [int(x) for x in args.ks.split(",")]:
                ke, kv = (k, 0) if stack_name == "extra" else (0, k)
                fw, bw, uf = [], [], []
                t_start = time.time()
                for step in range(args.steps):
                    gc.collect()
                    # untaped forward: the inference cost of the same blocks, same inputs
                    m, z = dev.up(m0), dev.up(z0)
                    dev.sync()
                    t0 = time.time()
                    mo, zo = dev.stack(m, z, ke, kv)
                    dev.sync()
                    uf.append(time.time() - t0)
                    for t in (mo, zo):
                        dev.ttnn.deallocate(t)
                    ml, zl = dev.leaf(m0), dev.leaf(z0)
                    dev.sync()
                    t0 = time.time()
                    with dev.tt.tape():
                        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=args.ckpt)
                    dev.sync()
                    t1 = time.time()
                    roots = [zo] if stack_name == "extra" else [mo, zo]
                    seeds = [dev.seed(wz, zo)] if stack_name == "extra" else \
                        [dev.seed(wm, mo), dev.seed(wz, zo)]
                    dev.sync()
                    t2 = time.time()
                    ag.backward(roots, seeds)
                    dev.sync()
                    t3 = time.time()
                    fw.append(t1 - t0)
                    bw.append(t3 - t2)
                    if args.ckpt:
                        ag.release_pins()
                    del mo, zo, ml, zl, roots, seeds
                t_end = time.time()
                w = slice(args.warm, None)
                pt = {"n": n, "stack": stack_name, "K": k, "ckpt": args.ckpt,
                      "fwd_taped": _stats(fw[w]), "bwd": _stats(bw[w]),
                      "fwd_untaped": _stats(uf[w]),
                      "step": _stats([a + b for a, b in zip(fw[w], bw[w])]),
                      "first_step": {"fwd": fw[0], "bwd": bw[0], "untaped": uf[0]},
                      "aiclk": window(trace, t_start, t_end), "loadavg": os.getloadavg(),
                      "wall": [t_start, t_end]}
                blob["points"].append(pt)
                print(json.dumps({k_: pt[k_] for k_ in ("n", "stack", "K", "aiclk")}),
                      "fwd", round(pt["fwd_taped"]["mean"], 4), "bwd",
                      round(pt["bwd"]["mean"], 4), "untaped", round(pt["fwd_untaped"]["mean"], 4),
                      flush=True)
                save(args.out, blob)
    trace.stop()
    blob["fits"] = fits(blob["points"])
    save(args.out, blob)


def _stats(xs):
    xs = np.asarray(xs, float)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
            "min": float(xs.min()), "max": float(xs.max()), "n": int(len(xs))}


def fits(points):
    from perf.hallgrad.blocksweep import fit
    out = {}
    groups = collections.defaultdict(list)
    for p in points:
        groups[(p["n"], p["stack"], p["ckpt"])].append(p)
    for (n, s, c), ps in groups.items():
        ps = sorted(ps, key=lambda p: p["K"])
        if len(ps) < 2:
            continue
        ks = [p["K"] for p in ps]
        key = f"n{n}_{s}{'_ckpt' if c else ''}"
        out[key] = {m: fit(ks, [p[m]["mean"] for p in ps])
                    for m in ("fwd_taped", "bwd", "step", "fwd_untaped")}
    return out


# ------------------------------------------------------------------------------ fit (memory)


def cmd_fit(args):
    """Largest K of Evoformer blocks whose taped forward + backward completes uncheckpointed."""
    dm, ref = load_models(args.params)
    dev = Dev(dm)
    ag, ttnn = dev.ag, dev.ttnn
    n = args.n
    m0, z0 = embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
    m0, z0 = m0.detach(), z0.detach()
    res = {"stamp": stamp(args.card), "n": n, "ckpt": args.ckpt, "tries": []}

    def dram_used():
        mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

    for k in [int(x) for x in args.ks.split(",")]:
        gc.collect()
        rec = {"K": k}
        base = dram_used()
        try:
            ml, zl = dev.leaf(m0), dev.leaf(z0)
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, 0, k, ckpt=args.ckpt)
            dev.sync()
            rec["after_fwd_bytes"] = dram_used() - base
            ag.backward([mo, zo], [dev.seed(torch.randn(m0.shape), mo),
                                   dev.seed(torch.randn(z0.shape), zo)])
            dev.sync()
            rec["after_bwd_bytes"] = dram_used() - base
            rec["ok"] = True
        except Exception as e:                      # an allocation refusal is the answer
            rec["ok"] = False
            rec["error"] = str(e).splitlines()[0][:300]
        finally:
            if args.ckpt:
                ag.release_pins()
            mo = zo = ml = zl = None
            gc.collect()
        res["tries"].append(rec)
        print(json.dumps(rec), flush=True)
        save(f"fit_n{n}{'_ckpt' if args.ckpt else ''}.json", res)
        if not rec["ok"]:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vjp", "stack", "reach", "time", "fit"])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "3")))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--blocks", default=None, help="boundary indices to score (vjp)")
    ap.add_argument("--controls-all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", default="1e-1,3e-2,1e-2,3e-3,1e-3")
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--ns", default="128,256")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", default="time.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"vjp": cmd_vjp, "stack": cmd_stack, "reach": cmd_reach, "time": cmd_time,
     "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
