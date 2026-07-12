"""Profile the token-level DiT ATTENTION half (AttentionPairBias) in Protenix-v2.

The token-level DiffusionTransformer (24 blocks) is the largest measured component
share in the codebase (docs/atomattention-kernel-scout.md: 67.1% of a Boltz-2
diffusion step). Each block is an ATTENTION half (AdaLN -> AttentionPairBias ->
s-gate -> residual) and an FFN half (ConditionedTransitionBlock, scouted separately).
This scout isolates the attention half.

Methodology matches the prior kernel scouts (warm, device-synchronized, real
checkpoint weights, real target). Every timed region is the SECOND same-shape
forward in one process and is bracketed by a device sync.

Subcommands:
  share    coarse: denoise/fold + token_dit/denoise (2 syncs/step, no inner wraps)
  attn     fine:   apb/token_dit + adaln_a/token_dit (inner apb+adaln wraps)
  decomp   ttnn op-launch counts inside ONE warm token-level apb call
  ab       gate-pack fusion A/B on the real captured apb input (bit-exact + timing)
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import torch
import ttnn
import yaml


class _Timer:
    """Replace a bound callable with a sync-bracketed timer (transparent proxy)."""

    def __init__(self, fn, sink, device):
        self._fn = fn
        self._sink = sink
        self._dev = device

    def __call__(self, *a, **k):
        ttnn.synchronize_device(self._dev)
        t0 = time.perf_counter()
        out = self._fn(*a, **k)
        ttnn.synchronize_device(self._dev)
        self._sink.append(time.perf_counter() - t0)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fn"), name)


class _Capture:
    """Proxy that records the first call's (args, kwargs) then delegates.

    Its class defines __call__, so apb(...) inside the model resolves here
    (an instance attribute would not, since dunders bind on the type)."""

    def __init__(self, fn):
        self._fn = fn
        self.grabbed = None

    def __call__(self, *a, **k):
        if self.grabbed is None:
            self.grabbed = (a, dict(k))
        return self._fn(*a, **k)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fn"), name)


def _load(args):
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device

    spec = yaml.safe_load(Path(args.input).read_text())
    sequence = spec["sequences"][0]["protein"]["sequence"]
    feats = build_protein_features(sequence)
    device = get_device()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    model = Protenix.load_from_checkpoint(
        args.checkpoint, compute_kernel_config=config, device=device
    )
    return model, feats, device, len(sequence)


def _wrap_dit_apb(diffusion, apb_sink, adaln_sink, device):
    """Wrap the 24 per-block apb (and attention-side AdaLN) instances in place."""
    new = []
    for (adaln_a, apb, ctb_adaln, A, Cc) in diffusion._dit:
        wa = _Timer(adaln_a, adaln_sink, device) if adaln_sink is not None else adaln_a
        wp = _Timer(apb, apb_sink, device) if apb_sink is not None else apb
        new.append((wa, wp, ctb_adaln, A, Cc))
    diffusion._dit = new


def _share(args):
    model, feats, device, ntok = _load(args)
    diff = model.diffusion
    denoise_t, dit_t = [], []
    diff.denoise = _Timer(diff.denoise, denoise_t, device)
    diff._token_dit_device = _Timer(diff._token_dit_device, dit_t, device)

    def fold():
        denoise_t.clear(); dit_t.clear()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        model.fold(feats, n_step=args.steps, n_sample=args.samples, seed=0,
                   return_confidence=True)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0

    fold()                    # warm
    total = fold()            # timed
    denoise = sum(denoise_t)
    dit = sum(dit_t)
    steps = len(denoise_t)
    print(json.dumps({
        "mode": "share",
        "tokens": ntok, "dit_blocks": diff.DIT_BLOCKS,
        "sampling_steps": args.steps, "samples": args.samples,
        "fold_total_s": total,
        "denoise_total_s": denoise, "denoise_share_of_fold": denoise / total,
        "denoise_per_step_s": denoise / max(steps, 1),
        "token_dit_total_s": dit, "token_dit_share_of_denoise": dit / denoise,
        "token_dit_per_step_s": dit / max(steps, 1),
    }, sort_keys=True), flush=True)


def _attn(args):
    model, feats, device, ntok = _load(args)
    diff = model.diffusion
    dit_t, apb_t, adaln_t = [], [], []
    diff._token_dit_device = _Timer(diff._token_dit_device, dit_t, device)
    _wrap_dit_apb(diff, apb_t, adaln_t, device)

    def fold():
        dit_t.clear(); apb_t.clear(); adaln_t.clear()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        model.fold(feats, n_step=args.steps, n_sample=args.samples, seed=0,
                   return_confidence=True)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0

    fold()                    # warm
    fold()                    # timed
    dit = sum(dit_t); apb = sum(apb_t); adaln = sum(adaln_t)
    steps = len(dit_t)
    print(json.dumps({
        "mode": "attn",
        "tokens": ntok, "dit_blocks": diff.DIT_BLOCKS,
        "sampling_steps": args.steps, "samples": args.samples,
        "token_dit_total_s": dit,
        "apb_total_s": apb, "apb_share_of_token_dit": apb / dit,
        "adaln_a_total_s": adaln, "adaln_a_share_of_token_dit": adaln / dit,
        "attn_half_share_of_token_dit": (apb + adaln) / dit,
        "apb_per_block_call_s": apb / max(len(apb_t), 1),
        "token_dit_per_step_s": dit / max(steps, 1),
    }, sort_keys=True), flush=True)


# --------------------------------------------------------------------------- #
# op-launch decomposition of a single warm token-level apb call
# --------------------------------------------------------------------------- #
def _capture_apb_input(model, feats, diff):
    """Insert a capturing proxy at DiT block 0, run a short warm fold, return
    (real_apb, args, kwargs) for the first per-step apb call."""
    real = diff._dit[0][1]
    cap = _Capture(real)
    t = diff._dit[0]
    diff._dit[0] = (t[0], cap, t[2], t[3], t[4])
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)
    diff._dit[0] = t   # restore
    a, k = cap.grabbed
    return real, a, k


def _decomp(args):
    model, feats, device, ntok = _load(args)
    diff = model.diffusion
    real_apb, cap_a, cap_k = _capture_apb_input(model, feats, diff)
    grabbed = {"s": cap_a[0], "z": cap_a[1], "kw": cap_k}
    real_call = real_apb.__call__

    # count ttnn op launches over one apb call on the captured input
    counts = collections.Counter()
    names = ["linear", "matmul", "layer_norm", "add", "multiply", "permute",
             "reshape", "unsqueeze", "squeeze", "to_layout", "pad", "slice",
             "softmax", "typecast", "deallocate"]
    exp_names = ["nlp_create_qkv_heads", "nlp_concat_heads"]
    tf_names = ["scaled_dot_product_attention"]
    saved = {}
    for n in names:
        saved[("ttnn", n)] = getattr(ttnn, n)
    for n in exp_names:
        saved[("exp", n)] = getattr(ttnn.experimental, n)
    for n in tf_names:
        saved[("tf", n)] = getattr(ttnn.transformer, n)

    def mk(orig_fn, key):
        def wrapped(*a, **k):
            counts[key] += 1
            return orig_fn(*a, **k)
        return wrapped

    for n in names:
        setattr(ttnn, n, mk(saved[("ttnn", n)], n))
    for n in exp_names:
        setattr(ttnn.experimental, n, mk(saved[("exp", n)], n))
    for n in tf_names:
        setattr(ttnn.transformer, n, mk(saved[("tf", n)], n))
    try:
        _ = real_call(grabbed["s"], grabbed["z"], **grabbed["kw"])
        ttnn.synchronize_device(device)
    finally:
        for n in names:
            setattr(ttnn, n, saved[("ttnn", n)])
        for n in exp_names:
            setattr(ttnn.experimental, n, saved[("exp", n)])
        for n in tf_names:
            setattr(ttnn.transformer, n, saved[("tf", n)])

    print(json.dumps({
        "mode": "decomp", "tokens": ntok,
        "apb_bias_precomputed": bool(grabbed["kw"].get("bias_precomputed")),
        "op_launches": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "total_launches": sum(counts.values()),
    }, sort_keys=True), flush=True)


# --------------------------------------------------------------------------- #
# gate-pack A/B: fold proj_g into the packed QKV linear (both consume the same s)
# --------------------------------------------------------------------------- #
def _ab(args):
    from tt_bio.tenstorrent import CORE_GRID_MAIN, _sdpa_program_config_for_lengths, _dtype
    model, feats, device, ntok = _load(args)
    diff = model.diffusion
    apb, cap_a, _ = _capture_apb_input(model, feats, diff)
    ckc = apb.compute_kernel_config
    s = cap_a[0]; bias = cap_a[1]

    # baseline forward = the shipping token-level apb (bias_precomputed=True)
    def baseline():
        return apb(s, bias, bias_precomputed=True)

    # build a packed [qkv | g] weight + bias; g and qkv both left-multiply s
    qkv_w = ttnn.to_torch(apb.qkv_weight)          # (c_in, qkv_out)
    g_w = ttnn.to_torch(apb.g_weight)              # (c_in, g_out)
    qkv_b = ttnn.to_torch(apb.qkv_bias)            # (qkv_out,)
    qkv_out = qkv_w.shape[-1]
    packed = torch.cat([qkv_w, g_w], dim=-1)
    packed_b = torch.cat([qkv_b, torch.zeros(g_w.shape[-1], dtype=qkv_b.dtype)])
    packed_w = ttnn.from_torch(packed, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    packed_bias = ttnn.from_torch(packed_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    n_heads, head_dim = apb.n_heads, apb.head_dim
    scale = head_dim ** -0.5
    o_weight = apb.o_weight

    def packed_fwd():
        qkvg = ttnn.linear(s, packed_w, bias=packed_bias, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        qkv = qkvg[:, :, :qkv_out]
        g = qkvg[:, :, qkv_out:]
        ttnn.deallocate(qkvg)
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=n_heads, num_kv_heads=n_heads, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]))
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = o[:, :, :, :head_dim]
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
        o = ttnn.permute(o, (0, 2, 1))
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], dtype=_dtype())
        ttnn.deallocate(g)
        x = ttnn.linear(o, o_weight, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(o)
        return x

    # parity
    ref = torch.Tensor(ttnn.to_torch(baseline())).float()
    got = torch.Tensor(ttnn.to_torch(packed_fwd())).float()
    max_abs = float((ref - got).abs().max())
    ref_abs_max = float(ref.abs().max())
    rel = max_abs / ref_abs_max if ref_abs_max > 0 else 0.0
    denom = (ref.norm() * got.norm())
    pcc = float((ref.flatten() @ got.flatten()) / denom) if float(denom) > 0 else 0.0

    def timeit(fn, reps):
        fn(); ttnn.synchronize_device(device)          # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(device)
        return (time.perf_counter() - t0) / reps

    reps = args.repeats
    base_s = timeit(baseline, reps)
    pack_s = timeit(packed_fwd, reps)
    print(json.dumps({
        "mode": "ab", "tokens": ntok, "reps": reps,
        "baseline_s": base_s, "packed_gate_qkv_s": pack_s,
        "speedup": base_s / pack_s,
        "pcc": pcc, "max_abs": max_abs, "ref_abs_max": ref_abs_max, "rel_err": rel,
    }, sort_keys=True), flush=True)


class _GatePacked:
    """Drop-in for a token-level AttentionPairBias whose proj_g is folded into the
    packed QKV linear (both left-multiply the same input s). Delegates any call that
    is not the token-level bias_precomputed path (the DiT per-step case) to the
    original apb, so it is safe to install unconditionally."""

    def __init__(self, apb, device):
        from tt_bio.tenstorrent import CORE_GRID_MAIN, _sdpa_program_config_for_lengths, _dtype
        self._apb = apb
        self._ckc = apb.compute_kernel_config
        self._cg = CORE_GRID_MAIN
        self._sdpa_cfg = _sdpa_program_config_for_lengths
        self._dtype = _dtype
        self._nh, self._hd = apb.n_heads, apb.head_dim
        self._scale = apb.head_dim ** -0.5
        self._o_weight = apb.o_weight
        qkv_w = ttnn.to_torch(apb.qkv_weight)
        g_w = ttnn.to_torch(apb.g_weight)
        qkv_b = ttnn.to_torch(apb.qkv_bias)
        self._qkv_out = qkv_w.shape[-1]
        self._packed_w = ttnn.from_torch(torch.cat([qkv_w, g_w], dim=-1),
                                         layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        self._packed_b = ttnn.from_torch(torch.cat([qkv_b, torch.zeros(g_w.shape[-1], dtype=qkv_b.dtype)]),
                                         layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    def __call__(self, s, z, keys_indexing=None, seq_mask=None, bias_precomputed=False):
        if self._apb.atom_level or not bias_precomputed or keys_indexing is not None or seq_mask is not None:
            return self._apb(s, z, keys_indexing=keys_indexing, seq_mask=seq_mask, bias_precomputed=bias_precomputed)
        qkvg = ttnn.linear(s, self._packed_w, bias=self._packed_b, compute_kernel_config=self._ckc, core_grid=self._cg)
        qkv = qkvg[:, :, :self._qkv_out]
        g = qkvg[:, :, self._qkv_out:]
        ttnn.deallocate(qkvg)
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self._nh, num_kv_heads=self._nh, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=z, is_causal=False, scale=self._scale,
            program_config=self._sdpa_cfg(q.shape[2], k.shape[2]))
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = o[:, :, :, :self._hd]
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
        o = ttnn.permute(o, (0, 2, 1))
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], dtype=self._dtype())
        ttnn.deallocate(g)
        x = ttnn.linear(o, self._o_weight, compute_kernel_config=self._ckc, core_grid=self._cg)
        ttnn.deallocate(o)
        return x

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_apb"), name)


def _e2e(args):
    import numpy as np
    model, feats, device, ntok = _load(args)
    diff = model.diffusion

    def timed_fold():
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        coords, _ = model.fold(feats, n_step=args.steps, n_sample=1, seed=0, return_confidence=True)
        ttnn.synchronize_device(device)
        return coords[0].float().cpu().numpy(), time.perf_counter() - t0

    # baseline (shipping apb) + self-determinism check (same seed twice)
    base_xyz0, _ = timed_fold()        # warm
    base_xyz, base_t = timed_fold()
    self_rmsd = float(np.sqrt(((base_xyz0 - base_xyz) ** 2).sum(-1).mean()))

    # install gate-pack on all 24 DiT apbs, then re-fold (fresh trace/bias caches per fold)
    diff._dit = [(a, _GatePacked(apb, device), c, A, Cc) for (a, apb, c, A, Cc) in diff._dit]
    timed_fold()                       # warm
    pack_xyz, pack_t = timed_fold()

    rmsd = float(np.sqrt(((base_xyz - pack_xyz) ** 2).sum(-1).mean()))
    max_dev = float(np.abs(base_xyz - pack_xyz).max())
    print(json.dumps({
        "mode": "e2e", "tokens": ntok, "sampling_steps": args.steps,
        "baseline_fold_s": base_t, "gatepack_fold_s": pack_t,
        "e2e_speedup": base_t / pack_t,
        "coord_rmsd_A": rmsd, "coord_max_dev_A": max_dev,
        "baseline_self_rmsd_A": self_rmsd,
    }, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("share", "attn", "decomp", "ab", "e2e"))
    p.add_argument("--input", default="examples/prot.yaml")
    p.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    {"share": _share, "attn": _attn, "decomp": _decomp, "ab": _ab, "e2e": _e2e}[args.mode](args)


if __name__ == "__main__":
    main()
