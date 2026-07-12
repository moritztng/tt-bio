"""Profile the AF3-style atom-attention encoder/decoder (windowed local SDPA +
per-window gather + per-head pair bias) in Protenix-v2 and Boltz-2.

Matches the actual-input, warm, synchronized methodology of the prior kernel
scouts (docs/permodel-kernel-scout.md, docs/boltz2-protenix-kernel-scout.md):
every number below is the SECOND same-shape forward in one process, with a
device sync bracketing each timed region.

Subcommands:
  protenix   share of AtomAttentionEncoder (input embedder, once) + the two
             per-step diffusion AtomTransformers (atxE + atxD) in a full fold.
  boltz2     share of the atom-level windowed DiffusionTransformer (encoder +
             decoder, 2x/step) in a full Boltz-2 predict.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn
import yaml


# --------------------------------------------------------------------------- #
# shared timing wrapper: replace a bound-callable attribute with a timer.
# --------------------------------------------------------------------------- #
class _Timer:
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
        # transparent proxy for everything except the wrapped __call__
        # (e.g. atxE.precompute_biases, called by DiffusionModule._atom_cond)
        return getattr(object.__getattribute__(self, "_fn"), name)


def _protenix(args):
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

    enc_input = []   # AtomAttentionEncoder (input embedder), once per fold
    enc_diff = []    # diffusion atxE, once per sampling step
    dec_diff = []    # diffusion atxD, once per sampling step

    model.input_aae = _Timer(model.input_aae, enc_input, device)
    model.diffusion.atxE = _Timer(model.diffusion.atxE, enc_diff, device)
    model.diffusion.atxD = _Timer(model.diffusion.atxD, dec_diff, device)

    def fold():
        enc_input.clear(); enc_diff.clear(); dec_diff.clear()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        model.fold(feats, n_step=args.steps, n_sample=args.samples, seed=0,
                   return_confidence=True)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0

    warm = fold()
    total = fold()
    ein = sum(enc_input); edf = sum(enc_diff); ddf = sum(dec_diff)
    atom_total = ein + edf + ddf
    print(json.dumps({
        "model": "protenix-v2",
        "tokens": len(sequence),
        "sampling_steps": args.steps,
        "samples": args.samples,
        "warmup_total_s": warm,
        "timed_total_s": total,
        "input_encoder_calls": len(enc_input),
        "input_encoder_s": ein,
        "diff_encoder_calls": len(enc_diff),
        "diff_encoder_s": edf,
        "diff_decoder_calls": len(dec_diff),
        "diff_decoder_s": ddf,
        "atom_attention_total_s": atom_total,
        "atom_attention_share": atom_total / total,
        "diff_atom_share": (edf + ddf) / total,
        "free_atom_ceiling": total / (total - atom_total),
    }, sort_keys=True), flush=True)


_OPS = ("matmul", "linear", "softmax", "permute", "embedding", "pad", "reshape",
        "add", "multiply", "mul", "to_layout", "slice", "layer_norm", "unsqueeze",
        "relu", "concat", "transpose")


def _protenix_decomp(args):
    """Count ttnn op launches per category inside ONE diffusion atxE call, on real
    fold inputs. Atom attention is tiny-tensor (nb windows x 32 queries) and
    dispatch-bound, so launch count per category is the decomposition."""
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device

    spec = yaml.safe_load(Path(args.input).read_text())
    sequence = spec["sequences"][0]["protein"]["sequence"]
    feats = build_protein_features(sequence)
    device = get_device()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(args.checkpoint, compute_kernel_config=config, device=device)

    counts = {op: 0 for op in _OPS}
    recording = {"on": False}
    originals = {}
    for op in _OPS:
        fn = getattr(ttnn, op, None)
        if fn is None:
            continue
        originals[op] = fn

        def make(_op, _fn):
            def wrapped(*a, **k):
                if recording["on"]:
                    counts[_op] += 1
                return _fn(*a, **k)
            return wrapped
        setattr(ttnn, op, make(op, fn))

    orig_call = model.diffusion.atxE.__call__.__func__ if hasattr(model.diffusion.atxE.__call__, "__func__") else None
    # wrap the atxE instance: record ops for exactly the FIRST call of the timed fold
    atxE = model.diffusion.atxE
    done = {"v": False}
    orig_atxE_call = type(atxE).__call__

    def timed_atxE(self, *a, **k):
        if not done["v"]:
            recording["on"] = True
            out = orig_atxE_call(self, *a, **k)
            recording["on"] = False
            done["v"] = True
            return out
        return orig_atxE_call(self, *a, **k)

    type(atxE).__call__ = timed_atxE

    # warm then measure (recording triggers on first diffusion step of 2nd fold)
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)
    done["v"] = False
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)

    total = sum(counts.values())
    print(json.dumps({
        "model": "protenix-v2-atxE-one-call",
        "n_blocks": atxE.n_blocks,
        "ops_total": total,
        "counts": {k: v for k, v in counts.items() if v},
    }, sort_keys=True), flush=True)


def _pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0, 0.0
    va = a - a.mean(); vb = b - b.mean()
    denom = (va.norm() * vb.norm()).clamp_min(1e-12)
    pcc = float((va @ vb) / denom)
    return pcc, float((a - b).abs().max())


def _protenix_ab(args):
    """A/B the atom-transformer windowed attention on REAL captured diffusion inputs:
    baseline (separate K,V projection + separate KV windowing) vs a fused variant that
    packs the K+V projection into one linear and windows the packed KV once, then splits.
    Bit-exactness (identity refactor) + warm timing over many repeats."""
    from tt_bio.protenix import AtomTransformer
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device

    spec = yaml.safe_load(Path(args.input).read_text())
    sequence = spec["sequences"][0]["protein"]["sequence"]
    feats = build_protein_features(sequence)
    device = get_device()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(args.checkpoint, compute_kernel_config=config, device=device)

    atxE = model.diffusion.atxE
    stash = {}
    orig_call = type(atxE).__call__

    def capture(self, a, s, p, mask_trunked, bias_cache=None):
        if "args" not in stash:
            stash["args"] = (a, s, p, mask_trunked, bias_cache)
        return orig_call(self, a, s, p, mask_trunked, bias_cache=bias_cache)

    type(atxE).__call__ = capture
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)  # warm + capture
    type(atxE).__call__ = orig_call
    a, s, p, mask_trunked, bias_cache = stash["args"]

    # ---- fused KV-pack _attention (monkeypatched onto a clone of atxE) ----
    H, dh, nq, nk, PADL = atxE.N_HEADS, atxE.HEAD_DIM, atxE.N_QUERIES, atxE.N_KEYS, atxE.PAD_LEFT

    def build_packed_kv(apb):
        wk = atxE._w[apb + "attention.linear_k.weight"]  # (out,in)
        wv = atxE._w[apb + "attention.linear_v.weight"]
        wkv = torch.cat([wk, wv], dim=0)                 # (2*out, in)
        return ttnn.from_torch(wkv.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                               device=device, dtype=ttnn.bfloat16)

    packed_cache = {}

    def fused_windows_kv_packed(x, N, NP):
        # window packed KV (last dim 2*H*dh) once, return (Kb, Vb) each (nb,H,nk,dh)
        nb = NP // nq
        C = x.shape[-1]  # 2*H*dh
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        Lp = PADL + NP + nk
        x = ttnn.pad(x, [[0, 0], [PADL, Lp - PADL - N], [0, 0]], 0.0)
        x = ttnn.reshape(x, (Lp, C))
        idx = atxE._kv_window_idx(nb, nq, nk, NP)
        x = ttnn.embedding(idx, x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (nb, nk, 2, H, dh))
        x = ttnn.permute(x, (2, 0, 3, 1, 4))            # (2,nb,H,nk,dh)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        Kb = x[0]; Vb = x[1]
        return Kb, Vb

    def fused_attention(self, q_norm, kv_norm, p, apb, N, NP, pad_bias, z_pre=None):
        Q = self._lin(q_norm, apb + "attention.linear_q.weight", apb + "attention.linear_q.bias")
        wkv = packed_cache.setdefault(apb, build_packed_kv(apb))
        KV = ttnn.linear(kv_norm, wkv, compute_kernel_config=self.compute_kernel_config,
                         core_grid=CORE_GRID_MAIN)
        Qb = self._windows_q(Q, N, NP)
        Kb, Vb = fused_windows_kv_packed(KV, N, NP)
        z = z_pre if z_pre is not None else self._pair_bias(p, apb)
        sc = ttnn.matmul(Qb, ttnn.permute(Kb, (0, 1, 3, 2)), compute_kernel_config=self.compute_kernel_config)
        sc = ttnn.multiply(sc, dh ** -0.5)
        sc = ttnn.add(ttnn.add(sc, z), pad_bias)
        o = ttnn.matmul(ttnn.softmax(sc, dim=-1), Vb, compute_kernel_config=self.compute_kernel_config)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (NP, H * dh))
        o = ttnn.slice(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [0, 0], [N, H * dh])
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    from tt_bio.protenix import CORE_GRID_MAIN

    def run(fused):
        if fused:
            type(atxE)._attention = fused_attention
        out = orig_call(atxE, a, s, p, mask_trunked, bias_cache=bias_cache)
        if fused:
            type(atxE)._attention = base_attention
        return out

    base_attention = type(atxE)._attention

    # correctness
    out_base = orig_call(atxE, a, s, p, mask_trunked, bias_cache=bias_cache)
    hb = torch.Tensor(ttnn.to_torch(out_base)).float()
    out_fused = run(True)
    hf = torch.Tensor(ttnn.to_torch(out_fused)).float()
    pcc, maxabs = _pcc(hb, hf)

    # timing (warm, synchronized, median of repeats)
    import statistics
    def timed(fused):
        if fused:
            type(atxE)._attention = fused_attention
        samples = []
        for _ in range(args.repeats):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            orig_call(atxE, a, s, p, mask_trunked, bias_cache=bias_cache)
            ttnn.synchronize_device(device)
            samples.append(time.perf_counter() - t0)
        if fused:
            type(atxE)._attention = base_attention
        return statistics.median(samples), samples

    base_s, base_samples = timed(False)
    fused_s, fused_samples = timed(True)
    print(json.dumps({
        "model": "protenix-v2-atxE-kvpack-ab",
        "tokens": len(sequence),
        "repeats": args.repeats,
        "pcc": pcc, "max_abs": maxabs,
        "baseline_median_s": base_s,
        "fused_median_s": fused_s,
        "atxE_speedup": base_s / fused_s,
        "baseline_samples_s": base_samples,
        "fused_samples_s": fused_samples,
    }, sort_keys=True), flush=True)


# Boltz-2 note: its predict fans jobs to mp-spawn worker processes, so the
# atom-level DiffusionTransformer (encoder + decoder) is timed with a spawn-safe
# scripts/aa_site/sitecustomize.py that patches the class at interpreter startup
# (reaches the workers too). See the Reproduce section of
# docs/atomattention-kernel-scout.md. Within one diffusion sample the 200 sampling
# steps are warm same-shape repeats, so the per-step median is the warm number.


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("protenix")
    p.add_argument("--input", default="examples/prot.yaml")
    p.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--samples", type=int, default=1)
    p.set_defaults(fn=_protenix)

    d = sub.add_parser("protenix-decomp")
    d.add_argument("--input", default="examples/prot.yaml")
    d.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    d.set_defaults(fn=_protenix_decomp)

    ab = sub.add_parser("protenix-ab")
    ab.add_argument("--input", default="examples/prot.yaml")
    ab.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    ab.add_argument("--repeats", type=int, default=50)
    ab.set_defaults(fn=_protenix_ab)

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    args.fn(args)


if __name__ == "__main__":
    main()
