"""Scout the diffusion CONDITIONING path of Protenix-v2 (the unexplored component
after the shared trunk primitives and the DiT attention half were closed).

Targets the 24-block token DiT's per-block conditioning: the two AdaLN modulations
(pre-attention + pre-transition) and the two s-gate sigmoids, plus the swish-gate.
These run 24x per sampling step and were NOT measured by the prior
atom-attention / DiT-attention scouts (which covered the attention half only) or
the difftransformer-swiglu scout (which covered ConditionedTransitionBlock's
swish/gates matmuls, not the AdaLN modulation or the s-gates).

Subcommands:
  profile   share of AdaLN + s-gates in a warm 200-step fold, and per-DiT-block
            dispatch decomposition.
  decomp    ttnn op-launch counts inside ONE _token_dit_device call (24 blocks).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, ttnn, yaml
from tt_bio.tenstorrent import _adaln_memory_config  # noqa: E402 (optional dep, lazy)


class _Timer:
    """Wrap a callable Module; time + count its invocations, device-synced."""
    def __init__(self, fn, sink, cnt, device):
        self._fn = fn; self._sink = sink; self._cnt = cnt; self._dev = device
    def __call__(self, *a, **k):
        ttnn.synchronize_device(self._dev)
        t0 = time.perf_counter()
        out = self._fn(*a, **k)
        ttnn.synchronize_device(self._dev)
        self._sink.append(time.perf_counter() - t0)
        self._cnt.append(1)
        return out
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fn"), name)


def _load(args):
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device
    spec = yaml.safe_load(Path(args.input).read_text())
    seq = spec["sequences"][0]["protein"]["sequence"]
    feats = build_protein_features(seq)
    device = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(args.checkpoint, compute_kernel_config=cfg, device=device)
    return model, feats, device, seq


def _profile(args):
    model, feats, device, seq = _load(args)
    diff = model.diffusion
    dit_times, dit_cnt = [], []
    adaln_times, adaln_cnt = [], []   # both AdaLN modulations per block

    # wrap _token_dit_device
    orig_dit = diff._token_dit_device
    def dit_wrapped(a_t, s_t, biases, NT):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = orig_dit(a_t, s_t, biases, NT)
        ttnn.synchronize_device(device)
        dit_times.append(time.perf_counter() - t0)
        return out
    diff._token_dit_device = dit_wrapped

    # wrap each AdaLN in the _dit tuples (positions 0 and 2)
    new_dit = []
    for (adaln_a, apb, ctb_adaln, A, Cc) in diff._dit:
        new_dit.append((_Timer(adaln_a, adaln_times, adaln_cnt, device), apb,
                        _Timer(ctb_adaln, adaln_times, adaln_cnt, device), A, Cc))
    diff._dit = new_dit

    def fold():
        dit_times.clear(); adaln_times.clear(); adaln_cnt.clear()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        model.fold(feats, n_step=args.steps, n_sample=1, seed=0, return_confidence=False)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0

    warm = fold()
    total = fold()
    print(json.dumps({
        "model": "protenix-v2-conditioning-profile",
        "tokens": len(seq),
        "steps": args.steps,
        "warmup_total_s": warm,
        "timed_total_s": total,
        "dit_calls": len(dit_times),
        "dit_total_s": sum(dit_times),
        "dit_share": sum(dit_times) / total,
        "adaln_calls": len(adaln_times),
        "adaln_total_s": sum(adaln_times),
        "adaln_share_of_fold": sum(adaln_times) / total,
        "adaln_share_of_dit": sum(adaln_times) / sum(dit_times) if dit_times else 0,
        "dit_per_call_ms": 1000 * sum(dit_times) / max(1, len(dit_times)),
        "adaln_per_call_ms": 1000 * sum(adaln_times) / max(1, len(adaln_times)),
        "adaln_free_dit_ceiling": sum(dit_times) / (sum(dit_times) - sum(adaln_times)) if dit_times else 0,
    }, sort_keys=True), flush=True)


_OPS = ("linear", "sigmoid", "multiply", "multiply_", "add", "add_",
        "layer_norm", "to_memory_config", "deallocate", "reshape", "matmul")


def _decomp(args):
    model, feats, device, seq = _load(args)
    diff = model.diffusion
    counts = {op: 0 for op in _OPS}
    rec = {"on": False}
    originals = {}
    for op in _OPS:
        fn = getattr(ttnn, op, None)
        if fn is None:
            continue
        originals[op] = fn
        def make(_op, _fn):
            def wrapped(*a, **k):
                if rec["on"]:
                    counts[_op] += 1
                return _fn(*a, **k)
            return wrapped
        setattr(ttnn, op, make(op, fn))

    orig_dit = diff._token_dit_device
    done = {"v": False}
    def dit_wrapped(a_t, s_t, biases, NT):
        if not done["v"]:
            rec["on"] = True
            out = orig_dit(a_t, s_t, biases, NT)
            rec["on"] = False
            done["v"] = True
            return out
        return orig_dit(a_t, s_t, biases, NT)
    diff._token_dit_device = dit_wrapped

    # warm then capture on 2nd fold's first DiT call
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)
    done["v"] = False
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)
    total = sum(counts.values())
    print(json.dumps({
        "model": "protenix-v2-dit-one-call-decomp",
        "dit_blocks": diff.DIT_BLOCKS,
        "ops_total": total,
        "counts": {k: v for k, v in counts.items() if v},
    }, sort_keys=True), flush=True)


def _adaln_fused(adaln, a, s, large_seq_len=False):
    """Reimplement AdaLN.__call__ with ttnn.addcmul fusing multiply_+add_ into one eltwise.
    math: a_out = LN(a) * sigmoid(s_scale) + s_bias,  s_scale=lin(LN(s,w)), s_bias=lin(LN(s,w))."""
    memory_config = _adaln_memory_config(adaln.atom_level, large_seq_len)
    if adaln.atom_level:
        a = ttnn.to_memory_config(a, memory_config=memory_config)
        s = ttnn.to_memory_config(s, memory_config=memory_config)
    a = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=adaln.compute_kernel_config)
    s = ttnn.layer_norm(s, weight=adaln.s_norm_weight, epsilon=1e-5,
                        compute_kernel_config=adaln.compute_kernel_config)
    s_scale = ttnn.linear(s, adaln.s_scale_weight, bias=adaln.s_scale_bias,
                          compute_kernel_config=adaln.compute_kernel_config, memory_config=memory_config)
    s_bias = ttnn.linear(s, adaln.s_bias_weight, compute_kernel_config=adaln.compute_kernel_config,
                         memory_config=memory_config)
    sig = ttnn.sigmoid(s_scale)
    ttnn.deallocate(s_scale)
    a = ttnn.addcmul(s_bias, a, sig, memory_config=memory_config)
    ttnn.deallocate(s_bias)
    a = ttnn.to_memory_config(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return a


def _ab(args):
    model, feats, device, seq = _load(args)
    diff = model.diffusion
    adaln_a0 = diff._dit[0][0]
    stash = {}
    orig_call = type(adaln_a0).__call__

    class _Cap:
        def __init__(self, inner): self._inner = inner
        def __call__(self, a, s, large_seq_len=False):
            if "a" not in stash:
                ttnn.synchronize_device(device)
                stash["a"] = ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=a.dtype)
                stash["s"] = ttnn.clone(s, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=s.dtype)
            return orig_call(self._inner, a, s, large_seq_len=large_seq_len)
        def __getattr__(self, n): return getattr(self._inner, n)
    adaln_a, apb, ctb_adaln, A, Cc = diff._dit[0]
    diff._dit[0] = (_Cap(adaln_a), apb, ctb_adaln, A, Cc)
    model.fold(feats, n_step=2, n_sample=1, seed=0, return_confidence=False)  # warm + capture
    diff._dit[0] = (adaln_a, apb, ctb_adaln, A, Cc)   # restore
    a, s = stash["a"], stash["s"]

    def _pcc(x, y):
        x = ttnn.to_torch(x).float().flatten(); y = ttnn.to_torch(y).float().flatten()
        if torch.allclose(x, y): return 1.0, 0.0
        vx = x - x.mean(); vy = y - y.mean()
        d = (vx.norm() * vy.norm()).clamp_min(1e-12)
        return float((vx @ vy) / d), float((x - y).abs().max())

    import statistics
    def _clone_inputs():
        return (ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=a.dtype),
                ttnn.clone(s, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=s.dtype))
    # correctness: baseline vs fused on FRESH cloned inputs (AdaLN mutates a in place)
    ttnn.synchronize_device(device)
    base = orig_call(adaln_a0, *_clone_inputs())
    ttnn.synchronize_device(device)
    fused = _adaln_fused(adaln_a0, *_clone_inputs())
    ttnn.synchronize_device(device)
    pcc, maxabs = _pcc(base, fused)

    def timed(fn, n):
        samples = []
        for _ in range(n):
            ca, cs = _clone_inputs()
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            fn(ca, cs)
            ttnn.synchronize_device(device)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples), samples
    n = args.repeats
    base_med, base_s = timed(lambda a_, s_: orig_call(adaln_a0, a_, s_), n)
    fused_med, fused_s = timed(lambda a_, s_: _adaln_fused(adaln_a0, a_, s_), n)
    print(json.dumps({
        "model": "protenix-v2-adaln-addcmul-ab",
        "tokens": len(seq),
        "repeats": n,
        "pcc": pcc, "max_abs": maxabs,
        "baseline_median_ms": 1000 * base_med,
        "fused_median_ms": 1000 * fused_med,
        "speedup": base_med / fused_med,
        "baseline_samples_ms": [1000 * x for x in base_s],
        "fused_samples_ms": [1000 * x for x in fused_s],
    }, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("--input", default="examples/prot.yaml")
    p.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    p.add_argument("--steps", type=int, default=200); p.set_defaults(fn=_profile)
    d = sub.add_parser("decomp"); d.add_argument("--input", default="examples/prot.yaml")
    d.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    d.set_defaults(fn=_decomp)
    ab = sub.add_parser("ab"); ab.add_argument("--input", default="examples/prot.yaml")
    ab.add_argument("--checkpoint", default="/home/ttuser/.boltz/protenix-v2.pt")
    ab.add_argument("--repeats", type=int, default=50); ab.set_defaults(fn=_ab)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    args.fn(args)


if __name__ == "__main__":
    main()
