"""Kernel scout: profile ESMC's Attention.forward share of a warm forward pass.

Reuses the shipped tt_bio.esmc load path + real checkpoints. Instruments the
Attention/Block forward with device-synchronized timing to decompose attention
into its sub-ops (layernorm, QKV proj, q/k LN, head split, rotary, SDPA, merge,
out proj) and to measure attention's share of the whole block stack, mirroring
docs/boltz2-protenix-kernel-scout.md's TriangleAttention decomposition.

Usage:
    ESM_ROOT=/path/to/esm TT_VISIBLE_DEVICES=1 \
      python3 esmc_attention_profile.py --model esmc-300m --lengths 76,384
"""
from __future__ import annotations

import argparse
import time

import ttnn

from tt_bio import esmc as tt_esmc
from tt_bio import tenstorrent

# real, well-folded proteins to derive deterministic sequences of a target length
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def make_seq(n: int) -> str:
    """Deterministic length-n protein sequence (tiled ubiquitin), no randomness."""
    s = (UBQ * (n // len(UBQ) + 1))[:n]
    return s


def timed(dev, fn):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    r = fn()
    ttnn.synchronize_device(dev)
    return r, time.perf_counter() - t0


PROF: dict[str, float] = {}


def acc(key: str, dt: float):
    PROF[key] = PROF.get(key, 0.0) + dt


def instrument(dev):
    """Monkeypatch Attention.__call__ + Block.__call__ with sync-timed versions."""
    Attention = tt_esmc.Attention
    Block = tt_esmc.Block
    apply_rotary = tt_esmc.apply_rotary

    def attn_forward(self, x, cos, sin, attn_mask=None, key_valid=None):
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads

        x_norm, dt = timed(dev, lambda: ttnn.layer_norm(
            x, weight=self.in_norm_weight, bias=self.in_norm_bias,
            epsilon=1e-5, compute_kernel_config=ck))
        acc("in_layernorm", dt)

        qkv, dt = timed(dev, lambda: self._lin(x_norm, self.qkv_weight))
        acc("qkv_proj", dt)
        ttnn.deallocate(x_norm)

        (q, k, v), dt = timed(dev, lambda: ttnn.chunk(qkv, 3, dim=-1))
        acc("chunk", dt)
        ttnn.deallocate(qkv)

        q, dt = timed(dev, lambda: ttnn.layer_norm(
            q, weight=self.q_ln_weight, epsilon=1e-5, compute_kernel_config=ck))
        acc("qk_layernorm", dt)
        k, dt = timed(dev, lambda: ttnn.layer_norm(
            k, weight=self.k_ln_weight, epsilon=1e-5, compute_kernel_config=ck))
        acc("qk_layernorm", dt)

        qkv, dt = timed(dev, lambda: ttnn.concat([q, k, v], dim=-1))
        acc("concat_repack", dt)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        (q, k, v), dt = timed(dev, lambda: self._split_heads(qkv, self.n_heads))
        acc("split_heads", dt)

        q, dt = timed(dev, lambda: apply_rotary(q, cos, sin))
        acc("rotary", dt)
        k, dt = timed(dev, lambda: apply_rotary(k, cos, sin))
        acc("rotary", dt)
        if key_valid is not None:
            k, dt = timed(dev, lambda: ttnn.multiply(k, key_valid)); acc("key_valid", dt)
            v, dt = timed(dev, lambda: ttnn.multiply(v, key_valid)); acc("key_valid", dt)

        o, dt = timed(dev, lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim ** -0.5,
            program_config=tt_esmc._sdpa_program_config_for_lengths(q.shape[2], k.shape[2])))
        acc("sdpa", dt)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)

        o, dt = timed(dev, lambda: self._merge_heads(o))
        acc("merge_heads", dt)
        out, dt = timed(dev, lambda: self._lin(o, self.out_weight))
        acc("out_proj", dt)
        ttnn.deallocate(o)
        return out

    def block_forward(self, x, cos, sin, attn_mask=None, key_valid=None):
        r1, dt = timed(dev, lambda: self.attn(x, cos, sin, attn_mask, key_valid))
        acc("ATTN_TOTAL", dt)
        (x2), dt = timed(dev, lambda: ttnn.add(x, ttnn.multiply(r1, self.inv_scale)))
        acc("residual", dt)
        x = x2
        ttnn.deallocate(r1)
        r3, dt = timed(dev, lambda: self.ffn(x))
        acc("FFN_TOTAL", dt)
        (x2), dt = timed(dev, lambda: ttnn.add(x, ttnn.multiply(r3, self.inv_scale)))
        acc("residual", dt)
        ttnn.deallocate(r3)
        return x2

    Attention.__call__ = attn_forward
    Block.__call__ = block_forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.MODELS))
    ap.add_argument("--lengths", default="76,384")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    dev = tenstorrent.get_device()

    print(f"Loading {args.model} on device …", flush=True)
    model = tt_esmc.load_esmc(args.model)
    cfg = tt_esmc.CONFIGS[args.model][0] if args.model in tt_esmc.CONFIGS else \
        {"n_layers": 80, "n_heads": 40, "d_model": 2560}
    n_layers = cfg["n_layers"]

    toks = {L: tt_esmc.tokenize(make_seq(L)) for L in lengths}

    # --- Phase 1: clean end-to-end warm latency, no instrumentation ---
    e2e_med = {}
    for L in lengths:
        tokens = toks[L]
        for _ in range(args.warmup):
            model(tokens)
        e2e = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            model(tokens)
            ttnn.synchronize_device(dev)
            e2e.append(time.perf_counter() - t0)
        e2e_med[L] = sorted(e2e)[len(e2e) // 2]
        print(f"[clean] {args.model} L={L} (tokens={tokens.shape[-1]}): "
              f"{e2e_med[L]*1e3:.2f} ms warm forward (median of 5)", flush=True)

    # --- Phase 1b: clean skip-based share (real pipelined wall-clock) ---
    # Replace attn (or ffn) with one cheap eltwise producing a correct-shape new
    # tensor. Delta vs the full forward is that component's true pipelined cost,
    # free of the per-op-sync inflation that biases dispatch-bound components.
    def skip(self, x, *a, **k):
        return ttnn.multiply(x, 0.0)

    orig_attn = tt_esmc.Attention.__call__
    orig_ffn = tt_esmc.SwiGLUFFN.__call__

    def clean_median(tokens):
        v = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            model(tokens)
            ttnn.synchronize_device(dev)
            v.append(time.perf_counter() - t0)
        return sorted(v)[len(v) // 2]

    skip_share = {}
    for L in lengths:
        tokens = toks[L]
        tt_esmc.Attention.__call__ = skip
        for _ in range(args.warmup):
            model(tokens)
        no_attn = clean_median(tokens)
        tt_esmc.Attention.__call__ = orig_attn

        tt_esmc.SwiGLUFFN.__call__ = skip
        for _ in range(args.warmup):
            model(tokens)
        no_ffn = clean_median(tokens)
        tt_esmc.SwiGLUFFN.__call__ = orig_ffn

        full = e2e_med[L]
        d_attn, d_ffn = full - no_attn, full - no_ffn
        skip_share[L] = (d_attn, d_ffn)
        print(f"[skip] {args.model} L={L}: full={full*1e3:.2f} no_attn={no_attn*1e3:.2f} "
              f"no_ffn={no_ffn*1e3:.2f} ms | attn_cost={d_attn*1e3:.2f} ms "
              f"({100*d_attn/full:.1f}% of forward), ffn_cost={d_ffn*1e3:.2f} ms "
              f"({100*d_ffn/full:.1f}%)", flush=True)

    # --- Phase 2: instrumented decomposition (per-op sync) ---
    instrument(dev)
    for L in lengths:
        tokens = toks[L]
        actual_L = tokens.shape[-1]
        print(f"\n=== {args.model}  L={L} (tokens={actual_L})  n_layers={n_layers}  "
              f"clean_e2e={e2e_med[L]*1e3:.2f} ms ===", flush=True)
        PROF.clear()
        model(tokens)  # one instrumented pass accumulates over all n_layers blocks

        attn = PROF.get("ATTN_TOTAL", 0.0)
        ffn = PROF.get("FFN_TOTAL", 0.0)
        resid = PROF.get("residual", 0.0)
        block_stack = attn + ffn + resid
        print(f"instrumented block stack ({n_layers} blocks, per-op sync):", flush=True)
        print(f"  ATTN_TOTAL  {attn*1e3:8.2f} ms  ({100*attn/block_stack:5.1f}% of block stack)", flush=True)
        print(f"  FFN_TOTAL   {ffn*1e3:8.2f} ms  ({100*ffn/block_stack:5.1f}%)", flush=True)
        print(f"  residual    {resid*1e3:8.2f} ms  ({100*resid/block_stack:5.1f}%)", flush=True)
        print(f"  block stack {block_stack*1e3:8.2f} ms", flush=True)

        # attention sub-op decomposition
        subops = ["in_layernorm", "qkv_proj", "chunk", "qk_layernorm", "concat_repack",
                  "split_heads", "rotary", "key_valid", "sdpa", "merge_heads", "out_proj"]
        print(f"  attention decomposition (sum over {n_layers} blocks):", flush=True)
        sub_sum = sum(PROF.get(k, 0.0) for k in subops)
        for k in subops:
            v = PROF.get(k, 0.0)
            if v > 0:
                print(f"    {k:16s} {v*1e3:8.2f} ms  ({100*v/sub_sum:5.1f}% of attn)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
