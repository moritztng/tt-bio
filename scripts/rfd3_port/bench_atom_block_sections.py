"""Attribute the cost inside one real RFD3 atom block, by truncated replay.

Why not the existing instruments:

* ``--sync-profile`` runs with ``enable_fast_runtime_mode=False``, which adds
  ~1.5 ms of host validation to *every* op. A 452 ms step measures as 5348 ms
  there, so only ops far above that floor mean anything.
* ``bench_atom_attn_ops.py`` replays the op sequence with hand-built tensors but
  omits ``core_grid=CORE_GRID_MAIN``, which the shipped code passes. Letting ttnn
  auto-pick the grid makes the pair-bias projection measure 6.32 ms against
  0.42 ms with the real grid -- a 15x artifact, not a real cost.

So: capture the real inputs of a real encoder block from a live forward, then
replay that block back-to-back in steady state with fast runtime mode on and the
shipped compute config. Replaying truncated at each stage attributes cost by
difference, and the untruncated total is checkable against the ``--stage-profile``
per-block number.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 -m scripts.rfd3_port.bench_atom_block_sections [--contig ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

# cut points inside RFD3AtomBlock.__call__, in execution order
STAGES = [
    "adaln_1",        # rms_norms + gain/bias linears
    "qkvg",           # 4 projections + q/k rms_norm
    "heads",          # reshape + permute x4
    "pair_bias",      # pair-bias linear + permute
    "qk",             # dense QK matmul
    "bias_scatter",   # scatter sparse pair bias into the dense LxL mask
    "score_fp32",     # typecast/scale/add in fp32 over the dense LxL
    "softmax",        # dense fp32 softmax
    "av",             # attention->bf16 + AV matmul
    "attn_out",       # gate, out projection, residual add
    "transition",     # adaln_2 + SwiGLU + residual add
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    import ttnn

    from tt_bio.rfd3 import (
        BATCH_INVARIANT_GRID, CORE_GRID_MAIN, RFD3AtomBlock,
        build_diffusion_module, build_token_initializer,
    )
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    spec_data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in features.items()
    }
    tw = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                    map_location="cpu", weights_only=True)
    module = build_diffusion_module(dw)
    with torch.no_grad():
        initial = build_token_initializer(tw)(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in features.items()}
        )

    length = features["ref_pos"].shape[0]
    noisy = torch.randn(args.batch, length, 3,
                        generator=torch.Generator().manual_seed(42)) * 16.0
    times = torch.full((args.batch,), 8.0)

    # --- capture the live arguments of the first encoder block -----------------
    captured = {}
    original = RFD3AtomBlock.__call__

    def spy(self, q, c, p, additive_mask=None, sparse_qk=None):
        if not captured:
            captured.update(block=self, q=q, c=c, p=p, mask=additive_mask,
                            sparse_qk=sparse_qk)
        return original(self, q, c, p, additive_mask, sparse_qk)

    RFD3AtomBlock.__call__ = spy
    with torch.no_grad():
        module(X_noisy_L=noisy, t=times, f=features, **initial)
    RFD3AtomBlock.__call__ = original
    if not captured:
        raise SystemExit("no atom block ran -- is the encoder traced away?")

    device = module.device
    block, q0, c, p = captured["block"], captured["q"], captured["c"], captured["p"]
    mask, sparse_qk = captured["mask"], captured["sparse_qk"]
    print(f"atoms={length} batch={args.batch} q={tuple(q0.shape)} c={tuple(c.shape)} "
          f"p={tuple(p.shape)} sparse={sparse_qk is not None} "
          f"n_head={block.n_head} head_dim={block.head_dim}", flush=True)

    ckc, dt = block.compute_kernel_config, block.dtype

    def run(stop):
        """The shipped RFD3AtomBlock.__call__ body, truncated after `stop`."""
        q = q0
        batch, tokens = q.shape[0], q.shape[1]
        norm = block._adaln(q, c, block.a_ln_s, block.a_gain_w, block.a_gain_b,
                            block.a_bias_w)
        if stop == "adaln_1":
            return norm
        L = dict(compute_kernel_config=ckc, dtype=dt, core_grid=BATCH_INVARIANT_GRID)
        qq = ttnn.linear(norm, block.q_w, **L)
        kk = ttnn.linear(norm, block.k_w, **L)
        vv = ttnn.linear(norm, block.v_w, **L)
        gg = ttnn.linear(norm, block.g_w, **L)
        qq = ttnn.rms_norm(qq, weight=block.q_ln, epsilon=1e-6, compute_kernel_config=ckc)
        kk = ttnn.rms_norm(kk, weight=block.k_ln, epsilon=1e-6, compute_kernel_config=ckc)
        if stop == "qkvg":
            return kk

        def heads(x):
            x = ttnn.reshape(x, (batch, tokens, block.n_head, block.head_dim))
            return ttnn.permute(x, (0, 2, 1, 3))

        qq, kk, vv, gg = map(heads, (qq, kk, vv, gg))
        if stop == "heads":
            return gg
        n_keys, attn_idx_dev, dense_bias = sparse_qk
        pair_bias = ttnn.linear(p, block.b_w, compute_kernel_config=ckc, dtype=dt,
                                core_grid=CORE_GRID_MAIN)
        pair_bias = ttnn.permute(pair_bias, (0, 3, 1, 2))
        if stop == "pair_bias":
            return pair_bias
        scores = ttnn.matmul(qq, ttnn.permute(kk, (0, 1, 3, 2)),
                             compute_kernel_config=ckc)
        if stop == "qk":
            return scores
        bias = ttnn.scatter(dense_bias, 3, attn_idx_dev, pair_bias)
        if stop == "bias_scatter":
            return bias
        scores = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
        scores = ttnn.multiply(scores, block.head_dim ** -0.5)
        bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
        scores = ttnn.add(scores, bias_f)
        if stop == "score_fp32":
            return scores
        attention = ttnn.softmax(scores, dim=-1)
        if stop == "softmax":
            return attention
        attention = ttnn.typecast(attention, dt, memory_config=attention.memory_config())
        out = ttnn.matmul(attention, vv, compute_kernel_config=ckc, dtype=dt)
        if stop == "av":
            return out
        out = ttnn.multiply(out, ttnn.sigmoid(gg))
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (batch, tokens, block.n_head * block.head_dim))
        out = ttnn.linear(out, block.o_w, **L)
        gate = ttnn.linear(c, block.a_out_w, bias=block.a_out_b, **L)
        q = ttnn.add(q, ttnn.multiply(out, ttnn.sigmoid(gate)))
        if stop == "attn_out":
            return q
        norm = block._adaln(q, c, block.t_ln_s, block.t_gain_w, block.t_gain_b,
                            block.t_bias_w)
        left = ttnn.linear(norm, block.t_fc1, activation="silu", **L)
        right = ttnn.linear(norm, block.t_fc2, **L)
        update = ttnn.linear(ttnn.multiply(left, right), block.t_fc3, **L)
        gate = ttnn.linear(c, block.t_out_w, bias=block.t_out_b, **L)
        return ttnn.add(q, ttnn.multiply(update, ttnn.sigmoid(gate)))

    def measure(stop):
        for _ in range(2):
            run(stop)
        ttnn.synchronize_device(device)
        samples = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            run(stop)
            ttnn.synchronize_device(device)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        samples.sort()
        return samples[len(samples) // 2]

    print(f"\n{'stage':<16s} {'cumulative':>11s} {'this stage':>11s}")
    prev = 0.0
    for stage in STAGES:
        cum = measure(stage)
        print(f"{stage:<16s} {cum:11.3f} {cum - prev:11.3f}", flush=True)
        prev = cum
    print(f"\nfull block {prev:.3f} ms  x9 invocations/step = {prev * 9:.1f} ms")


if __name__ == "__main__":
    main()
