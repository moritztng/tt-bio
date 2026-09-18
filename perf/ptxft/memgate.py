#!/usr/bin/env python3
"""Phase 3: how many pairformer blocks can be taped, at what token count, in 34.23 GB.

Measured, not extrapolated. hallgrad-build found the current chain binds at 512 aa and
NOT where the feasibility study predicted, so this sweeps (depth, tokens) and reports the
boundary.

The instrument matters and hallgrad-build already paid for the lesson: `get_memory_view`
is itself a pipeline drain, so it cannot be called inside a timed region, and a polling
thread starves against a tight op-enqueue loop holding the GIL -- its first attempt
reported a 0.034 GB output tensor as the forward peak while a 0.27 GB score tensor went
past unseen. So this reports DRAM ALLOCATED AT SYNCHRONISATION POINTS, which is exactly
what it is called, and treats the OOM BOUNDARY as the decisive number.

Two arms:

--equivalence  per-block checkpointing must reproduce the un-checkpointed gradient, for
               the INPUT and for the WEIGHTS. Depth is bought with checkpointing, so if
               it changes the gradient then every depth below is meaningless.
--sweep        (depth, tokens) -> allocated DRAM after forward and after backward, plus
               OK or the allocator's own refusal.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

CARD_GB = 34.23  # measured off the live device by hallgrad-build: 8 banks x 4278190016 B


def dram(device):
    """DRAM allocated, read at a synchronisation point. Not a peak, and not called a peak."""
    import ttnn
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * mv.num_banks / 1e9


def dram_total(device):
    import ttnn
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return mv.total_bytes_per_bank * mv.num_banks / 1e9


def build_stack(sd_blocks, device, *, n_heads, head_dim, lora, adapt, dtype,
                adapter_dtype, chunk, q_chunk, rng):
    from perf.ptxft.tape_block import PairTrackBlock
    blocks, params = [], {}
    for i, sd in enumerate(sd_blocks):
        b = PairTrackBlock(sd, device, n_heads=n_heads, head_dim=head_dim, dtype=dtype,
                           adapter_dtype=adapter_dtype, lora=lora,
                           adapt=adapt, rng=rng, chunk=chunk, q_chunk=q_chunk)
        blocks.append(b)
        for k, v in b.params.items():
            params[f"blocks.{i}.{k}"] = v
    return blocks, params


def run_stack(blocks, z, *, checkpointed):
    """Forward the taped stack, with or without per-block checkpointing."""
    from tt_bio import autograd as ag
    for b in blocks:
        if checkpointed:
            z = ag.checkpoint(b, z, params=tuple(b.params.values()))
        else:
            z = b(z)
    return z


def arm_equivalence(sd_blocks, device, a, lora, adapt):
    """Checkpointed vs not: same loss, same input gradient, same WEIGHT gradients."""
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    print("# --equivalence: per-block checkpointing must not change the gradient.")
    print("# Depth is bought with checkpointing, so this gates every depth below it.")
    print("# The WEIGHT gradients are the half that is new here: the recomputed tape calls")
    print("# add_grad on the original parameter objects, so they need no duplication, but")
    print("# the checkpoint node is PRUNED unless they are named as parents.")
    N = a.eq_n
    rng = np.random.default_rng(11)
    z0 = (rng.standard_normal((N, N, 256)) * 0.5).astype(np.float32)
    seedg = (rng.standard_normal((N, N, 256)) * 0.1).astype(np.float32)
    arms = {}
    for label, ckpt in (("plain", False), ("checkpointed", True)):
        blocks, params = build_stack(
            sd_blocks[:a.eq_depth], device, n_heads=a.heads, head_dim=a.head_dim,
            lora=lora, adapt=adapt, dtype=ttnn.bfloat16, adapter_dtype=ttnn.float32,
            chunk=a.chunk, q_chunk=a.q_chunk, rng=np.random.default_rng(7))
        zt = ag.Tensor(ft.to_device(z0, device, dtype=ttnn.bfloat16), requires_grad=True)
        out = run_stack(blocks, zt, checkpointed=ckpt)
        loss = float(ft.to_host(out.value).astype(np.float64).sum())
        out.backward(seed=ft.to_device(seedg, device, dtype=ttnn.bfloat16))
        arms[label] = {
            "loss": loss,
            "dz": ft.to_host(zt.grad).astype(np.float64) if zt.grad is not None else None,
            "w": {k: (ft.to_host(v.grad).astype(np.float64) if v.grad is not None else None)
                  for k, v in params.items()},
        }
    failures = []
    p, c = arms["plain"], arms["checkpointed"]
    print(f"{'quantity':<30} {'plain':>14} {'checkpointed':>14} {'rel diff':>11}  verdict")
    dl = abs(c["loss"] - p["loss"]) / max(abs(p["loss"]), 1e-30)
    print(f"{'loss':<30} {p['loss']:>14.6e} {c['loss']:>14.6e} {dl:>11.2e}  "
          f"{'PASS' if dl < 1e-6 else 'FAIL'}")
    if dl >= 1e-6:
        failures.append(f"checkpoint changes the loss by {dl:.2e}")
    if p["dz"] is None or c["dz"] is None:
        failures.append("no input gradient in one arm")
    else:
        d = np.linalg.norm(c["dz"] - p["dz"]) / max(np.linalg.norm(p["dz"]), 1e-30)
        print(f"{'d(loss)/dz':<30} {np.linalg.norm(p['dz']):>14.6e} "
              f"{np.linalg.norm(c['dz']):>14.6e} {d:>11.2e}  "
              f"{'PASS' if d < 1e-6 else 'FAIL'}")
        if d >= 1e-6:
            failures.append(f"checkpoint changes dz by {d:.2e}")
    n_missing = sum(1 for k in p["w"] if c["w"][k] is None)
    worst, worst_k = 0.0, None
    for k in p["w"]:
        if p["w"][k] is None or c["w"][k] is None:
            continue
        nb = np.linalg.norm(p["w"][k])
        if nb == 0:
            continue
        d = np.linalg.norm(c["w"][k] - p["w"][k]) / nb
        if d > worst:
            worst, worst_k = d, k
    print(f"{'weight grads present':<30} {len(p['w']) - sum(1 for k in p['w'] if p['w'][k] is None):>14d} "
          f"{len(c['w']) - n_missing:>14d} {'':>11}  "
          f"{'PASS' if n_missing == 0 else 'FAIL (' + str(n_missing) + ' absent)'}")
    print(f"{'worst weight-grad diff':<30} {'':>14} {'':>14} {worst:>11.2e}  "
          f"{'PASS' if worst < 1e-6 else 'FAIL'} ({worst_k})")
    if n_missing:
        failures.append(f"{n_missing} weight gradients absent under checkpointing -- the "
                        f"node was pruned")
    if worst >= 1e-6:
        failures.append(f"checkpoint changes weight gradient {worst_k} by {worst:.2e}")
    return failures


def arm_sweep(sd_blocks, device, a, lora, adapt):
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    total = dram_total(device)
    print(f"# --sweep: DRAM allocated at synchronisation points, card total "
          f"{total:.2f} GB (read live)")
    print(f"# adapters: rank {lora.rank} on {len(adapt)} of 34 pair-track linears per block")
    print(f"{'N':>5} {'depth':>6} {'ckpt':>5} {'base':>8} {'fwd':>8} {'bwd':>8} "
          f"{'fwd s':>8} {'bwd s':>8}  outcome")
    rows = []
    for N in [int(x) for x in a.tokens.split(",")]:
        for depth in [int(x) for x in a.depths.split(",")]:
            for ckpt in ([True] if a.checkpointed_only else [False, True]):
                rng = np.random.default_rng(3)
                z0 = (rng.standard_normal((N, N, 256)) * 0.5).astype(np.float32)
                seedg = (rng.standard_normal((N, N, 256)) * 0.05).astype(np.float32)
                base = fwd = bwd = float("nan")
                tf = tb = float("nan")
                outcome = "OK"
                try:
                    blocks, params = build_stack(
                        sd_blocks[:depth], device, n_heads=a.heads, head_dim=a.head_dim,
                        lora=lora, adapt=adapt, dtype=ttnn.bfloat16,
                        adapter_dtype=ttnn.float32, chunk=a.chunk, q_chunk=a.q_chunk,
                        rng=np.random.default_rng(7))
                    base = dram(device)
                    zt = ag.Tensor(ft.to_device(z0, device, dtype=ttnn.bfloat16),
                                   requires_grad=True)
                    t0 = time.perf_counter()
                    out = run_stack(blocks, zt, checkpointed=ckpt)
                    ttnn.synchronize_device(device)
                    tf = time.perf_counter() - t0
                    fwd = dram(device)
                    t0 = time.perf_counter()
                    out.backward(seed=ft.to_device(seedg, device, dtype=ttnn.bfloat16))
                    ttnn.synchronize_device(device)
                    tb = time.perf_counter() - t0
                    bwd = dram(device)
                    got = sum(1 for v in params.values() if v.grad is not None)
                    if got != len(params):
                        outcome = f"ONLY {got}/{len(params)} GRADS"
                except Exception as exc:                                     # noqa: BLE001
                    msg = str(exc).splitlines()
                    line = next((m for m in msg if "Not enough space" in m or "Out of Memory"
                                 in m), msg[0] if msg else "")
                    outcome = f"OOM/ERR: {line.strip()[:70]}"
                print(f"{N:>5} {depth:>6} {str(ckpt):>5} {base:>8.2f} {fwd:>8.2f} "
                      f"{bwd:>8.2f} {tf:>8.2f} {tb:>8.2f}  {outcome}")
                rows.append((N, depth, ckpt, base, fwd, bwd, tf, tb, outcome))
                blocks = params = out = zt = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--tokens", default="64,128,256")
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--q-chunk", type=int, default=None)
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--checkpointed-only", action="store_true")
    ap.add_argument("--eq-n", type=int, default=32)
    ap.add_argument("--eq-depth", type=int, default=2)
    ap.add_argument("--max-blocks", type=int, default=8)
    a = ap.parse_args()

    from tt_bio import tenstorrent as tt
    from tt_bio import finetune as ft
    from perf.clocksample import during
    from perf.ptxft.block_parity import load_blocks
    from perf.ptxft.tape_block import PAIR_TRACK_TARGETS

    idxs = list(range(48 - a.max_blocks, 48))
    blocks_sd, _head = load_blocks(a.ckpt, idxs)
    sd_blocks = [blocks_sd[i] for i in idxs]
    device = tt.get_device()
    lora = ft.LoraConfig(rank=a.rank, alpha=a.alpha)
    adapt = PAIR_TRACK_TARGETS

    failures = []
    with during() as clk:
        if a.equivalence:
            failures += arm_equivalence(sd_blocks, device, a, lora, adapt)
            print()
        if a.sweep:
            arm_sweep(sd_blocks, device, a, lora, adapt)
            print()
    print(clk.line(0))
    print()
    if failures:
        print(f"MEMGATE FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("MEMGATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
