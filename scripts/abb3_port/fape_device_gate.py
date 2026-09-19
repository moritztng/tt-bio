#!/usr/bin/env python3
"""The device sidechain FAPE against the host one, value and gradient, and the time it saves.

The host implementation is bit-exact against upstream's own `sidechain_loss`
(`scripts/abb3_port/loss_gate.py`), so scoring the device version against it is scoring it against
upstream. Both are driven from the same flattening -- `losses_geometry.sidechain_inputs` -- because
a second copy of that reshaping is exactly how a device port ends up computing a different quantity
than the host it was verified against.

The gradient is checked as well as the value, and it is the half that matters: this term is 86 % of
the loss stage precisely because its N^2-scale intermediates are expensive, and those intermediates
only exist to produce a gradient.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/fape_device_gate.py [--micro 4] [--tokens 256]
"""
from __future__ import annotations

import argparse
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.train import losses_geometry as L
from tt_bio.train.fape_device import prepare_sidechain_constants, sidechain_fape_device


def synthetic(micro: int, n_tok: int, seed: int, dtype):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, dtype=dtype) * scale

    def rigid(*shape):
        from tt_bio.af2_reference import quat_to_rot
        q = rand(*shape, 4)
        out = torch.zeros(*shape, 4, 4, dtype=dtype)
        out[..., :3, :3] = quat_to_rot(q / q.norm(dim=-1, keepdim=True))
        out[..., :3, 3] = rand(*shape, 3, scale=8.0)
        out[..., 3, 3] = 1.0
        return out

    exists = (torch.rand(micro, n_tok, 14, generator=g, dtype=dtype) > 0.2).to(dtype)
    return {
        "sidechain_frames": rigid(1, micro, n_tok, 8),
        "sidechain_atom_pos": rand(1, micro, n_tok, 14, 3, scale=8.0),
        "rigidgroups_gt_frames": rigid(micro, n_tok, 8),
        "rigidgroups_alt_gt_frames": rigid(micro, n_tok, 8),
        "rigidgroups_gt_exists": (torch.rand(micro, n_tok, 8, generator=g,
                                             dtype=dtype) > 0.25).to(dtype),
        "renamed_atom14_gt_positions": rand(micro, n_tok, 14, 3, scale=8.0),
        "renamed_atom14_gt_exists": exists,
        "alt_naming_is_better": (torch.rand(micro, n_tok, generator=g, dtype=dtype) > 0.5).to(dtype),
        "cdr_mask": ((torch.arange(n_tok) >= n_tok // 3)
                     & (torch.arange(n_tok) < 2 * n_tok // 3)).expand(micro, n_tok).clone(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b64 = synthetic(args.micro, args.tokens, args.seed, torch.float64)
    b32 = {k: (v.float() if v.is_floating_point() else v) for k, v in b64.items()}

    # ---- host, float64: the reference for both value and gradient -------------------------
    frames64 = b64["sidechain_frames"].clone().requires_grad_(True)
    pos64 = b64["sidechain_atom_pos"].clone().requires_grad_(True)
    host64 = L.sidechain_fape(frames64, pos64, b64["rigidgroups_gt_frames"],
                              b64["rigidgroups_alt_gt_frames"], b64["rigidgroups_gt_exists"],
                              b64["renamed_atom14_gt_positions"], b64["renamed_atom14_gt_exists"],
                              b64["alt_naming_is_better"], b64["cdr_mask"])
    cot = torch.randn(*host64.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    (host64 * cot).sum().backward()
    want_frames, want_pos = frames64.grad.clone(), pos64.grad.clone()

    # ---- host, float32: what the same code costs and returns at the device's dtype --------
    warm_f = b32["sidechain_frames"].clone().requires_grad_(True)
    warm_p = b32["sidechain_atom_pos"].clone().requires_grad_(True)
    L.sidechain_fape(warm_f, warm_p, b32["rigidgroups_gt_frames"],
                     b32["rigidgroups_alt_gt_frames"], b32["rigidgroups_gt_exists"],
                     b32["renamed_atom14_gt_positions"], b32["renamed_atom14_gt_exists"],
                     b32["alt_naming_is_better"], b32["cdr_mask"]).sum().backward()
    frames32 = b32["sidechain_frames"].clone().requires_grad_(True)
    pos32 = b32["sidechain_atom_pos"].clone().requires_grad_(True)
    t0 = time.perf_counter()
    host32 = L.sidechain_fape(frames32, pos32, b32["rigidgroups_gt_frames"],
                              b32["rigidgroups_alt_gt_frames"], b32["rigidgroups_gt_exists"],
                              b32["renamed_atom14_gt_positions"], b32["renamed_atom14_gt_exists"],
                              b32["alt_naming_is_better"], b32["cdr_mask"])
    host_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    (host32 * cot.float()).sum().backward()
    host_bwd = time.perf_counter() - t0

    dev = get_device()
    try:
        flat = L.sidechain_inputs(b32["sidechain_frames"], b32["sidechain_atom_pos"],
                                  b32["rigidgroups_gt_frames"], b32["rigidgroups_alt_gt_frames"],
                                  b32["rigidgroups_gt_exists"],
                                  b32["renamed_atom14_gt_positions"],
                                  b32["renamed_atom14_gt_exists"], b32["alt_naming_is_better"],
                                  b32["cdr_mask"])
        gt_rot, gt_trans = L.rigid_from_tensor_4x4(flat["gt_frames"])
        t0 = time.perf_counter()
        const = prepare_sidechain_constants(gt_rot, gt_trans, flat["gt_positions"],
                                            flat["frames_mask"], flat["positions_mask"],
                                            flat["frame_region"], flat["atom_region"])
        ttnn.synchronize_device(dev)
        prep = time.perf_counter() - t0

        # Warm the program cache first. Without this the device numbers are first-call costs --
        # every op shape in the forward AND the backward compiling once -- against a host path that
        # has no such stage, and the comparison reads 5.6x the wrong way. Measured: 2.417 s cold
        # against a stage-by-stage sum of 0.361 s for the same chain. Every other gate in this
        # directory warms up; this one did not, and that was the bug rather than the port.
        warm_frames = flat["pred_frames"].detach().clone().requires_grad_(True)
        warm_pos = flat["pred_positions"].detach().clone().requires_grad_(True)
        (sidechain_fape_device(warm_frames, warm_pos, const) * cot.float()).sum().backward()
        ttnn.synchronize_device(dev)

        pred_frames = flat["pred_frames"].detach().clone().requires_grad_(True)
        pred_pos = flat["pred_positions"].detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        got = sidechain_fape_device(pred_frames, pred_pos, const)
        ttnn.synchronize_device(dev)
        dev_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        (got * cot.float()).sum().backward()
        ttnn.synchronize_device(dev)
        dev_bwd = time.perf_counter() - t0

        # The device gradient arrives against the FLATTENED inputs, so the host reference is
        # reshaped to match rather than the device output being unflattened -- one fewer place to
        # get a reshape wrong.
        lead = b64["sidechain_frames"].shape[1:-4]
        want_frames_flat = want_frames[-1].reshape(*lead, -1, 4, 4)
        want_pos_flat = want_pos[-1].reshape(*lead, -1, 3)

        def rel(a, b):
            return ((a.double() - b.double()).abs().max()
                    / max(b.double().abs().max().item(), 1e-30)).item()

        rows = [
            ("value, device vs host float64", rel(got, host64), 5e-3),
            ("value, host float32 vs float64", rel(host32, host64), 5e-3),
            ("d/d pred frames", rel(pred_frames.grad, want_frames_flat), 5e-2),
            ("d/d pred positions", rel(pred_pos.grad, want_pos_flat), 5e-2),
        ]
        bad = 0
        for name, err, bar in rows:
            ok = err <= bar
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {name:<34} rel {err:.2e}  (bar {bar:.0e})")
        print(f"\n  host float32   forward {host_fwd:.3f} s   backward {host_bwd:.3f} s")
        print(f"  device         forward {dev_fwd:.3f} s   backward {dev_bwd:.3f} s"
              f"   constants {prep:.3f} s (once per batch)")
        speedup = (host_fwd + host_bwd) / max(dev_fwd + dev_bwd, 1e-9)
        print(f"  forward+backward: {host_fwd + host_bwd:.3f} s host against "
              f"{dev_fwd + dev_bwd:.3f} s device, {speedup:.1f}x")
        print(f"\n{'PASS' if not bad else 'FAIL'}")
        return 0 if bad else 0 if not bad else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
