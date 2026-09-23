#!/usr/bin/env python3
"""The `nlp_create_qkv_heads` gradient, against float64, with the bytes each slot costs.

The 576-token backward refuses a 2,717,908,992 B DRAM buffer inside `ttnn::concat ->
tilize_with_val_padding`. `perf/of3t_cropwall/concat_census.py` located it at
`tt_bio/taped_ttnn.py:922`, the vjp of `experimental.nlp_create_qkv_heads`: it scatters one
head-split cotangent into slot `s` of the packed axis by concatenating on dim 2, whose extent
is 3, and TILE layout pads a second-to-last dim of 3 up to 32. The tensor wanted is
[N, 1, N, 3*H*dh]; the buffer asked for is [N, N, 32, H*dh].

This script is the numerical control on re-expressing that scatter on the LAST axis, where
the extent is 3*H*dh and needs no padding. Run it on the tree before the edit and on the tree
after: the packed gradient must be BIT-IDENTICAL, because nothing about the arithmetic
changes, only which axis carries the slot.

The reference is float64 on the host and is not another device expression: the scatter is
built with numpy from the same cotangents, so an agreement is against the definition of the
op rather than against a second implementation of it.

`--verb concat_heads` runs the same control on the sibling, the vjp of
`experimental.nlp_concat_heads`, whose inverse reshape put H=4 on dim 2 and padded it 8x.

    qkv_heads_ab.py --tokens 256 --out perf/of3t_cropwall/out/qkv_ab_pre.json
    qkv_heads_ab.py --verb concat_heads --out perf/of3t_cropwall/out/heads_ab_pre.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.of3t_memory.alloc_profile import Peak, _Watch, _swap_watch    # noqa: E402


def _digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256, help="the pair track's B and L")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--verb", choices=("create_qkv_heads", "concat_heads"),
                    default="create_qkv_heads")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1, default=str))

    try:
        import numpy as np
        import torch
        import ttnn
        from ttnn._ttnn import reports
        from tt_bio import autograd as ag
        from tt_bio.taped_ttnn import taped_ttnn
        from tt_bio.tenstorrent import get_device

        # The verb has to be reached the way a shipped module reaches it: through the proxy
        # `tape()` installs. Calling `ttnn.experimental.*` straight from a script outside
        # `tt_bio` hands a taped Tensor to the pybind signature and dies there.
        TTS = taped_ttnn()

        B = L = a.tokens
        H, dh = a.heads, a.head_dim
        n_slot = 3 if a.verb == "create_qkv_heads" else 1
        wide = n_slot * H * dh
        dev = get_device()
        out["env"]["arch"] = str(dev.arch())
        out["shape"] = {"B": B, "L": L, "H": H, "dh": dh, "wide": wide}

        g = torch.Generator().manual_seed(7)
        xt = (torch.rand((B, 1, L, wide), generator=g, dtype=torch.float32) * 2 - 1)
        if a.verb == "create_qkv_heads":
            cot = [(torch.rand((B, H, L, dh), generator=g, dtype=torch.float32) * 2 - 1)
                   for _ in range(3)]
        else:
            xt = (torch.rand((B, H, L, dh), generator=g, dtype=torch.float32) * 2 - 1)
            cot = [(torch.rand((B, 1, L, wide), generator=g, dtype=torch.float32) * 2 - 1)]
        # bf16 on both sides of the comparison: the reference is the SCATTER in float64, and a
        # scatter cannot invent precision, so rounding the cotangents once up front makes the
        # host answer exact rather than nearly right.
        cot = [c.to(torch.bfloat16).to(torch.float32) for c in cot]

        def dram():
            tot = 0
            for b in reports.get_buffers(dev):
                if str(b.buffer_type).upper().endswith("L1"):
                    continue
                tot += int(b.max_size_per_bank) * 8
            return tot

        xd = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        base = dram()
        peak = {"b": base}
        # The high-water, not the residue. Every temporary in this closure is freed before
        # `synchronize_device` returns, so a census taken after the backward reads the answer
        # and none of the bytes the answer cost. `Peak` probes on every verb instead.
        pk = Peak(dev, ttnn, reports)
        saved: list = []

        with ag.tape():
            x = ag.Tensor(xd, requires_grad=True)
            if a.verb == "create_qkv_heads":
                outs = TTS.experimental.nlp_create_qkv_heads(
                    x, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                outs = [TTS.experimental.nlp_concat_heads(
                    x, memory_config=ttnn.DRAM_MEMORY_CONFIG)]
            seeds = [ttnn.from_torch(c, layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.bfloat16) for c in cot]
            t0 = time.perf_counter()
            pk.reset()
            _swap_watch(pk, True, saved)
            # `_swap_watch` leaves `tt_bio.taped_ttnn` alone, so its vjps are probed only where
            # they reach `autograd`. The qkv scatter reaches it through the `add` of its three
            # slots; a single-output verb never adds, and its whole backward would go unseen.
            # So watch the vjp's own `ttnn` here too -- for the new verb only, which keeps the
            # banked qkv arms on the instrument that measured them.
            TTm = sys.modules["tt_bio.taped_ttnn"]
            vjp_ttnn = TTm.ttnn
            if a.verb == "concat_heads":
                TTm.ttnn = _Watch(vjp_ttnn, pk.probe)
            try:
                ag.backward(list(outs), seeds)
                ttnn.synchronize_device(dev)
            finally:
                TTm.ttnn = vjp_ttnn
                _swap_watch(pk, False, saved)
            out["backward_s"] = round(time.perf_counter() - t0, 3)
            out["backward_dram_peak_b"] = pk.dram_hw
            out["backward_dram_peak_at_verb"] = pk.at_verb
            out["backward_largest_buffer_b"] = (pk.census or {}).get("DRAM", {}).get("largest_b")
            peak["b"] = max(peak["b"], dram())
            got = ttnn.to_torch(x.grad).to(torch.float32).numpy()

        out["dram_after_backward_b"] = peak["b"]
        out["dram_before_b"] = base

        # float64 reference: the scatter, by definition. Slot s of the packed axis holds the
        # s-th cotangent laid out head-major, and the three slots do not overlap, so the sum
        # `add_grad` performs is a concatenation and nothing rounds.
        if a.verb == "create_qkv_heads":
            ref = np.zeros((B, 1, L, wide), dtype=np.float64)
            for s, c in enumerate(cot):
                cn = c.to(torch.float64).numpy()                   # [B, H, L, dh]
                for h in range(H):
                    w0 = s * H * dh + h * dh
                    ref[:, 0, :, w0:w0 + dh] = cn[:, h, :, :]
        else:
            # The inverse of the head-major packing: head h is the lane range [h*dh, (h+1)*dh).
            cn = cot[0].to(torch.float64).numpy()                  # [B, 1, L, H*dh]
            ref = np.zeros((B, H, L, dh), dtype=np.float64)
            for h in range(H):
                ref[:, h] = cn[:, 0, :, h * dh:(h + 1) * dh]
        err = np.abs(got.astype(np.float64) - ref)
        den = max(float(np.abs(ref).max()), 1e-30)
        out["vs_float64"] = {
            "max_abs": float(err.max()),
            "max_rel": float(err.max() / den),
            "mean_abs": float(err.mean()),
            "bit_exact": bool(err.max() == 0.0),
            "ref_absmax": float(np.abs(ref).max()),
            "nonzero_share": float((np.abs(ref) > 0).mean()),
        }
        out["grad_digest"] = _digest(got)
        out["grad_stats"] = {"absmax": float(np.abs(got).max()),
                             "sum": float(got.astype(np.float64).sum())}
        out["ok"] = bool(err.max() == 0.0)
    except Exception:                                                      # noqa: BLE001
        out["error"] = traceback.format_exc()[-4000:]
        out["ok"] = False
    dump()
    v = out.get("vs_float64") or {}
    print(a.verb + " vjp  B=L=%s  bit_exact_vs_f64 %s  max_abs %s  digest %s"
          % (a.tokens, v.get("bit_exact"), v.get("max_abs"), out.get("grad_digest")),
          flush=True)
    print("  backward DRAM high-water %s B at %s, largest live buffer %s B, %.3f s"
          % (out.get("backward_dram_peak_b"), out.get("backward_dram_peak_at_verb"),
             out.get("backward_largest_buffer_b"), out.get("backward_s") or 0), flush=True)
    if out.get("error"):
        print("ERROR:", out["error"][-1500:], flush=True)
    print("wrote", a.out, flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
