#!/usr/bin/env python3
"""The shipped fused SDPA's gradient, against a float64 reference.

`ttnn.transformer.scaled_dot_product_attention` is the one op in the pairformer with no
backward, and `state/ptx/LEDGER.md` K17 makes this implementation the shared one --
`ptx-diffusion`'s `AttentionPairBias` sites consume the same entry. So it is verified here
the way `ptxft` proved its gradients and not more loosely.

Four levels of evidence:

1. The float64 reference is validated against float64 central finite differences before
   anything on device is compared to it.
2. The device gradient against that reference, on operands rounded to the device dtype
   first, so what is measured is the op's error and not input quantisation.
3. Chunking invariance. The backward recomputes the scores in blocks, and the block size
   is a memory decision. Two different chunkings must produce the same gradient, which is
   a check no single run can make of itself.
4. A negative control that must FAIL, so a passing run means something.

The forward is the production fused kernel, called by the tape and handed to
`autograd.triangle_attention` as its value; nothing here computes attention twice.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallgrad.gradcheck import COS_BAR, REL_L2_BAR, fd_check, metrics   # noqa: E402


def clocks(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for d in json.loads(r.stdout).get("device_info", []):
                c = d.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(1.0)


def ref_attention(q, k, v, bias, scale):
    """What the FUSED KERNEL computes, which is what the tape must differentiate.

    The mask is added before the scale. That is measured, not assumed (see
    `tt_bio/taped_ttnn.py::_v_sdpa`), and production compensates for it by pre-scaling the
    pair bias by sqrt(head_dim) at `tenstorrent.py:7205`, so the composite production runs
    is standard attention. Writing the other ordering here is how this check would have
    passed against a gradient that disagrees with the forward -- it did, at 6.5e-03, on
    the first run of this file.
    """
    s = (q @ k.transpose(-1, -2) + bias) * scale
    return torch.softmax(s, dim=-1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    B, H, N, d = a.B, a.H, a.N, a.d
    scale = d ** -0.5

    stop, samples = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, samples), daemon=True)
    th.start()
    try:
        ok = run(B, H, N, d, scale, a.seed, a.probes)
    finally:
        stop.set(); th.join(timeout=3)
    if samples:
        print(f"  aiclk during run: min {min(samples)} max {max(samples)} MHz "
              f"({len(samples)} samples)")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run(B, H, N, d, scale, seed, probes):
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tp

    rng = np.random.default_rng(seed)
    t64 = lambda *s: torch.tensor(rng.standard_normal(s), dtype=torch.float64)
    q64, k64, v64 = t64(B, H, N, d), t64(B, H, N, d), t64(B, H, N, d)
    b64 = t64(1, H, N, N) * 0.5
    lw = t64(B, H, N, d)
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)

    print(f"sdpa  B={B} H={H} N={N} d={d}  scale={scale:.6f}")

    # -- 1. the reference against its own finite differences, all float64 on host --------
    ps = [x.clone().requires_grad_(True) for x in (q64, k64, v64, b64)]
    names = ["q", "k", "v", "bias"]
    worst, probed, elig = fd_check(
        lambda: (ref_attention(ps[0], ps[1], ps[2], ps[3], scale) * lw).sum(),
        ps, n_probe=probes, seed=seed)
    print(f"  reference vs float64 central differences: worst {worst:.2e} "
          f"over {probed} of {elig} eligible")
    if worst > 1e-6:
        print("  REFERENCE FAILED its own check; nothing below is evidence")
        return False

    # -- 2. the same reference on rounded operands ---------------------------------------
    rp = [bf(x).requires_grad_(True) for x in (q64, k64, v64, b64)]
    ((ref_attention(*rp, scale) * lw).sum()).backward()
    ref_out = ref_attention(*[p.detach() for p in rp], scale)

    dev = tt.get_device()
    D = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn_taped = tp.taped_ttnn()

    def one(budget):
        """One taped forward+backward at a given score-block budget."""
        prev = tp.SDPA_SCORE_BUDGET
        tp.SDPA_SCORE_BUDGET = budget
        try:
            ts = [ag.Tensor(D(x), requires_grad=True) for x in (q64, k64, v64, b64)]
            with ag.tape():
                out = ttnn_taped.transformer.scaled_dot_product_attention(
                    ts[0], ts[1], ts[2], attn_mask=ts[3], is_causal=False, scale=scale)
            out.backward(seed=D(lw))
            return out, [t.grad for t in ts]
        finally:
            tp.SDPA_SCORE_BUDGET = prev

    # A budget that holds the whole score tensor, and one that forces many blocks.
    whole = B * H * N * N * 2
    out, gs = one(whole)
    print(f"  chunking A: budget {whole} B (one block)")

    ok = True
    # The forward is REPORTED, not gated. It is the shipped fused kernel and its softmax is
    # the documented approximate one -- `tenstorrent.py::_accurate_softmax` measures the same
    # kernel at rel_rms 2.73e-02 against fp64 and ships it anyway, opt-in accurate path aside.
    # Holding the tape to a bar the production forward does not meet would be measuring ttnn.
    fwd = metrics(ttnn.to_torch(out.value).to(torch.float64).numpy(), ref_out.numpy())
    print(f"  forward        rel_l2 {fwd['rel_l2']:.3e}  cos {fwd['cos']:.6f}  "
          f"(fused kernel's own softmax deficit, reported not gated)")
    got = {}
    for nm, g, p in zip(names, gs, rp):
        gd = ttnn.to_torch(g).to(torch.float64).numpy()
        got[nm] = gd
        m = metrics(gd, p.grad.detach().numpy())
        good = m["rel_l2"] <= REL_L2_BAR and m["cos"] >= COS_BAR
        ok = ok and good
        print(f"  d{nm:12s} rel_l2 {m['rel_l2']:.3e}  cos {m['cos']:.6f}  "
              f"max_abs {m['max_abs']:.3e}  {'PASS' if good else 'FAIL'}")

    # -- 3. chunking invariance: a memory lever must not move the gradient ---------------
    small = H * N * 2 * 8            # forces a query-axis chunk as well as a leading one
    _, gs2 = one(small)
    print(f"  chunking B: budget {small} B (many blocks)")
    for nm, g in zip(names, gs2):
        gd = ttnn.to_torch(g).to(torch.float64).numpy()
        same = np.array_equal(gd, got[nm])
        m = metrics(gd, got[nm])
        tag = "BIT-IDENTICAL" if same else f"rel_l2 {m['rel_l2']:.3e}"
        print(f"  d{nm:12s} across two chunkings: {tag}")
        ok = ok and (same or m["rel_l2"] <= REL_L2_BAR)

    # -- 4. the check must be able to fail ------------------------------------------------
    bad = got["q"].copy()
    bad[..., 0] *= 1.5
    mb = metrics(bad, rp[0].grad.detach().numpy())
    fails = mb["rel_l2"] > REL_L2_BAR
    print(f"  negative control (dq head-0 column scaled 1.5x): rel_l2 {mb['rel_l2']:.3e} "
          f"{'-- rejected, the check can fail' if fails else '-- CHECK IS BLIND'}")
    return ok and fails


if __name__ == "__main__":
    sys.exit(main())
