"""p24: ttnn trace-capture spike for the RFD3 18-block token DiT (LocalTokenTransformer),
in isolation, at the REAL fixture's shape (I=40 tokens, the IAI_protein.pdb + p12/p21/p23
fixture).

Per p23's handoff, this is the first incremental step toward tracing the RFD3 diffusion
step: the DiT block stack's run_device(a, s, z, additive_mask) already takes fixed-shape
device tensors with no dynamic control flow, so it is the simplest sub-piece to spike
before deciding whether to extend outward into the full per-step __call__ (where
attn_indices/dit_idx are recomputed every step from changing coordinates -- NOT static,
see p23's writeup).

Measures: eager per-call ms vs traced per-call ms (same real weights, same shared
random activations used by verify_dit.py), and PCC(traced_output, eager_output) to prove
the capture is lossless (same instruction stream, just replayed).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/spike_dit_trace.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import tt_bio.tenstorrent as _TTd
from tt_bio.tenstorrent import get_device
from tt_bio.rfd3 import build_dit, _dense_attention_mask, _tt

sys.path.insert(0, os.path.dirname(__file__))
from verify_dit import load, scoped, pcc, build_shared_inputs  # reuse the same shared-input builder

GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
I_REAL = 40   # real IAI_protein fixture token count (p23/p24 measured)
N_KEYS = 128  # RFD3DiffusionModule.N_ATTN_KEYS; min(N_KEYS, I) = I for this fixture


def main():
    # Reserve a trace region BEFORE building the module (get_device() is a
    # process-wide singleton -- the first call fixes the device layout).
    get_device(trace_region_size=1 << 28)  # 256MB: plenty for [1,40,768]-scale tensors
    import ttnn

    weights = load(GOLDEN_DIR, "diffusion_module.real_weights")
    dit_weights = scoped(weights, "diffusion_transformer")
    dit = build_dit(dit_weights)
    dev, dt = dit.device, dit.dtype

    A_I, S_I, Z_II, indices = build_shared_inputs(I=I_REAL, n_keys=min(N_KEYS, I_REAL))
    mask_host = _dense_attention_mask(indices)

    # ---- eager: fresh from_torch each call (today's actual __call__ path) ----
    N_REPS = 15
    a = _tt(A_I, dev, dt); s = _tt(S_I, dev, dt); z = _tt(Z_II, dev, dt)
    mask = _tt(mask_host, dev, dt)
    eager_out = dit.run_device(a, s, z, mask)
    ttnn.synchronize_device(dev)
    eager_ref = ttnn.to_torch(eager_out).float().clone()

    t0 = time.time()
    for _ in range(N_REPS):
        a = _tt(A_I, dev, dt); s = _tt(S_I, dev, dt); z = _tt(Z_II, dev, dt)
        mask = _tt(mask_host, dev, dt)
        out = dit.run_device(a, s, z, mask)
        ttnn.synchronize_device(dev)
    eager_ms = (time.time() - t0) / N_REPS * 1000
    print(f"[eager] {eager_ms:.3f} ms/call ({N_REPS} reps, I={I_REAL}, 18 blocks)")

    # ---- traced: persistent input buffers, capture once, replay N times ----
    def to_persistent(x_host, dtype):
        host_t = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, dev)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    a_p = to_persistent(A_I, dt)
    s_p = to_persistent(S_I, dt)
    z_p = to_persistent(Z_II, dt)
    mask_p = to_persistent(mask_host, dt)

    # warmup (compiles kernels; disallowed to allocate/write during actual capture)
    _ = dit.run_device(a_p, s_p, z_p, mask_p)
    _ = dit.run_device(a_p, s_p, z_p, mask_p)
    ttnn.synchronize_device(dev)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    traced_out = dit.run_device(a_p, s_p, z_p, mask_p)
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    # correctness: replay once, compare to the eager reference (same inputs -> must match)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    traced_ref = ttnn.to_torch(traced_out).float().clone()
    p = pcc(traced_ref, eager_ref)
    print(f"[parity] PCC(traced, eager) = {p:.6f}  (same fixed inputs -> must be ~1.0)")

    t0 = time.time()
    for _ in range(N_REPS):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(A_I, layout=ttnn.TILE_LAYOUT, dtype=dt), a_p)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    traced_ms = (time.time() - t0) / N_REPS * 1000
    print(f"[traced] {traced_ms:.3f} ms/call ({N_REPS} reps, I={I_REAL}, 18 blocks)")

    speedup = eager_ms / traced_ms if traced_ms > 0 else float("nan")
    print(f"[result] eager={eager_ms:.3f}ms traced={traced_ms:.3f}ms speedup={speedup:.3f}x")

    ttnn.release_trace(dev, tid)
    if p < 0.999:
        raise AssertionError(f"trace/eager PCC {p:.6f} < 0.999 -- capture is NOT lossless")


if __name__ == "__main__":
    main()
