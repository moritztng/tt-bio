"""p24: ttnn trace-capture spike for the RFD3 3-block atom encoder (LocalAtomTransformer),
in isolation, at the REAL fixture's atom-level shape (L=419 atoms, IAI_protein.pdb fixture).

Companion to spike_dit_trace.py -- the encoder/decoder operate at the (larger, L=419)
atom level rather than the (smaller, I=40) token level, so it is worth checking whether
the atom-level dispatch/compute ratio differs enough for trace to land a real win even
though the token-level DiT spike measured ~0 (1.02x).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/spike_atom_encoder_trace.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.tenstorrent import get_device
from tt_bio.rfd3 import build_atom_encoder, _dense_attention_mask, _tt

sys.path.insert(0, os.path.dirname(__file__))
from verify_dit import load, scoped, pcc

GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
L_REAL = 419  # real IAI_protein fixture atom count
N_KEYS = 128  # RFD3DiffusionModule.N_ATTN_KEYS (encoder/decoder use f["attn_indices"])


def build_shared_atom_inputs(L, n_keys, seed=42):
    g = torch.Generator().manual_seed(seed)
    Q_L = (torch.randn(1, L, 128, generator=g) * 0.1).bfloat16()
    C_L = (torch.randn(1, L, 128, generator=g) * 0.1).bfloat16()
    P_LL = (torch.randn(1, L, L, 16, generator=g) * 0.1).bfloat16()
    idx = torch.zeros(1, L, n_keys, dtype=torch.long)
    for i in range(L):
        others = torch.randperm(L, generator=g)
        others = others[others != i][: n_keys - 1]
        idx[0, i] = torch.cat([torch.tensor([i]), others])
    return Q_L, C_L, P_LL, idx


def main():
    get_device(trace_region_size=1 << 28)
    import ttnn

    weights = load(GOLDEN_DIR, "diffusion_module.real_weights")
    enc_weights = scoped(weights, "encoder")
    enc = build_atom_encoder(enc_weights)
    dev, dt = enc.device, enc.dtype

    Q_L, C_L, P_LL, indices = build_shared_atom_inputs(L_REAL, min(N_KEYS, L_REAL))
    mask_host = _dense_attention_mask(indices)

    N_REPS = 15
    q = _tt(Q_L, dev, dt); c = _tt(C_L, dev, dt); p = _tt(P_LL, dev, dt)
    mask = _tt(mask_host, dev, dt)
    eager_out = enc.run_device(q, c, p, mask)
    ttnn.synchronize_device(dev)
    eager_ref = ttnn.to_torch(eager_out).float().clone()

    t0 = time.time()
    for _ in range(N_REPS):
        q = _tt(Q_L, dev, dt); c = _tt(C_L, dev, dt); p = _tt(P_LL, dev, dt)
        mask = _tt(mask_host, dev, dt)
        out = enc.run_device(q, c, p, mask)
        ttnn.synchronize_device(dev)
    eager_ms = (time.time() - t0) / N_REPS * 1000
    print(f"[eager] {eager_ms:.3f} ms/call ({N_REPS} reps, L={L_REAL}, 3 blocks)")

    def to_persistent(x_host, dtype):
        host_t = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, dev)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    q_p = to_persistent(Q_L, dt)
    c_p = to_persistent(C_L, dt)
    p_p = to_persistent(P_LL, dt)
    mask_p = to_persistent(mask_host, dt)

    _ = enc.run_device(q_p, c_p, p_p, mask_p)
    _ = enc.run_device(q_p, c_p, p_p, mask_p)
    ttnn.synchronize_device(dev)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    traced_out = enc.run_device(q_p, c_p, p_p, mask_p)
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    traced_ref = ttnn.to_torch(traced_out).float().clone()
    p_val = pcc(traced_ref, eager_ref)
    print(f"[parity] PCC(traced, eager) = {p_val:.6f}  (same fixed inputs -> must be ~1.0)")

    t0 = time.time()
    for _ in range(N_REPS):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(Q_L, layout=ttnn.TILE_LAYOUT, dtype=dt), q_p)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    traced_ms = (time.time() - t0) / N_REPS * 1000
    print(f"[traced] {traced_ms:.3f} ms/call ({N_REPS} reps, L={L_REAL}, 3 blocks)")

    speedup = eager_ms / traced_ms if traced_ms > 0 else float("nan")
    print(f"[result] eager={eager_ms:.3f}ms traced={traced_ms:.3f}ms speedup={speedup:.3f}x")

    ttnn.release_trace(dev, tid)
    if p_val < 0.999:
        raise AssertionError(f"trace/eager PCC {p_val:.6f} < 0.999 -- capture is NOT lossless")


if __name__ == "__main__":
    main()
