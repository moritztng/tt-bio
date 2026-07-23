"""p24: ttnn trace-capture spike for the RFD3 decoder (CompactStreamingDecoder), the
component profile_step_breakdown.py identifies as the single largest contributor to
per-step wall-clock (~40% in a rough 11-step profile). Companion to spike_dit_trace.py
(dead end, 1.02x) and spike_atom_encoder_trace.py (real win, 5.56x on the pure
run_device path).

CompactStreamingDecoder.__call__ (tt_bio/rfd3.py) already separates cleanly into (a) a
handful of host uploads / cheap host index math at the top, and (b) a pure-device loop
(pack -> upcast.run_device -> unpack -> atom_block, x3, then downcast) that only takes
device tensors -- no from_torch inside the loop body itself. This script re-runs that
exact loop body against PERSISTENT device buffers (the pattern a real trace integration
would use), instead of modifying the shipped class.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/spike_decoder_trace.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.tenstorrent import get_device
from tt_bio.rfd3 import build_decoder, _dense_attention_mask, _tt, _tt_idx, _build_valid_mask

sys.path.insert(0, os.path.dirname(__file__))
from verify_dit import load, scoped, pcc

GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
L_REAL = 419
I_REAL = 40
N_KEYS = 128


def build_shared_decoder_inputs(L, I, n_keys, seed=42):
    g = torch.Generator().manual_seed(seed)
    A_I = (torch.randn(1, I, 768, generator=g) * 0.1).bfloat16()
    S_I = (torch.randn(1, I, 384, generator=g) * 0.1).bfloat16()
    Q_L = (torch.randn(1, L, 128, generator=g) * 0.1).bfloat16()
    C_L = (torch.randn(1, L, 128, generator=g) * 0.1).bfloat16()
    P_LL = (torch.randn(1, L, L, 16, generator=g) * 0.1).bfloat16()
    tok_idx = torch.sort(torch.randint(0, I, (L,), generator=g)).values  # monotonic, like real atom_to_token_map
    idx = torch.zeros(1, L, n_keys, dtype=torch.long)
    for i in range(L):
        others = torch.randperm(L, generator=g)
        others = others[others != i][: n_keys - 1]
        idx[0, i] = torch.cat([torch.tensor([i]), others])
    return A_I, S_I, Q_L, C_L, P_LL, tok_idx, idx


def grouping_indices(tok_idx, batch):
    valid = _build_valid_mask(tok_idx)
    length = tok_idx.numel()
    padded = torch.full(valid.shape, length, dtype=torch.int64)
    padded[valid] = torch.arange(length)
    pack = torch.cat([padded.reshape(-1) + b * (length + 1) for b in range(batch)])
    flat_valid = valid.flatten().nonzero(as_tuple=False).squeeze(1)
    unpack = torch.cat([flat_valid + b * valid.numel() for b in range(batch)])
    return valid, pack, unpack


def run_device_loop(dec, a_split, q, c, p, mask, upcast_mask_dev, pack_idx_dev, unpack_idx_dev, valid, length):
    """Mirrors CompactStreamingDecoder.__call__'s pure-device loop body exactly
    (tt_bio/rfd3.py:961-967), operating on already-persistent device tensors."""
    for upcast, atom_block in zip(dec.upcast, dec.atom_blocks):
        q_grouped = dec._pack_atoms_device(q, pack_idx_dev, valid)
        q_grouped = ttnn.add(q_grouped, upcast.run_device(q_grouped, a_split, attn_mask_dev=upcast_mask_dev))
        q = dec._unpack_atoms_device(q_grouped, unpack_idx_dev, length)
        q = atom_block(q, c, p, mask)
    return q


def main():
    get_device(trace_region_size=1 << 28)
    global ttnn
    import ttnn

    weights = load(GOLDEN_DIR, "diffusion_module.real_weights")
    dec_weights = scoped(weights, "decoder")
    dec = build_decoder(dec_weights)
    dev, dt = dec.device, dec.dtype

    A_I, S_I, Q_L, C_L, P_LL, tok_idx, indices = build_shared_decoder_inputs(L_REAL, I_REAL, min(N_KEYS, L_REAL))
    valid, pack_idx, unpack_idx = grouping_indices(tok_idx, 1)
    mask_host = _dense_attention_mask(indices)
    valid_q_host = valid.unsqueeze(-1).expand(-1, -1, 3)

    def upload_all():
        a = _tt(A_I, dev, dt)
        a_split = ttnn.reshape(a, (1, I_REAL, 3, 256))
        q = _tt(Q_L, dev, dt)
        c = _tt(C_L, dev, dt)
        p = _tt(P_LL, dev, dt)
        mask = _tt(mask_host, dev, dt)
        pack_idx_dev = _tt_idx(pack_idx, dev)
        unpack_idx_dev = _tt_idx(unpack_idx, dev)
        upcast_mask_dev = dec.upcast[0]._prepare_additive_mask(valid_q_host, 1, valid.shape[0], valid.shape[1], 3)
        return a_split, q, c, p, mask, upcast_mask_dev, pack_idx_dev, unpack_idx_dev

    N_REPS = 12
    args = upload_all()
    eager_out = run_device_loop(dec, *args, valid, L_REAL)
    ttnn.synchronize_device(dev)
    eager_ref = ttnn.to_torch(eager_out).float().clone()

    t0 = time.time()
    for _ in range(N_REPS):
        args = upload_all()
        out = run_device_loop(dec, *args, valid, L_REAL)
        ttnn.synchronize_device(dev)
    eager_ms = (time.time() - t0) / N_REPS * 1000
    print(f"[eager] {eager_ms:.3f} ms/call ({N_REPS} reps, L={L_REAL}, 3x upcast+atomblock)")

    def to_persistent(x_host, dtype, layout=None):
        host_t = ttnn.from_torch(x_host, layout=layout or ttnn.TILE_LAYOUT, dtype=dtype)
        dev_t = ttnn.allocate_tensor_on_device(host_t.spec, dev)
        ttnn.copy_host_to_device_tensor(host_t, dev_t)
        return dev_t

    a_p = to_persistent(A_I, dt)
    a_split_p = ttnn.reshape(a_p, (1, I_REAL, 3, 256))
    q_p = to_persistent(Q_L, dt)
    c_p = to_persistent(C_L, dt)
    p_p = to_persistent(P_LL, dt)
    mask_p = to_persistent(mask_host, dt)
    pack_idx_p = to_persistent(pack_idx.to(torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    unpack_idx_p = to_persistent(unpack_idx.to(torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    upcast_mask_p = dec.upcast[0]._prepare_additive_mask(valid_q_host, 1, valid.shape[0], valid.shape[1], 3)

    persistent_args = (a_split_p, q_p, c_p, p_p, mask_p, upcast_mask_p, pack_idx_p, unpack_idx_p)
    _ = run_device_loop(dec, *persistent_args, valid, L_REAL)
    _ = run_device_loop(dec, *persistent_args, valid, L_REAL)
    ttnn.synchronize_device(dev)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    traced_out = run_device_loop(dec, *persistent_args, valid, L_REAL)
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
    print(f"[traced] {traced_ms:.3f} ms/call ({N_REPS} reps, L={L_REAL}, 3x upcast+atomblock)")

    speedup = eager_ms / traced_ms if traced_ms > 0 else float("nan")
    print(f"[result] eager={eager_ms:.3f}ms traced={traced_ms:.3f}ms speedup={speedup:.3f}x")

    ttnn.release_trace(dev, tid)
    if p_val < 0.999:
        raise AssertionError(f"trace/eager PCC {p_val:.6f} < 0.999 -- capture is NOT lossless")


if __name__ == "__main__":
    main()
