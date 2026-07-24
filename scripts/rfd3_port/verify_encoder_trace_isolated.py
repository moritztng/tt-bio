"""p26: isolated verification of the PRODUCTION LocalAtomTransformer(trace=True) path
(RFD3_TRACE_ENCODER=1 integration wired in p25, unverified that pass -- device wedge).

Unlike p24's spike_atom_encoder_trace.py (which called enc.run_device() directly on
manually-persisted device tensors), this drives the class through its real __call__
entrypoint with fresh HOST tensors each call -- the same call shape
RFD3DiffusionModule._process_forward uses (self.encoder(Q_L, C_L, P_LL, indices=...)) --
to verify the _run_device_traced capture/refresh/replay path end to end, including a
shape change (forces release_trace + recapture) and repeat calls at the same shape
(exercise the copy_host_to_device_tensor refresh branch).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_encoder_trace_isolated.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.tenstorrent import get_device
from tt_bio.rfd3 import build_atom_encoder, _dense_attention_mask

sys.path.insert(0, os.path.dirname(__file__))
from verify_dit import load, scoped, pcc

GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")
N_KEYS = 128


def build_inputs(L, n_keys, seed):
    g = torch.Generator().manual_seed(seed)
    Q_L = torch.randn(1, L, 128, generator=g) * 0.1
    C_L = torch.randn(1, L, 128, generator=g) * 0.1
    P_LL = torch.randn(1, L, L, 16, generator=g) * 0.1
    idx = torch.zeros(1, L, n_keys, dtype=torch.long)
    for i in range(L):
        others = torch.randperm(L, generator=g)
        others = others[others != i][: n_keys - 1]
        idx[0, i] = torch.cat([torch.tensor([i]), others])
    return Q_L, C_L, P_LL, idx


def main():
    get_device(trace_region_size=1 << 28)

    weights = load(GOLDEN_DIR, "diffusion_module.real_weights")
    enc_weights = scoped(weights, "encoder")
    enc_eager = build_atom_encoder(enc_weights, trace=False)
    enc_traced = build_atom_encoder(enc_weights, trace=True)

    fails = []

    def check(label, L, n_keys, seed):
        Q_L, C_L, P_LL, idx = build_inputs(L, min(n_keys, L), seed)
        eager_out = enc_eager(Q_L, C_L, P_LL, indices=idx)
        traced_out = enc_traced(Q_L, C_L, P_LL, indices=idx)
        p = pcc(traced_out, eager_out)
        status = "OK" if p > 0.999 else "FAIL"
        if status == "FAIL":
            fails.append(label)
        print(f"[{status}] {label}: L={L} PCC(traced, eager) = {p:.6f}")

    # same shape, called 3x in a row with DIFFERENT data each time -- exercises the
    # copy_host_to_device_tensor refresh branch (capture happens once, replay 2x).
    check("call 1 (first capture)", 200, N_KEYS, seed=1)
    check("call 2 (same shape, refresh)", 200, N_KEYS, seed=2)
    check("call 3 (same shape, refresh)", 200, N_KEYS, seed=3)

    # shape change -- forces release_trace + recapture.
    check("call 4 (shape change -> recapture)", 419, N_KEYS, seed=4)
    check("call 5 (same new shape, refresh)", 419, N_KEYS, seed=5)

    # back to the original shape -- forces a SECOND recapture (not cached across
    # shape changes -- only one _trace_state slot).
    check("call 6 (shape change back -> recapture)", 200, N_KEYS, seed=6)

    if fails:
        raise AssertionError(f"encoder trace PCC failed for: {fails}")
    print("[result] all isolated production-class encoder trace checks PASS (PCC > 0.999)")


if __name__ == "__main__":
    main()
