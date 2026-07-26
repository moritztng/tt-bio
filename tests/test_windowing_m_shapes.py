"""CPU shape test for the M-aware (multiplicity-batched) ttnn windowing helpers.

Pure-torch mirror of the ttnn op sequence in tt_bio/protenix._window_q_m /
_window_kv_m (the M-aware variants used by the gated batched device denoise). Verifies
the SHAPE flow (not ttnn values) that the card-session on-device port will follow:
folding multiplicity M into the windowed-attention block dim B = M*nb via a leading-M
3D pad, with per-sample-correct windows and no cross-sample bleed.

No device needed. Run: python3 tests/test_windowing_m_shapes.py
"""
import torch

NQ, NK, PAD_LEFT = 32, 128, 48


def window_q_m(x, M, N, NP, nq=NQ):
    """(M,N,C) -> (M*nb, nq, C) via leading-M 3D pad on dim 1, then reshape."""
    x = torch.nn.functional.pad(x, (0, 0, 0, NP - N))      # (M, NP, C), per-sample pad
    return x.reshape(M * (NP // nq), nq, x.shape[-1])


def window_q_single(x, N, NP, nq=NQ):
    x = torch.nn.functional.pad(x, (0, 0, 0, NP - N))
    return x.reshape(NP // nq, nq, x.shape[-1])


def run():
    torch.manual_seed(0)
    cases = [(N, NP) for (N, NP) in [(37, 64), (64, 64), (96, 128), (128, 128)] if NP >= N and NP % NQ == 0]
    M = 4
    max_diff = 0.0
    for N, NP in cases:
        C = 128
        x = torch.randn(M, N, C)
        q = window_q_m(x, M, N, NP)                       # (M*nb, nq, C)
        nb = NP // NQ
        assert q.shape == (M * nb, NQ, C), f"window_q_m shape {q.shape}"
        for k in range(M):
            qk = q[k * nb:(k + 1) * nb]                 # sample k's windows
            qk_ref = window_q_single(x[k], N, NP)        # single-sample windowing of x[k]
            d = (qk - qk_ref).abs().max().item()
            max_diff = max(max_diff, d)
    print(f"[CHECK] window_q_m per-sample correctness max abs diff = {max_diff:.3e} (expect 0.0)")
    assert max_diff == 0.0, "window_q_m bleeds across samples"
    print("ALL WINDOWING-M-SHAPE TESTS PASS")


if __name__ == "__main__":
    run()
