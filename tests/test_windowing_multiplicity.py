"""Pure-torch reference + correctness test for the M-aware (multiplicity-batched)
windowing that the Protenix/OpenDDE device-denoise batch will need.

This is a SPEC for the upcoming ttnn port, not the port itself. It proves the
structural insight (recorded in state/tt-bio-diffusion-multiplicity-batching.md)
that folding the multiplicity M into the windowed-attention block dim B (= M*nb)
via a LEADING-M 3D pad produces per-sample-correct windows with NO cross-sample
bleed -- the trap to avoid is a single trailing pad on a flattened (M*N, C) tensor,
which puts all padding at the end and makes the last sample's windows all-pad while
earlier samples bleed across the boundary.

Mirrors the ttnn op sequence in tt_bio/protenix.py (_window_q / _window_kv and
AtomTransformer._windows_q / _windows_kv) but in pure torch so it runs with no device.

Verified here:
  1. M=1: the M-aware functions are bit-exact with the existing single-sample
     windowing (the current M=1 path is the special case).
  2. M>1: each (sample k, block b) window equals the single-sample window (block b)
     of sample k's atoms -- no cross-sample bleed, no all-pad windows.
  3. The NAIVE single-trailing-pad approach (the trap) DOES bleed across samples --
     included as a negative control to demonstrate the failure mode the ttnn port
     must avoid.

Run: python3 tests/test_windowing_multiplicity.py   (no device needed)
"""
import torch

NQ, NK, PAD_LEFT = 32, 128, 48


def window_q_single(x, N, NP, nq=NQ):
    """(N, C) -> (nb, nq, C), right-padded to NP. Mirrors _window_q at M=1."""
    C = x.shape[-1]
    x = torch.nn.functional.pad(x, (0, 0, 0, NP - N))
    return x.reshape(NP // nq, nq, C)


def window_kv_single(x, N, NP, nq=NQ, nk=NK, pad_left=PAD_LEFT):
    """(N, C) -> (nb, nk, C), overlapping windows with a left pad of (nk-nq)/2.
    Mirrors _window_kv at M=1 (the gather semantics: window i, key j <- padded row
    i*nq + j)."""
    C = x.shape[-1]
    Lp = pad_left + NP + nk
    xp = torch.nn.functional.pad(x, (0, 0, pad_left, Lp - pad_left - N))
    nb = NP // nq
    # window i, key j <- padded row i*nq + j  (i in [0,nb), j in [0,nk))
    idx = (torch.arange(nb).reshape(nb, 1) * nq + torch.arange(nk).reshape(1, nk)).reshape(nb * nk)
    return xp[idx].reshape(nb, nk, C)


def window_q_m(x, N, NP, nq=NQ):
    """M-aware: (M, N, C) -> (M*nb, nq, C). Per-sample right-pad via the leading-M
    3D pad (torch.nn.functional.pad on dim 1 pads each of the M samples independently),
    then reshape to fold M into B = M*nb. Sample k's nb windows stay contiguous and
    correct: block k*nb .. k*nb+nb-1 are sample k's windows."""
    M, _, C = x.shape
    x = torch.nn.functional.pad(x, (0, 0, 0, NP - N))      # (M, NP, C), per-sample pad
    return x.reshape(M * (NP // nq), nq, C)


def window_kv_m(x, N, NP, nq=NQ, nk=NK, pad_left=PAD_LEFT):
    """M-aware KV windowing: (M, N, C) -> (M*nb, nk, C). Per-sample left+right pad via
    the leading-M 3D pad, then a per-sample gather (the table is (M, Lp, C); gather
    per sample with the same (nb*nk,) indices)."""
    M, _, C = x.shape
    Lp = pad_left + NP + nk
    x = torch.nn.functional.pad(x, (0, 0, pad_left, Lp - pad_left - N))   # (M, Lp, C), per-sample
    nb = NP // nq
    idx = (torch.arange(nb).reshape(nb, 1) * nq + torch.arange(nk).reshape(1, nk)).reshape(nb * nk)
    # gather per sample: (M, nb*nk, C) -> (M*nb, nk, C)
    out = x[:, idx]                                         # (M, nb*nk, C)
    return out.reshape(M * nb, nk, C)


def window_q_naive_flat(x, N, NP, nq=NQ):
    """The TRAP: flatten (M, N, C) -> (M*N, C), single trailing pad to (M*NP, C)? No --
    a single trailing pad of (M*N -> M*NP) puts ALL padding at the end. Demonstrates
    the bleed. (For a fair bleed demo we pad (M*N) to (M*N + M*(NP-N)) = M*NP with a
    single trailing pad, then reshape (M*nb, nq, C).)"""
    M, _, C = x.shape
    flat = x.reshape(M * N, C)
    flat = torch.nn.functional.pad(flat, (0, 0, 0, M * (NP - N)))   # ALL pad at the end
    return flat.reshape(M * (NP // nq), nq, C)


def run():
    torch.manual_seed(0)
    cases = [(N, N) for N in (37, 64, 96, 128)] + [(37, 64), (64, 96)]   # (N, NP) with NP>=N, NQ-multiple
    cases = [(N, NP) for (N, NP) in cases if NP >= N and NP % NQ == 0 and N <= NP]
    M = 4
    max_diff_m1 = 0.0
    max_bleed = 0.0
    naive_bleed = 0.0
    for N, NP in cases:
        C = 128
        x = torch.randn(M, N, C)
        for nq, nk, pl, wq_m, wkv_m, wq1, wkv1 in [
            (NQ, NK, PAD_LEFT, window_q_m, window_kv_m, window_q_single, window_kv_single),
        ]:
            # 1. M=1 bit-exact with single-sample windowing
            x1 = x[:1]
            q_m1 = wq_m(x1, N, NP, nq) if wq_m is window_q_m else wq_m(x1, N, NP, nq, nk, pl)
            q_ref = wq1(x1[0], N, NP, nq)
            d = (q_m1.reshape(q_ref.shape) - q_ref).abs().max().item()
            max_diff_m1 = max(max_diff_m1, d)
            kv_m1 = wkv_m(x1, N, NP, nq, nk, pl)
            kv_ref = wkv1(x1[0], N, NP, nq, nk, pl)
            d = (kv_m1.reshape(kv_ref.shape) - kv_ref).abs().max().item()
            max_diff_m1 = max(max_diff_m1, d)

            # 2. M>1 per-sample correctness: block k*nb..k*nb+nb-1 == single-sample windows of x[k]
            nb = NP // nq
            q_full = wq_m(x, N, NP, nq)                       # (M*nb, nq, C)
            for k in range(M):
                qk = q_full[k * nb:(k + 1) * nb]
                qk_ref = wq1(x[k], N, NP, nq)
                d = (qk - qk_ref).abs().max().item()
                max_bleed = max(max_bleed, d)
            kv_full = wkv_m(x, N, NP, nq, nk, pl)              # (M*nb, nk, C)
            for k in range(M):
                kvk = kv_full[k * nb:(k + 1) * nb]
                kvk_ref = wkv1(x[k], N, NP, nq, nk, pl)
                d = (kvk - kvk_ref).abs().max().item()
                max_bleed = max(max_bleed, d)

            # 3. NAIVE single-trailing-pad trap: demonstrate it bleeds (negative control)
            q_naive = window_q_naive_flat(x, N, NP, nq)        # (M*nb, nq, C)
            for k in range(M):
                qk = q_naive[k * nb:(k + 1) * nb]
                qk_ref = wq1(x[k], N, NP, nq)
                d = (qk - qk_ref).abs().max().item()
                naive_bleed = max(naive_bleed, d)

    print(f"[CHECK] M=1 M-aware vs single-sample windowing  max abs diff = {max_diff_m1:.3e}  (expect 0.0)")
    print(f"[CHECK] M>1 per-sample correctness (no bleed)   max abs diff = {max_bleed:.3e}  (expect 0.0)")
    print(f"[CTRL]  naive single-trailing-pad bleed        max abs diff = {naive_bleed:.3e}  (expect >0, the trap)")
    ok = (max_diff_m1 == 0.0 and max_bleed == 0.0 and naive_bleed > 0.0)
    print("ALL WINDOWING-MULTIPLICITY TESTS PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
