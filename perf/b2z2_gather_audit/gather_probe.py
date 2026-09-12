#!/usr/bin/env python3
"""Score one atom-key-gather implementation against the gather the cached matrix defines.

The reference is not another implementation. It is `boltz2.get_indexing_matrix` at the REAL
window count, zero-padded to the atom bucket exactly as `tenstorrent.py` pads it before caching
it, contracted with the atom sequence in float64 on the host. That is what the model computes,
so an implementation that claims to be the gather has to reproduce it.

One arm per process, because each arm lives in a different checkout and a process gets one
device context. Each run writes its output tensor and its score to --out.

  --impl onehot   the shipped chain (reshape, permute, matmul, permute, reshape), transcribed
                  from AttentionPairBias.__call__ where it is inline rather than a function
  --impl window   tt_bio.tenstorrent._atom_key_window        (TT_BIO_ATOM_KEY_WINDOW)
  --impl shift    tt_bio.tenstorrent._atom_shift_gather      (TT_BIO_ATOM_SHIFT_GATHER)

--tail live|zero decides whether the atom sequence carries values in the bucket's padded tail.
Inside the diffusion transformer it always does: the pad rows go through linears with bias and
through layer norms. `zero` is the fixture a probe would build by hand, and it is the control
that shows why this defect survived every check.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

W = 32                  # ATOM_WINDOW
H = 128                 # ATOM_DIM, the key window


def padded_keys_indexing(k_real, k_pad):
    """The cached matrix: built at the real window count, zero-padded to the bucket."""
    import torch
    from tt_bio.boltz2 import get_indexing_matrix
    ki = get_indexing_matrix(k_real, W, H, torch.device("cpu"))
    return torch.nn.functional.pad(ki, (0, 8 * k_pad - ki.shape[1], 0, 2 * k_pad - ki.shape[0]))


def reference(s_pt, ki):
    """`single_to_keys` on the cached matrix, in float64. b j i d, j k -> b k i d."""
    import torch
    b, k, w, d = s_pt.shape
    x = s_pt.to(torch.float64).view(b, 2 * k, w // 2, d)
    out = torch.einsum("bjid,jk->bkid", x, ki.to(torch.float64))
    return out.reshape(b, k, H, d)


def arm_onehot(ttnn, T, s, ki_tt, _ki_pt):
    b, k, w, d = s.shape
    x = ttnn.reshape(s, (b, 2 * k, w // 2, -1))
    x = ttnn.permute(x, (0, 2, 3, 1))
    x = ttnn.matmul(x, ki_tt, compute_kernel_config=_CKC[0], core_grid=T.CORE_GRID_MAIN)
    x = ttnn.permute(x, (0, 3, 1, 2))
    return ttnn.reshape(x, (b, k, -1, d))


def arm_window(ttnn, T, s, ki_tt, _ki_pt):
    out = T._atom_key_window(s, ki_tt)
    if out is None:
        raise SystemExit("_atom_key_window declined this geometry")
    return out


def arm_shift(ttnn, T, s, _ki_tt, ki_pt):
    windows = T._atom_gather_shift_windows(ki_pt)
    if windows is None:
        raise SystemExit("_atom_gather_shift_windows declined this matrix")
    print(f"shift gate recovered windows={windows}", flush=True)
    return T._atom_shift_gather(s, windows)


ARMS = {"onehot": arm_onehot, "window": arm_window, "shift": arm_shift}
_CKC = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=sorted(ARMS))
    ap.add_argument("--windows-real", type=int, default=129)
    ap.add_argument("--windows-pad", type=int, default=140)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tail", default="live", choices=("live", "zero"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T

    ki_pt = padded_keys_indexing(a.windows_real, a.windows_pad)
    torch.manual_seed(a.seed)
    s_pt = torch.randn(1, a.windows_pad, W, a.dim).to(torch.bfloat16).float()
    if a.tail == "zero":
        flat = s_pt.view(1, -1, a.dim)
        flat[:, a.windows_real * W:, :] = 0.0
        s_pt = flat.view(1, a.windows_pad, W, a.dim)
    ref = reference(s_pt, ki_pt)

    T.get_device(trace_region_size=512 * 1024 * 1024)
    dev = T.get_device()
    _CKC[0] = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    s = ttnn.from_torch(s_pt, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ki_tt = ttnn.from_torch(ki_pt, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)

    got = ttnn.to_torch(ARMS[a.impl](ttnn, T, s, ki_tt, ki_pt)).float()
    if list(got.shape) != list(ref.shape):
        raise SystemExit(f"shape {list(got.shape)} != reference {list(ref.shape)}")

    diff = (got - ref.float()).abs()
    per_window = diff.amax(dim=(0, 2, 3))
    bad = (per_window > 0).nonzero().flatten().tolist()
    res = {
        "impl": a.impl, "tail": a.tail, "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "seed": a.seed,
        "windows_real": a.windows_real, "windows_pad": a.windows_pad, "dim": a.dim,
        "ki_shape": list(ki_pt.shape), "ki_nonzero_cols": int((ki_pt != 0).any(0).sum()),
        "out_shape": list(got.shape),
        "max_abs": float(diff.max()),
        "bit_exact": bool(torch.equal(got, ref.float())),
        "n_bad_windows": len(bad),
        "first_bad_window": bad[0] if bad else None,
        "bad_windows": bad,
        "per_window_max_abs": [float(x) for x in per_window],
        "t": time.time(),
    }
    print(json.dumps({k: res[k] for k in ("impl", "tail", "ki_shape", "ki_nonzero_cols",
                                          "max_abs", "bit_exact", "n_bad_windows",
                                          "first_bad_window")}, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
