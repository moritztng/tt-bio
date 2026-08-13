#!/usr/bin/env python3
"""A fused 2D complex FFT on Blackhole: correctness against numpy.fft, and throughput vs the floor.

The transform is Y = F . X . F blocked into 32x32 tiles, with the image and the DFT matrix both L1
resident for the whole transform, so DRAM sees one read and one write per image instead of the four
a two-pass GPU FFT needs. Arm A is the DFT-by-matmul path the feasibility pass measured, built out
of stock ttnn ops on the same data; arm B is the fused kernel.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels" / "fft2d"
CB_F, CB_X, CB_T, CB_O = 0, 1, 2, 16
DTYPES = {"bf16": (ttnn.bfloat16, torch.bfloat16, 2), "fp32": (ttnn.float32, torch.float32, 4)}


def dft_blocks(N, tor):
    """F, i*F and -i*F as an NT x NT grid of 32x32 tiles, interleaved Fr, Fi, -Fi.

    Built in float64 on the host and cast once. Nothing trigonometric runs on device, which deletes
    the sin/cos precision problem the #21412 thread was still working on in February 2026 for the
    cost of one O(N^2) constant.
    """
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N)
    NT = N // 32
    out = torch.empty(3 * NT * NT, 32, 32, dtype=tor)
    for m in range(NT):
        for j in range(NT):
            b = F[m * 32:(m + 1) * 32, j * 32:(j + 1) * 32]
            k = m * NT + j
            out[3 * k] = torch.from_numpy(b.real.copy()).to(tor)
            out[3 * k + 1] = torch.from_numpy(b.imag.copy()).to(tor)
            out[3 * k + 2] = torch.from_numpy((-b.imag).copy()).to(tor)
    return out


def to_tiles(img, tor):
    """Complex [NIMG, N, N] -> tile pages, Xr/Xi interleaved, row-major over (i, j) tile blocks."""
    nimg, N, _ = img.shape
    NT = N // 32
    out = torch.empty(nimg * 2 * NT * NT, 32, 32, dtype=tor)
    for b in range(nimg):
        for i in range(NT):
            for j in range(NT):
                k = b * 2 * NT * NT + 2 * (i * NT + j)
                blk = img[b, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
                out[k] = torch.from_numpy(blk.real.copy()).to(tor)
                out[k + 1] = torch.from_numpy(blk.imag.copy()).to(tor)
    return out


def from_tiles(t, nimg, N):
    NT = N // 32
    a = t.to(torch.float32).numpy()
    out = np.empty((nimg, N, N), dtype=np.complex128)
    for b in range(nimg):
        for i in range(NT):
            for j in range(NT):
                k = b * 2 * NT * NT + 2 * (i * NT + j)
                out[b, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = a[k] + 1j * a[k + 1]
    return out


def pack(t):
    """[P, 32, 32] -> the [1, 1, 32P, 32] page layout ttnn's TensorAccessor indexes."""
    return t.reshape(1, 1, -1, 32)


def build(dev, N, nimg, dt, ftt, xtt, ott):
    NT, (tdt, _, nb) = N // 32, DTYPES[dt]
    tb = 32 * 32 * nb
    nf, ntile = 3 * NT * NT, 2 * NT * NT
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=tdt, page_size=tb)
        return ttnn.CBDescriptor(total_size=d * tb, core_ranges=cg, format_descriptors=[f])

    # The two depths that matter, and they matter for different reasons.
    #
    # cb_x at one image deep forces the reader and the compute to alternate: the kernel does
    # cb_wait_front(cb_x, ntile) and cannot start until a whole image has landed, and the reader
    # cannot start the next one until compute pops. At two deep the reader runs a full image ahead
    # and the DRAM read hides behind the matmuls. No kernel change is needed for this -- the depth
    # alone buys the pipelining, because the CB protocol already expresses it.
    #
    # cb_o is deliberately small: pass 2 emits two tiles at a time and the writer drains them, so a
    # whole output image is never resident and the write overlaps the matmuls that follow it.
    xdepth = int(os.environ.get("FFT_XDEPTH", "2"))
    odepth = int(os.environ.get("FFT_ODEPTH", "32"))
    cbs = [cb(CB_F, nf), cb(CB_X, xdepth * ntile), cb(CB_T, ntile), cb(CB_O, odepth)]

    rct = [CB_F, CB_X, tb, nf, ntile, nimg]
    rct += list(ttnn.TensorAccessorArgs(ftt).get_compile_time_args())
    rct += list(ttnn.TensorAccessorArgs(xtt).get_compile_time_args())
    # The writer drains in chunks of `chunk` tiles behind one barrier. It must divide ntile and fit
    # in cb_o, so cb_o is sized as two chunks: one being written out, one being filled by pass 2.
    chunk = int(os.environ.get("FFT_CHUNK", "16"))
    wct = [CB_O, tb, ntile, nimg, chunk] + list(ttnn.TensorAccessorArgs(ott).get_compile_time_args())

    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [ftt.buffer_address(), xtt.buffer_address(), c * nimg * ntile]
            crt[cx][cy] = [0]
            wrt[cx][cy] = [ott.buffer_address(), c * nimg * ntile]
            c += 1
    mk = lambda s, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(KDIR / s), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    fid = getattr(ttnn.MathFidelity, os.environ.get("FFT_FIDELITY", "HiFi2"))
    return ttnn.ProgramDescriptor(kernels=[
        mk("reader_fft2d.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk("writer_fft2d.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute_fft2d.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=[CB_F, CB_X, CB_T, CB_O, NT, nimg],
            runtime_args=crt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True)),
    ], semaphores=[], cbs=cbs)


def run_box(dev, N, dt, nimg, reps=5):
    tdt, tor, nb = DTYPES[dt]
    g = dev.compute_with_storage_grid_size()
    ncores = g.x * g.y
    NT = N // 32
    ntile = 2 * NT * NT
    total = ncores * nimg

    rng = np.random.default_rng(0)
    img = (rng.standard_normal((total, N, N)) + 1j * rng.standard_normal((total, N, N))) / N

    ftt = ttnn.from_torch(pack(dft_blocks(N, tor)), dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)
    xtt = ttnn.from_torch(pack(to_tiles(img, tor)), dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)
    ott = ttnn.from_torch(pack(torch.zeros(total * ntile, 32, 32, dtype=tor)),
                          dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)

    pd = build(dev, N, nimg, dt, ftt, xtt, ott)
    ttnn.generic_op([ftt, xtt, ott], pd)
    ttnn.synchronize_device(dev)

    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.generic_op([ftt, xtt, ott], pd)
        ttnn.synchronize_device(dev)
        best = min(best, time.perf_counter() - t0)

    # Verify a subset. The transform is identical per image and every core runs the same program, so
    # checking all of a large batch costs host time without adding evidence; the batch exists to
    # amortise the DFT matrix, not to broaden the correctness claim. nv is reported so the claim
    # says what it covers.
    nv = min(total, int(os.environ.get("FFT_VERIFY_N", "130")))
    got = from_tiles(ttnn.to_torch(ott).reshape(-1, 32, 32)[:nv * ntile], nv, N)
    ref = np.fft.fft2(img[:nv], axes=(-2, -1))
    err = got - ref
    rel = float(np.linalg.norm(err) / np.linalg.norm(ref))

    # Error against spatial frequency: FSC at 0.143 is decided by the high-frequency shells, so a
    # single norm is not enough to judge a cryo-EM FFT.
    ky = np.minimum(np.arange(N), N - np.arange(N))
    r = np.round(np.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2)).astype(int)
    shells = {}
    for lo, hi, nm in ((0, N // 8, "low"), (N // 8, N // 4, "mid"), (N // 4, N, "high")):
        m = (r >= lo) & (r < hi)
        shells[nm] = float(np.linalg.norm(err[:, m]) / np.linalg.norm(ref[:, m]))

    bytes_moved = 2 * total * ntile * 32 * 32 * nb          # one read + one write per image
    return {
        "box": N, "dtype": dt, "images_per_core": nimg, "images": total,
        "ms": best * 1e3, "images_per_s": total / best,
        "rel_l2": rel, "rel_l2_by_shell": shells,
        "max_abs_over_max_ref": float(np.abs(err).max() / np.abs(ref).max()),
        "achieved_GBps_image_only": bytes_moved / best / 1e9,
        # The DFT matrix is re-read by every core on every launch, so it is real DRAM traffic and
        # belongs in any honest bandwidth figure. It is separated out because it is the term the
        # batch size amortises, and hiding it inside one number would hide the lever.
        "f_bytes_frac": 3 * (N // 32) ** 2 / (2 * nimg * 2 * (N // 32) ** 2),
        "achieved_GBps_total": (bytes_moved + ncores * 3 * (N // 32) ** 2 * 32 * 32 * nb)
                               / best / 1e9,
        "verified_images": nv,
        "xdepth": int(os.environ.get("FFT_XDEPTH", "2")),
        "odepth": int(os.environ.get("FFT_ODEPTH", "32")),
        "chunk": int(os.environ.get("FFT_CHUNK", "16")),
        "fidelity": os.environ.get("FFT_FIDELITY", "HiFi2"),
        "l1_tiles": 3 * (N // 32) ** 2 + int(os.environ.get("FFT_XDEPTH", "2")) * 2 * (N // 32) ** 2
                    + 2 * (N // 32) ** 2 + int(os.environ.get("FFT_ODEPTH", "32")),
    }


def main():
    dev = ttnn.open_device(device_id=0)
    out = {"arms": []}
    res_path = Path(__file__).resolve().parent.parent / "fftprobe" / "fft2d_result.json"
    nimg = int(os.environ.get("FFT_NIMG", "8"))
    boxes = [int(b) for b in os.environ.get("FFT_BOXES", "256").split(",")]
    sweep = os.environ.get("FFT_SWEEP", "")
    try:
        for N in boxes:
            for dt in os.environ.get("FFT_DTYPES", "bf16").split(","):
                for combo in (sweep.split(";") if sweep else [""]):
                    for kv in combo.split(","):
                        if "=" in kv:
                            k, v = kv.split("=")
                            os.environ[k] = v
                    try:
                        r = run_box(dev, N, dt, nimg)
                        out["arms"].append(r)
                        print(f"box {r['box']} {dt} x{r['xdepth']} o{r['odepth']} c{r['chunk']} "
                              f"{r['fidelity']}: {r['images_per_s']:10.0f} img/s  "
                              f"{r['achieved_GBps_total']:6.1f} GB/s tot  "
                              f"rel_l2 {r['rel_l2']:.3e}  "
                              f"L1 {r['l1_tiles']} tiles", flush=True)
                    except Exception as e:                               # noqa: BLE001
                        out["arms"].append({"box": N, "dtype": dt, "combo": combo,
                                            "error": str(e)[:400]})
                        print(f"box {N} {dt} [{combo}]: ERROR {str(e)[:220]}", flush=True)
                    json.dump(out, open(res_path, "w"), indent=1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
