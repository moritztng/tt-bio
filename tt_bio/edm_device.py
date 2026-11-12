"""The EDM sampler step on the card (lever ``edm_device``), shared by OpenFold3 and the Protenix family.

The coordinates stay resident between diffusion steps: the host draws each step's rotation, translation and
noise in its own RNG order and stages them, and the captured step does the augmentation, the denoise and the
EDM update in fp32. Nothing is read back until the last step, so the device never waits on the host between
steps.
"""
import math

import ttnn


def edm_step_scalars(t: float, c_tau: float, sigma_data: float, step_scale: float):
    """The per-step scalars the ``edm_device`` step reads: ``(1/sqrt(t^2+sd^2), c_skip, c_out,
    1/t, step_scale*(c_tau-t))``, in python floats (float64), cast once to fp32 on upload."""
    sd2 = sigma_data ** 2
    return (1.0 / math.sqrt(t * t + sd2), sd2 / (sd2 + t * t),
            sigma_data * t / math.sqrt(sd2 + t * t), 1.0 / t, step_scale * (c_tau - t))


def matmul_f32(a, b, ckc):
    """fp32 ``a @ b`` from three bf16-operand matmuls (hi*hi + lo*hi + hi*lo).

    The FPU reads fp32 operands as tf32 (11-bit mantissa), which put ~5e-4 relative error on
    the mean and rotation of every edm_device step. Each operand is split into a bf16-exact
    head and its residual; the heads multiply exactly and the dropped lo*lo term is ~2^-16
    relative, so the result is within ~1e-6 of fp32. The operands here are [NP, 3] and [3, 3],
    so the two extra matmuls cost nothing measurable.
    """
    def split(x):
        h16 = ttnn.typecast(x, ttnn.bfloat16)
        h = ttnn.typecast(h16, ttnn.float32)
        ttnn.deallocate(h16)
        return h, ttnn.subtract(x, h)

    ah, al = split(a)
    bh, bl = split(b)
    hh = ttnn.matmul(ah, bh, compute_kernel_config=ckc)
    lh = ttnn.matmul(al, bh, compute_kernel_config=ckc)
    hl = ttnn.matmul(ah, bl, compute_kernel_config=ckc)
    s1 = ttnn.add(lh, hl)
    out = ttnn.add(hh, s1)
    for y in (ah, al, bh, bl, hh, lh, hl, s1):
        ttnn.deallocate(y)
    return out


def project_positions(r, w32, c, ckc, out_dtype=None):
    """``c + r @ w32``, the diffusion's first read of the atom positions, exact in fp32.

    ``r`` is the scaled absolute position (x / sqrt(t^2 + sigma_data^2)). Rounded to bf16 (or read
    as tf32 by an fp32 matmul) it keeps 8 (11) bits of |x|, so the network sees neighbouring atoms
    with an error that doubles every octave of |x|. Here ``r`` and ``w32`` stay fp32 through
    `matmul_f32`, and the sum is cast to ``out_dtype`` (``c``'s dtype by default) only at the end.
    The extra cost is two (N, 3) x (3, C) matmuls per step."""
    f32 = ttnn.float32
    out_dtype = out_dtype or c.dtype
    r32 = r if r.dtype == f32 else ttnn.typecast(r, f32)
    c32 = c if c.dtype == f32 else ttnn.typecast(c, f32)
    p = matmul_f32(r32, w32, ckc)
    q = ttnn.add(c32, p)
    for y, own in ((p, True), (c32, c32 is not c), (r32, r32 is not r)):
        if own:
            ttnn.deallocate(y)
    if out_dtype == f32:
        return q
    out = ttnn.typecast(q, out_dtype)
    ttnn.deallocate(q)
    return out


def f32_ckc(device):
    """HiFi4 with fp32 accumulation, the config `matmul_f32` needs to be exact."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
