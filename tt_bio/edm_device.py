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
