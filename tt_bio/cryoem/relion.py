"""The kernels RELION calls, on the RELION side of the bridge.

RELION hands these functions raw memoryviews over its own buffers, so nothing is copied on the way
in. Every entry point returns True when it handled the call and False to decline it, in which case
RELION runs its own kernel: an unsupported shape, a missing device or a disagreeing environment must
degrade to the CPU path, never fail the refinement.

The numerics are RELION's, transcribed from src/acc/cpu/cpu_kernels/diff2.h, cpu_utils.h and
src/acc/acc_projectorkernel_impl.h, for the 3D-reference / 2D-data case that 3D auto-refine runs.
One deliberate difference: RELION accumulates per 256-pixel block, this reduces the whole image at
once. That changes the last bits and nothing else, and TT_RELION_CHECK=1 measures the residual
against RELION's own value rather than asserting it.

Backend selected by TT_RELION_BACKEND:
  torch   torch on the host. Exact, slow, and the arm that proves the plumbing.
  ttnn    the device path.
"""
from __future__ import annotations

import os

import numpy as np

_BACKEND = os.environ.get("TT_RELION_BACKEND", "torch")
_ORI_CHUNK = int(os.environ.get("TT_RELION_ORI_CHUNK", "32"))
_CHECK = os.environ.get("TT_RELION_CHECK", "") not in ("", "0")
# TT_RELION_TRACE=<path>: append one line per call with the shape, a hash of the euler set and a
# hash of the reference. Answers "is the coarse orientation set shared across particles" from a
# real run rather than from reading RELION's sampling code.
_TRACE = os.environ.get("TT_RELION_TRACE", "")
# TT_RELION_DUMP=<path>: write the first call's raw inputs to <path>.<pid>.npz and keep going.
# The separable-interpolant study needs RELION's own padded model and its own orientation set at
# the real operating point; reconstructing either from RELION's source is guesswork.
_DUMP = os.environ.get("TT_RELION_DUMP", "")
_DUMP_N = int(os.environ.get("TT_RELION_DUMP_N", "1"))
_dumped = [0]
_stats = {"handled": 0, "declined": 0, "resid_max": 0.0, "resid_n": 0}


def _torch():
    import torch
    torch.set_grad_enabled(False)
    return torch


def _project(t, mdl, x, y, e, mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ,
             maxR2_padded, padding_factor):
    """AccProjectorKernel::project3Dmodel, the 2D-data overload, PROJECTOR_NO_TEXTURES.

    e is [C, 9], x and y are [P]. Returns ref_real, ref_imag as [C, P].
    """
    pf = padding_factor
    xp = (e[:, 0:1] * x + e[:, 1:2] * y) * pf
    yp = (e[:, 3:4] * x + e[:, 4:5] * y) * pf
    zp = (e[:, 6:7] * x + e[:, 7:8] * y) * pf

    # RELION truncates the radius test to int, so 4.9 becomes 4. trunc, not round and not floor.
    inside = (xp * xp + yp * yp + zp * zp).trunc().to(t.int64) <= maxR2_padded

    # Hermitian pair for the negative-x half space.
    invers = xp < 0
    xp = t.where(invers, -xp, xp)
    yp = t.where(invers, -yp, yp)
    zp = t.where(invers, -zp, zp)

    x0 = xp.floor()
    fx = xp - x0
    y0 = yp.floor()
    fy = yp - y0
    z0 = zp.floor()
    fz = zp - z0

    mdlXY = mdlX * mdlY
    base = ((z0.to(t.int64) - mdlInitZ) * mdlXY
            + (y0.to(t.int64) - mdlInitY) * mdlX
            + x0.to(t.int64))
    # Outside the radius RELION never evaluates the interpolation; here the gather is
    # unconditional, so clamp it in bounds and mask the result afterwards.
    nvox = mdlXY * mdlZ
    base = base.clamp(0, nvox - mdlXY - mdlX - 2)

    off = (0, 1, mdlX, mdlX + 1, mdlXY, mdlXY + 1, mdlXY + mdlX, mdlXY + mdlX + 1)
    c = [mdl.index_select(0, (base + o).reshape(-1)).reshape(base.shape + (2,)) for o in off]

    def lerp(a, b, f):
        return a + (b - a) * f.unsqueeze(-1)

    dx00 = lerp(c[0], c[1], fx)
    dx10 = lerp(c[2], c[3], fx)
    dx01 = lerp(c[4], c[5], fx)
    dx11 = lerp(c[6], c[7], fx)
    dxy0 = lerp(dx00, dx10, fy)
    dxy1 = lerp(dx01, dx11, fy)
    ref = lerp(dxy0, dxy1, fz)

    ref_r = ref[..., 0]
    ref_i = t.where(invers, -ref[..., 1], ref[..., 1])
    zero = t.zeros((), dtype=ref_r.dtype)
    return t.where(inside, ref_r, zero), t.where(inside, ref_i, zero)


def diff2_coarse(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, out_mv,
                 mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                 padding_factor, imgX, imgY,
                 orientation_num, translation_num, image_size):
    """runDiff2KernelCoarse, 3D reference and 2D data. Accumulates onto out_mv."""
    if _BACKEND == "ttnn":
        _stats["declined"] += 1
        return False        # not wired yet, so RELION keeps its own kernel
    try:
        t = _torch()
        mdl = t.from_numpy(np.frombuffer(mdl_mv, dtype=np.float32).reshape(-1, 2).copy())
        eul = t.from_numpy(np.frombuffer(eul_mv, dtype=np.float32)
                           .reshape(orientation_num, 9).copy())
        tx = t.from_numpy(np.frombuffer(tx_mv, dtype=np.float32).copy())
        ty = t.from_numpy(np.frombuffer(ty_mv, dtype=np.float32).copy())
        img_r = t.from_numpy(np.frombuffer(re_mv, dtype=np.float32).copy())
        img_i = t.from_numpy(np.frombuffer(im_mv, dtype=np.float32).copy())
        w = t.from_numpy(np.frombuffer(corr_mv, dtype=np.float32).copy()) * 0.5

        pix = t.arange(image_size, dtype=t.int64)
        x = (pix % imgX).to(t.float32)
        yi = pix // imgX
        y = t.where(yi > maxR, yi - imgY, yi).to(t.float32)

        # The shift stack. A phase shift preserves magnitude, so this is the only place the
        # translations enter and it does not depend on the orientation.
        ph = x.unsqueeze(0) * tx.unsqueeze(1) + y.unsqueeze(0) * ty.unsqueeze(1)
        s, c = t.sin(ph), t.cos(ph)
        sh_r = c * img_r - s * img_i
        sh_i = c * img_i + s * img_r

        acc = np.empty(orientation_num * translation_num, dtype=np.float32)
        for o0 in range(0, orientation_num, _ORI_CHUNK):
            o1 = min(o0 + _ORI_CHUNK, orientation_num)
            ref_r, ref_i = _project(t, mdl, x, y, eul[o0:o1], mdlX, mdlY, mdlZ,
                                    mdlInitY, mdlInitZ, maxR2_padded, padding_factor)
            dr = ref_r.unsqueeze(1) - sh_r.unsqueeze(0)
            di = ref_i.unsqueeze(1) - sh_i.unsqueeze(0)
            d2 = ((dr * dr + di * di) * w).sum(-1)
            acc[o0 * translation_num:o1 * translation_num] = d2.reshape(-1).numpy()

        if _DUMP and _dumped[0] < _DUMP_N:
            _dumped[0] += 1
            np.savez("%s.%d.%d.npz" % (_DUMP, os.getpid(), _dumped[0]),
                     mdl=mdl.numpy(), eul=eul.numpy(), tx=tx.numpy(), ty=ty.numpy(),
                     img_r=img_r.numpy(), img_i=img_i.numpy(), w=w.numpy(),
                     diff2=acc,
                     geom=np.array([mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                                    padding_factor, imgX, imgY, orientation_num,
                                    translation_num, image_size], dtype=np.int64))
        if _TRACE:
            import hashlib
            eh = hashlib.sha256(np.frombuffer(eul_mv, dtype=np.float32).tobytes()).hexdigest()[:16]
            mh = hashlib.sha256(np.frombuffer(mdl_mv, dtype=np.float32).tobytes()).hexdigest()[:16]
            with open("%s.%d" % (_TRACE, os.getpid()), "a") as fh:
                fh.write("%d %d %d %s %s\n"
                         % (orientation_num, translation_num, image_size, eh, mh))
        out = np.frombuffer(out_mv, dtype=np.float32)
        if _CHECK:
            _stats["resid_ours"] = acc.copy()
        out += acc
        _stats["handled"] += 1
        return True
    except Exception as exc:                        # declining beats killing the refinement
        import traceback
        traceback.print_exc()
        print("tt_bio.cryoem.relion: diff2_coarse declined: %r" % (exc,))
        _stats["declined"] += 1
        return False


def stats():
    return dict(_stats)
