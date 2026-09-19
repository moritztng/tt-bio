"""Mirror of ttnn's own matmul program-config derivation for an interleaved input + core_grid.

Transcribed from tt-metal `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`
(`create_matmul_program_config`, `create_matmul_1d_systolic_array_program_config`,
`get_multi_dim_per_core_factor`, `get_subblock_sizes`). It exists so a ladder arm can differ from
ttnn's own choice in exactly ONE parameter: without the mirror, an explicit program config is a
second change and the measurement is of two things at once.

The load-bearing line of that file, and the reason this row exists:

    if (size < max_l1_space) { return {per_core_M, per_core_N, in0_block_w}; }

`get_multi_dim_per_core_factor` returns the block equal to the whole per-core output whenever the
circular buffers fit. So `out_block_h = per_core_M` is not a tuned choice, it is the absence of
one: the block is only ever cut down when L1 refuses it. Nothing in the derivation considers the
DRAM write the block schedule is supposed to overlap.

`out_block_h`/`out_block_w` here assume the CBs fit (the common case). The identity control on
device -- an explicit mirror against ttnn's own choice -- is what verifies that assumption per
shape, rather than re-implementing `get_estimated_size_of_cbs`.
"""
from __future__ import annotations

SUBBLOCK_HW_CHOICES = [(4, 2), (2, 4), (8, 1), (1, 8), (7, 1), (1, 7), (3, 2), (2, 3), (6, 1),
                       (1, 6), (5, 1), (1, 5), (2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1),
                       (1, 2), (1, 1)]


def subblocks(obh: int, obw: int, fp32_dest_acc: bool) -> tuple[int, int]:
    """get_subblock_sizes: the tuples are (w, h), and fp32 dest caps h*w at 4."""
    for w, h in SUBBLOCK_HW_CHOICES:
        if (h * w) <= 4 or not fp32_dest_acc:
            if obh % h == 0 and obw % w == 0:
                return h, w
    raise ValueError(f"no subblock for {obh}x{obw}")


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def derive(mt: int, kt: int, nt: int, gx: int, gy: int, fp32_dest_acc: bool,
           m_size: int, k_size: int, n_size: int) -> dict:
    """ttnn's derived config for a non-sharded in0 at `core_grid=(gx, gy)`.

    `mt` is batch*M in tiles (fuse_batch is always on for these paths)."""
    cores = gx * gy
    height, width = mt * 32, n_size
    ratio = max(height, width) // min(height, width)
    within_tile = k_size <= 32 or m_size <= 32 or n_size <= 32
    if ratio > 8 or within_tile:
        tall = mt > nt
        if tall:
            pcm, ibw, pcn = -(-mt // cores), -(-kt // cores), nt
        else:
            pcm, ibw, pcn = mt, -(-kt // cores), -(-nt // cores)
        while kt % ibw:
            ibw -= 1
        obh, obw = pcm, pcn
        sh, sw = subblocks(obh, obw, fp32_dest_acc)
        return {"cls": "1d", "per_core_M": pcm, "per_core_N": pcn, "in0_block_w": ibw,
                "out_block_h": obh, "out_block_w": obw, "out_subblock_h": sh,
                "out_subblock_w": sw, "mcast_in0": not tall, "grid": (gx, gy)}
    pcm, pcn = -(-mt // gy), -(-nt // gx)
    ibw = 4
    while kt % ibw:
        ibw -= 1
    pcn = max(pcn, 1)
    obh, obw = pcm, pcn
    sh, sw = subblocks(obh, obw, fp32_dest_acc)
    if sw != pcn:
        sh = 1
    return {"cls": "2d", "per_core_M": pcm, "per_core_N": pcn, "in0_block_w": ibw,
            "out_block_h": obh, "out_block_w": obw, "out_subblock_h": sh, "out_subblock_w": sw,
            "mcast_in0": False, "grid": (gx, gy)}


def build(ttnn, c: dict, out_block_h: int | None = None, in0_block_w: int | None = None):
    """The config `c` with at most one parameter moved, subblocks recomputed ttnn's own way."""
    obh = c["out_block_h"] if out_block_h is None else out_block_h
    ibw = c["in0_block_w"] if in0_block_w is None else in0_block_w
    obw = c["out_block_w"]
    sh, sw = subblocks(obh, obw, c["fp32_dest_acc"])
    if c["cls"] == "2d" and sw != c["per_core_N"]:
        sh = 1
    gx, gy = c["grid"]
    if c["cls"] == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh,
            out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=c["per_core_M"],
            per_core_N=c["per_core_N"], fuse_batch=True, fused_activation=None,
            mcast_in0=c["mcast_in0"])
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=c["per_core_M"],
        per_core_N=c["per_core_N"], transpose_mcast=False, fused_activation=None)
