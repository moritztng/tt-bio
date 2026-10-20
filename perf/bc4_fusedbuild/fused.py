"""Host side of the fused AF2 trimul forward (tt_bio/kernels/trimul_fused), bc4-bet-fusedbuild.

Build step 1: phase 1 alone, `phase1(z, w, mask)` -> (A, B) channel-major [C, S, S] in DRAM, a = mask * (LN(z) @ Wp_a
+ bp_a) * sigmoid(LN(z) @ Wg_a + bg_a) and b likewise without the mask: what the contraction consumes.
"""
import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
KDIR = Path(__file__).resolve().parents[2] / "tt_bio" / "kernels" / "trimul_fused"
_CACHE = {}


def _cb(idx, grid, tiles, dtype=ttnn.bfloat16, page=None):
    tb = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page or tb)
    return ttnn.CBDescriptor(total_size=tiles * tb, core_ranges=grid, format_descriptors=[fmt])


def _dram(t, dev):
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def weights(w, dev, C=128):
    """Device tensors of the module's weights, `w` in af2_reference names."""
    pw, gw, pb, gb = w["p_in.weight"], w["g_in.weight"], w["p_in.bias"], w["g_in.bias"]
    wt = torch.cat([pw[:C], gw[:C], pw[C:], gw[C:]])                    # [4C, C]: rows = out channel
    bias = torch.cat([pb[:C], gb[:C], pb[C:], gb[C:]])
    rows = lambda v: v.float()[None, :].expand(TILE, -1)                  # noqa: E731
    return {"wt": _dram(wt.float(), dev),
            "bias": _dram(bias.float()[:, None].expand(-1, TILE), dev),     # tile t rows c = bias[32 t + c]
            "ln_w": _dram(rows(w["norm_in.weight"]), dev), "ln_b": _dram(rows(w["norm_in.bias"]), dev)}


def _grid(dev):
    g = dev.compute_with_storage_grid_size()
    return g.x, g.y


def phase1(Z, W, MK, S, fid, sig=1, abl=0, out=None):
    dev = Z.device()
    C = 128
    St, Kt = S // TILE, C // TILE
    gx, gy = _grid(dev)
    units = St * St * 2
    assert units <= gx * gy, (units, gx, gy)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    if out is None:
        out = [ttnn.allocate_tensor_on_device(ttnn.Shape([C, S, S]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                              ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
    key = ("p1", S, sig, abl, MK is not None, str(fid))
    e = _CACHE.get(key)
    if e is None:
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in range(gx * gy):
            n = 1 if c < units else 0
            for ra in (rd, wr, cp):
                ra[c % gx][c // gx] = [c if n else 0, n]
        inv = struct.unpack("<I", struct.pack("<f", 1.0 / C))[0] >> 16
        eps = struct.unpack("<I", struct.pack("<f", 1e-5))[0]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        mk = MK if MK is not None else Z
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "reader.cpp"), source_type=src, core_ranges=grid,
            compile_time_args=[Kt, St, int(MK is not None), (inv << 16) | inv, eps] + acc(Z) + acc(W["wt"])
            + acc(W["bias"]) + acc(W["ln_w"]) + acc(W["ln_b"]) + acc(mk),
            runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "writer_p1.cpp"), source_type=src, core_ranges=grid,
            compile_time_args=[St] + acc(out[0]) + acc(out[1]),
            runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute.cpp"), source_type=src, core_ranges=grid,
            compile_time_args=[Kt, int(MK is not None), sig, abl], runtime_args=cp,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                                dst_full_sync_en=False))
        f32 = ttnn.float32
        cbs = [_cb(0, grid, 16 * Kt), _cb(1, grid, 2 * Kt), _cb(2, grid, 16 * Kt), _cb(3, grid, 16),
               _cb(4, grid, 16), _cb(5, grid, 32), _cb(6, grid, 8), _cb(7, grid, 1),
               _cb(8, grid, Kt), _cb(9, grid, Kt), _cb(10, grid, 1), _cb(11, grid, 1, f32), _cb(12, grid, 1, f32),
               _cb(13, grid, 1, f32), _cb(14, grid, 1, f32), _cb(16, grid, Kt, f32), _cb(17, grid, Kt, f32)]
        e = _CACHE[key] = {"kernels": [reader, writer, compute], "cbs": cbs}
    reader, writer, _ = e["kernels"]
    mk = MK if MK is not None else Z
    reader.common_runtime_args = [Z.buffer_address(), W["wt"].buffer_address(), W["bias"].buffer_address(),
                                  W["ln_w"].buffer_address(), W["ln_b"].buffer_address(), mk.buffer_address()]
    writer.common_runtime_args = [out[0].buffer_address(), out[1].buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=[], cbs=e["cbs"])
    ttnn.generic_op([Z, W["wt"], W["bias"], W["ln_w"], W["ln_b"], mk] + out, pd)
    return out
