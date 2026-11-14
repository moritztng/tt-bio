"""Device time and float64 error of the atom transformer's linears (Boltz-2 diffusion, AtomAttention windowed path).

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/atomlin.py --out atomlin.jsonl

cen10 (Boltz-2 --fast, Wormhole, 1000 MHz, 512 tokens, 140 atom windows): the kv projection
[1, 140, 128, 128] x [128, 256] reads 248.7 us a call at 0.22 of roof (4.7 TFLOP/s) on 1200 calls a fold
(0.30 s), against ~55 us of DRAM traffic. q [1, 140, 32, 128] x [128, 128] + bias and the two gate linears
read 40-42 us against ~9 us. The model passes them 4D with core_grid=CORE_GRID_MAIN. The arms ask whether
the same matmul folded to 2D, or given an explicit 1D program (each core its own rows of the activation,
the whole weight), runs at its roof. Same K order, so the output should be bit-identical to the model's.

Device time is a captured trace replayed back to back (op_probe.dev_time), median of 7 batches, AICLK of
the opened node. Error is against float64 with op_probe's wrong-pixel rule; `same` = bit-equal to `model`.
"""
import argparse, json, math, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import aiclk, dev_time, opened_nodes, wrong  # noqa: E402


def pc1d(m_t, k_t, n_t, grid, sbw, kb=None):
    """1D program, activation rows split over the cores, weight multicast to all of them."""
    gx, gy = grid
    pm = math.ceil(m_t / (gx * gy))
    sw = max(w for w in range(1, min(sbw, n_t) + 1) if n_t % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=kb or k_t, out_subblock_h=1,
        out_subblock_w=sw, per_core_M=pm, per_core_N=n_t, fuse_batch=True, fused_activation=None,
        mcast_in0=False)


def bench(dev, nodes, emit, name, shp, N, bias, seed=7, diag=False):
    torch.manual_seed(seed)
    K = shp[-1]
    A = torch.randn(*shp); W = torch.randn(K, N) / K ** 0.5; Bv = torch.randn(N) * 0.5 if bias else None
    R = A.double() @ W.double()
    if bias:
        R = R + Bv.double()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    ta, tw = up(A), up(W)
    tb = up(Bv.reshape(1, N)) if bias else None
    M = math.prod(shp[:-1])
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    flat = lambda: ttnn.reshape(ta, (1, 1, M, K))  # noqa: E731
    grid = T.COMPUTE_GRID_MAIN
    arms = [
        # as AtomAttention calls it
        ("model", lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                      dtype=ttnn.bfloat16)),
        ("model_nogrid", lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc, dtype=ttnn.bfloat16)),
        ("flat", lambda: ttnn.linear(flat(), tw, bias=tb, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                     dtype=ttnn.bfloat16)),
        ("flat_nogrid", lambda: ttnn.linear(flat(), tw, bias=tb, compute_kernel_config=ckc, dtype=ttnn.bfloat16)),
    ]
    for sbw in (4, 2):
        pc = pc1d(M // 32, K // 32, N // 32, grid, sbw)
        arms.append((f"pc1d_sb{sbw}", lambda pc=pc: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc,
                                                                 program_config=pc, dtype=ttnn.bfloat16)))
        arms.append((f"flat_pc1d_sb{sbw}", lambda pc=pc: ttnn.linear(flat(), tw, bias=tb, compute_kernel_config=ckc,
                                                                      program_config=pc, dtype=ttnn.bfloat16)))
    if diag:
        # What the call waits on: math (fidelity), the fp32 DEST pack, K blocking (read/compute overlap), DRAM writes
        # or reads (L1 output / L1 input), and the fixed cost of one launch (M cut to one row tile a core).
        def ck(fid=ttnn.MathFidelity.HiFi4, acc=True):
            return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                          fp32_dest_acc_en=acc, packer_l1_acc=acc)
        call = lambda x=ta, c=ckc, **kw: ttnn.linear(x, tw, bias=tb, compute_kernel_config=c,  # noqa: E731
                                                    core_grid=T.CORE_GRID_MAIN, dtype=ttnn.bfloat16, **kw)
        l1 = ttnn.L1_MEMORY_CONFIG
        ta_l1 = ttnn.to_memory_config(ta, l1)
        small = ttnn.from_torch(A.reshape(-1, K)[:32 * grid[0] * grid[1]].reshape(1, 1, -1, K), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)
        arms += [
            ("d_lofi", lambda: call(c=ck(ttnn.MathFidelity.LoFi))),
            ("d_hifi2", lambda: call(c=ck(ttnn.MathFidelity.HiFi2))),
            ("d_acc0", lambda: call(c=ck(acc=False))),
            ("d_lofi_acc0", lambda: call(c=ck(ttnn.MathFidelity.LoFi, acc=False))),
            ("d_out_l1", lambda: call(memory_config=l1)),
            ("d_in_l1", lambda: call(x=ta_l1)),
            ("d_in_out_l1", lambda: call(x=ta_l1, memory_config=l1)),
            ("d_one_row_tile", lambda: call(x=small)),
        ]
        for kb in (1, 2):
            if K // 32 % kb == 0 and kb < K // 32:
                pc = pc1d(M // 32, K // 32, N // 32, grid, 4, kb)
                arms.append((f"d_pc1d_kb{kb}", lambda pc=pc: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc,
                                                                          program_config=pc, dtype=ttnn.bfloat16)))
    base = None
    for lab, fn in arms:
        try:
            y = fn()
            Yt = ttnn.to_torch(y)
            ttnn.deallocate(y)
        except Exception as e:
            emit(ev="arm", op=name, arm=lab, error=str(e).splitlines()[0][:240])
            continue
        if lab == "d_one_row_tile":
            us, sp, ck_ = dev_time(dev, fn, nodes=nodes)
            emit(ev="arm", op=name, arm=lab, us=round(us, 1), spread=round(sp, 3), aiclk=ck_)
            continue
        Yt = Yt.reshape(R.shape)
        if lab == "model":
            base = Yt
        nb, mx, rel = wrong(R, Yt.double())
        same = bool(base is not None and torch.equal(base, Yt))
        us, sp, ck_ = dev_time(dev, fn, nodes=nodes)
        emit(ev="arm", op=name, arm=lab, us=round(us, 1), spread=round(sp, 3), wrong=nb, max_err=round(mx, 5),
             rel_rms=float(f"{rel:.3e}"), same=same, aiclk=ck_)
    for t in (ta, tw, tb):
        if t is not None:
            ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=140)
    ap.add_argument("--diag", action="store_true", help="also the arms that say what the call waits on")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = T.get_device()
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=T.arch_name(), t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    w, d = a.windows, a.diag
    bench(dev, nodes, emit, "kv", (1, w, 128, 128), 256, False, diag=d)
    bench(dev, nodes, emit, "q", (1, w, 32, 128), 128, True, diag=d)
    bench(dev, nodes, emit, "gate", (1, w, 32, 128), 128, False)
    # the atom transition's fc1 / fc2 (63 / 56 us a call in cen10, 0.33 / 0.62 of the kernel in the math thread)
    bench(dev, nodes, emit, "tr_up", (1, w, 32, 128), 256, False, diag=d)
    bench(dev, nodes, emit, "tr_down", (1, w, 32, 256), 128, False, diag=d)
    emit(ev="end")


if __name__ == "__main__":
    main()
