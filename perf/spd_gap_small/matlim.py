"""What holds a Wormhole linear at the token DiT's shapes to ~0.4 of its roof.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/matlim.py --out matlim.jsonl

cen15 (Boltz-2 WH --fast, 1000 MHz): [512, 768] x [768, 1536] reads 53 us plain (75 us with the sigmoid epilogue)
against a HiFi4 math floor of ~21 us and a DRAM floor of ~19 us; the qkv [768, 3072] 107 us, the transition linears
34-44 us. Linears are ~510 us of a ~920 us layer-step. HiFi2 reads the same time (dfid) and bfp8 weights barely move
it, so neither math nor the weight bytes are the limit. Each arm moves one thing:
  place   in0 / in1 / out in DRAM or L1 (interleaved). If L1 operands reach the math floor, the reader's DRAM latency
          is the cost and the remedy is staging weights ahead (prefetch); if not, it is inside the compute pipeline.
  fid     LoFi / HiFi2 / HiFi4, fp32 acc on / off.
  m       M = 512 .. 4096 at the same K, N: time vs M gives the fixed cost a program carries (fill, drain, sync).
  cfg     the 2D multicast programs op_probe enumerates, every in0_block_w, at M 512.
Device time is a captured trace replayed back to back (op_probe.dev_time), so it includes the ~5.3 us a program
costs on this chip (lfl). AICLK of the opened node rides on every row.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import aiclk, dev_time, opened_nodes, programs  # noqa: E402

K = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[1536, 3072, 768])
    a = ap.parse_args()
    dev = T.get_device(trace="protenix")
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=T.arch_name(), t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    mem = {"D": ttnn.DRAM_MEMORY_CONFIG, "L": ttnn.L1_MEMORY_CONFIG}
    fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}
    grid = dev.compute_with_storage_grid_size()
    emit(ev="start", grid=[grid.x, grid.y])

    def ckc(f="hifi4", acc=True):
        return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid[f], math_approx_mode=False,
                                                      fp32_dest_acc_en=acc, packer_l1_acc=True)

    def run(tag, N, M, fn, flops):
        try:
            us, sp, ck = dev_time(dev, fn, nodes=nodes)
        except Exception as e:
            emit(ev="arm", tag=tag, n=N, m=M, error=str(e).splitlines()[0][:240])
            return
        emit(ev="arm", tag=tag, n=N, m=M, us=round(us, 1), spread=round(sp, 3), tflops=round(flops / us / 1e6, 1),
             aiclk=ck)

    torch.manual_seed(0)
    for N in a.n:
        W = torch.randn(K, N) / K ** 0.5
        for M in (512, 1024, 2048, 4096):
            A = torch.randn(1, M, K)
            fl = 2 * M * K * N
            up = lambda t, mc, dt=ttnn.bfloat16: ttnn.from_torch(  # noqa: E731
                t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            ta = {p: up(A, mem[p]) for p in "DL"}
            tw = {p: up(W, mem[p]) for p in "DL"}
            tw8 = up(W, mem["D"], ttnn.bfloat8_b)
            if M == 512:
                for pa in "DL":
                    for pw in "DL":
                        for po in "DL":
                            run(f"place_a{pa}_w{pw}_o{po}", N, M, lambda: ttnn.linear(
                                ta[pa], tw[pw], compute_kernel_config=ckc(), memory_config=mem[po]), fl)
                for f in fid:
                    for acc in (False, True):
                        run(f"fid_{f}_acc{int(acc)}", N, M,
                            lambda: ttnn.linear(ta["D"], tw["D"], compute_kernel_config=ckc(f, acc)), fl)
                        run(f"fid_{f}_acc{int(acc)}_w8", N, M,
                            lambda: ttnn.linear(ta["D"], tw8, compute_kernel_config=ckc(f, acc)), fl)
                for acc in (False, True):
                    for lab, pc in programs(M // 32, K // 32, N // 32, (grid.x, grid.y), acc, (1, 2, 3, 4, 6, 8)):
                        run(f"cfg_{lab}_acc{int(acc)}", N, M, lambda: ttnn.linear(
                            ta["D"], tw["D"], compute_kernel_config=ckc("hifi4", acc), program_config=pc), fl)
            run("m_hifi4_acc1", N, M, lambda: ttnn.linear(ta["D"], tw["D"], compute_kernel_config=ckc()), fl)
            run("m_lofi_acc0_LL", N, M, lambda: ttnn.linear(
                ta["L"], tw["L"], compute_kernel_config=ckc("lofi", False), memory_config=mem["L"]), fl)
            for t in (*ta.values(), *tw.values(), tw8):
                ttnn.deallocate(t)
    emit(ev="end")


if __name__ == "__main__":
    main()
