#!/usr/bin/env python3
"""Direct NOC transaction counters for the trimul channel move's DRAM reads.

The question this row exists for is why trimul moves bytes at 62 % of the combined DRAM roof when
the volume is there. A roofline cannot answer it: it reports that a transaction was inefficient,
never why. This reads the counters the NIU keeps -- `NIU_MST_RD_REQ_SENT`,
`NIU_MST_RD_DATA_WORD_RECEIVED`, `NIU_MST_RD_RESP_RECEIVED` -- out of every core that ran, so
transaction count and transaction size are measured rather than derived from bytes over a wall.

The kernel replays `reader_reblock_permute.cpp`'s page walk with no compute and no writer behind
it, so the read side is isolated. Arms move one thing each: barrier batching, the page stride
(which is what decides the DRAM bank), and the loop nest order.

Read arms only. Nothing here is a correctness claim: the bank arms deliberately read the wrong
pages, in bounds, to time a bank distribution the shipped kernel cannot reach.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from pathlib import Path

import torch
import ttnn

KERNEL = Path(__file__).resolve().parent / "kernels" / "reader_noc_probe.cpp"
TILE_BYTES = 2048
SLOTS = 8
CARDTEL = Path("/home/ttuser/qbcard/cardtel.tsv")
# Blackhole NOC payload is 512 bits, so a counted "data word" is 64 bytes, not 32.
# noc_parameters.h: NOC_DATA_WIDTH (512 + 3), NOC_WORD_BYTES = NOC_PAYLOAD_WIDTH / 8.
NOC_WORD_BYTES = 64


class Aiclk(threading.Thread):
    """20 Hz AICLK sampler. A Blackhole perf number without a clock sampled DURING the run is not
    a measurement, and the 2 Hz box telemetry logger misses a 0.4 ms arm entirely."""

    def __init__(self, card):
        super().__init__(daemon=True)
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(0.05):
            try:
                self.samples.append(int(self.path.read_text().strip()))
            except Exception:
                return

    def report(self):
        v = list(self.samples)
        if not v:
            return None
        return {"n": len(v), "min": min(v), "max": max(v),
                "median": int(statistics.median(v)), "mean": round(sum(v) / len(v), 1)}


def aiclk_window(t0, t1, card):
    """AICLK samples from the always-on 2 Hz card telemetry logger, inside [t0, t1]."""
    if not CARDTEL.is_file():
        return None
    col = f"c{card}_tt_aiclk"
    try:
        with CARDTEL.open() as fh:
            head = None
            vals = []
            for line in fh:
                if line.startswith("#"):
                    # `#BOOT ...` lines are interleaved with the header; only the real header
                    # carries the column names, and taking the last `#` line loses them.
                    if line.startswith("#epoch"):
                        head = line[1:].rstrip("\n").split("\t")
                    continue
                if head is None:
                    continue
                f = line.rstrip("\n").split("\t")
                ts = float(f[0])
                if t0 <= ts <= t1:
                    vals.append(int(f[head.index(col)]))
    except Exception as exc:  # telemetry is a convenience, never a blocker
        return {"error": repr(exc)}
    if not vals:
        return None
    return {"n": len(vals), "min": min(vals), "max": max(vals),
            "median": int(statistics.median(vals))}


def build(device, x, rep, groups, args_per_core):
    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                  ttnn.CoreCoord(g.x - 1, g.y - 1))])
    fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
                                  page_size=TILE_BYTES)
    cbs = [ttnn.CBDescriptor(total_size=(SLOTS + 1) * TILE_BYTES, core_ranges=core_grid,
                             format_descriptors=[fmt])]
    ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(rep).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    i = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rt[cx][cy] = args_per_core[i]
            i += 1
    k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=ct, runtime_args=rt,
        common_runtime_args=[0, 0], config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs), g.x * g.y


def split(num_groups, ncores):
    q, r = divmod(num_groups, ncores)
    out, first = [], 0
    for i in range(ncores):
        n = q + (1 if i < r else 0)
        out.append((first, n))
        first += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--out", default="noc_probe.json")
    a = ap.parse_args()

    N, C = a.n, a.c
    Nt, Ct = (N + 31) // 32, C // 32
    row_stride = Nt * Ct
    # Leave the last row-group out so a bank arm's extra stride cannot walk past the tensor.
    num_groups = (Nt - 1) * Nt

    from tt_bio import tenstorrent as tt_dev
    device = tt_dev.get_device()
    try:
        info = {"grid": None, "dram_banks": None, "l1_banks": None}
        g = device.compute_with_storage_grid_size()
        info["grid"] = [g.x, g.y]
        for name, bt in (("dram_banks", ttnn.BufferType.DRAM), ("l1_banks", ttnn.BufferType.L1)):
            for meth in ("num_banks",):
                try:
                    info[name] = getattr(device, meth)(bt)
                except Exception:
                    pass
        try:
            info["dram_grid"] = [device.dram_grid_size().x, device.dram_grid_size().y]
        except Exception:
            pass

        t = torch.randn(1, N, N, C, dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ncores = g.x * g.y
        rep = ttnn.from_torch(torch.zeros(1, 1, ncores, 8, dtype=torch.int32),
                              dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        parts = split(num_groups, ncores)

        arms = []
        for batch in (1, 2, 4, 8):
            arms.append((f"batch{batch}", dict(batch=batch, bank_step=0, ct_inner=0)))
        for step in (1, 2, 4):
            arms.append((f"bankstep{step}", dict(batch=1, bank_step=step, ct_inner=0)))
        arms.append(("ct_inner", dict(batch=1, bank_step=0, ct_inner=1)))
        arms.append(("ct_inner_b8", dict(batch=8, bank_step=0, ct_inner=1)))
        arms.append(("batch1_again", dict(batch=1, bank_step=0, ct_inner=0)))

        results = {}
        clk = Aiclk(a.card)
        clk.start()
        t_lo = time.time()
        for name, cfg in arms:
            for sample in (0, 1):
                per_core = [[f, n, Nt, Ct, 32, cfg["batch"], cfg["bank_step"], cfg["ct_inner"],
                             sample, i] for i, (f, n) in enumerate(parts)]
                pd, _ = build(device, x, rep, num_groups, per_core)
                pd.kernels[0].common_runtime_args = [x.buffer_address(), rep.buffer_address()]
                for _ in range(3):
                    ttnn.generic_op([x, rep], pd)
                ttnn.synchronize_device(device)
                ts = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, rep], pd)
                    ttnn.synchronize_device(device)
                    ts.append(time.perf_counter() - t0)
                r = ttnn.to_torch(rep).reshape(ncores, 8).to(torch.int64)
                key = name if sample == 0 else name + "_sampled"
                req = int(r[:, 0].sum())
                word = int(r[:, 1].sum())
                resp = int(r[:, 2].sum())
                nread = int(r[:, 3].sum())
                results[key] = {
                    "ms": round(statistics.median(ts) * 1e3, 4),
                    "ms_min": round(min(ts) * 1e3, 4),
                    "rd_req_sent": req,
                    "rd_data_word_received": word,
                    "rd_resp_received": resp,
                    "issued_read_pages": nread,
                    "bytes_per_req": round(word * NOC_WORD_BYTES / req, 3) if req else None,
                    "reqs_per_issued_read": round(req / nread, 4) if nread else None,
                    "max_outstanding": int(r[:, 4].max()),
                    "bytes": nread * TILE_BYTES,
                }
                results[key]["gbs"] = round(
                    nread * TILE_BYTES / (results[key]["ms"] * 1e-3) / 1e9, 2)
        t_hi = time.time()

        # Reference block, same process, same card, same session: what the SHIPPED op and a plain
        # copy of the same bytes do, so the probe's read-side isolation has something to sit beside.
        from tt_bio import reblock_permute as rbp
        ref = {}

        def timeit(fn, n=7):
            for _ in range(3):
                fn()
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                r = fn()
                ttnn.synchronize_device(device)
                ts.append(time.perf_counter() - t0)
                ttnn.deallocate(r)
            return round(statistics.median(ts) * 1e3, 4)

        for cw in (256, 32):
            tt_in = ttnn.from_torch(torch.randn(1, N, N, cw, dtype=torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            nb = N * N * cw * 2
            got = rbp.reblock_permute(tt_in, ttnn.DRAM_MEMORY_CONFIG)
            want = ttnn.permute(tt_in, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            exact = bool(torch.equal(ttnn.to_torch(got), ttnn.to_torch(want)))
            ttnn.deallocate(got)
            ttnn.deallocate(want)
            m_rbp = timeit(lambda: rbp.reblock_permute(tt_in, ttnn.DRAM_MEMORY_CONFIG))
            m_perm = timeit(lambda: ttnn.permute(tt_in, (0, 3, 1, 2),
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG))
            m_clone = timeit(lambda: ttnn.clone(tt_in, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ref[f"C{cw}"] = {
                "bytes_each_way": nb, "Ct": cw // 32, "row_stride": ((N + 31) // 32) * (cw // 32),
                "reblock_permute_ms": m_rbp,
                "reblock_permute_gbs_each_way": round(nb / (m_rbp * 1e-3) / 1e9, 2),
                "ttnn_permute_ms": m_perm,
                "ttnn_permute_gbs_each_way": round(nb / (m_perm * 1e-3) / 1e9, 2),
                "clone_ms": m_clone,
                "clone_gbs_each_way": round(nb / (m_clone * 1e-3) / 1e9, 2),
                "reblock_bit_exact_vs_ttnn_permute": exact,
            }
            ttnn.deallocate(tt_in)
        t_hi2 = time.time()
        clk.stop.set()
        clk.join(timeout=1.0)

        out = {
            "shape": [1, N, N, C], "Nt": Nt, "Ct": Ct, "row_stride": row_stride,
            "num_groups": num_groups, "ncores": ncores, "device": info,
            "aiclk_during_sysfs_20hz": clk.report(),
            "aiclk_during_boxtel_2hz": aiclk_window(t_lo, t_hi2, a.card),
            "noc_word_bytes": NOC_WORD_BYTES,
            "window": [t_lo, t_hi, t_hi2],
            "arms": results,
            "reference_ops": ref,
        }
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    finally:
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
