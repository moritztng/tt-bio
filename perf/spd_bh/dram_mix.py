"""Blackhole DRAM read/write NoC stall: tt-metal#59622's reproducer, run through tt-bio's runtime root.

110 cores read 16 KiB DRAM rows and write 1088 B pieces to DRAM, all on NoC 1, program after program. With the stock
``noc_async_read`` this hung p300 chips within 35-526 runs upstream; with the 2 KiB split (tt_bio.metal_overlay) it ran
3.5 million runs clean. A stall cannot be recovered from Python: a watchdog ends the process with status 3 when no run
finishes in 10 s, and the chip then needs a reset.

    TT_VISIBLE_DEVICES=<card> python perf/spd_bh/dram_mix.py [--stock] [--seconds 20]

``--stock`` turns the split off (TT_BIO_BH_DRAM_READ_SPLIT=0) to show the stall. Prints one RESULT line.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stock", action="store_true")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--read-bytes", type=int, default=16384)
a = ap.parse_args()
if a.stock:
    os.environ["TT_BIO_BH_DRAM_READ_SPLIT"] = "0"

import tt_bio  # noqa: E402,F401  (sets the runtime root before ttnn loads)
import torch  # noqa: E402
import ttnn  # noqa: E402

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "dram_read_write_mix.cpp")
COLS, ROWS = 11, 10
TABLE_ROWS, ROW_WORDS = 32, 4096
TASKS, PIECE_WORDS, SEED, NOC = 8192, 272, 52270, 1
root = os.environ.get("TT_METAL_RUNTIME_ROOT", "stock")
state = {"runs": 0, "last": time.monotonic(), "limit": 600.0}


def result(status, **kw):
    print("RESULT", f"status={status}", f"split={not a.stock}", f"runs={state['runs']}",
          *(f"{k}={v}" for k, v in kw.items()), f"root={root}", flush=True)


def watch():
    while True:
        time.sleep(1.0)
        idle = time.monotonic() - state["last"]
        if idle > state["limit"]:
            result("STALL", idle_s=round(idle))
            os._exit(3)


device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
if grid.x < COLS or grid.y < ROWS:
    result("SKIP", grid=f"{grid.x}x{grid.y}")
    sys.exit(0)
cores = [ttnn.CoreCoord(w % COLS, w // COLS) for w in range(COLS * ROWS)]
core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(COLS - 1, ROWS - 1))])
table_host = (torch.arange(TABLE_ROWS * ROW_WORDS, dtype=torch.int32) % TASKS).reshape(TABLE_ROWS, ROW_WORDS)
table = ttnn.from_torch(table_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
cache = ttnn.empty((TASKS * 32, PIECE_WORDS), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
rt = ttnn.RuntimeArgs()
for w, core in enumerate(cores):
    count, extra = divmod(TASKS, len(cores))
    first, n = w * count + min(w, extra), count + int(w < extra)
    rt[core.x][core.y] = [table.buffer_address(), cache.buffer_address(), first, n, SEED]
cta = ([NOC, a.read_bytes] + ttnn.TensorAccessorArgs(table).get_compile_time_args()
       + ttnn.TensorAccessorArgs(cache).get_compile_time_args())
kernel = ttnn.KernelDescriptor(
    kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_range,
    compile_time_args=cta, runtime_args=rt,
    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1))
cbs = [ttnn.CBDescriptor(total_size=s, core_ranges=core_range,
                         format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=s)])
       for i, s in ((0, ROW_WORDS * 4), (1, 4 * PIECE_WORDS * 4))]
program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

threading.Thread(target=watch, daemon=True).start()
ttnn.generic_op([table, cache], program)
ttnn.synchronize_device(device)
state.update(runs=1, last=time.monotonic(), limit=10.0)
start = time.monotonic()
while time.monotonic() - start < a.seconds:
    ttnn.generic_op([table, cache], program)
    ttnn.synchronize_device(device)
    state.update(runs=state["runs"] + 1, last=time.monotonic())
secs = time.monotonic() - start
expected = torch.tensor([(SEED * 65537 + i * 37 + 11) & 0x7FFFFFFF for i in range(PIECE_WORDS)], dtype=torch.int32)
ok = all(torch.equal(ttnn.to_torch(ttnn.slice(cache, [p, 0], [p + 1024, PIECE_WORDS])).to(torch.int32),
                     expected.expand(1024, PIECE_WORDS)) for p in (0, 4096 * 32))
result("CLEAN" if ok else "WRONG", seconds=round(secs, 1), us_per_run=round(1e6 * secs / max(state["runs"] - 1, 1), 1))
ttnn.close_device(device)
