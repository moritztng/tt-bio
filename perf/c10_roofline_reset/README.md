# Roofline counter control

The existing byte counter fails the exact dense-matmul control. Do not use it to publish a new roofline until its compulsory-byte accounting is corrected and calibrated.

The capture used qb2 physical card 0, tt-metal source `1452925b033c6608726b731a81500bd3e19f7894`, and tt-bio `71a306a8a5d94d880e85ce6829218ad671fbc246`. `FORCE_AICLK` requested 1350 MHz; all nine sysfs samples taken inside the capture read 1350 MHz. This is a counter validation, with no device-time, throughput, or fold-performance claim.

| 8192 × 8192 × 8192 bf16 matmul | Exact control | Counter |
|---|---:|---:|
| FLOPs, counting each multiply-add as two | 1,099,511,627,776 | 1,099,511,627,776 |
| Minimum bytes, two input reads and one output write | 402,653,184 | 536,870,912 |

Each operand is 134,217,728 bytes. `real_traffic.counts()` charges the terminal output both a write and a read because it has no recorded consumer. That extra read causes the 33.33% error. The minimum-byte control says nothing about physical rereads inside a tiled matmul; this graph counter does not measure those.

The failure is reproduced without modifying the existing counters. The program exits **2** for `INSTRUMENT_UNFIT`; it must exit **0** before a fold capture proceeds. The failure triggers the requested calibration stop, so no per-op roofline or revised fold floor accompanies this result.

Evidence: [result](control/control.json), [raw graph](control/matmul_graph.json), [clock samples](control/clock.jsonl), and [run log](control.log). Clock reads run in a separate process because a Python thread can miss an entire device call while the extension holds the GIL. Only reads whose start and end both fall within the synchronized capture are counted.

Reproduce from this worktree on qb2, with card 0 granted and free:

```bash
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=/home/ttuser/tt-metal-b2z/ttnn:/home/ttuser/tt-metal-b2z
export TT_METAL_HOME=/home/ttuser/tt-metal-b2z
export TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal-b2z
export LD_LIBRARY_PATH=/home/ttuser/tt-metal-b2z/build_Release/lib
export OMP_NUM_THREADS=2
export TT_BIO_AICLK=1350
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
TT_BIO_LEASE_HOLDER=worker:c10-roofline-reset \
python3 perf/c10_roofline_reset/control.py \
  --out perf/c10_roofline_reset/control
```

The harness loads the clock holder verbatim from `5c60137c2:tt_bio/aiclk.py` through `git show`; it does not install or change model code. It releases the pin and closes the device on normal completion. The source build is reused without rebuilding. The counters are imported from `perf/roof_budget/exec_flops.py` and `perf/b2x_difflayer/real_traffic.py`.
