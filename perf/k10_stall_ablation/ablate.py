#!/usr/bin/env python3
"""K10 Phase 1: what is the compute-side input stall a FUNCTION of?

One variable at a time, every arm interleaved inside one process and one device open, each
against its own A/A floor. A sequential A-then-B op bench is JIT-warmup-biased
(`op-ab-must-interleave-arms-compile-warmup-bias`), so nothing here runs arm A to completion
before arm B starts: the dispatch order is round-robin over every arm in the sweep.

The two counters this reads are tt-metal's shipped sum accumulators, and they are NOT the
math thread's:

    DEVICE COMPUTE CB WAIT FRONT [ns]     TRISC0 (unpack), blocked on INPUT tiles
    DEVICE COMPUTE CB RESERVE BACK [ns]   TRISC2 (pack), blocked on OUTPUT room

Both are summed over the cores that ran the op, so each is divided by that op's OWN
`CORE COUNT` before it is compared to anything. Nothing here reports a "useful math" or
"not stalled" figure: that quantity is TRISC1's own stall and no instrument in this campaign
measures it.

Run it under the profiler::

    . /home/mthuening/work/b2z2-profiler/profenv.sh
    $PY -m tracy -r --no-op-info-cache --enable-sum-profiling \
        -o perf/k10_stall_ablation/prof --op-support-count 30000 -- \
        perf/k10_stall_ablation/ablate.py --out perf/k10_stall_ablation/run.json

and then attribute with `split.py`, which pairs the CSV rows against the dispatch order this
file writes into the json. Bare (no `-m tracy`) it still runs and reports synced host wall,
which is only useful as a smoke test -- the stall numbers need the profiler.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# Mandatory on whglx: tt-bio caps the workers IT spawns, not the siblings a fleet launches
# beside it, and eight uncapped concurrent folds on this box read a 2.32x spread at loadavg 133.
# This is a dispatch rig rather than a fold, but the cap is free and the brief requires it be
# in effect and stated. Applied before torch/ttnn are imported, which is why it is at module
# scope and not inside main().
from tt_bio.runtime import host_thread_cap_env  # noqa: E402

HOST_WORKERS = int(os.environ.get("K10_HOST_WORKERS", "8"))
THREAD_CAP = host_thread_cap_env(HOST_WORKERS)
os.environ.update(THREAD_CAP)

FENCE_N = 3
FENCE_DIM = 32
TILE = 32

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


# --------------------------------------------------------------------------------------------
# CB depth. `mm_generic.build` double-buffers in0/in1/out (`num_tiles = block * 2`). Depth is
# only worth anything where the producer emits MORE THAN ONE block, so the classification comes
# first and the knob second. Scaling is done by wrapping mm_generic's own `_cb` here in perf/,
# not by editing it: `git diff origin/main...HEAD` outside perf/ has to stay empty.
_CB_DEPTH_SCALE = [1.0]


def install_cb_scale(G):
    base = G._cb

    def scaled(idx, core_grid, page_size, num_tiles, data_format):
        s = _CB_DEPTH_SCALE[0]
        # CB 3 is the fp32 accumulation intermediate, single-buffered by the factory and not a
        # pipeline stage -- scaling it would change what is measured, not how deep the pipe is.
        n = num_tiles if idx == 3 else max(2, int(round(num_tiles * s)))
        return base(idx, core_grid, page_size, n, data_format)
    G._cb = scaled


class Rig:
    """Everything that has to exist before a single measured op is dispatched."""

    def __init__(self, ttnn, dev, grid):
        self.ttnn, self.dev, self.grid = ttnn, dev, grid
        self.cache: dict = {}

    def t(self, key, shape, dtype, mem):
        """A cached device tensor. Every arm's operands are allocated in SETUP, never in the
        measured loop, so the dispatch order the loop records is exactly the op order the
        profiler sees."""
        ttnn = self.ttnn
        k = (key, tuple(shape), str(dtype), str(mem))
        if k not in self.cache:
            import torch
            th = torch.randn(*shape) * 0.05
            self.cache[k] = ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT, dtype=dtype,
                                            device=self.dev, memory_config=mem)
        return self.cache[k]

    def out(self, key, shape, dtype, mem):
        ttnn = self.ttnn
        k = ("out", key, tuple(shape), str(dtype), str(mem))
        if k not in self.cache:
            self.cache[k] = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, self.dev, mem)
        return self.cache[k]


def build_arms(ttnn, rig, G, grid, base_m, args):
    """(name, opcode, thunk, knob, level) for every arm. `knob` groups an arm with its control."""
    gx, gy = grid
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    L1 = ttnn.L1_MEMORY_CONFIG
    BF16 = ttnn.bfloat16
    HIFI4 = (ttnn.MathFidelity.HiFi4, False, False, False)
    HIFI2 = (ttnn.MathFidelity.HiFi2, False, False, False)
    LOFI = (ttnn.MathFidelity.LoFi, False, False, False)

    K, N = 128, 512            # boltz2 qkv+gate at c_z=128: kt=4, nt=16
    BLK = (4, 4, 1, 4, 1)      # the shipped _MM_BLOCK[(4, 16)] entry. K_block == kt == full K.
    arms = []

    def gen(name, knob, level, *, m=base_m, kblock=4, in0_mem=DRAM, in1_mem=DRAM,
            out_mem=DRAM, ckc=HIFI4, g=None, cbscale=1.0):
        g = g or grid
        mb, _kb, nb, sh, sw = BLK
        cfg = ((mb, kblock, nb, sh, sw), tuple(g))
        in0 = rig.t(f"in0m{m}", (1, m, 512, K), BF16, in0_mem)
        in1 = rig.t(f"w{N}", (1, 1, K, N), BF16, in1_mem)
        o = rig.out(f"o{m}n{N}", (1, m, 512, N), BF16, out_mem)

        def run():
            _CB_DEPTH_SCALE[0] = cbscale
            try:
                G.generic_minimal_matmul(rig.dev, in0, in1, [o], cfg, ckc)
            finally:
                _CB_DEPTH_SCALE[0] = 1.0
        arms.append((name, "GenericOpDeviceOperation", run, knob, level))

    def mm(name, knob, level, *, m=base_m, in0_mem=DRAM, in1_mem=DRAM, out_mem=DRAM,
           in0_dt=BF16, in1_dt=BF16, fid=ttnn.MathFidelity.HiFi4, g=None, n=128):
        g = g or grid
        in0 = rig.t(f"in0m{m}", (1, m, 512, K), in0_dt, in0_mem)
        in1 = rig.t(f"w{n}", (1, 1, K, n), in1_dt, in1_mem)
        ckcfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False)
        cg = ttnn.CoreGrid(y=g[1], x=g[0])

        def run():
            r = ttnn.matmul(in0, in1, memory_config=out_mem, dtype=BF16,
                            compute_kernel_config=ckcfg, core_grid=cg)
            ttnn.deallocate(r)
        arms.append((name, "MatmulDeviceOperation", run, knob, level))

    # ---- A/A floors. Two arms with identical configuration and identical operands. Any
    # difference between them is the rig's own noise and is the bar every knob below clears.
    gen("gen.AA.a", "AA", "a")
    gen("gen.AA.b", "AA", "b")
    mm("mm.AA.a", "AA", "a")
    mm("mm.AA.b", "AA", "b")

    # ---- K_block. The shipped entry sets K_block == kt == 4, the WHOLE contraction, so the
    # operand daisy chain has exactly one block to pipeline over.
    for kb in (2, 1):
        gen(f"gen.kblock{kb}", "kblock", str(kb), kblock=kb)

    # ---- operand residency, one operand at a time.
    gen("gen.in1L1", "residency_in1", "L1", in1_mem=L1)
    mm("mm.in1L1", "residency_in1", "L1", in1_mem=L1)
    if args.small_m:
        # in0 at the shipped M is 64 MiB and the output 256 MiB; neither fits L1 on this part.
        # The residency knob for the BIG operand is therefore measured at a reduced M, where
        # both arms fit, with its own same-M DRAM control so the comparison stays single-variable.
        sm = args.small_m
        gen("gen.smallM.dram", "residency_in0", "DRAM", m=sm)
        gen("gen.smallM.in0L1", "residency_in0", "L1", m=sm, in0_mem=L1)
        gen("gen.smallM.outL1", "residency_out", "L1", m=sm, out_mem=L1)
        gen("gen.smallM.dram2", "residency_out", "DRAM", m=sm)
        mm("mm.smallM.dram", "residency_in0", "DRAM", m=sm)
        mm("mm.smallM.in0L1", "residency_in0", "L1", m=sm, in0_mem=L1)

    # ---- CB depth. Only meaningful where the producer emits more than one block; at
    # K_block == kt it emits one, which is the classification this sweep is here to confirm.
    gen("gen.cb3x", "cbdepth", "3x", cbscale=1.5)
    gen("gen.cb4x", "cbdepth", "4x", cbscale=2.0)
    gen("gen.kb2.cb4x", "cbdepth_kb2", "4x", kblock=2, cbscale=2.0)
    gen("gen.kb2.cb2x", "cbdepth_kb2", "2x", kblock=2)

    # ---- math fidelity. HiFi3 end to end is a recorded null at 0.997-1.009x, so on the op this
    # should move the stall and not the wall. If it moves neither, the knob is not engaging.
    gen("gen.hifi2", "fidelity", "HiFi2", ckc=HIFI2)
    gen("gen.lofi", "fidelity", "LoFi", ckc=LOFI)
    mm("mm.hifi2", "fidelity", "HiFi2", fid=ttnn.MathFidelity.HiFi2)
    mm("mm.lofi", "fidelity", "LoFi", fid=ttnn.MathFidelity.LoFi)

    # ---- input dtype. A PROBE of what the stall responds to, not a candidate lever: bfp8 is a
    # recorded end-to-end loss (0.949x, +18.9 % bytes, 1.496 A). mm_generic's tile table covers
    # bf16/fp32 only, so this one runs on ttnn.matmul.
    mm("mm.bfp8.in1", "dtype", "bfp8_in1", in1_dt=ttnn.bfloat8_b)
    mm("mm.bfp8.both", "dtype", "bfp8_both", in0_dt=ttnn.bfloat8_b, in1_dt=ttnn.bfloat8_b)

    # ---- per-core tile count, grid held fixed.
    for d in (2, 4):
        gen(f"gen.mdiv{d}", "quantum", f"M/{d}", m=base_m // d)
        mm(f"mm.mdiv{d}", "quantum", f"M/{d}", m=base_m // d)

    # ---- grid shape AT A FIXED PER-CORE QUANTUM. The control is `util-grid-coverage`: 88 cores
    # engaged either way on 11x10 against 11x8, 61.98 us against 61.93 us, 0.08 % apart -- a null,
    # so a sweep that says otherwise is a broken sweep.
    #
    # 72 has no second factorisation inside an 8x9 device, so shape cannot be changed with the
    # core count held fixed on this part. It CAN be changed with the per-core quantum held fixed,
    # which is the interpretable version: `transpose` is true here so in0_axis_cores == gx, and
    # halving gx while halving M leaves M_tiles_per_core identical (8192/8 == 4096/4 == 1024) and
    # N_tiles_per_core untouched at gy. Each core sees exactly the work it saw before, on half the
    # cores and half the total problem.
    for gx2, gy2 in args.grids:
        assert base_m % (gx / gx2) == 0
        m2 = int(base_m * gx2 / gx)
        gen(f"gen.gridq{gx2}x{gy2}", "gridq", f"{gx2}x{gy2} @ M/{gx // gx2}", m=m2, g=(gx2, gy2))
        mm(f"mm.gridq{gx2}x{gy2}", "gridq", f"{gx2}x{gy2} @ M/{gx // gx2}", m=m2, g=(gx2, gy2))
        # the confound control: same core count, half the quantum (this IS `quantum` M/2 above,
        # restated here so the two rows sit next to each other in the table)
        gen(f"gen.gridq{gx}x{gy}.m{m2}", "gridq", f"{gx}x{gy} @ M/{gx // gx2} (quantum halved)",
            m=m2)

    return arms


def control_arms(ttnn, rig, args):
    """INSTRUMENT CONTROL: two stock ops whose binding limit is decidable from arithmetic
    intensity, so the answer is known before the rig speaks.

    PREDICTED, written before the run: the DRAM-bound elementwise add reads a per-core
    CB-WAIT-FRONT above 80 % of its TRISC0 residency; the compute-bound HiFi4 square matmul
    reads under 10 %. If they do not separate, nothing else in this file means anything.
    """
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    BF16 = ttnn.bfloat16
    arms = []
    n = args.control_n
    a = rig.t("ctl_a", (1, 1, n, n), BF16, DRAM)
    b = rig.t("ctl_b", (1, 1, n, n), BF16, DRAM)

    def starved():
        r = ttnn.add(a, b)
        ttnn.deallocate(r)
    arms.append(("ctl.starved.add", "BinaryNgDeviceOperation", starved, "control", "starved"))

    ckcfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=False,
                                             packer_l1_acc=False)
    m = args.control_mm
    x = rig.t("ctl_x", (1, 1, m, m), BF16, DRAM)
    y = rig.t("ctl_y", (1, 1, m, m), BF16, DRAM)

    def compute():
        r = ttnn.matmul(x, y, compute_kernel_config=ckcfg)
        ttnn.deallocate(r)
    arms.append(("ctl.compute.matmul", "MatmulDeviceOperation", compute, "control", "compute"))
    return arms


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--m", type=int, default=512, help="rows axis of the pair activation")
    ap.add_argument("--small-m", type=int, default=64,
                    help="reduced M for the residency arms (0 disables)")
    ap.add_argument("--control-n", type=int, default=4096)
    ap.add_argument("--control-mm", type=int, default=2048)
    ap.add_argument("--grids", default="", help="e.g. 8x8,4x16 -- extra grid-shape arms")
    ap.add_argument("--only", default="", help="comma-separated knob names to keep")
    a = ap.parse_args()
    a.grids = [tuple(int(v) for v in g.split("x")) for g in a.grids.split(",") if g]
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.mm_generic as G
    import tt_bio.tenstorrent as T

    install_cb_scale(G)

    OUT["env"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "card_TT_VISIBLE_DEVICES": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "ttnn": getattr(ttnn, "__file__", "?"),
        "host_thread_cap": THREAD_CAP,
        "host_workers_assumed": HOST_WORKERS,
        "cpu_count": os.cpu_count(),
        "loadavg_start": loadavg(),
        "reps": a.reps, "warmup": a.warmup, "m": a.m, "small_m": a.small_m,
    }
    dump()

    dev = T.get_device(trace_region_size=int(os.environ.get("TT_BIO_TRACE_REGION_SIZE",
                                                            512 * 1024 * 1024)))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    OUT["env"]["grid"] = list(grid)
    OUT["env"]["arch"] = str(dev.arch())
    print(f"  device open, grid {grid}, arch {OUT['env']['arch']}", flush=True)
    dump()

    rig = Rig(ttnn, dev, grid)
    arms = control_arms(ttnn, rig, a) + build_arms(ttnn, rig, G, grid, a.m, a)
    if a.only:
        keep = set(a.only.split(","))
        arms = [x for x in arms if x[3] in keep]
    OUT["arms"] = [{"name": n, "opcode": oc, "knob": k, "level": lv}
                   for n, oc, _, k, lv in arms]
    print(f"  {len(arms)} arms: " + ", ".join(x[0] for x in arms), flush=True)
    dump()

    fence_t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                              dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(fence_t)
        ttnn.synchronize_device(dev)

    # SETUP: every arm runs `warmup` times before the fence so JIT compile and program-cache
    # population are outside the measured region for ALL arms equally. This is the whole point
    # of interleaving -- a sequential A-then-B bench charges B nothing for a warm cache A paid.
    print("=== warmup (outside the fence) ===", flush=True)
    t0 = time.perf_counter()
    for i in range(a.warmup):
        for name, _oc, run, _k, _lv in arms:
            run()
        ttnn.synchronize_device(dev)
        print(f"  warmup {i + 1}/{a.warmup} done", flush=True)
    OUT["warmup_s"] = round(time.perf_counter() - t0, 2)
    dump()

    fence()
    print(f"=== measured region: {a.reps} interleaved reps x {len(arms)} arms ===", flush=True)
    order: list[str] = []
    wall: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for _ in range(a.reps):
        for name, _oc, run, _k, _lv in arms:
            s = time.perf_counter()
            run()
            ttnn.synchronize_device(dev)
            wall.setdefault(name, []).append(time.perf_counter() - s)
            order.append(name)
    OUT["measured_s"] = round(time.perf_counter() - t0, 2)
    fence()

    OUT["dispatch_order"] = order
    OUT["synced_wall_ms"] = {k: [round(1e3 * v, 4) for v in vs] for k, vs in wall.items()}
    OUT["env"]["loadavg_end"] = loadavg()
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
