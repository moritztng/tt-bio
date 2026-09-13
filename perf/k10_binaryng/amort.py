"""Pass 2. Amortised device timing, and the ablation that says what the gate rows are really paying.

Pass 1 (`site_roof.py`) timed one call between two `synchronize_device`s and found a **0.2555 ms
floor**: `add_(1x512x512x8)` and `add_(1x512x512x32)` move 12.6 MB and 50.3 MB and both take
0.2555 ms. A cost that does not move with the bytes at the small end is host round-trip, not device
work, and it inflates every per-call bandwidth in that pass. So this pass issues K calls back to
back and syncs once, which lets host dispatch overlap device execution the way production does.

It also runs the ablation pass 1 made necessary. Two rows came out at the same 1.276 ms while
moving very different DRAM byte counts (201.3 MB against 67.1 MB), which no bandwidth-bound model
produces. Both carry `input_tensor_b_activations=[SIGMOID]`. So: same shapes, same memory spaces,
gate on and gate off, interleaved.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                   # noqa: E402

torch.set_grad_enabled(False)
import ttnn                                                                    # noqa: E402
import tt_bio.tenstorrent as T                                                 # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
SIG = [ttnn.UnaryOpType.SIGMOID]
BLOCKS_PER_FOLD = 280

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--reps", type=int, default=16, help="calls issued between the two syncs")
ap.add_argument("--board", default="unlabelled")
args = ap.parse_args()

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
OUT = {"host": os.uname().nodename, "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
       "arch": str(dev.arch()), "board": args.board, "grid": [g.x, g.y],
       "rounds": args.rounds, "reps_per_round": args.reps,
       "host_thread_cap": os.environ.get("OMP_NUM_THREADS"), "groups": {}}


def mk(shape, mc, fill=0.0):
    return ttnn.from_torch(torch.full(shape, fill, dtype=torch.float32).bfloat16(),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


def nb(shape):
    n = 2
    for d in shape:
        n *= d
    return n


class Case:
    def __init__(self, key, setup, run, dram, l1, n_per_block=0, note=""):
        self.key, self.setup, self.run = key, setup, run
        self.dram, self.l1, self.n_per_block, self.note = dram, l1, n_per_block, note
        self.args = self.alias = self.err = None
        self.ts = []
        self.single_ms = None

    def alloc(self):
        try:
            self.args = self.setup()
        except Exception as e:                                                  # noqa: BLE001
            self.err = "ALLOC %s" % str(e)[:160]

    def warm(self):
        """One call: establishes the JIT, and settles whether the op is in place."""
        if self.err:
            return
        try:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = self.run(*self.args)
            ttnn.synchronize_device(dev)
            self.single_ms = (time.perf_counter() - t0) * 1e3
            self.alias = out is not None and out.buffer_address() in {
                a.buffer_address() for a in self.args}
            if out is not None and not self.alias:
                ttnn.deallocate(out)
        except Exception as e:                                                  # noqa: BLE001
            self.err = "RUN %s" % str(e)[:160]

    def tick(self, reps):
        if self.err:
            return
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            out = self.run(*self.args)
            if out is not None and not self.alias:
                ttnn.deallocate(out)
        ttnn.synchronize_device(dev)
        self.ts.append((time.perf_counter() - t0) * 1e3 / reps)

    def free(self):
        for a in (self.args or ()):
            try:
                ttnn.deallocate(a)
            except Exception:                                                   # noqa: BLE001
                pass
        self.args = None

    def row(self, roof):
        if self.err or not self.ts:
            return {"key": self.key, "error": self.err or "no samples", "note": self.note}
        ms = st.median(self.ts)
        tot = self.dram + self.l1
        r = {"key": self.key, "note": self.note, "ms": round(ms, 4),
             "single_call_ms": round(self.single_ms, 4) if self.single_ms else None,
             "host_overhead_ms": round(self.single_ms - ms, 4) if self.single_ms else None,
             "spread": round(max(self.ts) / min(self.ts), 4),
             "dram_MB": round(self.dram / 1e6, 3), "l1_MB": round(self.l1 / 1e6, 3),
             "dram_GBps": round(self.dram / (ms * 1e-3) / 1e9, 1),
             "mem_GBps": round(tot / (ms * 1e-3) / 1e9, 1),
             "in_place_alias": self.alias, "samples_ms": [round(x, 4) for x in self.ts]}
        if roof:
            r["pct_of_roof_dram"] = round(100 * r["dram_GBps"] / roof, 1)
            r["pct_of_roof_mem"] = round(100 * r["mem_GBps"] / roof, 1)
        if self.n_per_block:
            r["n_per_block"] = self.n_per_block
            r["s_per_fold"] = round(ms * self.n_per_block * BLOCKS_PER_FOLD / 1e3, 4)
        return r


def group(label, cases, roof=None):
    print("\n=== %s ===" % label, flush=True)
    for c in cases:
        c.alloc()
    for c in cases:
        c.warm()
    for _ in range(args.rounds):
        for c in cases:                       # round robin: never arm A then arm B
            c.tick(args.reps)
    rows = [c.row(roof) for c in cases]
    for c in cases:
        c.free()
    for r in rows:
        if "error" in r:
            print("%-50s %s" % (r["key"], r["error"]), flush=True)
        else:
            print("%-50s %8.4f ms (1-call %7.4f, host %+.4f)  DRAM %6.1f  mem %6.1f GB/s%s"
                  % (r["key"], r["ms"], r["single_call_ms"], r["host_overhead_ms"],
                     r["dram_GBps"], r["mem_GBps"],
                     "  %5.1f %% roof" % r["pct_of_roof_mem"] if roof else ""), flush=True)
    OUT["groups"][label] = rows
    return rows


P = (1, 512, 512, 128)
CH = (1, 128, 512, 512)
HM = (512, 4, 512, 32)
MK = (1, 1, 512, 512)
NP = nb(P)

# --- ROOF, amortised, swept. The roof is the best sustained rate any traffic shape reaches. ------
roof_cases = []
for shape in [(1, 512, 512, 32), (1, 512, 512, 64), P, (8192, 8192), (8192, 16384)]:
    n = nb(shape)
    tag = "x".join(map(str, shape))
    roof_cases.append(Case("roof add_  %-18s %6.1f MB" % (tag, 3 * n / 1e6),
                           (lambda s=shape: (mk(s, DRAM, 1.0), mk(s, DRAM, 0.0))),
                           (lambda a, b: ttnn.add_(a, b)), 3 * n, 0))
    roof_cases.append(Case("roof clone %-18s %6.1f MB" % (tag, 2 * n / 1e6),
                           (lambda s=shape: (mk(s, DRAM, 1.0),)),
                           (lambda a: ttnn.clone(a, memory_config=DRAM)), 2 * n, 0))
rr = group("ROOF amortised", roof_cases)
ok = [r for r in rr if "error" not in r]
ROOF = max(r["dram_GBps"] for r in ok)
OUT["roof_GBps"] = ROOF
OUT["roof_from"] = max(ok, key=lambda r: r["dram_GBps"])["key"].strip()
print("\nMEASURED DRAM ROOF (amortised) %.1f GB/s  (%s)" % (ROOF, OUT["roof_from"]), flush=True)

# --- CONTROLS, on the same footing -------------------------------------------------------------
MM = 2048
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=False, packer_l1_acc=False)
cc = group("INSTRUMENT-CONTROL amortised", [
    Case("ctl STARVED add 8192x8192 (predict ~roof)",
         (lambda: (mk((8192, 8192), DRAM, 1.0), mk((8192, 8192), DRAM, 0.0))),
         (lambda a, b: ttnn.add(a, b, memory_config=DRAM)), 3 * nb((8192, 8192)), 0),
    Case("ctl COMPUTE matmul 2048^3 HiFi4 (predict <<roof)",
         (lambda: (mk((MM, MM), DRAM, 1.0), mk((MM, MM), DRAM, 1.0))),
         (lambda a, b: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM)),
         3 * nb((MM, MM)), 0),
], ROOF)
for r in cc:
    if "matmul" in r["key"] and "error" not in r:
        r["TFLOPs"] = round(2.0 * MM ** 3 / (r["ms"] * 1e-3) / 1e12, 2)
        print("  matmul %.2f TFLOP/s at %.1f %% of the DRAM roof" % (r["TFLOPs"],
                                                                    r["pct_of_roof_dram"]))

# --- THE FIVE ROWS ------------------------------------------------------------------------------
group("ROWS amortised", [
    Case("row trimul pair mask  mul_(CH DRAM, mask DRAM)",
         (lambda: (mk(CH, DRAM, 1.0), mk(MK, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b)), 2 * NP + nb(MK), 0, 2),
    Case("row triatt out gate   mul_(HM DRAM, HM DRAM) sig",
         (lambda: (mk(HM, DRAM, 1.0), mk(HM, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)), 3 * NP, 0, 2),
    Case("row residual          add_(P DRAM, P DRAM)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 0.0))),
         (lambda a, b: ttnn.add_(a, b)), 3 * NP, 0, 2),
], ROOF)
group("ROWS amortised, L1 operand", [
    Case("row trimul out gate   mul_(P L1, P DRAM) sig",
         (lambda: (mk(P, L1, 1.0), mk(P, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)), NP, 2 * NP, 2),
], ROOF)
group("ROWS amortised, L1 operand b", [
    Case("row residual          add_(P DRAM, P L1)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, L1, 0.0))),
         (lambda a, b: ttnn.add_(a, b)), 2 * NP, NP, 3),
], ROOF)

# --- THE ABLATION: is the gate row paying for bytes or for the sigmoid? -------------------------
# Same shape, same memory spaces, same byte count; the ONLY difference is the fused activation.
group("SIGMOID ABLATION (identical bytes, gate on vs off)", [
    Case("abl P  DRAM mul_          (no activation)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b)), 3 * NP, 0, note="P shape, gate off"),
    Case("abl P  DRAM mul_ sigmoid(b)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)), 3 * NP, 0,
         note="P shape, gate on"),
    Case("abl HM DRAM mul_          (no activation)",
         (lambda: (mk(HM, DRAM, 1.0), mk(HM, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b)), 3 * NP, 0, note="HM shape, gate off"),
    Case("abl HM DRAM mul_ sigmoid(b)",
         (lambda: (mk(HM, DRAM, 1.0), mk(HM, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)), 3 * NP, 0,
         note="HM shape, gate on"),
    Case("abl P  DRAM add_          (reference)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 0.0))),
         (lambda a, b: ttnn.add_(a, b)), 3 * NP, 0, note="P shape, add"),
], ROOF)

# Is it the sigmoid specifically, or any fused SFPU activation? A cheap one against a dear one.
group("ACTIVATION COST (P DRAM, identical bytes)", [
    Case("act none",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b)), 3 * NP, 0),
    Case("act relu(b)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.RELU])),
         3 * NP, 0),
    Case("act sigmoid(b)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)), 3 * NP, 0),
    Case("act sigmoid standalone then mul_ (2 programs)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, ttnn.sigmoid(b, memory_config=DRAM))), 6 * NP, 0,
         note="what an unfused gate would cost: 2 programs, 6 operand-sizes"),
], ROOF)

# --- RESIDENCY: the only lever left if a row IS bandwidth-bound ---------------------------------
res = []
for c in (32, 64, 128):
    s = (1, 512, 512, c)
    n = nb(s)
    res.append(Case("res c=%-3d add_(DRAM, DRAM) %6.1f MB" % (c, 3 * n / 1e6),
                    (lambda ss=s: (mk(ss, DRAM, 1.0), mk(ss, DRAM, 0.0))),
                    (lambda a, b: ttnn.add_(a, b)), 3 * n, 0, note="c=%d dram" % c))
    res.append(Case("res c=%-3d add_(DRAM, L1)   %6.1f MB" % (c, 2 * n / 1e6),
                    (lambda ss=s: (mk(ss, DRAM, 1.0), mk(ss, L1, 0.0))),
                    (lambda a, b: ttnn.add_(a, b)), 2 * n, n, note="c=%d l1" % c))
group("RESIDENCY ABLATION", res, ROOF)

Path(args.out).write_text(json.dumps(OUT, indent=1))
print("\nWROTE " + args.out, flush=True)
