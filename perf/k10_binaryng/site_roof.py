"""Why the pairformer block's BinaryNg rows cost what they cost: bytes, roof, bound-by.

The census reads these five `ttnn.multiply_` / `ttnn.add_` sites as "no useful math", and
`util-op-deletes` already answered "can they be deleted" with NO. The open question is the cost
model, so this file measures three things on ONE part in ONE session:

  ROOF     a DRAM copy roof and a read+write roof, swept over sizes, on the card that ran the rows.
           A roof is measured, never asserted (memory `roofline-roof-must-be-measured-not-asserted`),
           and Wormhole's DRAM is not Blackhole's.
  CONTROL  the two known-answer ops `k10-instrument` calibrated on: a starved `ttnn.add` that must
           land at the roof, and a HiFi4 `ttnn.matmul` that must land far below it. An instrument
           that cannot separate those two has not measured anything.
  ROWS     the five sites at the exact shape, dtype and memory space `util-op-deletes`' BH op ledger
           records, plus a residency ablation that prices moving one operand off DRAM.

Bytes are deduped on BUFFER ADDRESS, not tensor id: every row here is in-place, so the output
aliases operand `a` and the traffic is `read a + read b + write a`, three operand-sizes and not
four. The aliasing is asserted at run time rather than assumed. DRAM bytes and L1 bytes are counted
separately, because an L1 operand crosses no DRAM interface and belongs on no DRAM roof.

Arms are INTERLEAVED round by round, never A-then-B: a sequential op bench charges the first arm
for the JIT warm-up (memory `op-ab-must-interleave-arms-compile-warmup-bias`).

Operand values are chosen to be fixed points of their own op (`add_` with b = 0, `multiply_` with
the mask = 1, the sigmoid gates with b = 20 so sigmoid(b) = 1) so a tensor can be re-timed for many
rounds without drifting to zero or to inf. Bandwidth does not depend on the values; determinism of
the operand state across rounds does.
"""
import argparse
import json
import math
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
BLOCKS_PER_FOLD = 280            # the same divisor `util-op-deletes` used, so rows compare

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=9)
ap.add_argument("--board", default="unlabelled")
args = ap.parse_args()

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
OUT = {
    "host": os.uname().nodename, "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
    "arch": str(dev.arch()), "board": args.board, "grid": [g.x, g.y], "rounds": args.rounds,
    "host_thread_cap": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TT_METAL_NUM_THREADS")},
    "roof": [], "control": [], "rows": [], "ablation": [], "sweep": [],
}


def mk(shape, mc, fill=0.0):
    t = torch.full(shape, fill, dtype=torch.float32).bfloat16()
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


def nbytes(shape):
    n = 2
    for d in shape:
        n *= d
    return n


class Case:
    """One timed op. `setup` allocates operands; `run` issues the op exactly once."""

    def __init__(self, key, setup, run, dram, l1, n_per_block=0, note=""):
        self.key, self.setup, self.run = key, setup, run
        self.dram, self.l1, self.n_per_block, self.note = dram, l1, n_per_block, note
        self.args = None
        self.ts = []
        self.alias = None
        self.err = None

    def alloc(self):
        try:
            self.args = self.setup()
        except Exception as e:                                                  # noqa: BLE001
            self.err = "ALLOC %s" % str(e)[:160]

    def tick(self, record):
        if self.err:
            return
        try:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = self.run(*self.args)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1e3
        except Exception as e:                                                  # noqa: BLE001
            self.err = "RUN %s" % str(e)[:160]
            return
        # Dedupe on buffer address: an in-place op hands back operand `a`'s own buffer, and the
        # byte count above is only right if it does. Assert it instead of believing it.
        if out is not None:
            addrs = {a.buffer_address() for a in self.args}
            self.alias = out.buffer_address() in addrs
            if not self.alias:
                ttnn.deallocate(out)
        if record:
            self.ts.append(dt)

    def free(self):
        for a in (self.args or ()):
            try:
                ttnn.deallocate(a)
            except Exception:                                                   # noqa: BLE001
                pass
        self.args = None

    def row(self):
        if self.err or not self.ts:
            return {"key": self.key, "error": self.err or "no samples", "note": self.note}
        ms = st.median(self.ts)
        tot = self.dram + self.l1
        r = {"key": self.key, "note": self.note, "ms": round(ms, 4),
             "spread": round(max(self.ts) / min(self.ts), 4),
             "dram_MB": round(self.dram / 1e6, 3), "l1_MB": round(self.l1 / 1e6, 3),
             "dram_GBps": round(self.dram / (ms * 1e-3) / 1e9, 1),
             "mem_GBps": round(tot / (ms * 1e-3) / 1e9, 1),
             "in_place_alias": self.alias, "samples_ms": [round(x, 4) for x in self.ts]}
        if self.n_per_block:
            r["n_per_block"] = self.n_per_block
            r["ms_per_block"] = round(ms * self.n_per_block, 4)
            r["s_per_fold"] = round(ms * self.n_per_block * BLOCKS_PER_FOLD / 1e3, 4)
        return r


def run_group(cases, rounds, label):
    """Allocate every case, warm every case, then time them ROUND-ROBIN. Never A-then-B."""
    for c in cases:
        c.alloc()
    for c in cases:                       # one untimed warm pass each, for the JIT
        c.tick(record=False)
    for _ in range(rounds):
        for c in cases:
            c.tick(record=True)
    rows = [c.row() for c in cases]
    for c in cases:
        c.free()
    for r in rows:
        if "error" in r:
            print("%-52s %s" % (r["key"], r["error"]), flush=True)
        else:
            print("%-52s %8.4f ms  DRAM %7.1f GB/s  mem %7.1f GB/s  alias=%s  spread %.3fx"
                  % (r["key"], r["ms"], r["dram_GBps"], r["mem_GBps"], r["in_place_alias"],
                     r["spread"]), flush=True)
    OUT[label].extend(rows)
    return rows


# ---------------------------------------------------------------------------------------------
# ROOF. Two shapes of traffic at several sizes; the roof is the best any of them achieves.
# ---------------------------------------------------------------------------------------------
print("\n=== ROOF (%s %s, grid %dx%d) ===" % (OUT["arch"], args.board, g.x, g.y), flush=True)
roof_cases = []
for shape in [(512, 512, 128), (1, 128, 512, 512), (2048, 2048), (4096, 4096), (8192, 8192)]:
    nb = nbytes(shape)
    roof_cases.append(Case(
        "copy   clone %-18s %6.1f MB" % ("x".join(map(str, shape)), nb / 1e6),
        (lambda s=shape: (mk(s, DRAM),)),
        (lambda a: ttnn.clone(a, memory_config=DRAM)), 2 * nb, 0))
    roof_cases.append(Case(
        "rw     add   %-18s %6.1f MB" % ("x".join(map(str, shape)), nb / 1e6),
        (lambda s=shape: (mk(s, DRAM), mk(s, DRAM))),
        (lambda a, b: ttnn.add(a, b, memory_config=DRAM)), 3 * nb, 0))
run_group(roof_cases, args.rounds, "roof")
good = [r for r in OUT["roof"] if "error" not in r]
ROOF = max(r["dram_GBps"] for r in good)
ROOF_KEY = max(good, key=lambda r: r["dram_GBps"])["key"]
print("\nMEASURED DRAM ROOF %.1f GB/s  (%s)" % (ROOF, ROOF_KEY.strip()), flush=True)
OUT["roof_GBps"] = ROOF
OUT["roof_from"] = ROOF_KEY.strip()

# ---------------------------------------------------------------------------------------------
# INSTRUMENT-CONTROL. Known answers, prediction written in the state doc before the run:
# the starved add must land at the roof, the HiFi4 matmul far below it.
# ---------------------------------------------------------------------------------------------
print("\n=== INSTRUMENT-CONTROL ===", flush=True)
MM_N = 2048
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=False, packer_l1_acc=False)
ctl = [
    Case("control STARVED add 8192x8192 bf16 (predict ~roof)",
         (lambda: (mk((8192, 8192), DRAM), mk((8192, 8192), DRAM))),
         (lambda a, b: ttnn.add(a, b, memory_config=DRAM)),
         3 * nbytes((8192, 8192)), 0, note="arithmetic intensity 0.17 FLOP/byte"),
    Case("control COMPUTE matmul 2048^3 HiFi4 (predict << roof)",
         (lambda: (mk((MM_N, MM_N), DRAM), mk((MM_N, MM_N), DRAM))),
         (lambda a, b: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM)),
         3 * nbytes((MM_N, MM_N)), 0, note="17.2 GFLOP, 341 FLOP/byte"),
]
run_group(ctl, args.rounds, "control")
for r in OUT["control"]:
    if "error" not in r:
        r["pct_of_roof"] = round(100 * r["dram_GBps"] / ROOF, 1)
        if "matmul" in r["key"]:
            r["TFLOPs"] = round(2.0 * MM_N ** 3 / (r["ms"] * 1e-3) / 1e12, 2)
        print("  %-52s %5.1f %% of roof%s" % (
            r["key"], r["pct_of_roof"],
            "  %.2f TFLOP/s" % r["TFLOPs"] if "TFLOPs" in r else ""), flush=True)

# ---------------------------------------------------------------------------------------------
# THE FIVE ROWS, at the production shapes.
# ---------------------------------------------------------------------------------------------
P = (1, 512, 512, 128)      # the pair tensor
CH = (1, 128, 512, 512)     # the same bytes, channel-major, inside the trimul
HM = (512, 4, 512, 32)      # the triangle attention's head-major activation
MK = (1, 1, 512, 512)
NP = nbytes(P)
SIG = [ttnn.UnaryOpType.SIGMOID]

print("\n=== THE FIVE ROWS (in-place; bytes = read a + read b + write a) ===", flush=True)
rows = [
    Case("row trimul pair mask  mul_(CH DRAM, mask DRAM)",
         (lambda: (mk(CH, DRAM, 1.0), mk(MK, DRAM, 1.0))),
         (lambda a, b: ttnn.multiply_(a, b)),
         2 * NP + nbytes(MK), 0, n_per_block=2),
    Case("row trimul out gate   mul_(P L1, P DRAM) sigmoid",
         (lambda: (mk(P, L1, 1.0), mk(P, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)),
         NP, 2 * NP, n_per_block=2),
    Case("row triatt out gate   mul_(HM DRAM, HM DRAM) sigmoid",
         (lambda: (mk(HM, DRAM, 1.0), mk(HM, DRAM, 20.0))),
         (lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG)),
         3 * NP, 0, n_per_block=2),
    Case("row residual          add_(P DRAM, P L1)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, L1, 0.0))),
         (lambda a, b: ttnn.add_(a, b)), 2 * NP, NP, n_per_block=3),
    Case("row residual          add_(P DRAM, P DRAM)",
         (lambda: (mk(P, DRAM, 1.0), mk(P, DRAM, 0.0))),
         (lambda a, b: ttnn.add_(a, b)), 3 * NP, 0, n_per_block=2),
]
# The two L1 rows each want a 67.1 MB L1-interleaved operand. Run them in their own group so a
# refusal on one does not take the DRAM rows down with it, and so two 67 MB L1 tensors are never
# asked for at once.
run_group([rows[0], rows[2], rows[4]], args.rounds, "rows")
run_group([rows[1]], args.rounds, "rows")
run_group([rows[3]], args.rounds, "rows")
for r in OUT["rows"]:
    if "error" not in r:
        r["pct_of_roof_dram"] = round(100 * r["dram_GBps"] / ROOF, 1)
        r["pct_of_roof_mem"] = round(100 * r["mem_GBps"] / ROOF, 1)

# ---------------------------------------------------------------------------------------------
# SWEEP: is time affine in DRAM bytes? The slope is a second, independent read of the roof, and
# the intercept is the part of the cost that moving fewer bytes cannot touch.
# ---------------------------------------------------------------------------------------------
print("\n=== SWEEP add_(DRAM, DRAM) vs size ===", flush=True)
sw = []
for c in (8, 16, 32, 64, 128):
    s = (1, 512, 512, c)
    sw.append(Case("sweep add_ %-18s %6.1f MB DRAM" % ("x".join(map(str, s)), 3 * nbytes(s) / 1e6),
                   (lambda ss=s: (mk(ss, DRAM, 1.0), mk(ss, DRAM, 0.0))),
                   (lambda a, b: ttnn.add_(a, b)), 3 * nbytes(s), 0))
run_group(sw, args.rounds, "sweep")
pts = [(r["dram_MB"] * 1e6, r["ms"] * 1e-3) for r in OUT["sweep"] if "error" not in r]
if len(pts) >= 3:
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    icpt = my - slope * mx
    ss_res = sum((p[1] - (icpt + slope * p[0])) ** 2 for p in pts)
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    OUT["fit"] = {"marginal_GBps": round(1 / slope / 1e9, 1),
                  "intercept_ms": round(icpt * 1e3, 4),
                  "r2": round(1 - ss_res / ss_tot, 5) if ss_tot else None,
                  "pct_of_roof": round(100 / slope / 1e9 / ROOF, 1), "n": n}
    print("FIT  marginal %.1f GB/s (%.1f %% of roof), intercept %.4f ms, r2 %.5f"
          % (OUT["fit"]["marginal_GBps"], OUT["fit"]["pct_of_roof"],
             OUT["fit"]["intercept_ms"], OUT["fit"]["r2"]), flush=True)

# ---------------------------------------------------------------------------------------------
# ABLATION: what does moving ONE operand off DRAM into L1 buy? That is the only lever left if the
# rows are bandwidth-bound -- fusing into a producer that already holds the tile is exactly "this
# operand never crosses DRAM". Swept, because the 67 MB L1 operand may not fit on every part, and
# interleaved arm-by-arm.
# ---------------------------------------------------------------------------------------------
print("\n=== ABLATION residency: add_(P DRAM, b DRAM) vs add_(P DRAM, b L1) ===", flush=True)
for c in (8, 16, 32, 64, 128):
    s = (1, 512, 512, c)
    nb = nbytes(s)
    pair = [
        Case("abl c=%-3d b DRAM %6.1f MB DRAM" % (c, 3 * nb / 1e6),
             (lambda ss=s: (mk(ss, DRAM, 1.0), mk(ss, DRAM, 0.0))),
             (lambda a, b: ttnn.add_(a, b)), 3 * nb, 0, note="c=%d arm=dram" % c),
        Case("abl c=%-3d b L1   %6.1f MB DRAM" % (c, 2 * nb / 1e6),
             (lambda ss=s: (mk(ss, DRAM, 1.0), mk(ss, L1, 0.0))),
             (lambda a, b: ttnn.add_(a, b)), 2 * nb, nb, note="c=%d arm=l1" % c),
    ]
    run_group(pair, args.rounds, "ablation")

Path(args.out).write_text(json.dumps(OUT, indent=1))
print("\nWROTE " + args.out, flush=True)
