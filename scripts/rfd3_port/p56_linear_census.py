"""Shape census of the DiT's linears, so the largest unroofed row can be placed on the roofline.

p49 put `linear@265` at 46.951 ms/step over 432 calls in the token DiT -- the biggest row left with no
roof on it. `265` is inside `_tuned_linear`, so one line covers every calibrated linear in the model
and the shapes have to be read off a real step before any FLOP or byte model means anything.

Groups by (caller of _tuned_linear, in shape, weight shape) and reports the measured sync-bracketed
cost per group with FLOPs and bytes, against this chip's roofs measured 2026-08-09
(perfwar-rfd3-esmfold2-sites.md): bf16 HiFi4 square matmul 102.02 TFLOP/s, DRAM read 390.0 GB/s,
write 269.6 GB/s.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p56_linear_census.py
"""
import collections
import json
import os
import pathlib
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                          # noqa: E402
from tt_bio.rfd3 import model as M                                     # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p56/linear_census.json")
STEPS, WARM = 6, 2
TFLOP_ROOF, RD_ROOF, WR_ROOF = 102.02e12, 390.0e9, 269.6e9

STEP = [0]
ACC = collections.defaultdict(lambda: [0.0, 0])
_tuned = M._tuned_linear


def w(x, weight, **k):
    fr = sys._getframe(1)
    site = "%s:%d" % (fr.f_code.co_filename.rsplit("/", 1)[-1], fr.f_lineno)
    xs = tuple(int(v) for v in x.padded_shape)
    ws = tuple(int(v) for v in weight.padded_shape)
    ttnn.synchronize_device(M.get_device())
    t0 = time.perf_counter()
    try:
        return _tuned(x, weight, **k)
    finally:
        ttnn.synchronize_device(M.get_device())
        dt = time.perf_counter() - t0
        if STEP[0] >= WARM:
            e = ACC[(site, xs, ws)]
            e[0] += dt
            e[1] += 1


M._tuned_linear = w


def main():
    call = M.RFD3DiffusionModule.__call__

    def stepped(self, *a, **k):
        try:
            return call(self, *a, **k)
        finally:
            STEP[0] += 1
    M.RFD3DiffusionModule.__call__ = stepped

    specs = json.loads(FIXTURE.read_text())
    os.system("rm -rf /tmp/rfd3_p56")
    rfd3_design.run_design(specs, "/tmp/rfd3_p56", checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=False)
    counted = STEP[0] - WARM

    rows = []
    for (site, xs, ws), (tot, n) in ACC.items():
        m = 1
        for d in xs[:-1]:
            m *= d
        kk, nn = xs[-1], ws[-1]
        calls = n / counted
        ms = 1000 * tot / counted
        flops = 2 * m * kk * nn * calls
        by_rd = 2 * (m * kk + kk * nn) * calls
        by_wr = 2 * m * nn * calls
        s = ms / 1e3
        rows.append({"site": site, "in": list(xs), "w": list(ws), "M": m,
                     "calls_per_step": round(calls, 1), "ms_per_step": round(ms, 3),
                     "TFLOP_s": round(flops / s / 1e12, 2),
                     "pct_compute": round(100 * (flops / s) / TFLOP_ROOF, 1),
                     "pct_read": round(100 * (by_rd / s) / RD_ROOF, 1),
                     "pct_write": round(100 * (by_wr / s) / WR_ROOF, 1)})
    rows.sort(key=lambda r: -r["ms_per_step"])

    print("%-16s %-22s %-14s %8s %7s %8s %7s %7s %7s" %
          ("site", "in", "weight", "ms/step", "calls", "TFLOP/s", "%comp", "%rd", "%wr"))
    for r in rows:
        if r["ms_per_step"] < 0.2:
            continue
        print("%-16s %-22s %-14s %8.3f %7.1f %8.2f %6.1f%% %6.1f%% %6.1f%%" %
              (r["site"], str(r["in"])[:22], str(r["w"])[:14], r["ms_per_step"],
               r["calls_per_step"], r["TFLOP_s"], r["pct_compute"], r["pct_read"],
               r["pct_write"]))
    print("\ntotal %.3f ms/step over %.0f calls"
          % (sum(r["ms_per_step"] for r in rows), sum(r["calls_per_step"] for r in rows)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "counted_steps": counted, "atoms": 6051,
                               "roofs": {"TFLOP_s": 102.02, "read_GB_s": 390.0,
                                         "write_GB_s": 269.6},
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
