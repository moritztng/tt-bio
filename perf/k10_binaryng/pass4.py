"""Pass 4. The one lever pass 3 turned up that nobody had priced: a 0.5 MB mask in L1.

The trimul's pair mask is 0.524 MB and its `multiply_` costs exactly what a full 67.1 MB second
operand costs, because a broadcast operand is re-read once per channel block rather than once.
So 0.5 MB of L1 should buy back a whole operand's worth of DRAM traffic -- a far better ratio than
the residency lever anywhere else in the block, where buying back 67.1 MB of DRAM costs 67.1 MB of
L1 the part does not have. Measured here, against its own DRAM arm, interleaved.
"""
import argparse, json, os, statistics as st, sys, time                          # noqa: E401
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch                                                                    # noqa: E402

torch.set_grad_enabled(False)
import ttnn                                                                     # noqa: E402
import tt_bio.tenstorrent as T                                                  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
SIG = [ttnn.UnaryOpType.SIGMOID]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--reps", type=int, default=16)
args = ap.parse_args()

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
OUT = {"host": os.uname().nodename, "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
       "arch": str(dev.arch()), "grid": [g.x, g.y], "rows": []}


def mk(shape, mc, fill=0.0):
    return ttnn.from_torch(torch.full(shape, fill, dtype=torch.float32).bfloat16(),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


CH, MK, HM = (1, 128, 512, 512), (1, 1, 512, 512), (512, 4, 512, 32)
NP = 67108864

CASES = [
    ("mask DRAM  mul_(CH DRAM, mask DRAM)", lambda: (mk(CH, DRAM, 1.0), mk(MK, DRAM, 1.0)),
     lambda a, b: ttnn.multiply_(a, b), 2 * NP + 524288),
    ("mask L1    mul_(CH DRAM, mask L1)  ", lambda: (mk(CH, DRAM, 1.0), mk(MK, L1, 1.0)),
     lambda a, b: ttnn.multiply_(a, b), 2 * NP),
    ("gate DRAM  mul_(HM DRAM, HM DRAM) sig", lambda: (mk(HM, DRAM, 1.0), mk(HM, DRAM, 20.0)),
     lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG), 3 * NP),
    ("gate L1    mul_(HM DRAM, HM L1)   sig", lambda: (mk(HM, DRAM, 1.0), mk(HM, L1, 20.0)),
     lambda a, b: ttnn.multiply_(a, b, input_tensor_b_activations=SIG), 2 * NP),
]
# Two 67 MB L1 tensors never coexist, so the gate pair runs after the mask pair frees.
for lo, hi in ((0, 2), (2, 4)):
    live, ts = [], {}
    for key, setup, run, dram in CASES[lo:hi]:
        try:
            live.append((key, setup(), run, dram))
        except Exception as e:                                                  # noqa: BLE001
            OUT["rows"].append({"key": key, "error": str(e)[:200]})
            print("%-40s ALLOC %s" % (key, str(e)[:70]), flush=True)
    for key, a, run, _ in live:
        ttnn.synchronize_device(dev)
        o = run(*a)
        ttnn.synchronize_device(dev)
        ts[key] = ([], o.buffer_address() in {x.buffer_address() for x in a})
    for _ in range(args.rounds):
        for key, a, run, _ in live:                      # interleaved, round by round
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                o = run(*a)
                if not ts[key][1]:
                    ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            ts[key][0].append((time.perf_counter() - t0) * 1e3 / args.reps)
    for key, a, run, dram in live:
        s = ts[key][0]
        ms = st.median(s)
        r = {"key": key, "ms": round(ms, 4), "dram_MB": round(dram / 1e6, 3),
             "dram_GBps": round(dram / (ms * 1e-3) / 1e9, 1), "in_place_alias": ts[key][1],
             "spread": round(max(s) / min(s), 4), "samples_ms": [round(x, 4) for x in s]}
        OUT["rows"].append(r)
        print("%-40s %8.4f ms  %7.2f MB DRAM  %6.1f GB/s  spread %.3fx"
              % (key, ms, r["dram_MB"], r["dram_GBps"], r["spread"]), flush=True)
        for x in a:
            try:
                ttnn.deallocate(x)
            except Exception:                                                   # noqa: BLE001
                pass

Path(args.out).write_text(json.dumps(OUT, indent=1))
print("\nWROTE " + args.out, flush=True)
