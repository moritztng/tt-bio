"""Pass 3. The two questions pass 2 left open, and nothing else.

1. **The broadcast row.** `mul_((1,128,512,512) DRAM, (1,1,512,512) DRAM)` reads 134.7 MB of DRAM
   on the naive count and comes out at 65.5 % of the roof, which is the only row under the 70 %
   kill line. Its non-broadcast twin reads 201.3 MB and runs in the SAME time. That is what you get
   if the 0.5 MB mask is not read once but once per channel block, i.e. its real byte count is
   67.1 MB and not 0.524 MB. Sweep the broadcast factor: if the explanation holds, time tracks the
   OUTPUT size and the broadcast row lands on the roof once its operand is counted C times.

2. **The gate rows.** The fused `SIGMOID` costs 0.39 ms on 33.55 M elements and `RELU` costs 0.014.
   Price the whole activation menu at one fixed byte count so the "make the gate cheaper" lever has
   a ceiling rather than a hope, and check the overhead is linear in elements (a compute cost) and
   not a fixed per-program cost.
"""
import argparse, json, os, statistics as st, sys, time                          # noqa: E401
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch                                                                    # noqa: E402

torch.set_grad_enabled(False)
import ttnn                                                                     # noqa: E402
import tt_bio.tenstorrent as T                                                  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--reps", type=int, default=16)
ap.add_argument("--board", default="unlabelled")
args = ap.parse_args()

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
OUT = {"host": os.uname().nodename, "card_env": os.environ.get("TT_VISIBLE_DEVICES"),
       "arch": str(dev.arch()), "board": args.board, "grid": [g.x, g.y],
       "rounds": args.rounds, "reps_per_round": args.reps, "groups": {}}


def mk(shape, fill=0.0):
    return ttnn.from_torch(torch.full(shape, fill, dtype=torch.float32).bfloat16(),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=DRAM)


def nbt(shape):
    n = 2
    for d in shape:
        n *= d
    return n


def timed(label, setup, run, dram, elems, note=""):
    try:
        a = setup()
    except Exception as e:                                                      # noqa: BLE001
        print("%-46s ALLOC %s" % (label, str(e)[:70]), flush=True)
        return {"key": label, "error": str(e)[:200]}
    try:
        ttnn.synchronize_device(dev)
        out = run(*a)
        ttnn.synchronize_device(dev)
        alias = out is not None and out.buffer_address() in {x.buffer_address() for x in a}
        if out is not None and not alias:
            ttnn.deallocate(out)
        ts = []
        for _ in range(args.rounds):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                o = run(*a)
                if o is not None and not alias:
                    ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3 / args.reps)
    except Exception as e:                                                      # noqa: BLE001
        print("%-46s RUN %s" % (label, str(e)[:70]), flush=True)
        return {"key": label, "error": str(e)[:200]}
    finally:
        for x in a:
            try:
                ttnn.deallocate(x)
            except Exception:                                                   # noqa: BLE001
                pass
    ms = st.median(ts)
    r = {"key": label, "note": note, "ms": round(ms, 4), "dram_MB": round(dram / 1e6, 3),
         "Melem": round(elems / 1e6, 2), "dram_GBps": round(dram / (ms * 1e-3) / 1e9, 1),
         "in_place_alias": alias, "spread": round(max(ts) / min(ts), 4),
         "samples_ms": [round(x, 4) for x in ts]}
    print("%-46s %8.4f ms  %6.1f MB DRAM  %6.1f GB/s  %6.2f Melem  spread %.3fx"
          % (label, ms, r["dram_MB"], r["dram_GBps"], r["Melem"], r["spread"]), flush=True)
    return r


# --- 1. BROADCAST -------------------------------------------------------------------------------
print("\n=== BROADCAST: is the 0.5 MB mask read once, or once per channel block? ===", flush=True)
bc = []
for C in (16, 32, 64, 128):
    S = (1, C, 512, 512)
    n = nbt(S)
    el = n // 2
    bc.append(timed("bcast  mul_((1,%3d,512,512), (1,1,512,512))" % C,
                    (lambda s=S: (mk(s, 1.0), mk((1, 1, 512, 512), 1.0))),
                    (lambda a, b: ttnn.multiply_(a, b)), 2 * n + nbt((1, 1, 512, 512)), el,
                    note="naive byte count: mask read ONCE"))
    bc.append(timed("full   mul_((1,%3d,512,512), same)" % C,
                    (lambda s=S: (mk(s, 1.0), mk(s, 1.0))),
                    (lambda a, b: ttnn.multiply_(a, b)), 3 * n, el, note="non-broadcast twin"))
OUT["groups"]["broadcast"] = bc

# --- 2. ACTIVATION MENU, one fixed byte count ---------------------------------------------------
print("\n=== ACTIVATION MENU on (1,512,512,128) DRAM, 201.3 MB, 33.55 Melem ===", flush=True)
P = (1, 512, 512, 128)
NP = nbt(P)
menu = []
for name in ("NONE", "RELU", "HARDSIGMOID", "TANH", "SILU", "SIGMOID", "EXP", "GELU"):
    if name == "NONE":
        run = lambda a, b: ttnn.multiply_(a, b)                                 # noqa: E731
    else:
        op = getattr(ttnn.UnaryOpType, name, None)
        if op is None:
            print("%-46s (no such UnaryOpType)" % ("act " + name), flush=True)
            continue
        run = (lambda a, b, o=op: ttnn.multiply_(a, b, input_tensor_b_activations=[o]))
    menu.append(timed("act %-12s mul_(P DRAM, P DRAM)" % name,
                      (lambda: (mk(P, 1.0), mk(P, 1.0))), run, 3 * NP, NP // 2,
                      note="identical bytes for every row"))
OUT["groups"]["activation_menu"] = menu

# --- 3. IS THE SIGMOID OVERHEAD LINEAR IN ELEMENTS? ---------------------------------------------
print("\n=== SIGMOID SCALING: overhead per element, or per program? ===", flush=True)
sc = []
for C in (32, 64, 128):
    S = (1, 512, 512, C)
    n = nbt(S)
    sc.append(timed("scal c=%-3d mul_          (gate off)" % C,
                    (lambda s=S: (mk(s, 1.0), mk(s, 1.0))),
                    (lambda a, b: ttnn.multiply_(a, b)), 3 * n, n // 2, note="c=%d off" % C))
    sc.append(timed("scal c=%-3d mul_ sigmoid  (gate on)" % C,
                    (lambda s=S: (mk(s, 1.0), mk(s, 20.0))),
                    (lambda a, b: ttnn.multiply_(
                        a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])),
                    3 * n, n // 2, note="c=%d on" % C))
OUT["groups"]["sigmoid_scaling"] = sc

Path(args.out).write_text(json.dumps(OUT, indent=1))
print("\nWROTE " + args.out, flush=True)
