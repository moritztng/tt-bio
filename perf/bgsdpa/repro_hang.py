"""Standalone repro for the device op the 2100-residue BoltzGen design wedges on.

`hang_probe --sync --ops all` (which drains the queue before every wrapped ttnn call, so the
stall names the device op and not the host's run-ahead limit) parks the fold here:

    208 #752 chunk ENTER [1, 2208, 2208, 64] bf16 TILE DRAM  <- tenstorrent.py:5183

`ttnn.chunk(gp_in_fused, chunks=4, dim=-1)` in TriangleMultiplication's in-projection loop.
The drain issued by the very next call never returns, so that chunk is the op the device never
finishes. This runs each arm in its own process behind a watchdog, logging every stage so a
HANG says WHICH stage hung, and resets the card between arms: a killed hang leaves the chip
dirty and every later arm would otherwise fail at device open with "failed to initialize FW".

    python3 perf/bgsdpa/repro_hang.py --out perf/bgsdpa/repro_hang.json --arms ladder
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "perf" / "bgsdpa" / "work"

# A chunk of a [1, S, S, 64] bf16 DRAM tensor into 4 along the last axis is what the fold does.
# The ladder asks at which S it stops returning; `split` asks whether the 16-wide (sub-tile)
# window is what does it, by giving the same tensor a 32-wide one.
LADDER = [{"op": "chunk", "shape": [1, s, s, 64], "chunks": 4} for s in
          (256, 512, 1024, 1408, 1856, 2208)]
SPLIT = [{"op": "chunk", "shape": [1, 2208, 2208, 64], "chunks": 2},
         {"op": "slice", "shape": [1, 2208, 2208, 64], "chunks": 4},
         {"op": "chunk", "shape": [1, 2208, 2208, 32], "chunks": 4},
         {"op": "chunk", "shape": [1, 2208, 2208, 64], "chunks": 4}]
# The multiply the fold enters right after the chunk, on a chunk-sized operand.
MUL = [{"op": "multiply_", "shape": [1, 2208, 2208, 16], "chunks": 0}]
# What the 1408-passes / 1856-hangs step is made of: the window width (16 is sub-tile, 32 is
# not), the bytes, the aspect ratio at a fixed element count, and where between the two it turns.
WHY = [{"op": "chunk", "shape": [1, 1856, 1856, 64], "chunks": 2},
       {"op": "chunk", "shape": [1, 1856, 1856, 128], "chunks": 4},
       {"op": "slice", "shape": [1, 1856, 1856, 64], "chunks": 4},
       {"op": "chunk", "shape": [1, 928, 3712, 64], "chunks": 4},
       {"op": "chunk", "shape": [1, 1536, 1536, 64], "chunks": 4},
       {"op": "chunk", "shape": [1, 1664, 1664, 64], "chunks": 4},
       {"op": "chunk", "shape": [1, 1792, 1792, 64], "chunks": 4}]
# The fold's own shape, at the two widths the byte cap picks either side of the fix:
# 4C=128 is a whole tile per piece (what `_TRIMUL_MIN_CHUNK` now floors it at), 4C=64 is the
# half tile that shipped. Same tensor, same op, same card.
FIX = [{"op": "chunk", "shape": [1, 2208, 2208, 128], "chunks": 4},
       {"op": "chunk", "shape": [1, 2208, 2208, 64], "chunks": 4}]
# OpenDDE's own configuration, on Blackhole. Its 1024-residue Wormhole fold is verified 3/3
# byte-identical at channel chunk 8 -- a quarter-tile piece out of a ONE-tile-wide row
# ([1, 2016, 2016, 32]) -- so a blanket "never go sub-tile" floor would refuse a fold that works.
# These arms ask whether the row width or the piece width is what wedges Blackhole.
ODDE = [{"op": "chunk", "shape": [1, 2016, 2016, 32], "chunks": 4},
        {"op": "chunk", "shape": [1, 2208, 2208, 32], "chunks": 4},
        {"op": "chunk", "shape": [1, 2016, 2016, 64], "chunks": 4}]
ARMSETS = {"ladder": LADDER, "split": SPLIT, "mul": MUL, "why": WHY, "fix": FIX, "odde": ODDE,
           "all": LADDER + SPLIT + MUL}


def one(spec, q, logpath):
    import torch
    import ttnn
    sys.path.insert(0, os.environ.get("WT", str(ROOT)))
    fh = open(logpath, "w", buffering=1)

    def stage(s):
        fh.write(f"{time.strftime('%H:%M:%S')} {s}\n")

    dev = None
    try:
        shape, n = spec["shape"], spec["chunks"]
        stage("open")
        from tt_bio import tenstorrent as T
        dev = T.get_device()
        stage("host-tensor")
        t = torch.randn(shape, dtype=torch.bfloat16)
        stage("upload")
        a = ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = (ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
             if spec["op"] == "multiply_" else None)
        ttnn.synchronize_device(dev)
        stage("op")
        t0 = time.perf_counter()
        if spec["op"] == "chunk":
            out = list(ttnn.chunk(a, chunks=n, dim=-1))
        elif spec["op"] == "slice":
            w = shape[-1] // n
            out = [ttnn.slice(a, [0, 0, 0, i * w],
                              [shape[0], shape[1], shape[2], (i + 1) * w], [1, 1, 1, 1])
                   for i in range(n)]
        else:
            out = [ttnn.multiply_(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])]
        stage("enqueued")
        ttnn.synchronize_device(dev)
        ms = round((time.perf_counter() - t0) * 1e3, 3)
        stage(f"drained {ms} ms")
        q.put(("ok", {"ms": ms, "out_shapes": [list(o.shape) for o in out]}))
        for o in out:
            ttnn.deallocate(o)
    except Exception as exc:                                          # noqa: BLE001
        stage(f"EXC {exc}")
        q.put(("err", str(exc).splitlines()[0][:240]))
    finally:
        try:
            if dev is not None:
                ttnn.close_device(dev)
        except Exception:                                             # noqa: BLE001
            pass
        stage("done")


def reset(card):
    for exe in (pathlib.Path.home() / ".local/bin/tt-smi", pathlib.Path("/usr/local/bin/tt-smi")):
        if exe.exists():
            subprocess.run([str(exe), "-r", str(card)], timeout=300,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchdog", type=int, default=180)
    ap.add_argument("--arms", default="ladder", choices=sorted(ARMSETS))
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "2"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, spec in enumerate(ARMSETS[args.arms]):
        logpath = WORK / f"repro_{args.arms}_{i}.log"
        q = mp.Queue()
        p = mp.Process(target=one, args=(spec, q, str(logpath)))
        p.start()
        p.join(args.watchdog)
        row = dict(spec, arm=i)
        if p.is_alive():
            p.kill()
            p.join()
            stages = logpath.read_text().splitlines() if logpath.exists() else []
            row.update(verdict="HANG", stage=stages[-1] if stages else "no log",
                       note=f"no result in {args.watchdog}s, killed")
        else:
            kind, val = q.get() if not q.empty() else ("err", f"exited rc={p.exitcode}")
            row.update(verdict="ok" if kind == "ok" else "error",
                       **(val if kind == "ok" else {"note": val}))
        if row["verdict"] != "ok":
            row["reset"] = reset(args.card)
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(args.out, "w") as fh:
            json.dump({"watchdog_s": args.watchdog, "arms": args.arms, "rows": rows,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
