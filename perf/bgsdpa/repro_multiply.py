"""Standalone repro attempt for the ttnn.multiply_ hang the 2100-residue BoltzGen design parks in.

The fold logs 400 returning calls and then enters call 401 with a = b = [1, 16, 160, 256] bf16
TILE L1 and never leaves. 160 is the ragged W tail of BoltzGen's padded 2208 under
TRANSITION_W_CHUNK_SIZE = 1024 (1024 + 1024 + 160). This walks the shapes around it, with a
watchdog, so a hang is reported as a hang instead of hanging this script too.

    python3 perf/bgsdpa/repro_multiply.py --out perf/bgsdpa/repro_multiply.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time


def one(shape, q):
    import torch
    import ttnn
    sys.path.insert(0, os.environ.get("WT", os.getcwd()))
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    try:
        t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        a = ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = ttnn.multiply_(a, b)
        ttnn.synchronize_device(dev)
        q.put(("ok", round((time.perf_counter() - t0) * 1e3, 3)))
        ttnn.deallocate(o)
    except Exception as exc:                                          # noqa: BLE001
        q.put(("err", str(exc).splitlines()[0][:200]))
    finally:
        try:
            ttnn.close_device(dev)
        except Exception:                                             # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchdog", type=int, default=90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # The suspect shape runs LAST: a killed hang leaves the chip dirty for whatever opens it
    # next, so every control has to be collected before one is provoked.
    shapes = [[1, 16, 1024, 256],   # a chunk the fold completes, the control
              [1, 16, 192, 256],    # one tile wider than the tail
              [1, 16, 128, 256],    # one tile narrower than the tail
              [1, 16, 160, 128],    # the tail width, half the channels
              [1, 16, 160, 256]]    # the chunk the fold hangs on
    rows = []
    for sh in shapes:
        q = mp.Queue()
        p = mp.Process(target=one, args=(sh, q))
        p.start()
        p.join(args.watchdog)
        if p.is_alive():
            p.kill()
            p.join()
            rows.append({"shape": sh, "verdict": "HANG",
                         "note": f"no result in {args.watchdog}s, killed"})
        else:
            kind, val = q.get() if not q.empty() else ("err", f"exited rc={p.exitcode}")
            rows.append({"shape": sh, "verdict": "ok" if kind == "ok" else "error",
                         "ms" if kind == "ok" else "note": val})
        print(json.dumps(rows[-1]), flush=True)
    with open(args.out, "w") as fh:
        json.dump({"watchdog_s": args.watchdog, "rows": rows,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
