#!/usr/bin/env python3
"""Gate 1 of the two-trajectory interleave: can the ttnn seam be driven from two threads?

The interleave does NOT need concurrent ttnn -- the card is one serial resource and a device
lock is the natural design. What it needs is weaker and still has to be true: a ttnn call made
from thread B must compute the same thing as the same call made from thread A, with the device
opened once by the main thread. A seam with thread-local device or config state fails this and
the design dies here rather than after a week of restructuring.

Arm A: both workloads on one thread, serially. The reference.
Arm B: the same two workloads on two threads, alternating under one lock, interleaved with a
host term in between so the threads really do change hands mid-stream.
Bit-exact equality of A and B is the verdict; anything else is reported as a difference and
not rounded away.
"""
import os
import sys
import threading
import time

import numpy as np
import torch
import ttnn

LOCK = threading.Lock()
N = int(os.environ.get("SEAM_N", "8"))


def work(dev, seed, out, idx):
    """One workload: a chain of matmuls on the card, returned to host as float64."""
    g = np.random.default_rng(seed)
    a = torch.tensor(g.standard_normal((256, 256)), dtype=torch.bfloat16)
    b = torch.tensor(g.standard_normal((256, 256)), dtype=torch.bfloat16)
    acc = None
    for i in range(N):
        with LOCK:
            ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
            tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
            tc = ttnn.matmul(ta, tb)
            c = ttnn.to_torch(tc)
            ttnn.deallocate(ta); ttnn.deallocate(tb); ttnn.deallocate(tc)
        acc = c if acc is None else acc + c
        a = (c / 16.0).to(torch.bfloat16)          # host work between device calls
        time.sleep(0.001)                          # force the hand-over
    out[idx] = acc.to(torch.float64).numpy()


def main():
    # NOT ttnn.open_device: a lone p300 is a CUSTOM topology and the bare call is a TT_FATAL
    # without a mesh graph descriptor. tt_bio.tenstorrent.get_device() sets it. The same trap
    # cost main c48e32670 (the host-report test opened a device without the p300 descriptor).
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    try:
        serial = [None, None]
        t0 = time.time()
        work(dev, 11, serial, 0)
        work(dev, 22, serial, 1)
        t_serial = time.time() - t0

        threaded = [None, None]
        t0 = time.time()
        ths = [threading.Thread(target=work, args=(dev, s, threaded, i))
               for i, s in enumerate((11, 22))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        t_threaded = time.time() - t0

        ok = True
        for i in (0, 1):
            d = np.abs(serial[i] - threaded[i]).max()
            same = bool((serial[i] == threaded[i]).all())
            ok &= same
            print(f"workload {i}: bit-identical={same} max|diff|={d:.6g} "
                  f"norm={np.abs(serial[i]).max():.6g}")
        print(f"serial {t_serial:.2f} s, two threads under one lock {t_threaded:.2f} s")
        print("GATE 1: PASS -- the seam is drivable from two threads under a device lock"
              if ok else
              "GATE 1: FAIL -- a threaded call did not reproduce the serial answer")
        return 0 if ok else 1
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main())
