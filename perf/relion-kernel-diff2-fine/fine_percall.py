#!/usr/bin/env python3
"""What a single diff2 call actually costs, on an idle box, one thread.

The refinement arms give a per-call cost via RELION's own Timer (thread-seconds / calls). That
number is collected at 30 busy threads on a 32-core box, through a bridge that holds the GIL. This
prices the same kernel alone so the two can be subtracted: if the isolated call is far cheaper than
the in-refinement call, the bridge loses its time to contention rather than to arithmetic.

Inputs are RELION's own, from the p8 dump, so the model, the euler set and the image are real.
"""
import glob
import os
import sys
import time

import numpy as np

os.environ.setdefault("TT_RELION_TORCH_THREADS", "1")
sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-kernel-diff2-fine")

from tt_bio.cryoem import relion as R  # noqa: E402

d = np.load(sorted(glob.glob("/home/ttuser/relion-scratch/p8/call.*.npz"))[0])
g = {k: int(v) for k, v in zip(
    "mdlX mdlY mdlZ mdlInitY mdlInitZ maxR maxR2_padded padding_factor imgX imgY "
    "orientation_num translation_num image_size".split(), d["geom"])}

mdl = d["mdl"].astype(np.float32)
eul = d["eul"].astype(np.float32)
img_r, img_i = d["img_r"].astype(np.float32), d["img_i"].astype(np.float32)
# _dense_diff2 halves corr itself, and the dump stored the halved copy.
corr = (d["w"].astype(np.float32) * 2.0)

t = R._torch()
print("torch threads:", __import__("torch").get_num_threads())
print("model %.1f MB, image_size %d" % (mdl.nbytes / 1e6, g["image_size"]))


def run(O, T, reps=3):
    e = np.ascontiguousarray(eul[:O]).astype(np.float32)
    tx = np.ascontiguousarray(np.resize(d["tx"].astype(np.float32), T))
    ty = np.ascontiguousarray(np.resize(d["ty"].astype(np.float32), T))
    args = (mdl.tobytes(), e.tobytes(), tx.tobytes(), ty.tobytes(),
            img_r.tobytes(), img_i.tobytes(), corr.tobytes(),
            g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"],
            g["maxR"], g["maxR2_padded"], g["padding_factor"], g["imgX"], g["imgY"],
            O, T, g["image_size"], "B")
    R._dense_diff2(t, *args)                     # warm
    best = min(_time_once(args) for _ in range(reps))
    print("  O=%-4d T=%-3d  %.3f s/call" % (O, T, best))
    return best


def _time_once(args):
    t0 = time.perf_counter()
    R._dense_diff2(t, *args)
    return time.perf_counter() - t0


print("\n-- our kernel, isolated, 1 thread --")
coarse = run(180, 9)
fine = run(48, 36)

print("\n-- model copy alone --")
t0 = time.perf_counter()
for _ in range(10):
    x = np.frombuffer(mdl.tobytes(), dtype=np.float32).reshape(-1, 2).copy()
cp = (time.perf_counter() - t0) / 10
print("  %.4f s/call  (%.1f%% of the coarse call, %.1f%% of the fine call)"
      % (cp, 100 * cp / coarse, 100 * cp / fine))

print("\n-- against the refinement's own Timer --")
for name, iso, inref, relion in (("coarse", coarse, 31277.65 / 5568, 6265.52 / 5568),
                                 ("fine", fine, 19254.56 / 5568, 556.30 / 5568)):
    print("  %-6s isolated %.3f s | in-refinement %.3f s (%.1fx) | RELION CPU %.3f s"
          % (name, iso, inref, inref / iso, relion))
