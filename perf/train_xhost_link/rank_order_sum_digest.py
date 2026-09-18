"""Rank-order float32 sum across two hosts: does every rank compute the same bits?

Mirrors hostreduce.master_hash's invariant without importing the module (that file belongs to
train-j-multihost). The arrays are built from integer arithmetic rather than an RNG so the test
is insensitive to the numpy version and only the float32 addition is under test -- qb1 and qb2
run different numpy majors.
"""
import hashlib, os, platform, sys, time
import numpy as np

N = 7_111_515          # the reproduction's trainable parameter count
WORLD = 2

def build(rank):
    i = np.arange(N, dtype=np.int64)
    q = (i * 1103515245 + 12345 + rank * 7919) % 2097152
    return (q.astype(np.float32) / np.float32(2097152.0) - np.float32(0.5)) * np.float32(1e-3)

arrs = [build(r) for r in range(WORLD)]
ih = hashlib.blake2b(digest_size=8)
for a in arrs:
    ih.update(a.tobytes())

t0 = time.perf_counter()
acc = np.zeros(N, dtype=np.float32)
for r in range(WORLD):                 # rank order 0..world-1 on every rank
    acc += arrs[r]
acc /= np.float32(WORLD)
t1 = time.perf_counter()

h = hashlib.blake2b(digest_size=16)
h.update(np.asarray(acc.shape, dtype=np.int64).tobytes())
h.update(acc.tobytes())
print(f"{platform.node()} numpy={np.__version__} py={sys.version.split()[0]} "
      f"cores={len(os.sched_getaffinity(0))}")
print(f"  inputs {ih.hexdigest()}  sum+scale {1000*(t1-t0):.1f} ms  "
      f"bytes {acc.nbytes}  digest {h.hexdigest()}")
