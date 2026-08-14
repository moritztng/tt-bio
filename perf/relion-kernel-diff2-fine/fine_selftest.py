"""diff2_fine against diff2_coarse on the same synthetic inputs.

The fine entry point reads a sparse subset out of the same dense matrix the coarse entry point
returns whole, so on identical inputs every requested entry must equal the coarse entry at
(rot_idx, trans_idx) plus sum_init, to the bit. This does not test the numerics against RELION --
that is the in-situ TT_RELION_CHECK arm -- it tests that the gather indexes what it claims to.
"""
import os
import pathlib
import sys

os.environ["TT_RELION_BACKEND"] = "torch"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from tt_bio.cryoem import relion as R

rng = np.random.default_rng(7)
mdlX, mdlY, mdlZ = 20, 39, 39
mdlInitY, mdlInitZ = -19, -19
imgX, imgY = 10, 18
image_size = imgX * imgY
maxR = 8
maxR2_padded = (2 * maxR) ** 2
pf = 2.0
O, T = 12, 36

mdl = rng.standard_normal(mdlX * mdlY * mdlZ * 2).astype(np.float32)
# random rotation matrices, row-major 3x3 as RELION stores them
eul = np.empty((O, 9), dtype=np.float32)
for i in range(O):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    eul[i] = (q * np.sign(np.linalg.det(q))).reshape(-1)
tx = (rng.standard_normal(T) * 0.3).astype(np.float32)
ty = (rng.standard_normal(T) * 0.3).astype(np.float32)
img_r = rng.standard_normal(image_size).astype(np.float32)
img_i = rng.standard_normal(image_size).astype(np.float32)
corr = np.abs(rng.standard_normal(image_size)).astype(np.float32)

geom = dict(mdlX=mdlX, mdlY=mdlY, mdlZ=mdlZ, mdlInitY=mdlInitY, mdlInitZ=mdlInitZ,
            maxR=maxR, maxR2_padded=maxR2_padded, padding_factor=pf, imgX=imgX, imgY=imgY)

dense = np.zeros(O * T, dtype=np.float32)
ok = R.diff2_coarse(memoryview(mdl), memoryview(eul.reshape(-1)), memoryview(tx), memoryview(ty),
                    memoryview(img_r), memoryview(img_i), memoryview(corr), memoryview(dense),
                    geom["mdlX"], geom["mdlY"], geom["mdlZ"], geom["mdlInitY"], geom["mdlInitZ"],
                    geom["maxR"], geom["maxR2_padded"], geom["padding_factor"],
                    geom["imgX"], geom["imgY"], O, T, image_size)
assert ok, "coarse declined"

# a significance mask with the structure the fine pass has: inherited in blocks of 4 translations
cs = (rng.random((O, T // 4)) < 0.35).astype(np.uint8)
sig = np.repeat(cs, 4, axis=1)
ridx, tidx = np.nonzero(sig)
rot_idx = ridx.astype(np.uint64)
trans_idx = tidx.astype(np.uint64)
n = len(rot_idx)
sum_init = 3.5

out = np.zeros(n, dtype=np.float32)
ok = R.diff2_fine(memoryview(mdl), memoryview(eul.reshape(-1)), memoryview(tx), memoryview(ty),
                  memoryview(img_r), memoryview(img_i), memoryview(corr),
                  memoryview(rot_idx), memoryview(trans_idx), memoryview(out),
                  geom["mdlX"], geom["mdlY"], geom["mdlZ"], geom["mdlInitY"], geom["mdlInitZ"],
                  geom["maxR"], geom["maxR2_padded"], geom["padding_factor"],
                  geom["imgX"], geom["imgY"], sum_init, O, T, n, image_size, 0)
assert ok, "fine declined"

want = dense.reshape(O, T)[ridx, tidx] + np.float32(sum_init)
print("significant entries:", n, "of", O * T)
print("bit-identical to the coarse matrix + sum_init:", np.array_equal(out, want))
print("max |delta|:", float(np.max(np.abs(out.astype(np.float64) - want.astype(np.float64)))))
print("stats:", R.stats())

# the accumulate contract: RELION zero-inits and both kernels use +=, so a second call doubles
out2 = out.copy()
R.diff2_fine(memoryview(mdl), memoryview(eul.reshape(-1)), memoryview(tx), memoryview(ty),
             memoryview(img_r), memoryview(img_i), memoryview(corr),
             memoryview(rot_idx), memoryview(trans_idx), memoryview(out2),
             geom["mdlX"], geom["mdlY"], geom["mdlZ"], geom["mdlInitY"], geom["mdlInitZ"],
             geom["maxR"], geom["maxR2_padded"], geom["padding_factor"],
             geom["imgX"], geom["imgY"], sum_init, O, T, n, image_size, 0)
print("accumulates rather than assigns:", np.allclose(out2, 2 * out, rtol=0, atol=0))
