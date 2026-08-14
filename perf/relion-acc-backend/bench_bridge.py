"""Cost of one diff2_coarse call at RELION's real shape, so the full-iteration arm can be priced
before it is launched. Shape read off the live hook: No=186 Nt=9 P=19404, mdlZ=199."""
import os, sys, time
import numpy as np

sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-acc-backend")
import tt_bio.cryoem.relion as R

mdlX, mdlY, mdlZ = 100, 199, 199
mdlInitY, mdlInitZ = -99, -99
maxR, pf = 98, 2.0
maxR2_padded = int(maxR * maxR * pf * pf)
imgX, imgY = 99, 196
P = imgX * imgY
No, Nt = 186, 9

rng = np.random.default_rng(0)
mdl = rng.standard_normal(mdlX * mdlY * mdlZ * 2).astype(np.float32) * 0.01
eul = np.zeros((No, 9), dtype=np.float32)
for i in range(No):                      # random rotations, near-identity spread is irrelevant here
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    eul[i] = q.reshape(-1).astype(np.float32)
tx = (rng.standard_normal(Nt) * 0.05).astype(np.float32)
ty = (rng.standard_normal(Nt) * 0.05).astype(np.float32)
re = rng.standard_normal(P).astype(np.float32)
im = rng.standard_normal(P).astype(np.float32)
corr = np.abs(rng.standard_normal(P)).astype(np.float32)
out = np.zeros(No * Nt, dtype=np.float32)

args = (memoryview(mdl), memoryview(eul.reshape(-1)), memoryview(tx), memoryview(ty),
        memoryview(re), memoryview(im), memoryview(corr), memoryview(out),
        mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded, pf, imgX, imgY, No, Nt, P)

ok = R.diff2_coarse(*args)               # warm: imports torch, builds the plan
print("handled:", ok, "nonzero out:", int((out != 0).sum()), "/", out.size)
reps = 5
t0 = time.perf_counter()
for _ in range(reps):
    R.diff2_coarse(*args)
dt = (time.perf_counter() - t0) / reps
print("per call: %.1f ms  chunk=%s  torch threads=%s"
      % (dt * 1e3, os.environ.get("TT_RELION_ORI_CHUNK", "32"), __import__("torch").get_num_threads()))
print("per follower for 2226 particles, serialised: %.0f s" % (dt * 2226))
