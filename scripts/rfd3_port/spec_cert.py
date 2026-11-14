#!/usr/bin/env python3
"""Certification rate of a speculative atom index on a spec_dump trajectory.

    python scripts/rfd3_port/spec_cert.py DUMP.npz [--kp 160,192,256] [--threads 2]

During the last recycle of step n the host is idle, and recycle 0 has already predicted the next step's coordinates:
X_spec = X_act + step_scale * d_t / t_hat * (X_r1 - X_r0) (the noise for step n+1 can be drawn early, the generator
stream is the same). Query k' > k neighbours of X_spec then; at step n+1 rescore only those candidates on X_act.
Row r is exact when its k-th rescored distance D_k < d'_spec(r) - |dX_r| - max_j |dX_j|, d'_spec(r) being its k'-th
speculative distance: every atom outside the candidates is then farther than D_k on X_act. Rows that fail go to a
real query. Prints the certified fraction per step band, the rescore cost and checks certified rows against cKDTree.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--kp", default="160,192,256")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--every", type=int, default=1, help="analyse every n-th step")
a = ap.parse_args()
z = np.load(a.dump)
xn, xr, th = z["x_noisy"].astype(np.float64), z["x_rec"].astype(np.float64), z["t_hat"]
smp = RFD3Sampler(num_timesteps=len(th) + 1)
sched = smp.noise_schedule("cpu").double().numpy()
L, k = xn.shape[1], a.k


def rescore(x, cand, k):
    d = ((x[cand] - x[:, None, :]) ** 2).sum(-1)               # [L, k'] squared, fp64
    part = np.argpartition(d, k - 1, axis=1)[:, :k]
    dk = np.sqrt(np.take_along_axis(d, part, 1).max(1))
    return np.take_along_axis(cand, part, 1), dk


out = {}
for kp in map(int, a.kp.split(",")):
    rows, ms_q, ms_r, bad = [], [], [], 0
    for n in range(0, len(th) - 1, a.every):
        fac = smp.step_scale * (sched[n + 1] - th[n]) / th[n]
        xa = xn[n + 1]
        dx = fac * (xr[n, -1] - xr[n, 0])
        xs = xa + dx
        delta = np.sqrt((dx ** 2).sum(-1))
        t = time.perf_counter()
        ds, cand = cKDTree(xs).query(xs, k=kp, workers=a.threads)
        ms_q.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        sel, dk = rescore(xa, cand, k)
        ok = dk < ds[:, -1] - delta - delta.max() - 1e-9
        ms_r.append((time.perf_counter() - t) * 1e3)
        _, truth = cKDTree(xa).query(xa, k=k, workers=a.threads)
        for r in np.flatnonzero(ok)[:: max(1, int(ok.sum()) // 400)]:
            bad += set(sel[r]) != set(truth[r])
        rows.append({"step": n, "t_hat": float(th[n]), "fac": float(fac), "dmax": float(delta.max()),
                     "dmed": float(np.median(delta)), "cert": float(ok.mean())})
    cert = np.array([r["cert"] for r in rows])
    out[kp] = {"cert_mean": float(cert.mean()), "rows_failed_per_step": float((1 - cert).mean() * L),
               "steps_all_cert": int((cert == 1).sum()), "steps": len(rows),
               "query_ms": float(np.median(ms_q)), "rescore_ms": float(np.median(ms_r)), "cert_wrong": bad,
               "bands": [{"steps": f"{i}-{i + 19}", "cert": float(cert[i // a.every:(i + 20) // a.every].mean()),
                          "dmax": float(np.median([r["dmax"] for r in rows[i // a.every:(i + 20) // a.every]]))}
                         for i in range(0, len(th) - 1, 20)]}
    print(json.dumps({kp: out[kp]}), flush=True)
