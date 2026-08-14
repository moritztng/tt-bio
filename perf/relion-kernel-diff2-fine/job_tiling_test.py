"""Does the job list tile [0, significant_num) exactly?

The fine hook's whole design rests on this: RELION's diff2_fine_2D writes
g_diff2s[job_idx[bid] + itrans] for itrans in [0, job_num[bid]), and the hook instead writes
out[w] for w in [0, significant_num). Those are the same set of entries, carrying the same
(rot, trans) pair, only if the jobs partition [0, w) in order.

makeJobsForDiff2Fine is transcribed here from acc_helper_functions_impl.h:30-105 and driven with
random significance masks over the shapes the fine pass actually sees.
"""
import numpy as np

CHUNK = 7          # D2F_CHUNK_REF3D, cpu_settings.h:40


def make_jobs(sig, O, T, chunk=CHUNK):
    """sig[i, j] == 1 where the coarse pass marked (i, j) significant. Returns the same three
    arrays RELION builds: rot_idx, trans_idx (length w) and jobOrigin, jobExtent (length k)."""
    rot_idx, trans_idx = [], []
    job_origin, job_extent = [0], [0]
    w = 0
    k = 0
    for i in range(O):
        job_extent[k] = 0
        tk = 0
        for j in range(T):
            if sig[i, j]:
                rot_idx.append(i)
                trans_idx.append(j)
                if tk >= chunk:
                    tk = 0
                    k += 1
                    job_origin.append(w)
                    job_extent.append(0)
                tk += 1
                job_extent[k] += 1
                w += 1
            elif tk != 0:
                tk = 0
                k += 1
                job_origin.append(w)
                job_extent.append(0)
        if tk > 0:
            k += 1
            job_origin.append(w)
            job_extent.append(0)
    if job_extent[k] != 0:
        k += 1
    return (np.array(rot_idx, dtype=np.uint64), np.array(trans_idx, dtype=np.uint64),
            np.array(job_origin[:k], dtype=np.uint64), np.array(job_extent[:k], dtype=np.uint64), w)


def relion_write(dense, rot_idx, trans_idx, job_origin, job_extent, w, sum_init):
    """diff2_fine_2D's output loop: per job, one orientation, a run of consecutive translations."""
    out = np.zeros(w, dtype=np.float64)
    for bid in range(len(job_origin)):
        o = int(rot_idx[job_origin[bid]])
        t0 = int(trans_idx[job_origin[bid]])
        for itrans in range(int(job_extent[bid])):
            out[int(job_origin[bid]) + itrans] += dense[o, t0 + itrans] + sum_init
    return out


def hook_write(dense, rot_idx, trans_idx, w, sum_init):
    """The hook: read the requested entries straight out of the dense matrix."""
    flat = dense.reshape(-1)
    take = rot_idx * np.uint64(dense.shape[1]) + trans_idx
    return flat[take] + sum_init


rng = np.random.default_rng(20260814)
bad = 0
for trial in range(4000):
    O = int(rng.integers(1, 65))          # 8 x (1..8) significant coarse rotations
    T = int(rng.choice([36, 36, 36, 4, 16, 64]))
    p = float(rng.choice([0.02, 0.1, 0.3, 0.7, 1.0]))
    # significance is inherited from the coarse mask, so it repeats in blocks of nr_over_trans=4
    coarse_T = max(1, T // 4)
    cs = (rng.random((O, coarse_T)) < p).astype(np.uint8)
    sig = np.repeat(cs, T // coarse_T, axis=1)[:, :T]
    if sig.sum() == 0:
        continue
    rot_idx, trans_idx, jo, je, w = make_jobs(sig, O, T)
    dense = rng.standard_normal((O, T)) * 100.0
    sum_init = float(rng.standard_normal())
    a = relion_write(dense, rot_idx, trans_idx, jo, je, w, sum_init)
    b = hook_write(dense, rot_idx, trans_idx, w, sum_init)
    assert int(je.sum()) == w, (trial, int(je.sum()), w)          # jobs tile exactly
    if not np.array_equal(a, b):
        bad += 1
        print("MISMATCH", trial, O, T, p)
print("trials with a mismatch:", bad)
print("jobs tile [0, significant_num) in every trial: OK")

# The other number the design turns on: jobs per distinct orientation.
for p in (0.02, 0.1, 0.3, 0.7, 1.0):
    ratios = []
    for _ in range(200):
        O, T, coarse_T = 24, 36, 9
        cs = (rng.random((O // 8, coarse_T)) < p).astype(np.uint8)
        cs = np.repeat(cs, 8, axis=0)                # all 8 oversampled rots share the coarse mask
        sig = np.repeat(cs, 4, axis=1)
        if sig.sum() == 0:
            continue
        _, _, jo, je, w = make_jobs(sig, O, T)
        live_o = int((sig.sum(1) > 0).sum())
        ratios.append(len(jo) / live_o)
    print("coarse significance p=%.2f  jobs per live orientation = %.2f" % (p, np.mean(ratios)))
