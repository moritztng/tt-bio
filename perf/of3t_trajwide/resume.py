"""Per-rung resume, because qb2 goes down about every 40 minutes under this load.

qb2 crashed twice on 2026-09-21 while this row was running -- 12:45:32Z and 13:27:09Z, both
preceded by ~60 s of `tenstorrent 0000:04:00.0: Failed to set initial power state: -5` on card
3, a card this row never opened. A 20-rung arm is 61 minutes on the device side and 2.35 hours
on the reference side, so without resume a rebooting host never finishes an arm no matter how
many times it is relaunched: every launch restarts at k=1 and dies before k=20.

The w_k dumps cannot be resumed from. They are float32 casts of float64 (reference) or of the
float32 master (ours), written for the scorer, and restoring a run from them would change the
trajectory. So a rung writes a SEPARATE exact checkpoint, `resume.pt` / `resume.npz`, holding
the state the next step actually reads: weights in their native dtype, both Adam moments, the
step counter, the scheduler, and the log rows so far.

Every restore is ASSERTED against the k it claims to be at: the restored weights, put through
the same cast the dump used, must equal `k{n}.npz` element for element. A resume that lands on
different weights is a silent trajectory fork, which is the one failure this row cannot absorb,
so it raises instead.

Written atomically (`.part` then `os.replace`) and kept at one file per arm: the checkpoint is
1-2 GB and only the newest rung is resumable.
"""
import json
import os

import numpy as np


def _atomic(path, write):
    tmp = path + ".part"
    write(tmp)
    os.replace(tmp, path)


def _quarantine(path, exc):
    """A checkpoint that does not restore is retired, loudly, so the relaunch starts clean.

    Without this a bad checkpoint is a trap: the supervisor restarts the arm, the restore
    raises again, and the arm loops forever at rc!=0 while the log says the same thing every
    five minutes. Renaming it means the next launch runs from k=1 and the evidence is kept.
    """
    bad = path + ".BAD"
    try:
        os.replace(path, bad)
    except OSError:
        pass
    raise AssertionError(f"{exc}\n  checkpoint retired to {bad}; the relaunch will start "
                         f"from k=1")


def _assert_matches_dump(d, k, got, what):
    """The restored state, cast the way the dump was cast, IS the dump at k. Not assumed."""
    p = os.path.join(d, f"k{k:02d}.npz")
    if not os.path.exists(p):
        raise AssertionError(f"{what}: resume claims k={k} but {p} does not exist")
    with np.load(p) as z:
        want = {n: z[n] for n in z.files}
    if set(want) != set(got):
        raise AssertionError(f"{what}: resume restored {len(got)} tensors, dump k={k} has "
                             f"{len(want)}; symmetric difference "
                             f"{sorted(set(want) ^ set(got))[:8]}")
    bad = [n for n in want if not np.array_equal(want[n], got[n])]
    if bad:
        n = bad[0]
        raise AssertionError(f"{what}: resume at k={k} does not reproduce the dump on "
                             f"{len(bad)} of {len(want)} tensors, worst example {n} "
                             f"max|d|={float(np.abs(want[n] - got[n]).max()):.6e}")
    print(f"resume: restored k={k} and it reproduces {len(want)}/{len(want)} dumped tensors "
          f"exactly", flush=True)


def clear_dumps(d):
    """An arm that starts at k=1 owns its dump directory, so it empties it first.

    Otherwise a run that dies at k=5 leaves k01..k05 from this epoch beside k06..k20 from the
    last one, and `--score` intersects `have_steps` across arms without knowing the difference.
    Two epochs of the same deterministic program probably agree; `probably` is not a reference.
    """
    import glob
    gone = 0
    for f in glob.glob(os.path.join(d, "k??.npz")) + glob.glob(os.path.join(d, "k??.part.npz")):
        os.remove(f)
        gone += 1
    if gone:
        print(f"resume: no checkpoint, starting at k=1 and clearing {gone} dumps from a "
              f"previous epoch of this arm", flush=True)


# ------------------------------------------------------------------- our side (numpy AdamW)

_OURS = "resume.npz"


def save_ours(d, k, opt, log, stale_hold, fwd_rel):
    meta = {"k": k, "steps": int(opt.steps), "log": log, "fwd_rel": fwd_rel,
            "has_stale_hold": stale_hold is not None}
    arrs = {}
    for tag, src in (("master", opt.master), ("m", opt.exp_avg), ("v", opt.exp_avg_sq)):
        for n, a in src.items():
            arrs[f"{tag}//{n}"] = a
    if stale_hold is not None:
        for n, a in stale_hold.items():
            arrs[f"stale//{n}"] = a
    # `np.savez` appends `.npz`, so the part file has to end in it too, same trap `save_step`
    # already documents.
    dst = os.path.join(d, _OURS)
    tmp = dst + ".part.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, dst)


def load_ours(d, opt, params, to_device, log, orient):
    """Returns (k_done, stale_hold, fwd_rel). (0, None, None) when there is nothing to resume."""
    p = os.path.join(d, _OURS)
    if not os.path.exists(p):
        clear_dumps(d)
        return 0, None, None
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        buckets = {"master": {}, "m": {}, "v": {}, "stale": {}}
        for key in z.files:
            if key == "__meta__":
                continue
            tag, n = key.split("//", 1)
            buckets[tag][n] = z[key]
    k = int(meta["k"])
    opt.master = buckets["master"]
    opt.exp_avg = buckets["m"]
    opt.exp_avg_sq = buckets["v"]
    opt.steps = int(meta["steps"])
    for n, t in params.items():
        t.value = to_device(opt.master[n], t.value.device(), dtype=t.value.dtype)
    params.rebind()
    try:
        _assert_matches_dump(d, k, orient(), "ours")
    except AssertionError as e:
        _quarantine(p, e)
    log.extend(meta["log"])
    return k, (buckets["stale"] or None) if meta["has_stale_hold"] else None, meta["fwd_rel"]


# ------------------------------------------------------- reference side (torch module + Adam)

_THEIRS = "resume.pt"


def save_theirs(d, k, A, B, log, aa_rows):
    import torch
    blob = {"k": k, "log": log, "aa_rows": aa_rows,
            "A": {"m": A["m"].state_dict(), "opt": A["opt"].state_dict(),
                  "sch": A["sch"].state_dict()}}
    if B is not None:
        blob["B"] = {"m": B["m"].state_dict(), "opt": B["opt"].state_dict(),
                     "sch": B["sch"].state_dict()}
    _atomic(os.path.join(d, _THEIRS), lambda p: torch.save(blob, p))


def load_theirs(d, A, B, log, aa_rows):
    import torch
    p = os.path.join(d, _THEIRS)
    if not os.path.exists(p):
        clear_dumps(d)
        return 0
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if ("B" in blob) != (B is not None):
        raise AssertionError(f"theirs: resume has B={'B' in blob} but this run has "
                             f"B={B is not None}; the A/A arm cannot resume a non-A/A run")
    for E, key in ((A, "A"), (B, "B")):
        if E is None:
            continue
        E["m"].load_state_dict(blob[key]["m"])
        E["opt"].load_state_dict(blob[key]["opt"])
        E["sch"].load_state_dict(blob[key]["sch"])
    k = int(blob["k"])
    got = {n: pp.detach().to(torch.float64).numpy().astype(np.float32)
           for n, pp in A["params"].items()}
    try:
        _assert_matches_dump(d, k, got, "theirs")
    except AssertionError as e:
        _quarantine(p, e)
    log.extend(blob["log"])
    aa_rows.extend(blob["aa_rows"])
    return k
