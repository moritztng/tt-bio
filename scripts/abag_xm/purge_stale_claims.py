"""Purge the claims, output dirs and logs of every cell in a galaxy window that has no rc=0
record, so a relaunch of that window's fleet script re-folds exactly the incomplete cells.

Was p31_purge_stale_claims.py with the window hardcoded. p32 needs the same treatment: its 48
large-target cells run under the same claim mechanism, and taking p32 down (which the boundary
sequence does, because the watchdog auto-launches it) leaves one poisoned claim per in-flight
fold. Usage:

    python3 purge_stale_claims.py p31
    python3 purge_stale_claims.py p32

SAFE ONLY WITH ZERO FOLDS ALIVE IN THAT WINDOW. It deletes output dirs, and a live fold's dir is
one of them. Verify with `pgrep -af abag_x[m]` first.
"""
import json, pathlib, shutil, sys

if len(sys.argv) != 2 or not sys.argv[1].startswith("p"):
    sys.exit("usage: purge_stale_claims.py <window>   e.g. p31, p32")
B = pathlib.Path("/home/cust-team/mthuening") / sys.argv[1]
if not (B / "tasks.txt").exists():
    sys.exit("no tasks.txt under %s -- window never ran, nothing to purge" % B)

MDIR = {"boltz2": "boltz2", "opendde-abag": "opendde",
        "protenix-v2": "protenix", "esmfold2": "esmfold2"}

ok = set()
res = B / "results.jsonl"
if res.exists():
    for line in res.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rung") == 512 and r.get("rc") == 0:
            ok.add((r["model"], r["target"], r.get("chunk")))

tasks = (B / "tasks.txt").read_text().splitlines()
n_claim = n_dir = n_log = n_try = 0
for i, line in enumerate(tasks, 1):
    m, t, rung, seed, c, k = line.split()
    if (m, t, int(c)) in ok:
        continue
    claim = B / "claims" / str(i)
    if claim.exists():
        shutil.rmtree(claim)
        n_claim += 1
    # the retry counter lives outside the claim dir so it survives a release; a purge is an
    # operator decision to start the cell over, so the counter goes too
    tries = B / "tries" / str(i)
    if tries.exists():
        tries.unlink()
        n_try += 1
    d = B / MDIR[m] / (t + "_c" + c)
    if d.exists():
        shutil.rmtree(d)
        n_dir += 1
    for log in B.glob(MDIR[m] + "_" + t + "_c" + c + "*.log"):
        log.unlink()
        n_log += 1
print("purged: %d claims, %d tries, %d outdirs, %d logs" % (n_claim, n_try, n_dir, n_log))
print("ok cells:", len(ok & {(m, t, int(c)) for m, t, _, _, c, _ in
                             (l.split() for l in tasks)}), "of", len(tasks))
