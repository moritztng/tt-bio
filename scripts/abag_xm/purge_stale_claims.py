"""Purge the claims, output dirs and logs of every cell in a galaxy window that has no rc=0
record, so a relaunch of that window's fleet script re-folds exactly the incomplete cells.

Was p31_purge_stale_claims.py with the window hardcoded. p32 needs the same treatment: its 48
large-target cells run under the same claim mechanism, and taking p32 down (which the boundary
sequence does, because the watchdog auto-launches it) leaves one poisoned claim per in-flight
fold. Usage:

    python3 purge_stale_claims.py p31
    python3 purge_stale_claims.py p32

Safe on a LIVE window since 2026-08-10: every cell with a fold running right now is skipped, so
neither its claim nor its outdir is touched. That is what lets a stranded residual be re-exposed
to the running driver's slots instead of waiting for the window to end. Pass --dry-run to report
without changing anything; there is no DRY env var, so a typo there purges for real.
"""
import json, pathlib, re, shutil, subprocess, sys

if not 2 <= len(sys.argv) <= 3 or not sys.argv[1].startswith("p") \
        or sys.argv[2:] not in ([], ["--dry-run"]):
    sys.exit("usage: purge_stale_claims.py <window> [--dry-run]   e.g. p31, p32")
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

DRY = bool(sys.argv[2:] and sys.argv[2] == "--dry-run")

# Live-fold exclusion. The fold command line carries --out_dir <B>/<mdir>/<target>_c<chunk>, which
# identifies the cell exactly, so the process table is an authoritative liveness oracle. mmap-only
# holds make fd counting a false negative, but a fold always has its own argv.
live = set()
for line in subprocess.run(["ps", "-eo", "args="], capture_output=True,
                           text=True).stdout.splitlines():
    mo = re.search(r"--out_dir\s+(\S+)", line)
    if not mo:
        continue
    p = pathlib.Path(mo.group(1))
    mo = re.match(r"(.+)_c(\d+)$", p.name)
    if mo and p.parent.parent == B:            # ignore another window's folds
        live.add((p.parent.name, mo.group(1), int(mo.group(2))))

tasks = (B / "tasks.txt").read_text().splitlines()
n_claim = n_dir = n_log = n_try = n_live = 0
for i, line in enumerate(tasks, 1):
    m, t, rung, seed, c, k = line.split()
    if (m, t, int(c)) in ok:
        continue
    if (MDIR[m], t, int(c)) in live:
        n_live += 1
        continue
    # Release the claim LAST. On a live window the claim is what gates a slot from starting this
    # cell, so deleting it first opens a race where a slot wins the claim and begins folding into
    # an out_dir this loop is about to delete.
    d = B / MDIR[m] / (t + "_c" + c)
    if d.exists():
        if not DRY:
            shutil.rmtree(d)
        n_dir += 1
    for log in B.glob(MDIR[m] + "_" + t + "_c" + c + "*.log"):
        if not DRY:
            log.unlink()
        n_log += 1
    # the retry counter lives outside the claim dir so it survives a release; a purge is an
    # operator decision to start the cell over, so the counter goes too
    tries = B / "tries" / str(i)
    if tries.exists():
        if not DRY:
            tries.unlink()
        n_try += 1
    claim = B / "claims" / str(i)
    if claim.exists():
        if not DRY:
            shutil.rmtree(claim)
        n_claim += 1
print("%s: %d claims, %d tries, %d outdirs, %d logs; %d live cells left alone"
      % ("would purge" if DRY else "purged", n_claim, n_try, n_dir, n_log, n_live))
print("ok cells:", len(ok & {(m, t, int(c)) for m, t, _, _, c, _ in
                             (l.split() for l in tasks)}), "of", len(tasks))
