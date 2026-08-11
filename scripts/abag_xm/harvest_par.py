#!/usr/bin/env python3
"""Parallel pre-warmer for the galaxy -> qb1 fold harvest.

p25_harvest.sh pulls one fold dir at a time, each rsync opening its own ssh +
cloudflared handshake through the japanfold tunnel. Measured 2026-08-10 on the p31
harvest: 8.6 s per 24 MiB fold dir, i.e. 2.8 MiB/s, and 2301 remaining cells = 5.5 h.
The tunnel is not the limit and neither is the qb1 sshfs mount (66 MB/s writes). The
cost is per-dir connection setup, and it parallelises:

    1 stream, own connection      6.8 s/dir
    4 streams, own connections    1.7 s/dir
    8 streams, own connections    1.05 s/dir
   12 streams, own connections    connection REFUSED on 1 of 12
    8 streams, one mux connection 0.58 s/dir   <- what this script does

So: one ssh ControlMaster, JOBS rsyncs multiplexed over it. 15x, and a single tunnel
handshake instead of 2301 of them. JOBS stays under sshd's default MaxSessions=10;
12 sessions on one master gets "Session open refused by peer".

This is a PRE-WARMER, not a replacement. It writes fold dirs and nothing else: no
sentinel, no state file, no reused_chunks.jsonl, no label propagation. Those belong to
watch_galaxy_drain_harvest.sh, which owns the chain. Run this ALONGSIDE the serial
harvest -- never kill that one. Its parent watcher treats a non-zero harvest exit as
FATAL, never writes deepn_harvested_<run>, and the label watcher then polls forever for
a state file that never appears. Let the serial harvest run; this script fills the slots
ahead of it, and it skips every complete slot instantly, so it finishes early and exits
0 with the chain intact.

Completeness is verified exactly as p25_harvest.sh does it (results.json status ok, one
CIF per all_runs entry). A slot that will not verify after --retries is removed, so it
is absent rather than half-present, and the completeness gate sees it.

The work list comes from `fleet_results.jsonl` in DEST, so a stale ledger used to mean
silent under-harvest: the fleet keeps finishing folds on the galaxy, the destination
ledger does not hear about them, and this script reports "0 to pull" while completed
cells sit unharvested. It is self-sealing, because nothing else refreshes the ledger
either, and it does not read as a harvest problem downstream -- the absent cell reads as
a fold that never succeeded. On 2026-08-11 that turned a 0.7 min network copy of 23
completed cells into a proposed 1 h JapanFold folding outage to refold them.

So the ledger is now refreshed from the galaxy's own `<run>/results.jsonl` before the
work list is built, appending only lines the destination is short of (multiset deficit,
so a legitimate repeated attempt is preserved and nothing is rewritten). Append-only, to
keep this safe against the other writers of that file. `--no-ledger-refresh` opts out.

usage:
  DEST=/home/moritz/qb1_galaxy python3 scripts/abag_xm/harvest_par.py p31 512
  ... --jobs 8 --models protenix-v2,esmfold2,opendde-abag,boltz2 --chunks 4-7
"""
import argparse
import collections
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

MD = {"boltz2": "boltz2", "opendde-abag": "opendde",
      "protenix-v2": "protenix", "esmfold2": "esmfold2"}
# A slot that exists, is incomplete, and was touched this recently is assumed to be a
# live rsync from the serial harvest. Skip it rather than race it for the same bytes.
FRESH_S = 120


def parse_chunks(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def slot_complete(rd):
    """results.json says ok and every all_runs entry has its CIF on disk."""
    rj = rd / "results.json"
    if not rj.exists():
        return False
    try:
        rec = json.loads(rj.read_text())[0]
        n = len(rec.get("all_runs") or [])
        return (rec.get("status") == "ok" and n > 0
                and len(list((rd / "structures").glob("*.cif"))) == n)
    except Exception:
        return False


def refresh_ledger(gal, run, fleet):
    """Append the galaxy run ledger's records that DEST is short of. Returns lines added.

    Keyed on the exact line and counted as a multiset, so an attempt that legitimately
    repeats (same target, chunk, seed, seconds) survives instead of being deduped away,
    and a line already present is never re-appended.
    """
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", gal,
                        f"cat /home/cust-team/mthuening/{run}/results.jsonl"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ledger refresh SKIPPED: cannot read {run}/results.jsonl on {gal}")
        return 0

    def wellformed(text):
        out = []
        for line in text.splitlines():            # the fleet appends concurrently: torn lines exist
            if not line.startswith("{"):
                continue
            try:
                json.loads(line)
            except ValueError:
                continue
            out.append(line)
        return out

    have = collections.Counter(wellformed(fleet.read_text() if fleet.exists() else ""))
    add = []
    for line, n in collections.Counter(wellformed(r.stdout)).items():
        add.extend([line] * (n - have.get(line, 0)))
    if add:
        with fleet.open("a") as fh:               # append-only; never rewrite a shared ledger
            fh.write("".join(x + "\n" for x in add))
    print(f"  ledger refresh: +{len(add)} record(s) from {run}/results.jsonl")
    return len(add)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("rung", type=int)
    ap.add_argument("--dest", default=os.environ.get("DEST", ""))
    ap.add_argument("--gal", default=os.environ.get("GAL", "japanfold-ssh"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--chunks", default="0-7")
    # boltz2 last on purpose: on 2026-08-10 the serial harvest was working through
    # boltz2, so leaving it for the end means its slots are already complete by the
    # time we get there and we skip them for free.
    ap.add_argument("--models",
                    default="protenix-v2,esmfold2,opendde-abag,boltz2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ledger-refresh", action="store_true",
                    help="trust DEST's ledger as-is; it can then hide a completed fold")
    args = ap.parse_args()

    if not args.dest:
        sys.exit("DEST is required (the qb1 analysis tree, e.g. /home/moritz/qb1_galaxy). "
                 "Defaulting it to $HOME would write a 57 GiB harvest onto pc's 12 GiB.")
    dest = pathlib.Path(args.dest)
    if not dest.is_dir():
        sys.exit(f"DEST {dest} is not a directory (sshfs mount down?)")
    fleet = dest / "fleet_results.jsonl"
    if not fleet.exists():
        sys.exit(f"no {fleet}; run p25_harvest.sh once first so it stages the ledger")

    models = args.models.split(",")
    chunks = parse_chunks(args.chunks)
    gb = f"/home/cust-team/mthuening/{args.run}"

    if not args.no_ledger_refresh:
        refresh_ledger(args.gal, args.run, fleet)

    ok = {}
    for line in fleet.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("rc") == 0 and r.get("cifs", 0) > 0 and r.get("rung") == args.rung:
            ok[(r["model"], r["target"], r.get("chunk"))] = r.get("chunks", 1)

    work = []
    skipped_fresh = 0
    for model in models:                       # model order is the collision guard
        for (m, t, ch), nch in sorted(ok.items()):
            if m != model or ch is None or nch <= 1 or ch not in chunks:
                continue
            mdir = MD[m]
            out = dest / mdir / f"{t}_n{args.rung}_c{ch}"
            rd = out / f"{mdir}_results_{t}"
            if slot_complete(rd):
                continue
            if out.exists():
                try:
                    if time.time() - out.stat().st_mtime < FRESH_S:
                        skipped_fresh += 1
                        continue
                except OSError:
                    pass
            work.append((m, t, ch, f"{gb}/{mdir}/{t}_c{ch}/{mdir}_results_{t}/", rd, out))

    print(f"harvest_par: {len(ok)} ok cells at rung {args.rung}; {len(work)} to pull "
          f"(jobs {args.jobs}, chunks {sorted(chunks)}, {skipped_fresh} skipped as live)")
    if args.dry_run or not work:
        for m, t, ch, src, _, _ in work[:20]:
            print(f"  would pull {m} {t} c{ch}")
        return 0

    cm = f"/tmp/ssh-cm/harvest_par.{args.run}.{os.getpid()}.sock"
    pathlib.Path("/tmp/ssh-cm").mkdir(exist_ok=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-M", "-S", cm,
                    "-o", "ControlPersist=900", "-fN", args.gal], check=True)
    lock = threading.Lock()
    state = {"done": 0, "failed": [], "t0": time.time()}

    def pull(item):
        m, t, ch, src, rd, out = item
        for attempt in range(args.retries + 1):
            out.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                ["rsync", "-az", "--timeout=300", "-e", f"ssh -S {cm}",
                 f"{args.gal}:{src}", str(rd) + "/"],
                capture_output=True, text=True)
            if r.returncode == 0 and slot_complete(rd):
                with lock:
                    state["done"] += 1
                    n, el = state["done"], time.time() - state["t0"]
                    if n % 50 == 0 or n == len(work):
                        pct = 100.0 * n / len(work)
                        rate = n / el if el else 0
                        eta = (len(work) - n) / rate / 60 if rate else 0
                        print(f"  {n}/{len(work)} ({pct:.1f} pct) "
                              f"{rate:.2f} dir/s eta {eta:.0f} min", flush=True)
                return True
            time.sleep(2 * (attempt + 1))
        # Never leave a half-present slot: the completeness gate must see it missing.
        shutil.rmtree(out, ignore_errors=True)
        with lock:
            state["failed"].append(f"{m} {t} c{ch}")
        return False

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(pull, work))
    finally:
        subprocess.run(["ssh", "-S", cm, "-O", "exit", args.gal],
                       capture_output=True)

    el = time.time() - state["t0"]
    print(f"harvest_par: {state['done']}/{len(work)} pulled in {el/60:.1f} min "
          f"({el/max(state['done'],1):.2f} s/dir)")
    if state["failed"]:
        print(f"dropped {len(state['failed'])} slots (removed, not half-present):")
        for f in state["failed"]:
            print(f"  DROP {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
