"""Run walk.py jobs on whichever whglx chips come free, at most --max at a time.

chain.sh pins a job list to one card, which fails on a box where every chip is taken: the card
chosen at launch is gone by the time the walk starts, and walk.py rightly refuses it. This waits
for a lease file that is absent or released, starts the next job's walk there, and requeues a
walk that was refused before its first fold (it wrote no row).

    python perf/mgxceil/sched.py --tag P --jobs jobsP.txt --max 3

--max counts every chip this row's holder has leased, so a second sched can take over a queue
while the first one's walks finish.

Job lines are chain.sh's: <model> <sizes csv>[@depth] [tt-bio args...]. Rows land in
<scratch>/<tag>_<model>.jsonl, one file per job so a refusal is visible as an empty one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = Path.home() / "scratch" / "mgxceil"
LEASES = Path.home() / "leases"
HOST = "j10glx02"
# The live japanfold app (24-27) and a co-tenant (1). Cardblocked; never ours to take.
AVOID = {1, 24, 25, 26, 27}
ENV = {"PYTHONPATH": str(ROOT), "TT_BIO_LEASE_DIR": str(LEASES),
       "TT_BIO_LEASE_HOLDER": "worker:mgx-ceilings", "TT_METAL_LOGGER_LEVEL": "FATAL",
       "TT_METAL_CACHE": str(Path.home() / ".cache" / "tt-metal-cache-mgxceil"),
       "TT_BIO_OPENFOLD3": str(Path.home() / "mgxi-weights" / "of3-p2-155k.pt"),
       "TT_BIO_OPENBIND": str(Path.home() / "mgxi-weights" / "of3-ob-2025-06-30-174k.pt")}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def pinned() -> set[int]:
    """Cards a live process has pinned. A lease file can read released while another row's fold
    is still pinned there and has not yet taken it (card 31, 2026-09-23: a released mgx-combos
    lease under a running mgx-diffusion fold)."""
    out = set()
    for env in Path("/proc").glob("[0-9]*/environ"):
        try:
            for kv in env.read_bytes().split(b"\0"):
                k, _, v = kv.partition(b"=")
                if k in (b"TT_VISIBLE_DEVICES", b"TT_BIO_LEASE_CARDS"):
                    out.update(int(c) for c in v.split(b",") if c.strip().isdigit())
        except OSError:
            continue
    return out


def free_cards(busy: set[int]) -> list[int]:
    out = []
    busy = busy | pinned()
    for c in range(32):
        if c in AVOID or c in busy:
            continue
        try:
            meta = json.loads((LEASES / f"{HOST}-card{c}.json").read_text() or "{}")
        except FileNotFoundError:
            meta = {}
        except (OSError, ValueError):
            continue
        if not meta or meta.get("released"):
            out.append(c)
    return out


def held(holder: str) -> set[int]:
    """Chips this row holds right now, including walks a previous sched started."""
    out = set()
    for f in LEASES.glob(f"{HOST}-card*.json"):
        try:
            meta = json.loads(f.read_text() or "{}")
        except (OSError, ValueError):
            continue
        if meta.get("holder") == holder and not meta.get("released"):
            out.add(int(meta["card"]))
    return out


def lost_lease(row: dict) -> bool:
    """A first fold that died on the lease, not the model: another row took the chip between
    this walk's check and its fold's acquire."""
    if row.get("verdict") != "ERROR":
        return False
    try:
        return "lease" in Path(row["log"]).read_text(errors="replace").lower()
    except (KeyError, OSError):
        return False


def parse(path: str) -> list[tuple[str, str, int, list[str]]]:
    jobs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        model, sizes, *rest = line.split()
        depth = 8192
        if "@" in sizes:
            sizes, d = sizes.split("@")
            depth = int(d)
        jobs.append((model, sizes, depth, rest))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    pending = parse(a.jobs)
    running: dict[int, tuple[subprocess.Popen, tuple, Path]] = {}
    env = {**os.environ, **ENV}
    while pending or running:
        for card, (p, job, out) in list(running.items()):
            if p.poll() is None:
                continue
            del running[card]
            rows = [json.loads(r) for r in out.read_text().splitlines()] if out.exists() else []
            if not rows or (len(rows) == 1 and lost_lease(rows[0])):
                log(f"card {card} {job[0]}: refused before its first fold, requeued")
                if rows:
                    out.rename(out.with_suffix(f".lost{int(time.time())}"))
                pending.insert(0, job)
            else:
                log(f"card {card} {job[0]}: done, {len(rows)} rows, last {rows[-1]['verdict']}")
        if pending and len(set(running) | held(ENV["TT_BIO_LEASE_HOLDER"])) < a.max:
            cards = free_cards(set(running))
            if cards:
                card, job = cards[0], pending.pop(0)
                model, sizes, depth, rest = job
                out = S / f"{a.tag}_{model}.jsonl"
                cmd = [sys.executable, str(ROOT / "perf" / "mgxceil" / "walk.py"),
                       "--model", model, "--card", str(card), "--sizes", sizes,
                       "--depth", str(depth), "--out", str(out),
                       "--out-root", str(S / "runs" / a.tag), "--timeout", "10800",
                       "--", "--host_threads", "2", *rest]
                logf = open(S / "logs" / f"{a.tag}_{model}.log", "a")
                env["TT_BIO_LEASE_CARDS"] = str(card)
                log(f"card {card} {model} {sizes} depth {depth} {' '.join(rest)}")
                running[card] = (subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf,
                                                  stdin=subprocess.DEVNULL), job, out)
                # Let the walk's first fold take the lease before choosing another chip.
                time.sleep(60)
                continue
        time.sleep(a.poll)
    log(f"sched {a.tag} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
