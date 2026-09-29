#!/usr/bin/env python3
"""One BindCraft 2 campaign per chip, on one or more chips of a japanfold Galaxy at once.

    scale.py --chips 30 --params P --out DIR            one chip, the trajectory price
    scale.py --chips 30,29 --params P --out DIR         two chips, the same workload on each

Each chip runs `perf/bcx_p10_campaign/campaign_run.py` to the campaign's own stop condition,
which is the unit JapanFold sells: seconds of ONE CHIP per completed trajectory. Same seed and
same binder on every chip, so the one-chip and two-chip runs are the same work and the only
difference between them is how many chips of the Galaxy are busy.

This process holds each chip's lease in the host's REAL lease dir and keeps its
`/dev/tenstorrent/<node>` open for the whole run, both for the reasons `perf/bwx_perf/sit.py`
paid for: the agent's chip workers take a chip the moment its holder lets go, and the agent's
per-chip recovery waits on open fds rather than on the lease.

AICLK for every chip on the box and the host's loadavg go to OUT/aiclk.jsonl at 1 Hz, sampled
DURING the campaigns. Each campaign also samples its own chip through the harness's own clock,
so the two instruments cross-check.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SYSFS = pathlib.Path("/sys/class/tenstorrent")


def device_node(chip: int) -> str:
    """`/dev/tenstorrent/<node>` of UMD chip `chip`. UMD counts chips in PCI order, the device
    nodes do not: on .107 chip 30 is node 6."""
    nodes = sorted(SYSFS.iterdir(), key=lambda d: os.path.basename(os.path.realpath(d / "device")))
    return "/dev/tenstorrent/" + nodes[chip].name.split("!", 1)[1]


def sample(dest: pathlib.Path, stop: threading.Event):
    nodes = sorted(int(p.name.split("!")[1]) for p in SYSFS.iterdir())
    tick = 0
    with open(dest / "aiclk.jsonl", "a", buffering=1) as f, \
            open(dest / "cotenants.txt", "a", buffering=1) as g:
        while not stop.is_set():
            row = {}
            for n in nodes:
                try:
                    row[n] = int((SYSFS / f"tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                except (OSError, ValueError, IndexError):
                    pass
            f.write(json.dumps({"t": time.time(), "load1": os.getloadavg()[0], "mhz": row}) + "\n")
            if tick % 60 == 0:
                ps = subprocess.run(["ps", "-eo", "pid,etime,pcpu,rss,args", "--sort=-pcpu"],
                                    capture_output=True, text=True).stdout.splitlines()[:8]
                g.write(f"--- {time.strftime('%FT%TZ', time.gmtime())} load "
                        f"{' '.join(f'{x:.2f}' for x in os.getloadavg())}\n")
                g.write("\n".join(line[:200] for line in ps) + "\n")
            tick += 1
            stop.wait(1.0)


def campaign(chip: int, out: pathlib.Path, args, results: dict):
    proj = out / f"chip{chip}"
    proj.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ,
               TT_VISIBLE_DEVICES=str(chip),
               TT_BIO_LEASE_CARDS=str(chip),
               # The parent holds the real lease for the whole sitting; the campaign takes a
               # private one so it does not wait on the parent and does not release the real one.
               TT_BIO_LEASE_DIR=str(proj / "leases"))
    cmd = [sys.executable, "-u", str(ROOT / "perf/bcx_p10_campaign/campaign_run.py"),
           "--max-trajectories", str(args.max_trajectories), "--binder", str(args.binder),
           "--seed", str(args.seed), "--params", args.params, "--out", str(proj / "proj")]
    t0 = time.time()
    print(f"=== chip {chip} campaign start {time.strftime('%FT%TZ', time.gmtime())} "
          f"load {os.getloadavg()[0]:.2f}", flush=True)
    with open(proj / "campaign.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    results[chip] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
    print(f"    chip {chip} rc={rc} {time.time() - t0:.0f}s "
          f"{time.strftime('%FT%TZ', time.gmtime())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True, help="comma-separated UMD chip ids")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-trajectories", type=int, default=2)
    ap.add_argument("--binder", type=int, default=146)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    chips = [int(c) for c in args.chips.split(",")]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from tt_bio.device_lease import DeviceLease
    leases, held = [], []
    for c in chips:
        leases.append(DeviceLease(card=str(c), timeout=900).acquire())
        held.append(os.open(device_node(c), os.O_RDWR))
        print(f"{time.strftime('%FT%TZ', time.gmtime())} holding {leases[-1].path} "
              f"and {device_node(c)}", flush=True)

    def let_go():
        for fd in held:
            try:
                os.close(fd)
            except OSError:
                pass
        for ls in leases:
            try:
                ls.release()
            except Exception as exc:                                     # noqa: BLE001
                print(f"lease release failed: {exc!r}", flush=True)

    # A lease whose holder is killed keeps `released: null`, and the Galaxy's agent reads that
    # metadata: on 2026-09-29 a crashed run's file held the box's own recovery for minutes.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (let_go(), sys.exit(128 + s)))

    stop = threading.Event()
    threading.Thread(target=sample, args=(out, stop), daemon=True).start()
    results: dict = {}
    t0 = time.time()
    try:
        threads = [threading.Thread(target=campaign, args=(c, out, args, results)) for c in chips]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        stop.set()
        let_go()
        (out / "sitting.json").write_text(json.dumps(
            {"chips": chips, "nodes": {str(c): device_node(c).rsplit("/", 1)[1] for c in chips},
             "seed": args.seed, "binder": args.binder,
             "max_trajectories": args.max_trajectories,
             "wall_seconds": round(time.time() - t0, 1), "campaigns": results,
             "finished_utc": time.strftime("%FT%TZ", time.gmtime())}, indent=1))
        print(f"{time.strftime('%FT%TZ', time.gmtime())} sitting done, leases released",
              flush=True)


if __name__ == "__main__":
    main()
