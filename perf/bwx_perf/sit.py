#!/usr/bin/env python3
"""One sitting of alternated BindCraft 2 round arms on ONE chip of a japanfold Galaxy.

    sit.py --chip 30 --rounds 7 --out OUT  a1:0:0:1 b1:1:1:1 b2:1:1:1 a2:0:0:1 ...

Each arm is `tag:extra_msa:template:trajectories` and runs as its own process, so the arms
alternate at the process boundary and neither owns the quiet half of the sitting, the way
`perf/bcx_default/sit.sh` does it. An N=1 arm is `perf/bcx_round/run_round.py --exact 0`,
which runs exactly `--rounds` gradient rounds; a real campaign screens a poor trajectory out
after two, and on 2026-09-29 that left an arm with one warm round. N>1 arms need the
interleave, so they are `perf/bcx_p10_duotraj/duo_round.py`.

The Galaxy is the difference. A japanfold agent's chip workers wait on tt-bio's per-card
`flock` and take a chip the moment its holder lets go, so a sitting whose arms each take the
lease themselves hands the chip back to the dev service between every pair of arms. This
process takes the card's lease in the host's REAL lease directory for the whole sitting and
gives every arm a private one, so the agent sees one holder from the first arm to the last and
the box's agent is restarted once per sitting rather than once per arm.

The lease is not enough on a box whose agent can reset one chip (PDB CPLD 1.16). The agent's
recovery waits for zero processes with `/dev/tenstorrent/<node>` open, not for the lease, so on
2026-09-29 it ran `tt-smi -r` on the sitting's chip in the gap between two arms and every later
arm died with `Query mappings failed`. So this process also holds the node open until the end.

Beside the arms it samples every chip's AICLK and the host's loadavg at 1 Hz into
OUT/aiclk.jsonl, and the top of `ps` once a minute into OUT/cotenants.txt.
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


def device_node(chip: str) -> str:
    """`/dev/tenstorrent/<node>` of UMD chip `chip`: UMD counts chips in PCI order, the nodes
    do not (on .107 chip 30 is node 6)."""
    root = pathlib.Path("/sys/class/tenstorrent")
    nodes = sorted(root.iterdir(), key=lambda d: os.path.basename(os.path.realpath(d / "device")))
    return "/dev/tenstorrent/" + nodes[int(chip)].name.split("!", 1)[1]


def sample(dest: pathlib.Path, stop: threading.Event):
    nodes = sorted(int(p.name.split("!")[1]) for p in pathlib.Path("/sys/class/tenstorrent").iterdir())
    tick = 0
    with open(dest / "aiclk.jsonl", "a", buffering=1) as f, \
            open(dest / "cotenants.txt", "a", buffering=1) as g:
        while not stop.is_set():
            row = {}
            for n in nodes:
                try:
                    row[n] = int(pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk")
                                 .read_text().split()[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--binder", type=int, default=146)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--private-leases", default=None,
                    help="lease dir the arms use; default OUT/leases")
    ap.add_argument("--shipped", action="store_true",
                    help="N=1 arms keep pdl1.json's five multimer_v3 models (run_round.py "
                         "--shipped); without it they pin one monomer trunk, as bring-up did")
    ap.add_argument("arms", nargs="+", metavar="tag:extra:template:N")
    args = ap.parse_args()
    arms = []
    for a in args.arms:
        tag, extra, tmpl, n = a.split(":")
        arms.append((tag, int(extra), int(tmpl), int(n)))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TT_VISIBLE_DEVICES"] = str(args.chip)
    from tt_bio.device_lease import DeviceLease

    lease = DeviceLease(card=args.chip, timeout=600).acquire()
    node = device_node(args.chip)
    held = os.open(node, os.O_RDWR)
    print(f"{time.strftime('%FT%TZ', time.gmtime())} holding {lease.path} and {node}", flush=True)
    # A lease whose holder is killed keeps `released: null` in its metadata. tt-bio itself is
    # fine with that -- the kernel drops the flock -- but the Galaxy's own agent reads the
    # metadata, and on 2026-09-29 a crashed run's file held `.107` in `box reset waits for
    # holders {'30': [704203]}` for minutes while the box tried to recover a bad chip. A
    # sitting is the long-running thing on a shared serving box, so it stamps its own release
    # on the way out of every signal that is not SIGKILL.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (lease.release(), sys.exit(128 + s)))
    stop = threading.Event()
    t = threading.Thread(target=sample, args=(out, stop), daemon=True)
    t.start()
    env = dict(os.environ, TT_BIO_LEASE_DIR=args.private_leases or str(out / "leases"),
               TT_BIO_LEASE_CARDS=str(args.chip), TT_BIO_MM_LAYOUT="1",
               TT_BIO_TAPED_CHANNEL_MOVE="1", TT_BIO_WIDEN_ADD="1")
    try:
        for tag, extra, tmpl, n in arms:
            arm = out / tag
            arm.mkdir(exist_ok=True)
            if n == 1:
                # `--exact 0` arms the product's whole fast round, and run_round.py stops an
                # arm whose levers do not match what it was told to expect, so it is told all
                # six: three by flag, three by the env vars it reads them from.
                cmd = [sys.executable, "-u", str(ROOT / "perf/bcx_round/run_round.py"),
                       "--exact", "0", "--triatt-hifi", "1", "--triatt-bw", "1",
                       "--rne-kernel", "1"] + (["--shipped"] if args.shipped else [])
            else:
                cmd = [sys.executable, "-u", str(ROOT / "perf/bcx_p10_duotraj/duo_round.py"),
                       "--interleave", "1", "--trajectories", str(n)]
            cmd += ["--rounds", str(args.rounds), "--binder", str(args.binder),
                    "--params", args.params, "--out", str(arm),
                    "--extra-msa", str(extra), "--template", str(tmpl)]
            t0 = time.time()
            print(f"=== {tag} extra_msa={extra} template={tmpl} N={n} "
                  f"{time.strftime('%FT%TZ', time.gmtime())} load {os.getloadavg()[0]:.2f}",
                  flush=True)
            with open(arm / "arm.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(f"    {tag} rc={rc} {time.time() - t0:.0f}s", flush=True)
    finally:
        stop.set()
        os.close(held)
        lease.release()
        print(f"{time.strftime('%FT%TZ', time.gmtime())} sitting done, lease released", flush=True)


if __name__ == "__main__":
    main()
