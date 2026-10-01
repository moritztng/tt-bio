#!/usr/bin/env python3
"""The BindCraft 2 size ladder on ONE chip of a Wormhole Galaxy.

    ladder.py --chip 30 --params ~/bwx/af2_params --out OUT hPDL1:50:t hPDL1:141:t ...

Each rung is `target:binder:mode[:KEY=VAL,...]`, `t` timed and `f` footprint, the optional
fourth field an environment the rung alone runs under (`TT_BIO_TRIATT_HIFI_PAD_UP=0`), and runs as its own process
through `perf/bgx_size/rung.py` -- the same harness that measured the p150a ladder, so the two
boards are compared through one instrument and not two. A rung that refuses does not stop the
ladder: a refusal is the measurement at the top of it.

Why this wrapper exists rather than a shell loop, and all three reasons are the Galaxy:

* A japanfold agent's chip workers wait on tt-bio's per-card `flock` and take a chip the moment
  its holder lets go, so a ladder whose rungs each take the lease themselves hands the chip back
  to the dev service between every pair of rungs. This holds the real lease for the whole ladder
  and gives every rung a private one, so the box's agent is restarted once per ladder.
* The lease is not enough where the agent can reset one chip (PDB CPLD 1.16): its recovery waits
  for zero processes with `/dev/tenstorrent/<node>` open, not for the lease, and on 2026-09-29 it
  ran `tt-smi -r` on `bwx-perf`'s chip between two arms and every later arm died with
  `Query mappings failed`. So the node is held open until the last rung ends.
* A lease whose holder is SIGKILLed keeps `released: null`, and the box's own recovery then waits
  on a dead pid. Every signal that can be caught releases it.

Timing and footprint are separate runs by construction (`rung.py`: `get_memory_view` drains the
pipeline), so a footprint rung reports no seconds and says so in `timing_valid`.
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


def device_node(chip: str) -> str:
    """`/dev/tenstorrent/<node>` of UMD chip `chip`. UMD counts chips in PCI order and the
    device nodes do not: on .107 chip 30 is node 6."""
    nodes = sorted(SYSFS.iterdir(), key=lambda d: os.path.basename(os.path.realpath(d / "device")))
    return "/dev/tenstorrent/" + nodes[int(chip)].name.split("!", 1)[1]


def sample(dest: pathlib.Path, stop: threading.Event) -> None:
    """Every chip's AICLK and the host's loadavg at 1 Hz, and the top of `ps` once a minute.

    Each rung samples its own clock inside its own rounds (`perf/bcx_round/meter.py`), which is
    the reading a rung quotes. This one covers the gaps between rungs and the other 30 chips,
    which is how a co-tenant that clamps the box gets attributed rather than guessed at.
    """
    nodes = sorted(int(p.name.split("!")[1]) for p in SYSFS.iterdir())
    tick = 0
    with open(dest / "aiclk.jsonl", "a", buffering=1) as f, \
            open(dest / "cotenants.txt", "a", buffering=1) as g:
        while not stop.is_set():
            row = {}
            for n in nodes:
                try:
                    row[n] = int((SYSFS / f"tenstorrent!{n}" / "tt_aiclk").read_text().split()[0])
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=6,
                    help="gradient rounds per timed rung. Round 1 compiles and is dropped, and "
                         "the last has no successor boundary, so N rounds give N-2 readings")
    ap.add_argument("--footprint-rounds", dest="fp_rounds", type=int, default=3)
    ap.add_argument("--trajectories", default="1",
                    help='what every rung is told explicitly. "auto" measures the default '
                         "instead of the size, which is the one thing a size ladder must not do")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds a single rung may take")
    ap.add_argument("--attrib", action="store_true",
                    help="run footprint rungs under perf/bcw_census/attrib.py, which lists every "
                         "live DRAM buffer at the high-water mark")
    ap.add_argument("rungs", nargs="+", metavar="target:binder:mode[:KEY=VAL,...]")
    args = ap.parse_args()

    plan = []
    for spec in args.rungs:
        target, binder, mode, *extra = spec.split(":", 3)
        if mode not in ("t", "f"):
            ap.error(f"{spec}: mode is t (timed) or f (footprint)")
        plan.append((target, int(binder), mode,
                     dict(kv.split("=", 1) for kv in extra[0].split(",")) if extra else {}))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TT_VISIBLE_DEVICES"] = str(args.chip)
    from tt_bio.device_lease import DeviceLease

    lease = DeviceLease(card=args.chip, timeout=600).acquire()
    node = device_node(args.chip)
    held = os.open(node, os.O_RDWR)
    say = lambda m: print(f"{time.strftime('%FT%TZ', time.gmtime())} {m}", flush=True)  # noqa: E731
    say(f"holding {lease.path} and {node}; {len(plan)} rungs")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (lease.release(), sys.exit(128 + s)))

    stop = threading.Event()
    threading.Thread(target=sample, args=(out, stop), daemon=True).start()
    env = dict(os.environ, TT_BIO_LEASE_DIR=str(out / "leases"),
               TT_BIO_LEASE_CARDS=str(args.chip))
    ledger = out / "ladder.jsonl"
    try:
        for target, binder, mode, extra in plan:
            tag = "_".join([target, str(binder), mode,
                            *("".join(k.split("_")[-2:]) + v for k, v in extra.items())])
            rung = out / tag
            rung.mkdir(exist_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "perf/bgx_size/rung.py"),
                   "--target", target, "--binder", str(binder),
                   "--trajectories", args.trajectories, "--max-trajectories", "1",
                   "--params", args.params, "--out", str(rung),
                   "--rounds", str(args.fp_rounds if mode == "f" else args.rounds)]
            if mode == "f":
                cmd.append("--footprint")
                if args.attrib:
                    cmd[2:3] = [str(ROOT / "perf/bcw_census/attrib.py"), "--every", "2",
                                "--step-mb", "24", "--"]
            say(f"=== {tag} load {os.getloadavg()[0]:.2f}")
            t0 = time.time()
            with open(rung / "rung.log", "w") as log:
                try:
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env={**env, **extra},
                                         timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            row = {"tag": tag, "target": target, "binder": binder, "mode": mode, "env": extra, "rc": rc,
                   "seconds": round(time.time() - t0, 1),
                   "finished_utc": time.strftime("%FT%TZ", time.gmtime())}
            # A rung that died still wrote its rung.json before teardown; read the axis back out
            # so the ledger alone says which axis each rung really ran.
            try:
                got = json.loads((rung / "rung.json").read_text())
                row.update({k: got.get(k) for k in
                            ("evoformer_axis", "complex_residues", "resident_peak_gb",
                             "free_at_peak_gb", "largest_free_block_at_peak_mb",
                             "device_total_gb", "rounds_done", "error", "design_tokens",
                             "auto_before_open", "auto_card_open", "triatt_reach")})
            except (OSError, ValueError) as exc:
                row["ledger_note"] = f"no rung.json: {exc!r}"
            with open(ledger, "a", buffering=1) as f:
                f.write(json.dumps(row) + "\n")
            say(f"    {tag} rc={rc} {row['seconds']:.0f}s axis={row.get('evoformer_axis')} "
                f"peak={row.get('resident_peak_gb')} err={str(row.get('error'))[:80]}")
    finally:
        stop.set()
        os.close(held)
        lease.release()
        say("ladder done, lease released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
