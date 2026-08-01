#!/usr/bin/env python3
"""Concurrency-scaling sweep for one-subprocess-per-chip `tt_bio.main predict` fanout.

Answers "how close to N-times throughput does N concurrent single-card folds get, and
what stops it" on a many-chip single-host box (the 32-chip Wormhole galaxy). Launches
N identical folds, one pinned chip each, and samples host and per-process CPU at 1 Hz
so a throughput number always comes with the CPU accounting that explains it.

Imports no ttnn: it is a launcher, so it can run anywhere and never opens a device
itself.

Two modes matter:

  --levels 1,2,4,8,16,24,28,32     the concurrency curve (throughput + throughput/chip)
  --level 32 --pin pack:M          confine every fold to M physical cores (both SMT
                                   siblings of each) and see whether throughput cares

The second is the discriminator: tt-metal's fast-dispatch back-pressure wait
(`SystemMemoryManager::fetch_queue_reserve_back`) busy-polls a device L1 address over
PCIe MMIO, which costs ~1 full host core per concurrent fold. If that core is pure
latency-tolerant spin, throughput is flat as M shrinks and packing hands the cores
back; if throughput falls with M, the host CPU is doing real work and the lever is
elsewhere. Run it before trusting either story.

Chips are assigned round-robin across PCIe root buses by default (--chip-order spread)
so the concurrency curve is not silently confounded with root-complex contention; pass
--chip-order linear to measure that contention on purpose.

Example (galaxy):

    python3 scripts/profiling/galaxy_conc_sweep.py \\
        --yaml examples/abag_xm/9xth.yaml --model opendde-abag \\
        --msa-dir ~/abag_xm/msa_cache --out-root /tmp/concsweep \\
        --levels 1,2,4,8,16,24,28,32 --host-threads 2 \\
        --jsonl ~/concsweep.jsonl

Every cell is one line of JSON in --jsonl. Discard the first cell of a session (cold
JIT / program cache is ~10x).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")


# ---------------------------------------------------------------- topology


def physical_cores() -> list[list[int]]:
    """[[cpu, sibling...], ...] one entry per physical core, read from sysfs.

    Never assume the 0..n-1 / n..2n-1 SMT layout -- it is a kernel enumeration
    detail, and pinning to the wrong halves silently packs two spinners onto one
    core when you meant to spread them.
    """
    seen: dict[tuple[int, ...], list[int]] = {}
    base = Path("/sys/devices/system/cpu")
    for d in sorted(base.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        f = d / "topology" / "thread_siblings_list"
        if not f.exists():
            continue
        sibs = []
        for part in f.read_text().strip().split(","):
            if "-" in part:
                lo, hi = part.split("-")
                sibs.extend(range(int(lo), int(hi) + 1))
            else:
                sibs.append(int(part))
        seen.setdefault(tuple(sorted(sibs)), sorted(sibs))
    return [v for _, v in sorted(seen.items())]


def tt_pci_root(dev: int) -> str | None:
    """The chip's PCIe ROOT complex, not its own bus.

    Every chip sits on its own bus (0000:c1, 0000:c2, ... one per device), so grouping
    by bus gives 32 groups of one and any "spread across root complexes" ordering
    silently degenerates into "sorted by bus" -- which packs a small cell onto a single
    root complex, the exact confound the spread is there to avoid. The root complex is
    the bus's high nibble (0x01-0x08 -> 0, 0x41-0x48 -> 4, ...).
    """
    p = Path(f"/sys/class/tenstorrent/tenstorrent!{dev}/device")
    if not p.exists():
        return None
    return os.path.basename(os.path.realpath(p)).split(":")[1][0]


def chip_order(n_chips: int, mode: str) -> list[int]:
    """Chip ids in the order cells consume them.

    'spread' interleaves the PCIe root buses so an N-way cell always uses N/4 chips
    per root complex; 'linear' fills one bus before moving on, which is what you want
    only when root-complex contention is the thing under test.
    """
    ids = list(range(n_chips))
    if mode == "linear":
        return ids
    by_bus: dict[str, list[int]] = {}
    for d in ids:
        by_bus.setdefault(tt_pci_root(d) or "?", []).append(d)
    buses = [by_bus[k] for k in sorted(by_bus)]
    out: list[int] = []
    i = 0
    while len(out) < len(ids):
        for b in buses:
            if i < len(b):
                out.append(b[i])
        i += 1
    return out


def parse_pin(spec: str | None) -> list[int] | None:
    """'pack:M' -> the cpu list of the first M physical cores, both siblings each."""
    if not spec:
        return None
    if not spec.startswith("pack:"):
        raise SystemExit(f"--pin: only 'pack:M' is supported, got {spec!r}")
    m = int(spec.split(":", 1)[1])
    cores = physical_cores()
    if m > len(cores):
        raise SystemExit(f"--pin pack:{m}: host has only {len(cores)} physical cores")
    return sorted(c for core in cores[:m] for c in core)


# ---------------------------------------------------------------- sampling


class CpuSampler(threading.Thread):
    """1 Hz host-wide and per-pid CPU sampling for the duration of one cell.

    Host busy-cores comes from /proc/stat deltas; per-fold cores from each pid's
    utime+stime delta. Reporting throughput without these two is how a core-starvation
    ceiling gets misread as a device ceiling.
    """

    def __init__(self, pids: list[int], period: float = 1.0):
        super().__init__(daemon=True)
        self.pids = pids
        self.period = period
        self.stop_evt = threading.Event()
        self.busy_cores: list[float] = []
        self.idle_frac: list[float] = []
        self.proc_cores: list[float] = []

    @staticmethod
    def _stat() -> tuple[float, float]:
        parts = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
        vals = [float(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        return sum(vals), idle

    @staticmethod
    def _descendants(roots: list[int]) -> list[int]:
        """Every pid in the launched processes' trees.

        A fold is not one process: `predict` forks a child that owns the device and
        spawns further multiprocessing children, and essentially all of the host CPU
        lives in those. Sampling only the pid we Popen'd reports ~0.01 cores per fold
        instead of ~1 and turns a core-starvation ceiling into an invisible one.
        """
        out, stack = [], list(roots)
        while stack:
            pid = stack.pop()
            out.append(pid)
            try:
                kids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except OSError:
                continue
            stack.extend(int(k) for k in kids)
        return out

    def _proc_ticks(self) -> float:
        total = 0.0
        for pid in self._descendants(self.pids):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            except (OSError, IndexError):
                continue
            # after the ') ' split, index 11/12 are utime/stime (fields 14/15, 1-based)
            total += (float(fields[11]) + float(fields[12])) / CLK_TCK
        return total

    def run(self) -> None:
        prev_tot, prev_idle = self._stat()
        prev_proc, prev_t = self._proc_ticks(), time.monotonic()
        while not self.stop_evt.wait(self.period):
            tot, idle = self._stat()
            proc, now = self._proc_ticks(), time.monotonic()
            d_tot, d_idle, dt = tot - prev_tot, idle - prev_idle, now - prev_t
            if d_tot > 0 and dt > 0:
                self.busy_cores.append((d_tot - d_idle) / d_tot * os.cpu_count())
                self.idle_frac.append(d_idle / d_tot)
                # a child exiting mid-cell drops its ticks out of the tree sum; that is a
                # bookkeeping artefact, not negative CPU
                self.proc_cores.append(max(0.0, (proc - prev_proc) / dt))
            prev_tot, prev_idle, prev_proc, prev_t = tot, idle, proc, now


# ---------------------------------------------------------------- one cell


def run_cell(a, n: int, chips: list[int], pin: list[int] | None) -> dict:
    """Launch n identical folds, one per chip, and wait for all of them."""
    tag = f"n{n}" if pin is None else f"n{n}_pin{len(pin)}"
    out_root = Path(a.out_root) / tag
    procs, logs, t0s = [], [], []
    started = time.monotonic()
    for i in range(n):
        chip = chips[i]
        out_dir = out_root / f"c{chip}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = []
        if pin is not None:
            cmd += ["taskset", "-c", ",".join(str(c) for c in pin)]
        cmd += [a.python, "-u", "-m", "tt_bio.main", "predict", a.yaml,
                "--model", a.model, "--out_dir", str(out_dir), "--override",
                "--diffusion_samples", str(a.diffusion_samples),
                "--host_threads", str(a.host_threads)]
        if a.msa_dir:
            cmd += ["--msa_dir", a.msa_dir, "--msa_cache_only"]
        if a.recycling_steps:
            cmd += ["--recycling_steps", str(a.recycling_steps)]
        cmd += a.extra
        env = dict(os.environ)
        env["TT_VISIBLE_DEVICES"] = str(chip)
        env["TT_BIO_LEASE_HOLDER"] = a.lease_holder
        log = open(out_dir / "run.log", "w")
        logs.append(log)
        t0s.append(time.monotonic())
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

    sampler = CpuSampler([p.pid for p in procs])
    sampler.start()

    walls, rcs = [], []
    for p, t0 in zip(procs, t0s):
        rc = p.wait(timeout=a.timeout) if a.timeout else p.wait()
        walls.append(time.monotonic() - t0)
        rcs.append(rc)
    elapsed = time.monotonic() - started

    sampler.stop_evt.set()
    sampler.join(timeout=5)
    for log in logs:
        log.close()

    ok = [w for w, rc in zip(walls, rcs) if rc == 0]
    mean = statistics.fmean
    return {
        "n": n,
        "pin_cpus": len(pin) if pin else None,
        "chips": chips[:n],
        "ok": len(ok),
        "failed": [c for c, rc in zip(chips[:n], rcs) if rc != 0],
        "wall_cell_s": round(elapsed, 2),
        "fold_wall_median_s": round(statistics.median(ok), 2) if ok else None,
        "fold_wall_max_s": round(max(ok), 2) if ok else None,
        # aggregate throughput over the whole cell, so a straggler is not hidden
        "folds_per_hour": round(3600.0 * len(ok) / elapsed, 3) if ok else 0.0,
        "folds_per_hour_per_chip": round(3600.0 * len(ok) / elapsed / n, 4) if ok else 0.0,
        "host_busy_cores_mean": round(mean(sampler.busy_cores), 2) if sampler.busy_cores else None,
        "host_idle_frac_mean": round(mean(sampler.idle_frac), 3) if sampler.idle_frac else None,
        "fold_cpu_cores_total_mean": round(mean(sampler.proc_cores), 2) if sampler.proc_cores else None,
        "fold_cpu_cores_each_mean": round(mean(sampler.proc_cores) / n, 3) if sampler.proc_cores else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, help="the ONE target every fold in every cell runs")
    ap.add_argument("--model", default="opendde-abag")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--jsonl", required=True, help="one JSON line appended per cell")
    ap.add_argument("--levels", default="1,2,4,8,16,24,28,32",
                    help="concurrency levels; ignored when --pin-sweep is given")
    ap.add_argument("--pin-sweep", default=None,
                    help="comma-separated M values for E2: fixed --level, pack into M physical cores")
    ap.add_argument("--level", type=int, default=32, help="fixed concurrency for --pin-sweep")
    ap.add_argument("--pin", default=None, help="single 'pack:M' applied to every cell")
    ap.add_argument("--chip-order", choices=["spread", "linear"], default="spread")
    ap.add_argument("--chips", default=None,
                    help="explicit comma-separated chip ids, used in the given order; overrides "
                         "--chip-order/--n-chips. Needed when another job owns part of the box.")
    ap.add_argument("--n-chips", type=int, default=32)
    ap.add_argument("--msa-dir", default=None,
                    help="MSA cache; implies --msa_cache_only so no cell can silently pay an "
                         "MSA search and stop being comparable to the others")
    ap.add_argument("--host-threads", type=int, default=2)
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--recycling-steps", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=3600, help="per-fold seconds; 0 disables")
    ap.add_argument("--lease-holder", default="worker:galaxy-32way-scaling-bottleneck")
    ap.add_argument("--warmup", action="store_true",
                    help="run a discarded n=1 cell first (cold JIT/program cache is ~10x)")
    ap.add_argument("extra", nargs="*", help="extra args passed through to tt_bio.main predict")
    a = ap.parse_args()

    if a.pin and a.pin_sweep:
        raise SystemExit("--pin and --pin-sweep are mutually exclusive")
    if not shutil.which("taskset") and (a.pin or a.pin_sweep):
        raise SystemExit("taskset not found but pinning was requested")

    if a.chips:
        chips = [int(x) for x in a.chips.split(",")]
    else:
        chips = chip_order(a.n_chips, a.chip_order)
    cores = physical_cores()
    print(f"host: {len(cores)} physical cores / {os.cpu_count()} logical, "
          f"{a.n_chips} chips, order={a.chip_order}", flush=True)
    print(f"chip order: {chips}", flush=True)

    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    jsonl = open(a.jsonl, "a")

    if a.warmup:
        print("warmup cell (discarded)...", flush=True)
        run_cell(a, 1, chips, None)

    if a.pin_sweep:
        cells = [(a.level, parse_pin(f"pack:{m}")) for m in
                 (int(x) for x in a.pin_sweep.split(","))]
    else:
        pin = parse_pin(a.pin)
        cells = [(int(x), pin) for x in a.levels.split(",")]

    for n, pin in cells:
        if n > len(chips):
            print(f"skip n={n}: only {len(chips)} chips", flush=True)
            continue
        print(f"--- cell n={n} pin={len(pin) if pin else 'none'} cpus", flush=True)
        rec = run_cell(a, n, chips, pin)
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["model"], rec["yaml"] = a.model, a.yaml
        rec["host_threads"] = a.host_threads
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        print(json.dumps(rec), flush=True)

    jsonl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
