"""The fair TT number for RFD4: one size, one chip, under state/rfd4/PROTOCOL.md section 10.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/fair/run.py --spec state/rfd4/inputs/pdl1-128.yaml \
        --weights <bundle> --out <dir> --part p150a [--ceiling 128] [--batches 5]

Two phases, one process, every line appended to `<out>/runs.jsonl`:

1. Climb. The sample width doubles from 1 until the card pushes back (a rung that errors, a rung
   the chunker had to run narrower because DRAM refused it, or two rungs in a row that buy under
   2 %) or the protocol's ceiling (b128 / b64 / b32 at 128 / 256 / 512 tokens). Each rung pays one
   compile call, then one warm screen batch.
2. Measure. At the best rung, `--batches` (>= 5) warm batches, each one full batch of finished
   designs on its own seed. These are the lines the evidence file is built from.

A warm line spans what a user waits for: spec parse, featurisation, all 60 sampler steps of the
released `thinking_medium` recipe (180 trunk passes), the decode and every CIF written. The engine
takes its depth from the recipe and refuses a shorter one, so nothing here can do less of the work.

Clock and contention are sampled DURING each batch, on threads, never once at a boundary:

* AICLK at 1 Hz (`perf/clocksample.py`). A median under 1200 MHz marks the line `clock_artifact`.
* Foreign CPU: busy cores on the box that are not this process or its children, from /proc/stat.
  A median over `--max-foreign-cores` marks the line `contended`.
* Foreign device holders: any other pid with a /dev/tenstorrent node open, from /proc/*/fd. A lease
  file does not see a short-lived process per arm; an open fd does. Any one marks it `contended`.

A line with a `discard` reason stays in runs.jsonl for the record and never reaches the evidence
file (`verdict.py`): a contended or under-clocked run is discarded, not averaged.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

CLOCK_FLOOR_MHZ = 1200
MIN_GAIN = 0.02
CEILING = {128: 128, 256: 64, 512: 32}  # PROTOCOL section 4


class Contention:
    """Foreign CPU cores and foreign device holders, sampled every `period` s while open."""

    def __init__(self, period=2.0):
        self.cores, self.holders = [], set()
        self._stop = threading.Event()
        self._period = period
        self._hz = os.sysconf("SC_CLK_TCK")

    @staticmethod
    def _box_busy():
        f = open("/proc/stat").readline().split()[1:]
        v = [int(x) for x in f]
        return sum(v) - v[3] - v[4]  # all minus idle and iowait

    @staticmethod
    def _own_busy():
        f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        return sum(int(x) for x in f[11:15])  # utime stime cutime cstime

    @staticmethod
    def device_holders():
        """Pids, other than this process and its children, with a /dev/tenstorrent node open."""
        me = os.getpid()
        out = set()
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) == me:
                continue
            try:
                if int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1]) == me:
                    continue  # our own tt-smi sampler
                for fd in os.listdir(f"/proc/{p}/fd"):
                    if os.readlink(f"/proc/{p}/fd/{fd}").startswith("/dev/tenstorrent"):
                        out.add(f"{p}:{open(f'/proc/{p}/comm').read().strip()}")
                        break
            except (OSError, ValueError, IndexError):
                continue
        return out

    def _run(self):
        box, own, t = self._box_busy(), self._own_busy(), time.monotonic()
        while not self._stop.wait(self._period):
            box2, own2, t2 = self._box_busy(), self._own_busy(), time.monotonic()
            self.cores.append(max(0.0, (box2 - box) - (own2 - own)) / self._hz / (t2 - t))
            box, own, t = box2, own2, t2
            self.holders |= self.device_holders()

    def __enter__(self):
        self.holders |= self.device_holders()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)
        return False

    def summary(self):
        c = sorted(self.cores)
        return {"foreign_cores_median": round(statistics.median(c), 2) if c else None,
                "foreign_cores_max": round(c[-1], 2) if c else None,
                "foreign_device_holders": sorted(self.holders)}


def board_type():
    """The granted chip's board type as tt-smi reports it (index 0 under TT_VISIBLE_DEVICES)."""
    try:
        d = json.loads(subprocess.run([clocksample.TT_SMI, "-s"], capture_output=True, text=True, timeout=60).stdout)
        return d["device_info"][0]["board_info"].get("board_type")
    except Exception as exc:  # noqa: BLE001 -- recorded, and --part is then unverified
        return f"unknown ({type(exc).__name__})"


def _median(xs):
    return round(statistics.median(xs), 1) if xs else None


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", required=True, help="p150a for the gated number; anything else is a dev line")
    ap.add_argument("--host", default=os.uname().nodename)
    ap.add_argument("--ceiling", type=int, help="widest rung; default the protocol's per-size ceiling")
    ap.add_argument("--ladder-start", type=int, default=1)
    ap.add_argument("--batch", type=int, help="skip the climb and measure at this width")
    ap.add_argument("--batches", type=int, default=5, help="warm batches at the best width (protocol: >= 5)")
    ap.add_argument("--seed", type=int, default=1000, help="batch i runs on seed + i")
    ap.add_argument("--max-foreign-cores", type=float, default=1.0)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import sampler

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    card = os.environ.get("TT_VISIBLE_DEVICES")
    sha, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    base = {"part": a.part, "host": a.host, "card": card, "sha": sha, "tree_dirty": dirty,
            "board_type": board_type(), "raw": str(out), "spec": a.spec}

    def emit(rec):
        rec = {**rec, **base, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    before = Contention.device_holders()
    if before:
        emit({"kind": "abort", "why": "another process holds a Tenstorrent device on this box",
              "foreign_device_holders": sorted(before)})
        sys.exit(2)

    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        engine = E.DesignEngine(a.weights, recipe="thinking_medium")
        t_build = time.perf_counter() - t0
    emit({"kind": "cold", "phase": "load", "what": "device open + weight upload", "wall_s": round(t_build, 2),
          "aiclk": clk.summary()})
    passes = int(sum(sampler.recycle_depths(engine.spec, engine.spec.nsteps)))
    size = None

    def batch(width, seed, kind, phase):
        """One full batch of `width` finished designs on `seed`, timed end to end."""
        nonlocal size
        engine.max_parallel_samples = width
        d = out / f"b{width}" / f"{phase}-s{seed}"
        with clocksample.during(period=1.0) as clk, Contention() as con:
            t0 = time.perf_counter()
            ex = rfd4.prepare_examples(a.spec, d / "inputs")[0]
            features, atom_array = rfd4.featurize(ex.cif, seed=seed)
            t_feat = time.perf_counter() - t0
            o = engine.design(features, num_designs=width, nsteps=engine.spec.nsteps, seed=seed, rng="per_design")
            t_design = time.perf_counter() - t0 - t_feat
            for k in range(width):
                rfd4.write_design(o, features, atom_array, d / f"{ex.example_id}_s{seed}_d{k}.cif", row=k)
            wall = time.perf_counter() - t0
        size = int(features["padding_mask"].shape[-1])
        cs = clk.summary().get(0) or {}
        mhz = cs.get("median")
        rec = {"kind": kind, "phase": phase, "size": size, "batch": width, "width_achieved": engine.chunk_width,
               "designs": width, "wall_s": round(wall, 3), "designs_per_s": round(width / wall, 5),
               "featurize_s": round(t_feat, 3), "design_s": round(t_design, 3),
               "write_s": round(wall - t_feat - t_design, 3), "recipe": "thinking_medium",
               "steps": engine.spec.nsteps, "trunk_passes": passes, "seed": seed, "rng": "per_design",
               "aiclk_mhz": mhz, "aiclk_min_mhz": cs.get("min"), "aiclk_samples": cs.get("n", 0),
               "power_w": _median(clk.power.get(0)), "power_w_peak": max(clk.power.get(0) or [None]),
               "host_load_per_core_median": round(statistics.median(clk.load), 3) if clk.load else None,
               **con.summary()}
        why = []
        if mhz is None or mhz < CLOCK_FLOOR_MHZ:
            why.append(f"clock_artifact: AICLK median {mhz} < {CLOCK_FLOOR_MHZ} MHz")
        if con.holders:
            why.append(f"contended: device held by {sorted(con.holders)}")
        if (rec["foreign_cores_median"] or 0) > a.max_foreign_cores:
            why.append(f"contended: {rec['foreign_cores_median']} foreign cores > {a.max_foreign_cores}")
        if engine.chunk_width < width:
            why.append(f"chunked: DRAM ran width {width} at {engine.chunk_width}")
        if why:
            rec["discard"] = "; ".join(why)
        emit(rec)
        return rec

    # Width 1 pays the first compile of everything; it is reported, never amortised.
    first = batch(1, a.seed, "cold", "compile")
    ceiling = a.ceiling or CEILING.get(size, 32)

    if a.batch:
        best = a.batch
        if a.batch != 1:
            batch(best, a.seed, "cold", "compile")
    else:
        rungs, best, top, flat = [], 1, 0.0, 0
        width = a.ladder_start
        while width <= ceiling:
            try:
                if width != 1:
                    batch(width, a.seed, "cold", "compile")
                r = batch(width, a.seed + 1, "warm", "screen")
            except Exception as exc:  # noqa: BLE001 -- the climb records why a rung failed, then stops
                emit({"kind": "error", "phase": "screen", "batch": width, "size": size,
                      "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
                break
            rungs.append(r)
            if r["width_achieved"] < width:
                break  # DRAM refused this width
            dps = r["designs_per_s"]
            flat = flat + 1 if dps < top * (1 + MIN_GAIN) else 0
            if dps > top:
                best, top = width, dps
            if flat >= 2:
                break
            width *= 2
        stop = ("ceiling" if width > ceiling else "pushback")
        emit({"kind": "climb", "size": size, "best_batch": best, "stopped": stop, "ceiling": ceiling,
              "rungs": [{k: r.get(k) for k in ("batch", "width_achieved", "designs_per_s", "aiclk_mhz",
                                               "foreign_cores_median", "discard")} for r in rungs]})

    lines = [batch(best, a.seed + 10 + i, "warm", "measure") for i in range(a.batches)]
    good = [r["designs_per_s"] for r in lines if "discard" not in r]
    emit({"kind": "summary", "size": size, "batch": best, "batches": len(lines), "clean": len(good),
          "designs_per_s_median": round(statistics.median(good), 5) if good else None,
          "designs_per_s_min": min(good, default=None), "designs_per_s_max": max(good, default=None),
          "compile_wall_s": first["wall_s"]})


if __name__ == "__main__":
    main()
