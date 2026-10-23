"""How RFD4 throughput scales with chips on one host: measured, not multiplied.

    python perf/rfd4/scaling/scale.py run --cards 0,1,3 --spec S --weights W --batch B --out DIR \
        [--lane-width L] [--reps 5] [--rounds "0;1;3;0,1;0,3;1,3;0,1,3;0@2;0,1,3@2"]
    python perf/rfd4/scaling/scale.py report DIR

`run` starts one worker per card (perf/rfd4/scaling/worker.py: one process, one chip, compiled once), then
plays ROUNDS `--reps` times, rotating the order each rep so no configuration always runs first or last.
A round names the cards that run one batch together, started at the same instant; `@K` pins each of those
workers to K logical CPUs of its own (physical cores first) and sizes its torch thread pool to K, which asks how
much host a chip needs to stay fed. Pinning without resizing the pool measures oversubscription instead: torch's
default pool time-sliced onto one core ran 20x slower while keeping that core flat out.
Cards a round does not name sit idle with their device open.

The driver samples at 1 Hz, from sysfs and /proc only so the sampler itself costs nothing: AICLK, board power
and temperature of every chip, CPU used by each worker, and the box's total busy cores. Busy cores that are
neither a worker nor this driver are `foreign_cores`: another row on the box. Every round is one line in
DIR/rounds.jsonl. A round with an active chip under 1200 MHz median is marked `clock_artifact`; one where the
box was out of CPU (busy cores >= 95 % of nproc) is marked `host_full`, which is a finding, not noise, when our
own workers are what filled it.

`report` writes DIR/report.json. A configuration's designs/s is the sum of each chip's own rate in the round
(a server does not make a fast chip wait for a slow one; the round's barrier is this harness's, so the
barrier-bound `designs_per_s_span` is quoted beside it). Efficiency is that sum over the same chips' own
single-chip medians, so a slower chip does not read as a scaling loss, and the single-chip spread across reps
is the A/A floor it is judged against.
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

HERE = Path(__file__).resolve().parent
CLOCK_FLOOR_MHZ = 1200
HZ = os.sysconf("SC_CLK_TCK")
NPROC = os.cpu_count() or 1
SYS = "/sys/class/tenstorrent/tenstorrent!{}"


def parse_round(spec):
    """'0,1,3@2' -> ((0, 1, 3), 2); '0' -> ((0,), None)."""
    cards, _, k = spec.partition("@")
    return tuple(int(c) for c in cards.split(",")), int(k) if k else None


def cpu_order():
    """Logical CPUs, one per physical core first, then their SMT siblings."""
    seen, first, rest = set(), [], []
    for c in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read_text().strip()
        pkg = Path(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read_text().strip()
        (rest if (pkg, p) in seen else first).append(c)
        seen.add((pkg, p))
    return first + rest


def pin(pid, cpus):
    """Every thread of `pid` to `cpus` (sched_setaffinity on a pid moves only its main thread)."""
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass  # a thread that exited between listdir and here


def read(path, scale=1.0):
    try:
        return float(Path(path).read_text().split()[0]) * scale
    except (OSError, ValueError, IndexError):
        return None


def text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hwmon(card, name, scale):
    d = Path(SYS.format(card)) / "device" / "hwmon"
    for h in d.glob("hwmon*"):
        return read(h / name, scale)
    return None


def proc_ticks(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])
    except (OSError, ValueError, IndexError):
        return None


def box_ticks():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v) - v[3] - v[4]


class Sampler:
    """1 Hz: per chip AICLK/power/temp, per worker CPU cores, box busy and foreign cores."""

    def __init__(self, cards, pids):
        self.cards, self.pids = cards, pids  # pids: {card: pid}
        self.rows, self._stop = [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)
        return False

    def _run(self):
        me = os.getpid()
        last = (time.monotonic(), box_ticks(), {c: proc_ticks(p) for c, p in self.pids.items()}, proc_ticks(me))
        while not self._stop.wait(1.0):
            now = (time.monotonic(), box_ticks(), {c: proc_ticks(p) for c, p in self.pids.items()}, proc_ticks(me))
            dt = now[0] - last[0]
            ours = {c: ((now[2][c] or 0) - (last[2][c] or 0)) / HZ / dt for c in self.pids}
            box = (now[1] - last[1]) / HZ / dt
            drv = ((now[3] or 0) - (last[3] or 0)) / HZ / dt
            self.rows.append({
                "t": time.time(), "box_cores": box, "worker_cores": ours,
                "foreign_cores": max(0.0, box - sum(ours.values()) - drv),
                "aiclk": {c: read(SYS.format(c) + "/tt_aiclk") for c in self.cards},
                "power_w": {c: hwmon(c, "power1_input", 1e-6) for c in self.cards},
                "temp_c": {c: hwmon(c, "temp1_input", 1e-3) for c in self.cards},
                "load1": os.getloadavg()[0]})
            last = now


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = [int(c) for c in a.cards.split(",")]
    rounds = [parse_round(r) for r in a.rounds.split(";")]
    assert all(set(c) <= set(cards) for c, _ in rounds), "a round names a card that has no worker"
    order = cpu_order()
    log = open(out / "rounds.jsonl", "a")

    def emit(rec):
        rec = {**rec, "host": os.uname().nodename, "nproc": NPROC, "tag": a.tag,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    procs = {}
    for c in cards:
        env = {**os.environ, "TT_VISIBLE_DEVICES": str(c), "TT_BIO_LEASE_CARDS": str(c)}
        cmd = [a.python, a.worker, "--spec", a.spec, "--weights", a.weights,
               "--batch", str(a.batch), "--out", str(out / "designs")]
        if a.lane_width:
            cmd += ["--lane-width", str(a.lane_width)]
        procs[c] = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=open(out / f"worker{c}.log", "a"), text=True, bufsize=1)
        # Serial bring-up: two chips compiling at once would race on the JIT cache for no gain.
        ready = json.loads(procs[c].stdout.readline() or '{"error": "worker exited before ready"}')
        emit({"kind": "ready", **ready})
        if not ready.get("ready"):
            sys.exit(f"card {c} did not come up: {ready}")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=os.environ.get("PYTHONPATH", str(HERE)).split(":")[0]).stdout.strip()
    fw = {c: text(SYS.format(c) + "/tt_fw_bundle_ver") for c in cards}
    serial = {c: text(SYS.format(c) + "/tt_serial") for c in cards}
    emit({"kind": "start", "cards": cards, "sha": sha, "fw": fw, "board_serial": serial,
          "batch": a.batch, "lane_width": a.lane_width, "spec": a.spec, "reps": a.reps, "rounds": a.rounds})

    seed = a.seed
    for rep in range(a.reps):
        k = rep % len(rounds)
        for active, ncpu in rounds[k:] + rounds[:k]:
            pins = {}
            for j, c in enumerate(active):
                cpus = set(order[j * ncpu:(j + 1) * ncpu]) if ncpu else set(order)
                pin(procs[c].pid, cpus)
                pins[c] = sorted(cpus)
            with Sampler(cards, {c: p.pid for c, p in procs.items()}) as s:
                for c in active:
                    procs[c].stdin.write(f"run {seed} {ncpu or 0}\n")
                    seed += 1
                res = {c: json.loads(procs[c].stdout.readline() or '{"error": "worker died"}') for c in active}
            for c in active:
                pin(procs[c].pid, set(order))
            bad = {c: r["error"] for c, r in res.items() if "error" in r}
            if bad:
                emit({"kind": "error", "rep": rep, "active": active, "errors": bad})
                break
            t0, t1 = min(r["t_start"] for r in res.values()), max(r["t_end"] for r in res.values())
            win = [x for x in s.rows if t0 <= x["t"] <= t1] or s.rows
            chips = {c: {**{k: r[k] for k in ("wall_s", "designs_per_s", "featurize_s", "design_s", "write_s",
                                              "cpu_s", "width_achieved", "seed", "threads")
                                              if k in r},
                         "cores": round(r["cpu_s"] / r["wall_s"], 2),
                         "aiclk_mhz": med([x["aiclk"][c] for x in win]),
                         "aiclk_min_mhz": min((x["aiclk"][c] for x in win if x["aiclk"][c]), default=None),
                         "power_w": med([x["power_w"][c] for x in win]),
                         "temp_c_max": max((x["temp_c"][c] for x in win if x["temp_c"][c]), default=None)}
                     for c, r in res.items()}
            designs = sum(r["designs"] for r in res.values())
            rec = {"kind": "round", "rep": rep, "active": list(active), "n": len(active), "cpus_per_chip": ncpu,
                   "pins": pins if ncpu else None, "designs": designs, "span_s": round(t1 - t0, 3),
                   "designs_per_s": round(designs / (t1 - t0), 5),
                   "sum_chip_designs_per_s": round(sum(r["designs_per_s"] for r in res.values()), 5),
                   "chips": chips, "idle_aiclk_mhz": {c: med([x["aiclk"][c] for x in win]) for c in cards
                                                      if c not in active},
                   "box_cores_median": med([x["box_cores"] for x in win]),
                   "box_cores_max": round(max(x["box_cores"] for x in win), 2) if win else None,
                   "worker_cores_median": med([sum(x["worker_cores"][c] for c in active) for x in win]),
                   "foreign_cores_median": med([x["foreign_cores"] for x in win]),
                   "load1_median": med([x["load1"] for x in win]), "samples": len(win)}
            why = [f"clock_artifact: chip {c} AICLK median {v['aiclk_mhz']}" for c, v in chips.items()
                   if (v["aiclk_mhz"] or 0) < CLOCK_FLOOR_MHZ]
            if (rec["box_cores_median"] or 0) >= 0.95 * NPROC:
                why.append(f"host_full: {rec['box_cores_median']} of {NPROC} cores busy")
            if any(v["width_achieved"] < min(a.batch, a.lane_width or a.batch) for v in chips.values()):
                why.append("chunked: DRAM refused the width")
            if why:
                rec["flags"] = why
            emit(rec)
        else:
            continue
        break
    for p in procs.values():
        try:
            p.stdin.write("quit\n")
            p.stdin.close()
            p.wait(timeout=120)
        except Exception:  # noqa: BLE001
            p.kill()
    emit({"kind": "end"})


def report(a):
    d = Path(a.dir)
    lines = [json.loads(x) for x in open(d / "rounds.jsonl")]
    rounds = [r for r in lines if r["kind"] == "round"]
    clean = [r for r in rounds if not any(f.startswith(("clock_artifact", "chunked")) for f in r.get("flags", []))]
    single = {}
    for r in clean:
        if r["n"] == 1 and r["cpus_per_chip"] is None:
            single.setdefault(r["active"][0], []).append(r["designs_per_s"])
    floor = {c: {"median": med(v), "spread": round((max(v) - min(v)) / statistics.median(v), 4), "n": len(v)}
             for c, v in single.items()}
    cfg = {}
    for r in clean:
        key = ",".join(map(str, r["active"])) + (f"@{r['cpus_per_chip']}" if r["cpus_per_chip"] else "")
        cfg.setdefault(key, []).append(r)
    table = {}
    for key, rs in cfg.items():
        agg = [r["sum_chip_designs_per_s"] for r in rs]
        ref = sum(floor[c]["median"] for c in rs[0]["active"]) if all(c in floor for c in rs[0]["active"]) else None
        table[key] = {
            "n": rs[0]["n"], "cpus_per_chip": rs[0]["cpus_per_chip"], "reps": len(rs),
            "designs_per_s": med(agg), "min": min(agg), "max": max(agg),
            "designs_per_s_span": med([r["designs_per_s"] for r in rs]),
            "efficiency": round(statistics.median(agg) / ref, 4) if ref else None,
            "efficiency_min_max": [round(min(agg) / ref, 4), round(max(agg) / ref, 4)] if ref else None,
            "worker_cores_per_chip": med([r["worker_cores_median"] / r["n"] for r in rs if r["worker_cores_median"]]),
            "cpu_s_per_design": med([v["cpu_s"] / (r["designs"] / r["n"]) for r in rs for v in r["chips"].values()]),
            "featurize_s": med([v["featurize_s"] for r in rs for v in r["chips"].values()]),
            "write_s": med([v["write_s"] for r in rs for v in r["chips"].values()]),
            "aiclk_mhz": med([v["aiclk_mhz"] for r in rs for v in r["chips"].values()]),
            "aiclk_min_mhz": min(v["aiclk_min_mhz"] or 9999 for r in rs for v in r["chips"].values()),
            "power_w": med([v["power_w"] for r in rs for v in r["chips"].values()]),
            "temp_c_max": max((v["temp_c_max"] or 0) for r in rs for v in r["chips"].values()),
            "box_cores_median": med([r["box_cores_median"] for r in rs]),
            "foreign_cores_median": med([r["foreign_cores_median"] for r in rs]),
            "flags": sorted({f.split(":")[0] for r in rs for f in r.get("flags", [])})}
    start = next((r for r in lines if r["kind"] == "start"), {})
    rep = {"dir": str(d), "start": start, "rounds": len(rounds), "clean": len(clean),
           "dropped": [r.get("flags") for r in rounds if r not in clean], "single_chip_floor": floor,
           "configs": dict(sorted(table.items(), key=lambda kv: (kv[1]["cpus_per_chip"] or 99, kv[1]["n"], kv[0])))}
    (d / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cards", required=True)
    r.add_argument("--spec", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--batch", type=int, required=True)
    r.add_argument("--lane-width", type=int)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--rounds", default="0;1;3;0,1;0,3;1,3;0,1,3")
    r.add_argument("--seed", type=int, default=2000)
    r.add_argument("--tag", default="")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--worker", default=str(HERE / "worker.py"), help="tests swap in a fake chip")
    p = sub.add_parser("report")
    p.add_argument("dir")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
