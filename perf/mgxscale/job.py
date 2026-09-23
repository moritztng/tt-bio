#!/usr/bin/env python3
"""One design JOB at production scale: N designs against one target, on one pinned chip.

`perf/bhdesign/ladder.py` walks the SIZE axis one design at a time and asks whether a size
runs. This asks the other question -- how many designs per hour a chip returns at a size that
already runs -- so it fixes the size and moves `--num_designs`, which is the axis every
designer ships a production default on (boltzgen 10000, rfd3 1 per spec, pxdesign 1) and which
nobody has moved off it on this hardware.

Three things a size walk does not have to check, and this does:

  * **the design COUNT is the result.** A job that asks for 8 and writes 1 is a silent
    failure, and per-design seconds computed from the request rather than from the artifacts
    would report it as an 8x speedup. Every row carries `n_designs` counted off disk.
  * **host load is the confounder, not the clock.** whglx runs 27 chips for the fleet; a
    design rung there read 9.0x its quiet-box time at the same AICLK
    (`state/mgx-design-ceiling.md`). So loadavg is sampled DURING alongside the clock, and a
    throughput row is only comparable to another at a stated load.
  * **the fold's own seconds.** Both design models write `runtime_s` per design into
    designs.json; wall_s carries import and a checkpoint load, which at 8 designs per job is
    amortised and at 1 is most of the row.

    python3 perf/mgxscale/job.py --model pxdesign --target-res 512 --designs 8 \
        --holder worker:mgx-design-scale --out perf/mgxscale/results/px512d8.jsonl

The fixtures come from the ladder, never from a second copy: a throughput number measured on a
differently-built target is not comparable with the ceiling row's PASS at the same size.
"""
import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perf.bhdesign.ladder import (boltzgen_fixture, cif_stats, classify,  # noqa: E402
                                  diagnosis, dram_numbers, pxdesign_fixture, rfd3_fixture,
                                  _FATAL)
from perf.clocksample import during  # noqa: E402

PY = os.environ.get("LADDER_PY") or sys.executable
LEASES = pathlib.Path(os.environ.get("TT_BIO_LEASE_DIR", "/tmp/tt-bio-device-leases"))
HOST = os.uname().nodename
# app.japanfold.com (24-27) and the tri_mech co-tenant (1), as in perf/mgxdesign/walk.py.
BLOCKED = {1, 24, 25, 26, 27}
CONTENTION = re.compile(r"device contention, nothing ran|is in use by worker:")


def free_cards() -> list[int]:
    """Chips whose lease flock can be taken right now. The flock IS the lease; the JSON beside
    it says `"released": null` forever for a holder that took a SIGKILL."""
    import fcntl
    out = []
    for c in range(32):
        if c in BLOCKED:
            continue
        try:
            fd = os.open(str(LEASES / f"{HOST}-card{c}.json"), os.O_RDWR | os.O_CREAT, 0o664)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            out.append(c)
        except OSError:
            pass
        finally:
            os.close(fd)
    return out


class loadwatch:
    """Sample /proc/loadavg for the duration. The design rungs on this box are host-bound
    before they are device-bound, and `scripts/speed_bar.py` voids on AICLK drift and chip
    identity -- neither of which can see a 64-core box at load 500."""

    def __init__(self, period=10.0):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._period = period

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(float(open("/proc/loadavg").read().split()[0]))
            except Exception:
                pass
            self._stop.wait(self._period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self):
        if not self.samples:
            return None
        s = sorted(self.samples)
        return {"min": round(s[0], 1), "median": round(s[len(s) // 2], 1),
                "max": round(s[-1], 1), "n": len(s), "nproc": os.cpu_count()}


def count_designs(model: str, out_dir: pathlib.Path) -> tuple[int, list[dict]]:
    """Designs actually on disk, and their per-design metrics. Counted, never assumed."""
    if not out_dir.is_dir():
        return 0, []
    rows = []
    dj = out_dir / "designs.json"
    if dj.is_file():
        try:
            rows = [r for r in json.loads(dj.read_text()) if isinstance(r, dict)]
        except Exception:
            rows = []
    if model == "boltzgen":
        # `intermediate_designs/` is what --num_designs asks for; `final_ranked_designs/` is
        # what --budget keeps after filtering. Counting *.cif under out_dir instead picks up
        # the run's own copy of the target structure at the top level and reports 8 designs
        # for 7.
        gen = sorted((out_dir / "intermediate_designs").glob("*.cif")) \
            if (out_dir / "intermediate_designs").is_dir() else []
        ranked = sorted((out_dir / "final_ranked_designs").rglob("*.cif")) \
            if (out_dir / "final_ranked_designs").is_dir() else []
        return len(gen), [{"ranked": len(ranked)}]
    cifs = sorted(out_dir.rglob("*.cif"))
    return len(cifs), rows


def run_job(args) -> dict:
    work = pathlib.Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.target_res}_d{args.designs}_s{args.steps}{args.tag}"
    out_dir = work / f"out_{tag}"
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    # Fixtures go in a per-job directory, not the shared work root. Four identical jobs
    # fanned across four chips otherwise write the same target YAML at the same moment and
    # one of them reads it half-written -- a data-parallelism harness that cannot run the
    # same job twice at once is not measuring data parallelism.
    fxdir = work / f"fx_{tag}"
    fxdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TT_VISIBLE_DEVICES"] = str(args.card)
    env["TT_BIO_LEASE_CARDS"] = str(args.card)
    env["TT_BIO_LEASE_HOLDER"] = args.holder
    env["TT_BIO_SIZE_LIMIT"] = "0"     # this harness measures the ceiling; it cannot obey one
    if args.host_threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            env[k] = str(args.host_threads)

    base = [PY, "-u", "-m", "tt_bio.main"]
    target = pathlib.Path(args.target)
    extra: dict = {}
    if args.model == "rfd3":
        fx = rfd3_fixture(fxdir, args.target_res, args.binder, target, contig=args.contig or "")
        cmd = base + ["design", str(fx), "--model", "rfd3", "--from_pdb",
                      "--out_dir", str(out_dir), "--num_timesteps", str(args.steps),
                      "--num_designs", str(args.designs), "--batch_size", str(args.batch_size)]
    elif args.model == "pxdesign":
        fx = pxdesign_fixture(fxdir, args.target_res, target, args.binder)
        cmd = base + ["design", str(fx), "--model", "pxdesign", "--out_dir", str(out_dir),
                      "--n_step", str(args.steps), "--num_designs", str(args.designs)]
    elif args.model == "boltzgen":
        fx, atoms, tres = boltzgen_fixture(fxdir, args.target_res, target, args.binder)
        cmd = base + ["design", str(fx), "--model", "boltzgen", "--out_dir", str(out_dir),
                      "--num_designs", str(args.designs), "--debug"]
        if args.bg_steps:
            cmd += ["--steps", args.bg_steps]
        if args.budget:
            cmd += ["--budget", str(args.budget)]
        extra = {"target_atoms": atoms, "target_res_actual": tres}
    else:
        raise SystemExit(f"unknown model {args.model}")

    log = work / f"log_{tag}.txt"
    t0 = time.time()
    ended = ""
    os.environ["TT_VISIBLE_DEVICES"] = str(args.card)   # tt-smi reads the granted chip as 0
    with log.open("w") as fh, during() as clk, loadwatch() as ld:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = t0 + args.timeout
        while proc.poll() is None:
            if time.time() >= deadline:
                ended = "TIMEOUT"
            elif _FATAL.search(log.read_text(errors="replace")):
                ended = "FATAL"
            if ended:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                fh.write(f"\n{ended}\n")
                fh.flush()
                break
            time.sleep(5)
        rc = proc.wait()
    wall = round(time.time() - t0, 1)
    blob = log.read_text(errors="replace")

    # A chip lost to another row between the flock probe and the engine's own open. The
    # engine refuses rather than colliding at the fd level and exits 75. That measured the
    # fleet, not the model, so it is NOT a row: `perf/mgxdesign/walk.py` learned the same
    # thing the hard way, where three such refusals were written as rfd3 failing at half its
    # recorded top. Both tests, because an exit code alone is a thin thing to key on.
    if CONTENTION.search(blob) or rc == 75:
        print(f"CONTENTION card {args.card}: retry elsewhere, nothing recorded", flush=True)
        sys.exit(75)

    n_written, rows = count_designs(args.model, out_dir)
    rec = {"model": args.model, "target_res": args.target_res, "asked": args.designs,
           "n_designs": n_written, "steps": args.steps, "binder": args.binder,
           "rc": rc, "wall_s": wall, "card": args.card, "host": HOST,
           "aiclk": clk.summary().get(0), "load": ld.summary(),
           "host_threads": args.host_threads, "batch_size": args.batch_size,
           "cmd": " ".join(cmd[3:]), "tag": args.tag.lstrip("_") or None,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}

    rts = [r["runtime_s"] for r in rows if isinstance(r.get("runtime_s"), (int, float))]
    if rts:
        rec["runtime_s"] = round(max(rts), 1)
    if rows and "ranked" in rows[0]:
        rec["ranked"] = rows[0]["ranked"]

    # The verdict is the artifact count, not the exit code. A job that asks for 8 and returns
    # 3 is a PARTIAL: it says something real about the batch axis and must not read as a pass.
    if n_written >= args.designs and rc == 0:
        rec["verdict"] = "PASS"
        rec["mechanism"] = "none"
    elif n_written > 0:
        rec["verdict"] = "PARTIAL"
        rec["mechanism"] = "timeout" if ended == "TIMEOUT" else classify(blob)
    else:
        rec["verdict"] = "FAIL"
        rec["mechanism"] = "timeout" if ended == "TIMEOUT" else classify(blob)
        rec.update(dram_numbers(blob))
    if rec["verdict"] != "PASS":
        rec["diag"] = diagnosis(blob)
    if n_written:
        rec["s_per_design"] = round(wall / n_written, 1)
        rec["designs_per_h"] = round(3600.0 * n_written / wall, 2)
        cifs = sorted(out_dir.rglob("*.cif"))
        if cifs:
            r, a = cif_stats(cifs[0])
            rec["first_cif"] = {"name": cifs[0].name, "residues": r, "atoms": a}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rfd3", "pxdesign", "boltzgen"])
    ap.add_argument("--target", default=str(ROOT / "perf/bhdesign/targets/big_1831.cif"))
    ap.add_argument("--target-res", type=int, required=True)
    ap.add_argument("--designs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400,
                    help="rfd3 --num_timesteps / pxdesign --n_step. Recorded either way.")
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8, help="rfd3 on-device design batch")
    ap.add_argument("--contig", default="", help="rfd3: verbatim contig instead of the A-crop")
    ap.add_argument("--bg-steps", default="", help="boltzgen --steps, e.g. 'design'")
    ap.add_argument("--budget", type=int, default=0, help="boltzgen --budget")
    ap.add_argument("--card", default="auto")
    ap.add_argument("--holder", default="worker:mgx-design-scale")
    ap.add_argument("--work", default=str(pathlib.Path.home() / "mgxscale-work"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--host-threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and not args.tag.startswith("_"):
        args.tag = "_" + args.tag

    if args.card == "auto":
        free = free_cards()
        if not free:
            raise SystemExit("no free chip")
        args.card = free[0]
    args.card = int(args.card)

    rec = run_job(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "diag"}, indent=2))


if __name__ == "__main__":
    main()
