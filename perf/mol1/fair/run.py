"""The fair TT number for Mol-1: one gated size on one or more chips of one box, under state/mol1/PROTOCOL.md
section 10.

    cd state/mol1/inputs && python <tt-bio>/perf/mol1/fair/run.py --input mol1-512.json --chips 0 \
        --part p150a --out <dir> [--rounds 5] [--model mol1]

One process per chip, each with its own device context, loads the model the way a JapanFold worker does
(`tt-bio predict` builds the run config, `_WorkerState.load_model` builds the model) and then folds through
`predict_one`, the serving path. The parent opens no device. Every line goes to `<out>/runs.jsonl`.

A round: every chip starts one prediction (the input on its own seed) at one barrier, and the round's
`wall_s` runs from the barrier to the last chip's written output. predictions = chips, so the rate is
predictions / wall_s and the gate divides it by `chips`. The first round per process compiles and is
`"kind":"cold"`; then `--rounds` warm rounds (protocol: >= 5), up to `--max-attempts` to get that many clean.

Sampled during each round, never at a boundary: AICLK per chip at 1 Hz (`perf/clocksample.py`; a chip under its
arch floor discards the round as `clock_artifact`), foreign CPU and foreign device holders (`perf/fairbox.py`).
With `--part p150a` a foreign holder or more than `--max-foreign-cores` discards the round; on a Wormhole Galaxy
(`--part wh`) the rest of the box is serving by design, so its load is recorded and nothing is discarded for it.

Each line carries the sha of the tree `tt_bio` is imported from (PYTHONPATH), whether it is dirty, the run config's
recycles / steps / samples next to the protocol's 20 / 200 / 5, and a sha256 of each chip's sampled coordinates,
so any line can be rerun on its seed and checked.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/, for clocksample and fairbox

import clocksample  # noqa: E402
from fairbox import Contention, board_type  # noqa: E402

WORK = {"recycling_steps": 20, "sampling_steps": 200, "diffusion_samples": 5}  # PROTOCOL section 1


def run_config(model, inp, out, extra=()):
    """The worker config `tt-bio predict <inp> --model <model>` builds, captured before it dispatches, and the
    staged job paths. Anything the CLI does first (input checks, MSA files read, weights fetched) happens here,
    outside the timed span, as for a user whose model is already loaded."""
    from tt_bio import main as M

    got = {}

    class Stop(Exception):
        pass

    def grab(*args, **kw):
        got["payload"] = args[1] if isinstance(args[0], str) else args[0]
        raise Stop

    M._dispatch_run = M._dispatch_to_controller = grab
    try:
        M.cli.main(["predict", str(inp), "--model", model, "--accelerator", "tenstorrent",
                    "--out_dir", str(out / "cli"), *extra], standalone_mode=False)
    except Stop:
        pass
    p = got["payload"]
    return dict(p["config"]), [stage_job(j, out / "jobs") for j in p["jobs"]]


def stage_job(job, d):
    """A dispatched job as the worker sees it: its input bytes written to a file (tt_bio/worker.py, run_job)."""
    if not isinstance(job, dict):
        return Path(job)
    d.mkdir(parents=True, exist_ok=True)
    from tt_bio.worker import stage_input
    return stage_input(job, d / job["name"])


def seeded(job, seed):
    """The staged input that folds ``seed``. An upstream JSON input is staged as ``<job>__seed_<s>.yaml`` and the
    worker takes the seed from that stem before ``cfg["seed"]`` (worker._target_seed), so each seed gets its own
    copy beside the shared sidecar."""
    from tt_bio.protenix_json import SEED_SEP, split_stem
    name, s = split_stem(job.stem)
    if s is None:
        return job
    path = job.with_name(f"{name}{SEED_SEP}{seed}{job.suffix}")
    if not path.exists():
        path.write_bytes(job.read_bytes())
    return path


def chip_main(chip, a, barrier, results):
    """One chip: load once, then one prediction per round, after the barrier."""
    os.environ["TT_VISIBLE_DEVICES"] = str(chip)
    os.environ["TT_BIO_LEASE_CARDS"] = str(chip)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    from tt_bio import worker as W

    out = Path(a.out) / f"chip{chip}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        cfg, jobs = run_config(a.model, Path(a.input).resolve(), out, a.cli_extra)
        W._ensure_local_artifacts(cfg)
        state = W._WorkerState("tenstorrent")
        t0 = time.time()
        state.load_model(cfg)
        state.bind_run("mol1-fair", cfg)
        results.put({"chip": chip, "phase": "load", "wall_s": round(time.time() - t0, 2),
                     "work": {k: cfg.get(k) for k in WORK}, "jobs": [str(j) for j in jobs]})
        m, coords = state.model, {}
        if a.extra_levers:
            from tt_bio.tenstorrent import parse_levers
            m._levers = m._levers | parse_levers(a.extra_levers.split(","))
        fold, fold_k = m.fold, m.fold_k

        def counted_fold(*args, **kw):
            r = fold(*args, **kw)
            c = r[0] if isinstance(r, tuple) else r
            coords[kw.get("seed")] = c.detach().float().cpu()
            return r

        def counted_fold_k(feats, seeds, **kw):
            rs = fold_k(feats, seeds, **kw)
            for sd, r in zip(seeds, rs):
                coords[sd] = (r[0] if isinstance(r, tuple) else r).detach().float().cpu()
            return rs

        m.fold, m.fold_k = counted_fold, counted_fold_k
    except Exception as exc:  # noqa: BLE001 -- the parent records it and stops the cell
        results.put({"chip": chip, "phase": "error", "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
        barrier.abort()
        return
    for i in range(1 + a.max_attempts):
        seeds = [a.seed + 1000 * a.chips.index(chip) + i * a.batch + j for j in range(a.batch)]
        try:
            barrier.wait()
        except Exception:  # noqa: BLE001 -- the parent ended the run
            return
        sdir = out / f"r{i}-s{seeds[0]}"
        sdir.mkdir(parents=True, exist_ok=True)  # the worker makes it before predict_one (run_job)
        paths = [seeded(jobs[0], sd) for sd in seeds]
        t0 = time.time()
        err = None
        coords.clear()
        try:
            rows = [r[0] for r in state.predict_many(paths, dict(cfg, seed=seeds[0], struct_dir=str(sdir)))]
        except Exception as exc:  # noqa: BLE001 -- a failing prediction is a result
            err, rows = f"{type(exc).__name__}: {str(exc)[:600]}", [{}] * len(seeds)
        t1 = time.time()
        for seed, metrics in zip(seeds, rows):
            c = coords.get(seed)
            results.put({"chip": chip, "phase": "round", "round": i, "seed": seed, "t0": t0, "t1": t1, "err": err,
                         "tokens": metrics.get("n_tokens"), "finite": bool(c is not None and torch.isfinite(c).all()),
                         "samples": int(c.shape[0]) if c is not None and c.dim() == 3 else None,
                         "coords_sha256": hashlib.sha256(c.numpy().tobytes()).hexdigest() if c is not None else None,
                         "plddt": metrics.get("plddt"), "ptm": metrics.get("ptm"), "iptm": metrics.get("iptm"),
                         "struct_dir": str(sdir)})


def git(*args):
    """git in the tree tt_bio is imported from: that is the code measured, not this harness."""
    import importlib.util
    root = Path(importlib.util.find_spec("tt_bio").origin).resolve().parent
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=root).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a frozen state/mol1/inputs/mol1-<size>.json; run with cwd there")
    ap.add_argument("--chips", required=True, help="comma-separated chip ids of ONE box, each in its own process")
    ap.add_argument("--part", required=True, choices=("p150a", "wh", "dev"),
                    help="p150a: the gated BH number, box held quiet; wh: Galaxy chips, box keeps serving; dev: neither")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mol1")
    ap.add_argument("--arm", default="normal", help="names the tt-bio mode measured; only a PASS-graded mode counts")
    ap.add_argument("--firmware", default=None, help="recorded on every line, e.g. 19.15.0.0")
    ap.add_argument("--host", default=os.uname().nodename)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1, help="predictions in flight per chip: K seeds of the input through "
                    "one engine call (PROTOCOL `batch`); predictions per round = chips x batch")
    ap.add_argument("--max-attempts", type=int, help="warm rounds to try for --rounds clean ones; default 3 x --rounds")
    ap.add_argument("--seed", type=int, default=101, help="upstream's default; chip k round i runs seed + 1000 k + i")
    ap.add_argument("--max-foreign-cores", type=float, default=1.0)
    ap.add_argument("--cli-extra", nargs=argparse.REMAINDER, default=[],
                    help="dev only: more `tt-bio predict` flags (e.g. an MSA cache); recorded, and verdict.py refuses the line")
    ap.add_argument("--quiet-timeout", type=float, default=1800)
    ap.add_argument("--extra-levers", default="", help="dev only: levers added to the model's graded set, for an A/B "
                    "arm (grade_ab) of a lever not yet in it; recorded, and verdict.py refuses the line")
    a = ap.parse_args()
    a.chips = [int(c) for c in a.chips.split(",")]
    grant = os.environ.get("TT_BIO_LEASE_CARDS")
    if grant and not set(a.chips) <= {int(c) for c in grant.split(",")}:
        ap.error(f"--chips {a.chips} outside this process's card grant TT_BIO_LEASE_CARDS={grant}")
    m = re.search(r"(\d+)\.json$", a.input)
    size = int(m.group(1)) if m else None  # the protocol's size label (256 / 512 / 1024), not the token count
    a.max_attempts = a.max_attempts or 3 * a.rounds
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    gated = a.part == "p150a"
    os.environ["TT_VISIBLE_DEVICES"] = ",".join(map(str, sorted(a.chips)))  # tt-smi then lists exactly these chips
    sha, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    base = {"model": a.model, "arm": a.arm, "part": a.part, "host": a.host, "chips": len(a.chips),
            "chip_ids": a.chips, "batch": a.batch, "cli_extra": a.cli_extra + (["--extra-levers", a.extra_levers] if a.extra_levers else []), "sha": sha, "tree_dirty": dirty, "firmware": a.firmware,
            "board_type": board_type() if gated else None, "input": a.input, "raw": str(out)}

    def emit(rec):
        rec = {**rec, **base, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    if gated and Contention.device_holders():
        emit({"kind": "abort", "why": "another process holds a Tenstorrent device on this box",
              "foreign_device_holders": sorted(Contention.device_holders())})
        sys.exit(2)

    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(len(a.chips) + 1), ctx.Queue()
    procs = [ctx.Process(target=chip_main, args=(c, a, barrier, results), daemon=True) for c in a.chips]
    for p in procs:
        p.start()
    work = None
    for _ in a.chips:
        r = results.get(timeout=3600)
        if r["phase"] == "error":
            emit({"kind": "error", "phase": "load", **r})
            sys.exit(3)
        emit({"kind": "cold", **r})
        work = r["work"]
    short = {k: (v, WORK[k]) for k, v in work.items() if v is not None and v != WORK[k]}
    if short and a.model == "mol1":  # the no-cheat rule is checked, not assumed
        emit({"kind": "error", "why": f"work_mismatch: run config {short} (configured, protocol)"})
        sys.exit(4)
    good = 0
    for i in range(1 + a.max_attempts):
        kind = "cold" if i == 0 else "warm"
        waited, loud = 0, None
        if gated:
            waited, loud, _ = Contention.wait_quiet(a.max_foreign_cores, timeout=a.quiet_timeout)
        with clocksample.during(period=1.0) as clk, Contention() as con:
            barrier.wait()
            start = time.time()
            rs = [results.get(timeout=7200) for _ in range(len(a.chips) * a.batch)]
        wall = max(r["t1"] for r in rs) - start
        cs = clk.summary()
        aiclk = {c: (cs.get(k) or {}).get("median") for k, c in enumerate(sorted(a.chips))}
        floor = clocksample.clock_floor_mhz()
        rec = {"kind": kind, "phase": "compile" if i == 0 else "measure", "round": i,
               "size": size or rs[0]["tokens"], "tokens": rs[0]["tokens"], "predictions": len(rs), "wall_s": round(wall, 3),
               "predictions_per_s": round(len(rs) / wall, 6), "work": work, "seed": [r["seed"] for r in rs],
               "per_chip": [{k: r[k] for k in ("chip", "seed", "err", "finite", "samples", "coords_sha256",
                                               "plddt", "ptm", "iptm", "struct_dir")}
                            | {"wall_s": round(r["t1"] - start, 3)} for r in rs],
               "aiclk_mhz": statistics.median(v for v in aiclk.values() if v) if any(aiclk.values()) else None,
               "aiclk_per_chip": aiclk, "aiclk_min_mhz": min((s["min"] for s in cs.values()), default=None),
               "aiclk_samples": min((s["n"] for s in cs.values()), default=0),
               "host_load_per_core_median": round(statistics.median(clk.load), 3) if clk.load else None,
               **con.summary(), "quiet_wait_s": waited}
        why = [f"error on chip {r['chip']}: {r['err']}" for r in rs if r["err"]]
        why += [f"non-finite on chip {r['chip']}" for r in rs if not r["err"] and not r["finite"]]
        why += [f"samples {r['samples']} on chip {r['chip']}, protocol 5" for r in rs
                if a.model == "mol1" and not r["err"] and r["samples"] != WORK["diffusion_samples"]]
        low = {c: m for c, m in aiclk.items() if m is None or m < floor}
        if low:
            why.append(f"clock_artifact: AICLK {low} < {floor} MHz")
        if gated and con.holders:
            why.append(f"contended: device held by {sorted(con.holders)}")
        if gated and (rec["foreign_cores_median"] or 0) > a.max_foreign_cores:
            why.append(f"contended: {rec['foreign_cores_median']} foreign cores > {a.max_foreign_cores}")
        if loud is not None and waited:
            rec["quiet_wait_foreign_cores"] = loud
        if why:
            rec["discard"] = "; ".join(why)
        elif kind == "warm":
            good += 1
        emit(rec)
        if any(r["err"] for r in rs) and i == 0:
            break
        if good >= a.rounds:
            break
    barrier.abort()
    for p in procs:
        p.join(timeout=60)
    emit({"kind": "summary", "clean": good, "rounds_tried": i})


if __name__ == "__main__":
    main()
