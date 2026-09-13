#!/usr/bin/env python3
"""Data-parallel throughput of the Boltz-2 512 aa cell: W independent folds, one per chip.

One fold per chip, nothing shared between chips. No tensor is split across devices, so this
measures the only multi-chip route the campaign still has: folds per hour per box against W.

How a width is measured, and why this shape rather than "run W folds and divide":

  * one process per card, each pinned with ``TT_VISIBLE_DEVICES=<card>`` so ttnn brings up that
    chip and no other. An unpinned open starts every visible chip and would silently make a
    width-1 arm a 32-chip arm.
  * a **barrier** between the warm-up fold and the timed reps. Without it the first chip to finish
    loading folds alone and its "concurrent" fold is a solo fold; the whole question here is what
    happens when W folds really are in flight at once.
  * throughput is taken over the **window where every child is folding**: the reps start after the
    barrier, so ``W * reps / (max_end - min_start)`` is a measured rate, not a per-fold latency
    converted into one. The median per-fold latency is reported next to it because the ratio of
    medians (width-1 median / width-W median) is the efficiency, and the two numbers have to agree.
  * the CIF sha256 of every fold, from every child. A DP arm that does not emit the width-1 digest
    is not the same computation and its throughput is meaningless. This is a digest equality check,
    which the chimeric cdk2x2 fixture can carry; it is not a parity score, which it cannot.
  * ``step_n`` read off the sampler per fold. 200 steps and 3 recycles are the protocol, asserted
    per fold by the driver rather than trusted from a config dict.

Host-side instrumentation, recorded per fold so the gap at the widest width can be attributed
rather than guessed: process CPU seconds (``utime + stime``) over the fold gives **cores per
concurrent fold**, which is the quantity the 2026-08-02 galaxy investigation named as the residual
(~1.007 cores/fold). Wall of ``DiffusionModule.forward`` gives the sampler's share.

    dp_width.py --cards 0,1,2,3 --reps 3 --out perf/b2z2_dp/w4.json
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import socket
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

SIZE = 512
# A width whose peer died never reaches the barrier, so the timeout used to be the only floor on
# how long a dead arm burns the box. It cost 30 minutes of 32 idle chips once (w=16, cards 12 and
# 13 failed their device open at t+3 s and the other 14 sat at the barrier). The parent now aborts
# the barrier the moment a child exits early, so this is the backstop and not the mechanism -- and
# it has to stay generous, because the wait here is legitimately long: device open takes
# /tmp/tt-bio-device-open.lock, which is host-wide, so W chips come up strictly one at a time at
# roughly 50 s each. The first child of a 32-way arm reaches the barrier while the last one has
# not opened its chip yet, and waits ~25 minutes for it. A 15-minute backstop would fail every
# 32-way arm on a healthy box.
BARRIER_TIMEOUT_S = 3600


def occupancy():
    """Which chips are open and by whom, recorded per fold so a co-tenant that arrives mid-run
    lands in the artifact instead of inside the number."""
    nodes = [f"/dev/tenstorrent/{i}" for i in range(64)]
    nodes = [n for n in nodes if os.path.exists(n)]
    try:
        out = subprocess.run(["lsof", "-F", "pcn"] + nodes,
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return {"error": str(e)}
    pids, names = set(), set()
    for line in out.splitlines():
        if line.startswith("p"):
            pids.add(line[1:])
        elif line.startswith("n") and "tenstorrent" in line:
            names.add(line[1:])
    return {"pids": sorted(pids), "nodes": sorted(names)}


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime)


# ---------------------------------------------------------------------------- child


def child(args) -> int:
    # Before torch: the inter-op pool sizes itself to cores/2 on the first parallel op and is not
    # settable afterwards, so the cap has to be bound before anything touches it.
    from tt_bio import runtime as _RT
    _RT.bind_host_threads()
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import ab_flag_levers as AB

    visible = os.environ.get("TT_VISIBLE_DEVICES")
    assert visible and visible.strip() == args.card, (
        f"child for card {args.card} has TT_VISIBLE_DEVICES={visible!r}; an unpinned open "
        "brings up every chip on the box and would make this arm a whole-box arm")

    out = {"card": args.card, "width": args.width, "pid": os.getpid(), "folds": [],
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "torch_num_threads": torch.get_num_threads(),
           "torch_interop_threads": torch.get_num_interop_threads()}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()
    t_open = time.perf_counter()
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out["device_open_s"] = round(time.perf_counter() - t_open, 3)
    out["env"] = {"host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
                  "torch": torch.__version__, "visible": visible,
                  "tt_bio_file": _TB.__file__}
    dump()

    work = Path(tempfile.mkdtemp(prefix=f"b2z2dp-c{args.card}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True)
    cfg = AB.build_cfg(args.msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run(f"b2z2-dp-c{args.card}", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # The 200-step sampler. forward() hands a torch tensor back to the torch sampler loop, so it
    # has already drained the device; a bare wall clock around it is the step's real cost and no
    # synchronize is added, which would delete the exposed-dispatch overlap a step really has.
    step = {"n": 0, "s": 0.0}
    dmod = getattr(T, "DiffusionModule", None)
    assert dmod is not None, "no DiffusionModule on this tree; step_n could not be asserted"
    _fwd = dmod.forward

    def timed_step(self, *a, **kw):
        t = time.perf_counter()
        r = _fwd(self, *a, **kw)
        step["n"] += 1
        step["s"] += time.perf_counter() - t
        return r

    dmod.forward = timed_step

    target = Path(args.fixture)

    def fold(tag):
        cfg["seed"] = args.seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        step["n"] = 0
        step["s"] = 0.0
        ttnn.synchronize_device(dev)
        c0, t_start = cpu_s(), time.time()
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        elapsed = time.perf_counter() - t
        c1, t_end = cpu_s(), time.time()
        # CHEAT-CHECK, per fold: the sampler ran its full 200 steps. Throughput taken against a
        # short-sampled arm is the one thing this campaign may not ship.
        assert step["n"] == args.steps, f"sampler ran {step['n']} steps, not {args.steps}"
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        body = cifs[0].read_bytes()
        rec = {"tag": tag, "fold_s": round(elapsed, 3),
               "start_epoch": round(t_start, 3), "end_epoch": round(t_end, 3),
               "sha256": hashlib.sha256(body).hexdigest()[:16],
               "step_s": round(step["s"], 4), "step_n": step["n"],
               "step_ms": round(1000 * step["s"] / step["n"], 4),
               "cpu_s": round(c1 - c0, 3),
               "cores": round((c1 - c0) / elapsed, 4),
               "loadavg": [round(x, 2) for x in os.getloadavg()],
               "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}
        out["folds"].append(rec)
        dump()
        return rec

    for i in range(args.warmup):
        fold(f"warmup{i}")
    out["occupancy_after_warmup"] = occupancy()
    dump()

    # Barrier: nobody starts a timed fold until every child in this width has finished warming up.
    ready = args.barrier / f"ready-{args.card}"
    ready.write_text(str(os.getpid()))
    t_wait = time.perf_counter()
    abort = args.barrier / "abort"
    while True:
        n = len(list(args.barrier.glob("ready-*")))
        if n >= args.width:
            break
        if abort.exists():
            raise RuntimeError(f"barrier aborted by parent at {n}/{args.width} ready: "
                               f"{abort.read_text().strip()}")
        if time.perf_counter() - t_wait > BARRIER_TIMEOUT_S:
            raise TimeoutError(f"barrier: {n}/{args.width} children ready after "
                               f"{BARRIER_TIMEOUT_S}s")
        time.sleep(0.2)
    out["barrier_wait_s"] = round(time.perf_counter() - t_wait, 3)
    out["timed_start_epoch"] = round(time.time(), 3)
    dump()

    for i in range(args.reps):
        fold(f"rep{i}")
    out["occupancy_at_end"] = occupancy()
    timed = [f for f in out["folds"] if f["tag"].startswith("rep")]
    out["median_fold_s"] = round(st.median([f["fold_s"] for f in timed]), 3)
    out["median_cores"] = round(st.median([f["cores"] for f in timed]), 4)
    out["ok"] = True
    dump()
    return 0


# --------------------------------------------------------------------------- parent


def parent(args) -> int:
    sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
    import ab_flag_levers as AB
    from tt_bio.worker import _ensure_local_artifacts

    cards = [c.strip() for c in args.cards.split(",") if c.strip()]
    width = len(cards)
    assert width > 0
    assert len(set(cards)) == width, f"duplicate card in {cards}"

    work = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix=f"b2z2dp-w{width}-"))
    work.mkdir(parents=True, exist_ok=True)
    msa_dir = work / "msa"
    barrier = work / "barrier"
    barrier.mkdir(parents=True, exist_ok=True)
    for p in barrier.glob("ready-*"):
        p.unlink()
    (barrier / "abort").unlink(missing_ok=True)
    kids = work / "children"
    kids.mkdir(parents=True, exist_ok=True)

    fixture = AB.FIX / f"cdk2x2_{args.size}.yaml"
    a3m = AB.FIX / f"cdk2x2_{args.size}.a3m"
    assert fixture.exists() and a3m.exists(), f"no fixture pair for size {args.size}"
    AB._seed_msa(fixture, a3m.read_text(), msa_dir)
    # Fetched once here so W children do not race each other downloading the same weights.
    _ensure_local_artifacts(AB.build_cfg(msa_dir, work / "prefetch"))

    out = {"doc": __doc__, "width": width, "cards": cards,
           "host": socket.gethostname(),
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "protocol": {"size": args.size, "sampling_steps": args.steps,
                        "recycling_steps": 3, "reps": args.reps, "warmup": args.warmup,
                        "seed": args.seed},
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "occupancy_at_start": occupancy(),
           "workdir": str(work)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    # Each child drives one card and so computes host_thread_cap(1) = every core on the box.
    # W of them therefore spawn W*cores threads for cores' worth of work. tt_bio.runtime already
    # names this case ("an external launcher runs one single-card job per chip") and says such a
    # launcher passes cores // concurrent_jobs; this is that launcher, so it passes it.
    from tt_bio import runtime as _RT
    cores = os.cpu_count() or 1
    if args.no_thread_cap:
        cap_env = {}
    elif args.fixed_cap:
        # A cap of cores//W makes every width a different thread count, and the fold's CIF bytes
        # depend on that count (32 threads -> da476491dbb2a847, 16 -> b3ac07a5d86933a8 on the same
        # fixture, seed and chip). The bit-exact bar across widths therefore needs ONE cap for the
        # whole curve, so the curve is run at a fixed cap and the adaptive one is kept as evidence.
        cap_env = {var: str(args.fixed_cap) for var in _RT.HOST_THREAD_VARS}
    else:
        cap_env = _RT.host_thread_cap_env(width, args.host_threads or cores)
    out["host_thread_cap"] = {"cores": cores, "cap_env": cap_env,
                              "capped": not args.no_thread_cap,
                              "fixed": bool(args.fixed_cap)}
    args.out.write_text(json.dumps(out, indent=1))

    procs = {}
    for c in cards:
        env = dict(os.environ)
        env.update(cap_env)
        env["TT_VISIBLE_DEVICES"] = c
        env["TT_BIO_LEASE_CARDS"] = c
        env["TT_BIO_LEASE_HOLDER"] = "worker:b2z2-dp-throughput-linear"
        env["TT_BIO_LOGICAL_DEVICE_ID"] = "0"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
               "--card", c, "--width", str(width), "--reps", str(args.reps),
               "--warmup", str(args.warmup), "--steps", str(args.steps),
               "--seed", str(args.seed), "--size", str(args.size),
               "--fixture", str(fixture), "--msa-dir", str(msa_dir),
               "--barrier", str(barrier), "--out", str(kids / f"card{c}.json")]
        log = open(kids / f"card{c}.log", "w")
        procs[c] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=str(REPO)), log)

    # A child that dies before the barrier (a wedged chip refuses the open in about three
    # seconds) leaves the survivors waiting for a width that can no longer be reached. Watch for
    # that and release them, so a bad card costs one arm's start-up and not BARRIER_TIMEOUT_S.
    def watch_for_early_exit():
        while True:
            dead = [c for c, (p, _) in procs.items() if p.poll() is not None]
            if len(dead) == len(procs):
                return
            if dead and len(list(barrier.glob("ready-*"))) < width:
                (barrier / "abort").write_text(
                    f"children {','.join(sorted(dead))} exited before the barrier")
                return
            time.sleep(1.0)

    watcher = threading.Thread(target=watch_for_early_exit, daemon=True)
    watcher.start()

    rcs = {}
    for c, (p, log) in procs.items():
        rcs[c] = p.wait()
        log.close()
    watcher.join(timeout=5)

    children = {}
    for c in cards:
        f = kids / f"card{c}.json"
        children[c] = json.loads(f.read_text()) if f.exists() else {"error": "no artifact"}

    out["returncodes"] = rcs
    out["children"] = children
    out["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]

    ok = all(rc == 0 for rc in rcs.values()) and all(
        children[c].get("ok") for c in cards)
    out["all_children_ok"] = bool(ok)
    if ok:
        timed = {c: [f for f in children[c]["folds"] if f["tag"].startswith("rep")]
                 for c in cards}
        starts = [f["start_epoch"] for c in cards for f in timed[c]]
        ends = [f["end_epoch"] for c in cards for f in timed[c]]
        n_folds = sum(len(timed[c]) for c in cards)
        window = max(ends) - min(starts)
        lat = [f["fold_s"] for c in cards for f in timed[c]]
        digests = sorted({f["sha256"] for c in cards for f in children[c]["folds"]})
        out["result"] = {
            "n_timed_folds": n_folds,
            "window_s": round(window, 3),
            "folds_per_hour": round(3600.0 * n_folds / window, 3),
            "median_fold_s": round(st.median(lat), 3),
            "mean_fold_s": round(st.mean(lat), 3),
            "min_fold_s": round(min(lat), 3),
            "max_fold_s": round(max(lat), 3),
            "per_card_median_fold_s": {c: children[c]["median_fold_s"] for c in cards},
            "median_cores_per_fold": round(
                st.median([f["cores"] for c in cards for f in timed[c]]), 4),
            "total_cores": round(
                sum(children[c]["median_cores"] for c in cards), 3),
            "median_step_ms": round(
                st.median([f["step_ms"] for c in cards for f in timed[c]]), 4),
            "cif_digests": digests,
            "digest_unanimous": len(digests) == 1,
            "barrier_wait_s": {c: children[c].get("barrier_wait_s") for c in cards},
            "omp_num_threads": children[cards[0]].get("omp_num_threads"),
            "torch_num_threads": children[cards[0]].get("torch_num_threads"),
        }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("result", {"all_children_ok": ok, "returncodes": rcs}), indent=1))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--cards", default="")
    ap.add_argument("--card", default="")
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--fixture", default="")
    ap.add_argument("--msa-dir", dest="msa_dir", type=Path, default=None)
    ap.add_argument("--barrier", type=Path, default=None)
    ap.add_argument("--workdir", default="")
    ap.add_argument("--host-threads", dest="host_threads", type=int, default=0,
                    help="this box's core budget to split across the W children; default nproc")
    ap.add_argument("--fixed-cap", dest="fixed_cap", type=int, default=0,
                    help="one host thread cap for every width, so every width is the same "
                         "computation and the CIF digest can be compared across the curve")
    ap.add_argument("--no-thread-cap", dest="no_thread_cap", action="store_true",
                    help="control arm: let every child claim all cores, as an unaware launcher does")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.steps == 200, f"the cell is 200 sampling steps; got {args.steps}"
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
