"""Docking throughput at K chips: the measured multi-chip scaling curve (row dfd-scaling).

Docking is parallel across complexes, so the honest multi-chip number is complexes/s with the
queue saturated: K independent single-chip workers, one complex each, not one complex split over
chips. Each worker runs the same fixed-shape score step `dfd-device` times on its own chip
(`static_forward` with `convs_on_device`), and throughput is counted only over the window in
which every worker is inside its timed loop, so no worker's warm-up or tail is counted as
parallel work.

    PYTHONPATH=<worktree> python3 scripts/diffdock_port/scaling_run.py \
        --chips 2,1,0 --fixture fx_8G43.pt --steps 20 --out k3.json

Per-complex arithmetic: PROTOCOL.md's recipe is 19 reverse-diffusion steps over 10 poses, and the
fixed-shape forward issues the identical program at every step (D-dfd-trace, DEVICE.md 5), so
complexes/s = steps/s / 19 for the score evaluation. That is the score rollout only; what it does
not cover is named in the output and in SCALING.md.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))
STEPS_PER_COMPLEX = 19


# --------------------------------------------------------------------------- worker

def worker(args):
    import tt_bio
    wt = os.environ.get("PYTHONPATH", "").split(":")[0]
    if wt and not tt_bio.__file__.startswith(wt):
        raise SystemExit(f"tt_bio resolved outside {wt}: {tt_bio.__file__}")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import device as ddev
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released
    from tt_bio.diffdock.static_net import StaticComplex, static_forward
    from tt_bio.tenstorrent import get_device

    rundir = Path(args.rundir)
    chip = os.environ["TT_VISIBLE_DEVICES"]
    fx = torch.load(args.fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    card = load_released(sd, cfg)
    graph = fx["graph"]
    rec_x = HostScoreNetwork(load_released(sd, cfg), lm_embeddings=fx["emb"])._receptor_x(
        graph.receptor)

    dev = get_device()
    convs = ddev.convs_on_device(card, dev, grade=False)

    # One shape for every pose and every step, so one build and one warm-up covers the loop.
    keys = sorted(fx["steps"])
    P = args.poses
    n_lig = fx["steps"][keys[0]].lig_pos.shape[1]
    c = StaticComplex.build(graph.ligand, rec_x, graph.receptor["pos"].float(),
                            graph.receptor["edge_index"], P)
    prepared = []
    for k in keys:
        step = fx["steps"][k]
        have = step.lig_pos.shape[0]
        # 10 poses on a 3-pose fixture: the poses of a batch are the same graph at different
        # coordinates, so tiling them is the same work (DEVICE.md 6). A timing run only.
        poses = step.lig_pos[torch.arange(P) % have]
        prepared.append((pad_nodes(poses.reshape(P * n_lig, 3).float(), c.sg.n_lig_padded),
                         (step.t_tr, step.tr_sigma, step.rot_sigma, step.tor_sigma),
                         step.crop_cutoff))

    with torch.no_grad():
        for lig_pos, sig, crop in prepared:            # warm: the first call on a shape JITs
            static_forward(card, c, lig_pos, *sig, crop_cutoff=crop)

        (rundir / f"ready.{chip}").write_text(str(os.getpid()))
        while not (rundir / "go").exists():
            time.sleep(0.05)

        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        rows = []
        for i in range(args.steps):
            lig_pos, sig, crop = prepared[i % len(prepared)]
            for m in convs.values():
                m.secs = 0.0
            t0 = time.time()
            static_forward(card, c, lig_pos, *sig, crop_cutoff=crop)
            t1 = time.time()
            rows.append({"step_key": keys[i % len(keys)], "t0": t0, "t1": t1,
                         "secs": t1 - t0,
                         "seconds_in_device_layers": sum(m.secs for m in convs.values())})
        ru1 = resource.getrusage(resource.RUSAGE_SELF)

    (rundir / f"worker.{chip}.json").write_text(json.dumps({
        "chip": chip, "pid": os.getpid(), "poses": P,
        "torch_threads": torch.get_num_threads(),
        "convs_on_device": len(convs),
        "cpu_s": (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime),
        "steps": rows}, indent=1))


# --------------------------------------------------------------------------- launcher

def tt_smi_path():
    for c in (Path(sys.executable).parent / "tt-smi", Path.home() / ".local/bin/tt-smi",
              Path("/usr/local/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def clock_sampler(stop, out, period=2.0):
    """One tt-smi poll for ALL chips per period: K pollers would perturb a host-bound run."""
    from tt_bio.runtime import aiclk_reading
    smi = tt_smi_path()
    if smi is None:
        return
    env = {k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"}
    while not stop.is_set():
        try:
            res = subprocess.run([smi, "-s"], capture_output=True, text=True, timeout=30, env=env)
            devs = json.loads(res.stdout).get("device_info", [])
            out.append({"t": time.time(),
                        "mhz": {str(i): aiclk_reading(int((d.get("telemetry") or {}).get(
                            "aiclk", 0))) for i, d in enumerate(devs)}})
        except Exception as e:                                   # a failed poll is recorded
            out.append({"t": time.time(), "error": repr(e)[:200]})
        stop.wait(period)


def cpu_sampler(stop, out, period=2.0):
    def read():
        f = open("/proc/stat").readline().split()[1:]
        v = [int(x) for x in f]
        return sum(v), v[3] + v[4]                               # total, idle+iowait
    prev = read()
    while not stop.is_set():
        stop.wait(period)
        cur = read()
        dt, di = cur[0] - prev[0], cur[1] - prev[1]
        prev = cur
        if dt > 0:
            out.append({"t": time.time(), "busy_frac": 1.0 - di / dt})


def launch(args):
    chips = [c.strip() for c in args.chips.split(",") if c.strip()]
    rundir = Path(args.rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    for f in rundir.glob("*"):
        f.unlink()

    procs = []
    for chip in chips:
        env = dict(os.environ)
        env.update(TT_VISIBLE_DEVICES=chip, TT_BIO_LEASE_CARDS=",".join(chips),
                   TT_BIO_LEASE_HOLDER="worker:dfd-scaling")
        log = open(rundir / f"worker.{chip}.log", "w")
        procs.append((chip, subprocess.Popen(
            [sys.executable, __file__, "--worker", "--chips", chip, "--fixture", args.fixture,
             "--steps", str(args.steps), "--poses", str(args.poses), "--rundir", str(rundir)],
            env=env, stdout=log, stderr=subprocess.STDOUT)))

    t_launch = time.time()
    while time.time() - t_launch < args.warm_timeout:
        if all((rundir / f"ready.{c}").exists() for c in chips):
            break
        for chip, p in procs:
            if p.poll() is not None:
                raise SystemExit(f"worker on chip {chip} died before ready, rc={p.returncode}; "
                                 f"see {rundir}/worker.{chip}.log")
        time.sleep(1.0)
    else:
        for _, p in procs:
            p.kill()
        raise SystemExit(f"workers not ready within {args.warm_timeout}s")

    clocks, cpu, stop = [], [], threading.Event()
    ths = [threading.Thread(target=clock_sampler, args=(stop, clocks), daemon=True),
           threading.Thread(target=cpu_sampler, args=(stop, cpu), daemon=True)]
    for th in ths:
        th.start()
    (rundir / "go").write_text(str(time.time()))
    rc = {chip: p.wait() for chip, p in procs}
    stop.set()
    for th in ths:
        th.join(timeout=35)
    if any(rc.values()):
        raise SystemExit(f"worker rc {rc}; see {rundir}/worker.*.log")

    out = summarise(chips, rundir, clocks, cpu, args)
    print(json.dumps({k: v for k, v in out.items() if k != "workers"}, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def summarise(chips, rundir, clocks, cpu, args):
    ws = [json.loads((rundir / f"worker.{c}.json").read_text()) for c in chips]
    # The window in which EVERY worker is inside its timed loop. A step counts only if it lies
    # entirely inside it, so no warm-up, start skew or tail is counted as parallel work.
    lo = max(w["steps"][0]["t0"] for w in ws)
    hi = min(w["steps"][-1]["t1"] for w in ws)
    window = hi - lo
    counted, busy = 0, []
    for w in ws:
        inside = [s for s in w["steps"] if s["t0"] >= lo and s["t1"] <= hi]
        w["steps_in_window"] = len(inside)
        w["median_step_s"] = sorted(s["secs"] for s in inside)[len(inside) // 2] if inside else None
        w["card_busy_fraction"] = (sum(s["seconds_in_device_layers"] for s in inside)
                                   / sum(s["secs"] for s in inside)) if inside else None
        counted += len(inside)
        busy.append(w["card_busy_fraction"])
    in_win = [c for c in clocks if "mhz" in c and lo <= c["t"] <= hi]
    per_chip = {}
    for ch in chips:
        vals = [c["mhz"].get(ch) for c in in_win if c["mhz"].get(ch)]
        per_chip[ch] = ({"median": sorted(vals)[len(vals) // 2], "min": min(vals),
                         "max": max(vals), "samples": len(vals)} if vals else None)
    host = [c["busy_frac"] for c in cpu if lo <= c["t"] <= hi]
    steps_per_s = counted / window if window > 0 else None
    # What the rest of the box was doing. qb2 is shared: a baseline taken while another campaign
    # runs on the other chips carries that campaign's host load, and the curve it anchors is not a
    # curve. Measured, not assumed: box CPU-seconds in the window minus this run's own.
    nproc = os.cpu_count()
    host_med = sorted(host)[len(host) // 2] if host else None
    external = ((host_med * nproc * window - sum(w["cpu_s"] for w in ws)) / window
                if host_med is not None and window > 0 else None)
    return {"chips": chips, "k": len(chips), "window_s": window,
            "steps_counted": counted, "steps_per_s": steps_per_s,
            "complexes_per_s": steps_per_s / STEPS_PER_COMPLEX if steps_per_s else None,
            "s_per_complex_per_chip": (len(chips) * STEPS_PER_COMPLEX / steps_per_s
                                       if steps_per_s else None),
            "aiclk_mhz": per_chip,
            "card_busy_fraction": busy,
            "host_cpu_busy_fraction": {
                "median": sorted(host)[len(host) // 2] if host else None,
                "max": max(host) if host else None, "samples": len(host)},
            "host_cpu_s_per_step": sum(w["cpu_s"] for w in ws) / counted if counted else None,
            "own_cpu_cores": sum(w["cpu_s"] for w in ws) / window if window > 0 else None,
            "external_cpu_cores": external,
            "contended": external is not None and external > 1.0,
            "nproc": nproc, "poses": args.poses,
            "steps_per_complex": STEPS_PER_COMPLEX,
            "t_start": lo, "t_end": hi, "workers": ws}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--chips", required=True, help="comma-separated TT_VISIBLE_DEVICES ids")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--steps", type=int, default=20, help="timed steps per worker")
    ap.add_argument("--poses", type=int, default=10, help="PROTOCOL.md samples_per_complex")
    ap.add_argument("--rundir", default="/tmp/dfd_scaling_run")
    ap.add_argument("--warm-timeout", type=float, default=1800)
    ap.add_argument("--out")
    args = ap.parse_args()
    (worker if args.worker else launch)(args)


if __name__ == "__main__":
    main()
