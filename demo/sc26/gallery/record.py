#!/usr/bin/env python3
"""Record one gallery fold on a qb2 chip and pack it for the repo.

    python3 demo/sc26/gallery/record.py <pick> --chip 3

The fold runs through the booth's own chip worker (engine/chipworker.py --model openfold3), the
same code that folds live at the booth, so a recording is what the chips show: every frame a
sampler state from OpenFold3.dump_fn, every stage event stamped once the chip has finished the
work before it. The worker folds the pick twice: the first fold compiles, the second is the
recording, so its time is a warm fold. The chip's AICLK is sampled every 0.2 s during it (the
worker's own clock) and every 0.2 s over the whole run (this script's).

MSAs are searched here, at recording time, with tt-bio's own search (engine/msa_search.py, the
ColabFold server) into runs/msa; the booth never needs the network because it only replays what
this writes:

    store/<pick>.json   metadata, atoms, per-frame times, stage events
    store/<pick>.bin    lzma: quantised sampler states, then the final structure as exact float32

build.py expands the store into protocol recordings and computes the display transforms.
Quantisation: each frame is stored as int16 on a grid of max(0.01 A, frame RMS spread / 1000), so
a structured frame is within 0.005 A of the sampler's number and a noise frame (spread up to
~2500 A, mostly off-screen) within 0.05 % of its spread. The final frame is stored exactly.

A worker that goes silent for --stall-s is stopped the fleet's way, never with SIGKILL: SIGINT,
then SIGTERM, then a reset of this chip's board.
"""
import argparse
import base64
import json
import lzma
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
ENGINE = HERE.parent / "engine"
PY = os.path.expanduser("~/tt-bio-dev/env/bin/python3")
Q_MIN, Q_REL = 0.01, 1000.0
MODEL = "openfold3"


def yaml_for(pick):
    lines = ["version: 1", "sequences:"]
    for c in pick["chains"]:
        lines += ["  - protein:", f"      id: {c['id']}", f"      sequence: {c['sequence']}"]
    for l in pick["ligands"]:
        lines += ["  - ligand:", f"      id: {l['id']}", f"      ccd: {l['ccd']}"]
    return "\n".join(lines) + "\n"


class Clock:
    def __init__(self, chip):
        self.node = Path(f"/sys/class/tenstorrent/tenstorrent!{chip}/tt_aiclk")
        self.samples, self.stop = [], threading.Event()

    def loop(self):
        while not self.stop.is_set():
            try:
                v = int(self.node.read_text().split()[0], 0)
                self.samples.append((time.time(), v if v != 0xFFFFFFFF else None))
            except (OSError, ValueError):
                self.samples.append((time.time(), None))
            self.stop.wait(0.2)

    def summary(self, t0, t1):
        v = [m for t, m in self.samples if t0 <= t <= t1 and m is not None]
        return {"min": min(v), "median": int(statistics.median(v)), "max": max(v), "n": len(v)} if v else None


def quantise(frames):
    q = []
    for f in frames:
        spread = float(np.sqrt(((f - f.mean(0)) ** 2).sum(1).mean()))
        q.append(max(Q_MIN, spread / Q_REL, float(np.abs(f).max()) / 32000.0))
    ints = np.stack([np.round(f / s) for f, s in zip(frames, q)]).astype(np.int16)
    return ints, q


def shuffled(a):
    return a.view(np.uint8).reshape(-1, a.itemsize).T.copy().tobytes()


def dec(s):
    return np.frombuffer(base64.b64decode(s), "<f4").reshape(-1, 3)


def run_worker(chip, jobs, msa_dir, log, stall_s):
    """Start a chip worker, fold `jobs` in order, return {job id: [events]}."""
    env = dict(os.environ, PYTHONPATH=str(WT), TT_VISIBLE_DEVICES=str(chip), TT_BIO_LEASE_CARDS=str(chip),
               TT_BIO_LEASE_HOLDER=os.environ.get("TT_BIO_LEASE_HOLDER", "worker:sc26-gallery"),
               HF_HUB_OFFLINE="1")
    p = subprocess.Popen([PY, "-u", str(ENGINE / "chipworker.py"), "--chip", str(chip), "--model", MODEL,
                          "--msa-dir", str(msa_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(log, "w"), env=env, text=True, bufsize=1, start_new_session=True)
    last = [time.time()]
    stalled = threading.Event()

    def watchdog():
        while p.poll() is None:
            time.sleep(5)
            if time.time() - last[0] < stall_s:
                continue
            stalled.set()
            print(f"stall: worker silent {stall_s:.0f} s, stopping", file=sys.stderr)
            for sig in (signal.SIGINT, signal.SIGTERM):
                os.killpg(p.pid, sig)
                try:
                    p.wait(30)
                    return
                except subprocess.TimeoutExpired:
                    pass
            subprocess.run([str(ENGINE.parent / "ops/reset_board.sh"), str(chip)], timeout=240)
            return
    threading.Thread(target=watchdog, daemon=True).start()

    todo, events = list(jobs), {}
    for line in p.stdout:
        last[0] = time.time()
        ev = json.loads(line)
        if ev.get("id"):
            events.setdefault(ev["id"], []).append(ev)
        if ev.get("type") == "fold_error":
            sys.exit(f"fold_error {ev}")
        if ev.get("type") == "chip" and ev.get("state") == "ready":
            p.stdin.write(json.dumps(todo.pop(0) if todo else {"type": "quit"}) + "\n")
            p.stdin.flush()
    p.wait()
    if stalled.is_set() or todo:
        sys.exit(f"worker ended rc={p.returncode} with {len(todo)} jobs left, see {log}")
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pick")
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--stall-s", type=float, default=600)
    a = ap.parse_args()
    pick = next(p for p in json.load(open(HERE / "picks.json")) if p["id"] == a.pick)
    run = Path(a.runs) / a.pick
    run.mkdir(parents=True, exist_ok=True)
    msa_dir = Path(a.runs) / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    seqs = run / "chains.json"
    seqs.write_text(json.dumps([{"sequence": c["sequence"]} for c in pick["chains"]]))
    subprocess.run([PY, str(ENGINE / "msa_search.py"), "--out", str(msa_dir), str(seqs)], check=True)

    yaml = yaml_for(pick)
    seq = ":".join(c["sequence"] for c in pick["chains"])
    job = dict(sequence=seq, yaml=yaml, seed=a.seed)
    clock = Clock(a.chip)
    th = threading.Thread(target=clock.loop, daemon=True)
    th.start()
    t_launch = time.time()
    ev = run_worker(a.chip, [dict(job, id="warm", frames=False), dict(job, id="rec")], msa_dir,
                    run / "worker.log", a.stall_s)
    clock.stop.set()
    th.join()
    (run / "events.jsonl").write_text("".join(json.dumps({k: v for k, v in e.items() if k not in ("xyz", "x0")})
                                              + "\n" for e in ev["rec"]))

    start = next(e for e in ev["rec"] if e["type"] == "fold_start")
    done = next(e for e in ev["rec"] if e["type"] == "fold_done")
    warm = next(e for e in ev["warm"] if e["type"] == "fold_done")
    frames = sorted((e for e in ev["rec"] if e["type"] == "frame"), key=lambda e: e["step"])
    stages = [[e["stage"], e["step"], e["total"], e["t"]] for e in ev["rec"] if e["type"] == "stage"]
    xyz = [dec(f["xyz"]) for f in frames]
    x0 = [dec(f["x0"]) for f in frames if f["step"] >= 0]
    final = dec(done["xyz"]).astype("<f4")
    assert np.array_equal(xyz[-1], final), "the last frame is not the scored structure"
    same_as_warmup = float(np.abs(dec(warm["xyz"]) - final).max())
    n = len(final)

    qx, qs_x = quantise(xyz)
    q0, qs_0 = quantise(x0)
    blob = lzma.compress(shuffled(qx) + shuffled(q0) + final.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    store = HERE / "store"
    store.mkdir(exist_ok=True)
    (store / f"{a.pick}.bin").write_bytes(blob)
    aa = sum(len(c["sequence"]) for c in pick["chains"])
    t_end = done["t_wall"]
    t0 = t_end - done["seconds"]
    meta = dict(
        id=a.pick, model=MODEL, seed=a.seed, steps=len(frames) - 1, recycling_steps=start["loops"],
        diffusion_samples=1, msa="ColabFold server (api.colabfold.com) via tt-bio, at recording time",
        chip=a.chip, host="tt-quietbox2", device="Blackhole p300 chip",
        recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        branch_head=subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        n_res=aa, n_entries=len(done["plddt"]), n_atoms=n,
        seconds=done["seconds"], seconds_compile_fold=warm["seconds"], stages=done["stages"],
        stage_events=stages, aiclk_mhz=done["aiclk_mhz"],
        aiclk_mhz_recording_run=clock.summary(t0, t_end), aiclk_mhz_whole_run=clock.summary(t_launch, time.time()),
        confidence=dict(ptm=done.get("ptm"), plddt=round(float(np.mean(done["plddt"])), 4)),
        warm_vs_compile_fold_max_A=same_as_warmup,
        frame_steps=[f["step"] for f in frames], frame_t=[f["t"] for f in frames],
        quantum_xyz=[round(q, 6) for q in qs_x], quantum_x0=[round(q, 6) for q in qs_0],
        atoms=start["atoms"], plddt=done["plddt"],
        rg_final=round(float(np.sqrt(((final - final.mean(0)) ** 2).sum(1).mean())), 2),
        bin=dict(file=f"{a.pick}.bin", codec="lzma(int16 xyz byte-shuffled | int16 x0 byte-shuffled | float32 final)",
                 bytes=len(blob)),
    )
    (store / f"{a.pick}.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(json.dumps({k: meta[k] for k in ("id", "n_res", "n_atoms", "seconds", "seconds_compile_fold", "stages",
                                           "aiclk_mhz", "confidence", "warm_vs_compile_fold_max_A")}
                     | {"bin_MB": len(blob) / 1e6}))


if __name__ == "__main__":
    main()
