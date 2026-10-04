#!/usr/bin/env python3
"""Record one gallery fold on a qb2 chip and pack it for the repo.

    python3 demo/sc26/gallery/record.py <pick> --chip 3

Boltz-2 folds the pick twice in one process with the trajectory hook on (TT_BIO_TRAJECTORY_DIR):
the first fold compiles, the second is the recording, so its time is a warm fold. The chip's AICLK
is read from sysfs every 0.2 s throughout. MSAs come from the ColabFold server at recording time;
the booth never needs the network because it only replays what this writes:

    store/<pick>.json   metadata, atoms, per-frame times and display transforms
    store/<pick>.bin    lzma: quantised sampler states, then the final structure as exact float32

build.py expands the store into protocol recordings and computes the display transforms. Quantisation: each frame is stored as int16
on a grid of max(0.01 A, frame RMS spread / 1000), so a structured frame is within 0.005 A of the
sampler's number and a noise frame (spread up to ~2500 A, mostly off-screen) within 0.05 % of its
spread. The final frame is stored exactly.
"""
import argparse
import json
import lzma
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
PY = os.path.expanduser("~/tt-bio-dev/env/bin/python3")
Q_MIN, Q_REL = 0.01, 1000.0


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


def parse_cif(path):
    import gemmi
    st = gemmi.read_structure(str(path))
    el, name, res, chain, plddt, xyz = [], [], [], [], [], []
    k = -1
    for ch in st[0]:
        for r in ch:
            k += 1
            plddt.append(round(r[0].b_iso / 100.0, 4))
            for a in r:
                el.append(a.element.name)
                name.append(a.name)
                res.append(k)
                chain.append(ch.name)
                xyz.append(a.pos.tolist())
    return dict(element=el, name=name, residue=res, chain=chain), plddt, np.array(xyz)


def predict(cmd, env, log_path, chip, stall_s):
    """Run predict; if its log is silent for stall_s, stop it the fleet's way, never with SIGKILL:
    SIGINT, then SIGTERM, then a reset of this chip (a fold hung inside a device read answers
    neither signal; the reset makes the read fail and the process exits)."""
    import signal
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while p.poll() is None:
            time.sleep(5)
            if time.time() - log_path.stat().st_mtime < stall_s:
                continue
            print(f"stall: log silent {stall_s:.0f} s, stopping", file=sys.stderr)
            for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 30)):
                os.killpg(p.pid, sig)
                try:
                    p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if p.poll() is None:
                subprocess.run([os.path.expanduser("~/.local/bin/tt-smi"), "-r", str(chip)], timeout=180)
                p.wait(120)
            log.write(f"\nSTALLED: no output for {stall_s:.0f} s, stopped\n")
            return "stalled"
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pick")
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--stall-s", type=float, default=300)
    a = ap.parse_args()
    pick = next(p for p in json.load(open(HERE / "picks.json")) if p["id"] == a.pick)
    run = Path(a.runs) / a.pick
    shutil.rmtree(run, ignore_errors=True)
    (run / "in").mkdir(parents=True)
    for tag in ("a-warmup", "b-record"):
        (run / "in" / f"{tag}.yaml").write_text(yaml_for(pick))

    clock = Clock(a.chip)
    th = threading.Thread(target=clock.loop, daemon=True)
    th.start()
    env = dict(os.environ, PYTHONPATH=str(WT), TT_VISIBLE_DEVICES=str(a.chip),
               TT_BIO_LEASE_CARDS=str(a.chip),
               TT_BIO_LEASE_HOLDER=os.environ.get("TT_BIO_LEASE_HOLDER", "worker:sc26-gallery"),
               TT_BIO_TRAJECTORY_DIR=str(run / "traj"))
    cmd = [PY, "-c", "import sys; sys.argv[0]='tt-bio'; from tt_bio.main import cli; cli()",
           "predict", str(run / "in"), "--model", "boltz2", "--accelerator", "tenstorrent",
           "--out_dir", str(run / "out"), "--seed", str(a.seed), "--output_format", "cif",
           "--use_msa_server", "--override"]
    t_launch = time.time()
    for attempt in range(3):  # the ColabFold server now and then hands back a broken archive
        rc = predict(cmd, env, run / "predict.log", a.chip, a.stall_s)
        if not (rc and rc != "stalled" and "run_mmseqs2" in (run / "predict.log").read_text(errors="replace")):
            break
        time.sleep(30)
    clock.stop.set()
    th.join()
    if rc:
        sys.exit(f"{a.pick}: predict rc={rc}, see {run / 'predict.log'}")

    text = (run / "predict.log").read_text(errors="replace")
    order = re.findall(r"✓ (\S+) — ([\d.]+)s", text)
    if [o[0] for o in order] != ["a-warmup", "b-record"]:
        sys.exit(f"{a.pick}: unexpected fold order {order}")
    results = {r["id"]: r for r in json.load(open(next((run / "out").rglob("results.json"))))}
    rec = results["b-record"]
    cifs = {t: next((run / "out").rglob(f"{t}.cif")) for t in ("a-warmup", "b-record")}

    traj = run / "traj"
    steps = sorted(int(m.group(1)) for m in (re.match(r"step_(-?\d+)\.npy", f.name) for f in traj.iterdir()) if m)
    atoms, plddt, cif_xyz = parse_cif(cifs["b-record"])
    n = len(cif_xyz)
    xyz = [np.load(traj / f"step_{s:03d}.npy")[0, :n] for s in steps]
    x0 = [np.load(traj / f"x0_{s:03d}.npy")[0, :n] for s in steps if s >= 0]
    mt = [os.stat(traj / f"step_{s:03d}.npy").st_mtime for s in steps]
    final = xyz[-1].astype("<f4")
    dev_cif = float(np.abs(final - cif_xyz).max())
    if dev_cif > 1e-3:
        sys.exit(f"{a.pick}: final sampler state is {dev_cif} A from the written CIF; atom order mismatch")
    _, _, warm_xyz = parse_cif(cifs["a-warmup"])
    same_as_warmup = float(np.abs(warm_xyz - cif_xyz).max())

    t_end = os.stat(cifs["b-record"]).st_mtime
    t0 = t_end - float(rec["runtime_s"])
    qx, qs_x = quantise(xyz)
    q0, qs_0 = quantise(x0)
    blob = lzma.compress(shuffled(qx) + shuffled(q0) + final.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    store = HERE / "store"
    store.mkdir(exist_ok=True)
    (store / f"{a.pick}.bin").write_bytes(blob)
    aa = sum(len(c["sequence"]) for c in pick["chains"])
    meta = dict(
        id=a.pick, model="boltz2", seed=a.seed, steps=len(steps) - 1, recycling_steps=4,
        diffusion_samples=1, msa="ColabFold server (api.colabfold.com), at recording time",
        chip=a.chip, host="tt-quietbox2", device="Blackhole p300 chip",
        recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        branch_head=subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        n_res=aa, n_entries=len(plddt), n_atoms=n,
        seconds=float(rec["runtime_s"]), seconds_compile_fold=float(order[0][1]),
        stages={"trunk": round(mt[0] - t0, 3), "diffusion": round(mt[-1] - mt[0], 3),
                "confidence": round(t_end - mt[-1], 3)},
        aiclk_mhz=clock.summary(t0, t_end), aiclk_mhz_whole_run=clock.summary(t_launch, time.time()),
        confidence=dict(ptm=rec.get("ptm"), iptm=rec.get("iptm"), plddt=rec.get("complex_plddt"),
                        confidence_score=rec.get("confidence_score")),
        final_vs_cif_max_A=dev_cif, warm_vs_compile_fold_max_A=same_as_warmup,
        frame_steps=steps, frame_t=[round(m - t0, 3) for m in mt],
        quantum_xyz=[round(q, 6) for q in qs_x], quantum_x0=[round(q, 6) for q in qs_0],
        atoms=atoms, plddt=plddt,
        rg_final=round(float(np.sqrt(((final - final.mean(0)) ** 2).sum(1).mean())), 2),
        bin=dict(file=f"{a.pick}.bin", codec="lzma(int16 xyz byte-shuffled | int16 x0 byte-shuffled | float32 final)",
                 bytes=len(blob)),
    )
    (store / f"{a.pick}.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(json.dumps({k: meta[k] for k in ("id", "n_res", "n_atoms", "seconds", "seconds_compile_fold", "stages",
                                           "aiclk_mhz", "confidence", "final_vs_cif_max_A",
                                           "warm_vs_compile_fold_max_A")} | {"bin_MB": len(blob) / 1e6}))


if __name__ == "__main__":
    main()
