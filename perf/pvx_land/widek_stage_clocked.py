#!/usr/bin/env python3
"""The wide-k trunk-stage arm, re-run with a clock.

The arm shipped in `docs/sdpa-wide-k-parity.md` (120.0 s -> 106.3 s on Protenix-v2 at 686 tokens)
records no AICLK, and on Blackhole the governor alone moves a fold 1.27-1.41x. Its on arm also
spreads 9 s against the off arm's 3 s, which is the shape of a governor ramping across a run. So
the ratio is an indication until the same arm is taken with the clock pinned and sampled while the
fold is running, which is what this does.

Three things it fixes about the original, beyond the clock:
  * the board is named in the output rather than inferred later;
  * every leg samples its own card's AICLK once a second and reports the median it actually held,
    not the one it asked for;
  * the two off legs at the same seed give the A/A floor in the same session as the A/B, so the
    floor is not borrowed from another run.

The trunk stage is read from the predict log's own `trunk 0/` and `diffusion 0/` stamps, exactly as
`widek_fold_stages.py` reads it, so the diffusion steps do not enter the number and both arms do
identical trunk work: same recycles, same calls, only the (q, k) pick moves.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "c14_bfp8"))
import aiclk_hold  # noqa: E402  (another row's module, read-only use)

SYSFS = Path("/sys/class/tenstorrent")

HOOK = """
import atexit, json, os
def _dump():
    try:
        import tt_bio.tenstorrent as T
    except Exception:
        return
    picks = {f"{a}x{b}": list(v) for (a, b), v in getattr(T, "SDPA_CHUNK_PICKS", {}).items()}
    if not picks and not any(getattr(T, "SDPA_K_CHUNK_STATS", [0, 0])):
        return
    with open(os.path.join(os.environ["WIDEK_DUMP"], f"pick_{os.getpid()}.json"), "w") as f:
        json.dump({"wide_k_resolved": bool(getattr(T, "SDPA_WIDE_K", False)),
                   "k_chunk_stats": list(getattr(T, "SDPA_K_CHUNK_STATS", [])),
                   "picks": picks}, f)
atexit.register(_dump)
"""


def aiclk(node: int) -> int:
    return int((SYSFS / f"tenstorrent!{node}" / "tt_aiclk").read_text())


def nodes() -> list:
    return sorted(int(p.name.split("!")[1]) for p in SYSFS.glob("tenstorrent!*"))


class Sampler(threading.Thread):
    """AICLK of every chip on the host, once a second, for the length of one leg."""

    def __init__(self, every: float = 1.0):
        super().__init__(daemon=True)
        self.every, self.stop = every, threading.Event()
        self.samples = {n: [] for n in nodes()}

    def run(self):
        while not self.stop.wait(self.every):
            for n, acc in self.samples.items():
                try:
                    acc.append(aiclk(n))
                except OSError:
                    pass

    def result(self) -> dict:
        out = {}
        for n, acc in self.samples.items():
            if acc:
                out[str(n)] = {"n": len(acc), "min": min(acc),
                               "median": int(statistics.median(acc)), "max": max(acc)}
        return out


def trunk_s(log: Path):
    """`trunk 0/` to `diffusion 0/` off the predict log's own stamps. One second of resolution."""
    marks = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = re.match(r"^(\d\d:\d\d:\d\d)\s", line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        for name, pat in (("trunk0", r"trunk 0/"), ("diff0", r"diffusion 0/")):
            if name not in marks and re.search(pat, line):
                marks[name] = t
    if "trunk0" in marks and "diff0" in marks:
        return round((marks["diff0"] - marks["trunk0"]).total_seconds(), 1)
    return None


def leg(a, arm: str, tag: str) -> dict:
    wd = Path(a.workdir)
    dump = wd / ("dump_" + tag)
    dump.mkdir(parents=True, exist_ok=True)
    hook = wd / "_hook"
    hook.mkdir(parents=True, exist_ok=True)
    (hook / "sitecustomize.py").write_text(HOOK)

    env = dict(os.environ)
    env.update({"TT_BIO_SDPA_WIDE_K": "1" if arm == "on" else "0",
                "TT_VISIBLE_DEVICES": str(a.card),
                "TT_BIO_LEASE_CARDS": str(a.card),
                "TT_BIO_LEASE_HOLDER": a.holder,
                "WIDEK_DUMP": str(dump),
                "PYTHONPATH": os.pathsep.join([str(hook), str(ROOT)])})
    cmd = [a.python, "-m", "tt_bio.main", "predict", a.input, "--model", a.model,
           "--out_dir", str(wd / tag), "--override", "--seed", str(a.seed),
           "--sampling_steps", str(a.steps), "--diffusion_samples", "1"]
    if a.msa_dir:
        cmd += ["--msa_dir", a.msa_dir]

    sam = Sampler()
    sam.start()
    t0 = time.perf_counter()
    with open(wd / (tag + ".log"), "w") as f:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    wall = round(time.perf_counter() - t0, 3)
    sam.stop.set()
    sam.join(timeout=3)

    served = fell_back = 0
    picks = {}
    for f in sorted(dump.glob("pick_*.json")):
        d = json.loads(f.read_text())
        st = d.get("k_chunk_stats") or [0, 0]
        served += int(st[0])
        fell_back += int(st[1]) if len(st) > 1 else 0
        picks.update(d.get("picks") or {})
    return {"arm": arm, "tag": tag, "rc": rc, "wall_s": wall,
            "trunk_s": trunk_s(wd / (tag + ".log")),
            "wide_k_served": served, "fell_back": fell_back, "picks": picks,
            "aiclk": sam.result()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--node", type=int, default=None,
                   help="/dev/tenstorrent node to pin the clock on")
    p.add_argument("--mhz", type=int, default=1350)
    p.add_argument("--order", default="off,on,off,on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--model", default="protenix-v2")
    p.add_argument("--input", default="examples/686.yaml")
    p.add_argument("--msa-dir", dest="msa_dir", default="")
    p.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    p.add_argument("--holder", default="worker:pvx-land")
    p.add_argument("--workdir", default="/home/ttuser/widek_stage_clocked")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    node = a.card if a.node is None else a.node
    Path(a.workdir).mkdir(parents=True, exist_ok=True)

    hold = aiclk_hold._Hold([node], a.mhz)
    legs = []
    try:
        time.sleep(1.0)
        print("[clock] node %d forced to %d MHz, reads %d MHz before the first leg"
              % (node, a.mhz, aiclk(node)), flush=True)
        for i, arm in enumerate(a.order.split(",")):
            tag = "%s%d_s%d" % (arm, i, a.seed)
            print("[leg] %s starting %s" % (tag, time.strftime("%H:%M:%SZ", time.gmtime())),
                  flush=True)
            r = leg(a, arm, tag)
            clk = r["aiclk"].get(str(node), {})
            print("[leg] %s rc=%s trunk=%ss wall=%ss served=%d fb=%d picks=%s "
                  "aiclk median=%s min=%s n=%s"
                  % (tag, r["rc"], r["trunk_s"], r["wall_s"], r["wide_k_served"], r["fell_back"],
                     r["picks"], clk.get("median"), clk.get("min"), clk.get("n")), flush=True)
            legs.append(r)
    finally:
        reasserts = hold.reasserts
        hold.release()

    off = [l["trunk_s"] for l in legs if l["arm"] == "off" and l["trunk_s"]]
    on = [l["trunk_s"] for l in legs if l["arm"] == "on" and l["trunk_s"]]
    rep = {"host": os.uname().nodename, "card": a.card, "node": node, "mhz_target": a.mhz,
           "reasserts": reasserts, "model": a.model, "input": a.input, "seed": a.seed,
           "sampling_steps": a.steps, "order": a.order, "legs": legs,
           "trunk_off_s": off, "trunk_on_s": on,
           "aa_floor_s": round(max(off) - min(off), 1) if len(off) > 1 else None,
           "mean_off_s": round(sum(off) / len(off), 1) if off else None,
           "mean_on_s": round(sum(on) / len(on), 1) if on else None}
    if off and on:
        rep["speedup"] = round((sum(off) / len(off)) / (sum(on) / len(on)), 4)
        rep["delta_s"] = round(sum(off) / len(off) - sum(on) / len(on), 1)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "legs"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
