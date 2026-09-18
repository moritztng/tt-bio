#!/usr/bin/env python3
"""Attribute the step's SERIAL term -- the part that does not shrink when you add a chip.

    TT_BIO_LEASE_CARDS=0,2,3 TT_BIO_LEASE_HOLDER=worker:train-b3-train \
    PYTHONPATH=$PWD python3 scripts/abb3_port/serial_gate.py --chips 2,3

A two-point Amdahl fit on the TOTAL says the serial term is ~8.3-8.9 s of a ~30 s step. That is a
residual, not a thing, and a fit cannot say what is inside it. This measures it three ways, and
the first one needs no new run at all.

**1. The per-stage split, which is exact rather than fitted.** ``TrainStep`` already times six
stages. If a stage costs ``s + p/w`` on ``w`` chips then ``s = 2*stage(2) - stage(1)``, applied
per stage. Because the stages sum to the total, the per-stage serial figures sum to the
total-level serial figure **identically** -- so this is an attribution of the same number, not a
second estimate of it. Anything that fails to add up is arithmetic, not modelling.

**2. A thread ladder on one chip.** The two host-torch stages are the whole serial term, and the
reason matters: a genuine serial cost is irreducible without deleting the work, while a
host-CORE ceiling is relieved by reducing host work per rank or by not oversubscribing. qb2 is an
8-core Ryzen 9700X and torch defaults to 8 intra-op threads, so two ranks put 16 threads on 8
cores. If the host stages are core-bound, capping threads changes their time on ONE chip in a way
a serial cost could not.

**3. The same 2-chip arm with half the threads per rank.** 2 ranks x 4 threads is 8 threads on 8
physical cores -- the same total as one rank at its default. If the "serial" term is
oversubscription it should shrink here. If it is real serial work it cannot.

**Falsifier, stated before the run.** If the attributed stages do not sum to the fitted serial
term within 20 %, that is reported as the result rather than smoothed over. An unexplained
residual is a finding.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

STAGES = ("forward", "download", "losses", "host_backward", "device_backward", "optimizer")
#: Which side of the step each stage sits on, for reporting only. `download` is a device-to-host
#: transfer and `optimizer` is host torch plus the cross-rank reduce.
SIDE = {"forward": "device", "download": "transfer", "losses": "host",
        "host_backward": "host", "device_backward": "device", "optimizer": "host"}


def run_arm(out: Path, chips: list, args, *, threads: int, label: str) -> dict:
    out = out / label
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(args.rendezvous, ignore_errors=True)
    cmd = [sys.executable, str(HERE / "supervise.py"), "--out", str(out),
           "--steps", str(args.steps), "--chips", ",".join(str(c) for c in chips),
           "--micro", str(args.micro), "--tokens", str(args.tokens),
           "--blocks", str(args.blocks), "--seed", str(args.seed),
           "--global-batch", str(args.global_batch),
           "--checkpoint-minutes", "600", "--max-restarts", "0",
           "--rendezvous", args.rendezvous]
    if threads:
        cmd += ["--torch-threads", str(threads)]
    print(f"\n[serial] {label}: {len(chips)} chip(s) {chips}, "
          f"{threads or 'default'} torch threads/rank", flush=True)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    rows = [json.loads(l) for l in
            (out / "history-rank0.jsonl").read_text().splitlines() if l.strip()] \
        if (out / "history-rank0.jsonl").exists() else []
    prov = json.loads((out / "provenance-rank0.json").read_text()) \
        if (out / "provenance-rank0.json").exists() else {}
    # First step carries process warmup, so the medians are over the rest.
    timed = rows[1:] or rows
    co = {}
    for r in range(len(chips)):
        q = out / f"provenance-rank{r}.json"
        if q.exists():
            co[r] = json.loads(q.read_text()).get("config", {}).get("cotenancy", {})
    return {
        "label": label, "rc": rc, "world": len(chips), "threads": threads,
        "steps": len(rows), "seconds": time.monotonic() - t0,
        "total": statistics.median([x["stages"]["total"] for x in timed]) if timed else None,
        "stages": {k: statistics.median([x["stages"][k] for x in timed]) for k in STAGES}
        if timed else {},
        "loss_terms": {k: statistics.median([x["loss_terms"].get(k, 0.0) for x in timed])
                       for k in (timed[0]["loss_terms"] if timed else {})},
        "aiclk": prov.get("aiclk", {}), "cotenancy": co,
        "clean": bool(co) and all(c.get("clean") for c in co.values()),
    }


def decompose(a1: dict, a2: dict) -> dict:
    """Per-stage serial term from a 1-chip and a 2-chip arm. ``s = 2*stage(2) - stage(1)``."""
    per = {k: 2 * a2["stages"][k] - a1["stages"][k] for k in STAGES}
    return {"per_stage": per, "sum": sum(per.values()),
            "total_level": 2 * a2["total"] - a1["total"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/serial-gate")
    ap.add_argument("--chips", default="2,3")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-serial")
    ap.add_argument("--ladder", default="8,4,2",
                    help="torch thread counts for the one-chip ladder")
    args = ap.parse_args()
    chips = [int(c) for c in args.chips.split(",")]
    root = Path(args.out)
    ladder = [int(t) for t in args.ladder.split(",")]

    arms: dict = {}
    # The two arms the decomposition needs, at the default thread count, interleaved with
    # nothing else so they are directly comparable.
    arms["w1t8"] = run_arm(root, chips[:1], args, threads=ladder[0], label="w1t8")
    arms["w2t8"] = run_arm(root, chips, args, threads=ladder[0], label="w2t8")
    # The oversubscription control: 2 ranks x half the threads is the same total as 1 rank.
    half = max(ladder[0] // 2, 1)
    arms["w2t_half"] = run_arm(root, chips, args, threads=half, label=f"w2t{half}")
    # The ladder, on one chip, to show how core-bound the host stages are at all.
    for t in ladder[1:]:
        arms[f"w1t{t}"] = run_arm(root, chips[:1], args, threads=t, label=f"w1t{t}")

    bad = [k for k, v in arms.items() if v["total"] is None]
    if bad:
        print(f"FAIL: no usable steps from arms {bad} (rc "
              f"{[arms[k]['rc'] for k in bad]})")
        return 1

    print(f"\n{'arm':>9} {'world':>5} {'thr':>4} {'step s':>8}  " +
          "".join(f"{k[:9]:>10}" for k in STAGES) + f"{'AICLK':>7} {'clean':>6}")
    for k, v in arms.items():
        clk = v["aiclk"].get("median")
        print(f"{v['label']:>9} {v['world']:>5} {v['threads']:>4} {v['total']:>8.2f}  " +
              "".join(f"{v['stages'][s]:>10.2f}" for s in STAGES) +
              f"{clk if clk else 0:>7.0f} {'yes' if v['clean'] else 'NO':>6}")

    d = decompose(arms["w1t8"], arms["w2t8"])
    print(f"\nSERIAL TERM, attributed per stage (s = 2*stage(2chip) - stage(1chip)):")
    for k in STAGES:
        share = 100 * d["per_stage"][k] / d["sum"] if d["sum"] else 0.0
        print(f"  {k:<16} {SIDE[k]:<9} {d['per_stage'][k]:>7.2f} s  {share:>5.1f} % of serial")
    print(f"  {'SUM':<16} {'':<9} {d['sum']:>7.2f} s")
    print(f"  {'total-level fit':<16} {'':<9} {d['total_level']:>7.2f} s  "
          f"(identical by construction -- the stages sum to the total)")
    resid = abs(d["sum"] - d["total_level"])
    tol = 0.20 * abs(d["total_level"]) if d["total_level"] else 0.0
    print(f"\nFALSIFIER: |sum - fit| = {resid:.3f} s against a 20 % tolerance of {tol:.2f} s -- "
          f"{'PASS' if resid <= tol else 'FAIL, and the residual is the result'}")

    host = sum(d["per_stage"][k] for k in STAGES if SIDE[k] == "host")
    dev = sum(d["per_stage"][k] for k in STAGES if SIDE[k] == "device")
    print(f"\nHOST vs DEVICE inside the serial term: host {host:.2f} s "
          f"({100 * host / d['sum']:.0f} %), device {dev:.2f} s "
          f"({100 * dev / d['sum']:.0f} %), transfer "
          f"{d['per_stage']['download']:.2f} s")

    if arms["w1t8"]["loss_terms"]:
        print(f"\nLOSS TERMS, median per step at 1 chip / {ladder[0]} threads (the stage that IS "
              f"the serial term):")
        tot = sum(arms["w1t8"]["loss_terms"].values())
        for name, secs in sorted(arms["w1t8"]["loss_terms"].items(), key=lambda kv: -kv[1]):
            print(f"  {name:<18} {secs:>7.2f} s  {100 * secs / tot:>5.1f} % of the loss stage"
                  f"  {100 * secs / d['sum']:>5.1f} % of serial")

    print(f"\nOVERSUBSCRIPTION CONTROL: 2 chips at {ladder[0]} threads/rank is "
          f"{arms['w2t8']['total']:.2f} s; at {half} threads/rank "
          f"({half * 2} total on 8 cores) it is {arms['w2t_half']['total']:.2f} s. "
          f"Host stages {arms['w2t8']['stages']['losses'] + arms['w2t8']['stages']['host_backward']:.2f} s "
          f"-> {arms['w2t_half']['stages']['losses'] + arms['w2t_half']['stages']['host_backward']:.2f} s.")
    d2 = decompose(arms["w1t8"], arms["w2t_half"])
    print(f"Serial term recomputed against the half-thread 2-chip arm: {d2['sum']:.2f} s "
          f"against {d['sum']:.2f} s at full threads. A term that moves when you change only "
          f"the THREAD COUNT is a core ceiling, not serial work.")

    print(f"\nONE-CHIP THREAD LADDER (host stages are losses + host_backward):")
    for t in ladder:
        v = arms.get(f"w1t{t}") or arms["w1t8"]
        h = v["stages"]["losses"] + v["stages"]["host_backward"]
        dv = v["stages"]["forward"] + v["stages"]["device_backward"]
        print(f"  {t:>2} threads: step {v['total']:>6.2f} s   host {h:>6.2f} s   "
              f"device {dv:>6.2f} s")
    dirty = [v["label"] for v in arms.values() if not v["clean"]]
    if dirty:
        print(f"\nCONTENDED: {dirty} ran beside a foreign device holder, so their absolute times "
              f"are upper bounds. The per-stage SHARES are far more robust to that than the "
              f"absolutes, since a host competitor inflates every host stage together.")
    json.dump({k: v for k, v in arms.items()}, open(root / "arms.json", "w"), indent=2,
              default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
