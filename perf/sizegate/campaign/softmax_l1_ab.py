#!/usr/bin/env python3
"""A/B the fp32-softmax L1 byte budget at one size-ladder rung.

`_FP32_SOFTMAX_L1_BYTES_PER_CORE` (tt_bio/tenstorrent.py) sets the height of the score
block the fp32-softmax tail keeps in L1. The shipped 768 KB was fitted at S=512 on one
part, and it is the only input to the block height at every site that reads it, so the top
of the ladder is where a one-size figure is most likely to be wrong.

Three things are measured per fold, because a runtime alone cannot tell a win from a
no-op:

  runtime_s      out of the fold's own results.json, the same field the ladder records
  fp32_softmax   the capacity census counters (TT_BIO_CAPACITY_CENSUS). If `l1_blocks`
                 and `l1_refused` do not move between arms the budget is INERT at this
                 shape and any runtime difference is noise, not the lever
  cif_sha256     the tail is documented bit-exact under a block-size change, so a digest
                 that moves between arms is a defect and not a tradeoff

Arms are interleaved rep by rep, never run as a block per arm: compile and warm-up bias
plus a drifting AICLK both track wall-clock, so a block design hands the first arm a
different machine from the last.

Usage: softmax_l1_ab.py --card 0 --model openbind --rung 1280 \
           --budgets 524288,786432,917504 --reps 2 --out results.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

WT = pathlib.Path(__file__).resolve().parents[3]
PY = "/home/ttuser/kisoji_p2_fresh/env/bin/python3"
sys.path.insert(0, str(WT / "perf" / "sizegate" / "campaign"))
import card_health as ch  # noqa: E402


def _structure(out_dir: pathlib.Path) -> dict | None:
    spec = importlib.util.spec_from_file_location(
        "sg", WT / "scripts" / "gpu_vs_tt" / "gpu5_accuracy_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cifs = sorted(out_dir.rglob("*.cif"))
    if not cifs:
        return None
    r = mod.gate(cifs[0], None, None)
    return {"cif": cifs[0].name,
            "sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16],
            "n_ca": r["n_ca"], "clash_frac": r.get("clash_frac"),
            "chain_break_frac": r.get("chain_break_frac"),
            "ca_ca_median_A": r.get("ca_ca_median_A"),
            "radius_of_gyration_A": r.get("radius_of_gyration_A"),
            "plddt_mean": r.get("plddt_mean"),
            "geometry_ok": r["pass"], "geometry_fail": r["fail"]}


def _runtime(out_dir: pathlib.Path):
    for f in sorted(out_dir.rglob("results.json")):
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        ts = [r["runtime_s"] for r in rows
              if r.get("status") == "ok" and r.get("runtime_s") is not None]
        if ts:
            return max(ts)
    return None


def _sample_clock(node: pathlib.Path, stop: threading.Event, out: list):
    while not stop.is_set():
        try:
            out.append(int((node / "tt_aiclk").read_text().strip()))
        except Exception:
            pass
        stop.wait(5.0)


def one_fold(card: int, node: pathlib.Path, model: str, rung: int, budget: int,
             work: pathlib.Path, tag: str, timeout_s: int) -> dict:
    out_dir = work / f"out_{tag}"
    census = work / f"cap_{tag}"
    log = work / f"{tag}.log"
    env = dict(os.environ)
    env.update({
        "TT_VISIBLE_DEVICES": str(card),
        "TT_BIO_LEASE_CARDS": f"0,{card}",
        "TT_BIO_LEASE_HOLDER": "worker:cov-ladder-p150a-p3",
        "PYTHONPATH": str(WT),
        "TT_BIO_FP32_SOFTMAX_L1_BYTES_PER_CORE": str(budget),
        "TT_BIO_CAPACITY_CENSUS": str(census),
    })
    cmd = [PY, "-u", "-m", "tt_bio.main", "predict",
           str(WT / "perf" / "size512" / "fixtures" / f"cdk2x2_{rung}.yaml"),
           "--model", model, "--single_sequence", "--sampling_steps", "6",
           "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out_dir)]
    clocks: list = []
    stop = threading.Event()
    t = threading.Thread(target=_sample_clock, args=(node, stop, clocks), daemon=True)
    t.start()
    t0 = time.monotonic()
    with open(log, "w") as fp:
        rc = subprocess.run(cmd, env=env, cwd=str(WT), stdout=fp,
                            stderr=subprocess.STDOUT, timeout=timeout_s).returncode
    wall = round(time.monotonic() - t0, 2)
    stop.set()
    t.join(timeout=15)
    cells = {"calls": 0, "blocked": 0, "blocks": 0, "l1": 0, "l1_blocks": 0,
             "l1_refused": 0, "l1_cores": 0, "dram_narrowed": 0}
    for f in sorted(census.glob("capacity_*.json")) if census.exists() else ():
        try:
            d = json.loads(f.read_text()).get("fp32_softmax") or {}
        except Exception:
            continue
        for k in cells:
            cells[k] += d.get(k) or 0
    cs = sorted(clocks)
    row = {"tag": tag, "budget": budget, "rc": rc, "wall_s": wall,
           "runtime_s": _runtime(out_dir),
           "aiclk": {"min": cs[0], "median": cs[len(cs) // 2], "max": cs[-1],
                     "n": len(cs)} if cs else None,
           "fp32_softmax": cells,
           "structure": _structure(out_dir) if rc == 0 else None}
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--model", default="openbind")
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--budgets", default="524288,786432,917504")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    if subprocess.run([PY, str(WT / "perf" / "sizegate" / "campaign" / "card_health.py"),
                       str(a.card)]).returncode != 0:
        print(f"card {a.card} is not healthy, refusing to measure on it")
        return 3
    node = pathlib.Path(str(ch.node_for_bdf(ch.bdf_for_umd(a.card))))
    budgets = [int(x) for x in a.budgets.split(",")]
    work = WT / "perf" / "sizegate" / "campaign" / f"ab_card{a.card}_{a.model}_{a.rung}"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    # A discarded fold first: the kernels for this shape compile on it, the same reason the
    # ladder recorder discards one fold PER RUNG.
    one_fold(a.card, node, a.model, a.rung, budgets[0], work, "warmup", a.timeout)
    for rep in range(a.reps):
        for b in budgets:
            rows.append(one_fold(a.card, node, a.model, a.rung, b, work,
                                 f"rep{rep}_b{b}", a.timeout))
            a.out.write_text(json.dumps(
                {"card": a.card, "model": a.model, "rung": a.rung, "node": str(node),
                 "budgets": budgets, "reps": a.reps, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
