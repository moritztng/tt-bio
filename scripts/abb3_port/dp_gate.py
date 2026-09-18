#!/usr/bin/env python3
"""Data-parallel scaling for the ABodyBuilder3 reproduction, measured on this model.

    TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:train-b3-train \
    PYTHONPATH=$PWD python3 scripts/abb3_port/dp_gate.py --chips 0,1 --steps 4

r5's 1.87x on two chips at 93.5 % is **Protenix**, whose step is almost entirely on the card.
ABodyBuilder3's is not: 58 % of its complete step is host torch, so its data-parallel scaling
is a different question with a different answer and it has to be measured rather than inherited.

**The arms are interleaved, 1 chip then N then 1 then N.** A compile cache warms and a host
settles, so two arms run back to back attribute that drift to whichever ran second. Interleaving
and taking the median per arm removes it, which is the standing rule for any A/B on this fleet.

The global batch is pinned at 64 on both arms, so what changes between them is only how the same
work is divided -- never the recipe. Each rank divides its micro-batch losses by the GLOBAL
accumulation count, so the cross-rank sum is the mean over 64 samples exactly, and a wider arm is
not quietly a larger learning rate.

The per-rank master-weight digests are compared every step by the run itself, so a speedup from
ranks that diverged cannot be reported as scaling. This script also reads the digests back out of
the histories and says so explicitly, because that is the claim a reader needs beside the ratio.
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


def arm(out: Path, chips: list, args, round_no: int) -> dict:
    out = out / f"w{len(chips)}-r{round_no}"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(args.rendezvous, ignore_errors=True)
    cmd = [sys.executable, str(HERE / "supervise.py"), "--out", str(out),
           "--steps", str(args.steps), "--chips", ",".join(str(c) for c in chips),
           "--micro", str(args.micro), "--tokens", str(args.tokens),
           "--blocks", str(args.blocks), "--seed", str(args.seed),
           "--global-batch", str(args.global_batch),
           "--checkpoint-minutes", "600", "--max-restarts", "0",
           "--rendezvous", args.rendezvous]
    print(f"\n[dp] world {len(chips)} on chips {chips}, round {round_no}", flush=True)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    rows = {}
    for r in range(len(chips)):
        path = out / f"history-rank{r}.jsonl"
        rows[r] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] \
            if path.exists() else []
    # The first step carries the warmup of a fresh process, so it is excluded from the median
    # and reported separately rather than dropped silently.
    walls = [x["wall"] for x in rows[0][1:]]
    prov = {}
    p = out / "provenance-rank0.json"
    if p.exists():
        prov = json.loads(p.read_text())
    digests = {r: [x["digest"] for x in rows[r]] for r in rows}
    agree = len({tuple(v) for v in digests.values()}) == 1 if rows[0] else False
    # Cleanliness is read from EVERY rank, not just rank 0: a cotenant on rank 1's chip
    # contaminates the arm just as thoroughly, and the arm's step time is set by its slowest
    # rank because each step ends at a rendezvous.
    co = {}
    for r in range(len(chips)):
        q = out / f"provenance-rank{r}.json"
        if q.exists():
            co[r] = json.loads(q.read_text()).get("config", {}).get("cotenancy", {})
    clean = bool(co) and all(c.get("clean") for c in co.values())
    return {"rc": rc, "chips": chips, "walls": walls, "first": rows[0][0]["wall"] if rows[0] else None,
            "median": statistics.median(walls) if walls else None,
            "seconds": time.monotonic() - t0, "aiclk": prov.get("aiclk", {}),
            "nodes": prov.get("device_nodes"), "digests_agree": agree, "cotenancy": co,
            "clean": clean,
            "steps": len(rows[0]), "stages": rows[0][-1]["stages"] if rows[0] else {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/dp-gate")
    ap.add_argument("--chips", default="0,1")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-dp-gate")
    args = ap.parse_args()
    chips = [int(c) for c in args.chips.split(",")]
    root = Path(args.out)

    results = {1: [], len(chips): []}
    for r in range(args.rounds):
        results[1].append(arm(root, chips[:1], args, r))
        if len(chips) > 1:
            results[len(chips)].append(arm(root, chips, args, r))

    print(f"\n{'world':>6} {'median step s':>14} {'per round':>22} {'speedup':>8} "
          f"{'efficiency':>11} {'AICLK during':>26} {'ranks agree':>12} {'clean':>7}")
    base = None
    lines = []
    for w in sorted(results):
        arms = [a for a in results[w] if a["median"] is not None]
        if not arms:
            print(f"{w:>6}  no usable arm (rc {[a['rc'] for a in results[w]]})")
            continue
        med = statistics.median([a["median"] for a in arms])
        if base is None:
            base = med
        clk = arms[-1]["aiclk"]
        clkstr = (f"{clk.get('median')} med / {clk.get('min')} min, {clk.get('samples')} smp"
                  if clk.get("median") else "NO SAMPLES")
        agree = all(a["digests_agree"] for a in arms)
        clean = all(a["clean"] for a in arms)
        lines.append((w, med, base / med, (base / med) / w, clkstr, agree, clean))
        print(f"{w:>6} {med:>14.3f} {str([round(a['median'], 2) for a in arms]):>22} "
              f"{base / med:>8.3f}x {100 * (base / med) / w:>10.1f}% {clkstr:>26} "
              f"{'yes' if agree else 'NO':>12} {'yes' if clean else 'NO':>7}")
        for a in arms:
            for r, c in sorted(a["cotenancy"].items()):
                if not c.get("clean"):
                    print(f"         rank {r} nodes {c.get('nodes')}: "
                          f"{c.get('same_node_samples')}/{c.get('samples')} samples with a "
                          f"foreign holder of OUR node {list(c.get('same_node', {}).values())[:2]}, "
                          f"{c.get('elsewhere_samples')}/{c.get('samples')} elsewhere "
                          f"{list(c.get('elsewhere', {}).values())[:2]}")
    last = results[max(results)][-1]
    print(f"\nSTAGES (last step of the widest arm, seconds): "
          f"{ {k: round(v, 2) for k, v in last['stages'].items()} }")
    print(f"NODES: rank 0 held {last['nodes']}; every rank's own node is read from its own "
          f"/proc/<pid>/fd by the provenance recorder, since TT_VISIBLE_DEVICES=N does not "
          f"select /dev/tenstorrent/N")
    bad = [ln[0] for ln in lines if not ln[5]]
    if bad:
        print(f"FAIL: master digests disagree across ranks on world {bad}; a throughput number "
              f"from diverged replicas is not scaling")
        return 1
    dirty = [ln[0] for ln in lines if not ln[6]]
    print("PASS: every rank's master digest matched at every step on every arm, so the ratios "
          "above are one model trained faster rather than N models trained separately")
    if dirty:
        # Not a FAIL: the ratio is still real, it is just an upper bound on time and therefore
        # a LOWER bound on the speedup. Said out loud because a contaminated number that is not
        # labelled gets quoted as a clean one, which is worse than a missing number.
        print(f"CONTENDED: world {dirty} ran beside a foreign device holder for part of the "
              f"measurement, so its step time is an UPPER bound. Re-take on an idle pair "
              f"before quoting it as the curve.")
    else:
        print("CLEAN: no foreign process held any Tenstorrent node on this host at any sample "
              "DURING either arm, so these medians are clean readings rather than upper bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
