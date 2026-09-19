#!/usr/bin/env python3
"""Data-parallel scaling for the ABodyBuilder3 reproduction, measured on this model.

    TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:train-b3-train \
    PYTHONPATH=$PWD python3 scripts/abb3_port/dp_gate.py --chips 0,1 --steps 4

r5's 1.87x on two chips at 93.5 % is **Protenix**, whose step is almost entirely on the card.
ABodyBuilder3's is not: 58 % of its complete step is host torch, so its data-parallel scaling
is a different question with a different answer and it has to be measured rather than inherited.

**It is a LADDER, not a pair.** ``--ladder 1,2,4`` runs three rungs off the same chip list, and
every rung is scored against the 1-chip arm measured in the SAME session on the SAME host. A
baseline from another host or another day is a comparison and never a denominator.

**The arms are interleaved, and the rung order reverses on odd rounds.** A compile cache warms and
a host settles, so two arms run back to back attribute that drift to whichever ran second.
Interleaving and taking the median per arm removes it, which is the standing rule for any A/B on
this fleet; with three rungs the reversal is what keeps the widest arm from always running last.

**The headline ratio is the CADENCE, not the device step.** ``wall`` times ``step.step()`` and
nothing else, so a loss to host contention between processes lands in ``data`` and ``outer`` and a
ladder scored on ``wall`` alone cannot see it. The history row is built so that
``t[n+1] - t[n] == data + wall + outer`` exactly, and that sum is what a throughput number is made
of. Both are printed, and the reduce is timed inside the optimizer stage it sits in, so a rung that
scales badly says whether it went to the exchange or to the host.

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
           "--rendezvous", args.rendezvous, "--data", args.data, "--split", args.split]
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
    # Attribution, per rank and settled (the first step carries a fresh process' warmup).
    # `data` is the host micro-batch build and upload, `losses`+`host_backward` is the rest of
    # the host torch, `reduce_s` is the exchange the wider arm pays and the narrow one does not.
    # These are read per rank because a step ends at a rendezvous, so the arm's cadence is set
    # by its SLOWEST rank and a mean over ranks would hide exactly that.
    att = {}
    for r in rows:
        s = rows[r][1:]
        if not s:
            continue
        att[r] = {k: round(statistics.median([x.get(k) or 0.0 for x in s]), 4)
                  for k in ("wall", "data", "outer", "reduce_s", "reduce_wait_s", "reduce_mb")}
        for k in ("forward", "device_backward", "losses", "host_backward", "optimizer"):
            att[r][k] = round(statistics.median([x["stages"].get(k, 0.0) for x in s]), 4)
        # The cadence a watcher sees, which is what a throughput number is made of:
        # t[n+1]-t[n] == data + wall + outer exactly, by construction of the history row.
        att[r]["cadence"] = round(att[r]["data"] + att[r]["wall"] + att[r]["outer"], 4)
    agree = len({tuple(v) for v in digests.values()}) == 1 if rows[0] else False
    # Cleanliness is read from EVERY rank, not just rank 0: a cotenant on rank 1's chip
    # contaminates the arm just as thoroughly, and the arm's step time is set by its slowest
    # rank because each step ends at a rendezvous.
    co, nodes = {}, {}
    for r in range(len(chips)):
        q = out / f"provenance-rank{r}.json"
        if q.exists():
            pv = json.loads(q.read_text())
            co[r] = pv.get("config", {}).get("cotenancy", {})
            # Each rank's node comes from THAT rank's provenance, which the sampler read off
            # this very process' own /proc/<pid>/fd. Rank 0's file names rank 0's chip and
            # nothing else, so reading the arm's node set out of it would report one node for a
            # four-rank arm -- and TT_VISIBLE_DEVICES=N is the mapping nobody may assume.
            nodes[r] = pv.get("device_nodes") or []
    clean = bool(co) and all(c.get("clean") for c in co.values())
    flat_nodes = [n for r in sorted(nodes) for n in nodes[r]]
    return {"rc": rc, "chips": chips, "walls": walls, "first": rows[0][0]["wall"] if rows[0] else None,
            "median": statistics.median(walls) if walls else None,
            "seconds": time.monotonic() - t0, "aiclk": prov.get("aiclk", {}),
            "nodes": flat_nodes, "nodes_per_rank": nodes,
            "digests_agree": agree, "cotenancy": co,
            "clean": clean, "attribution": att,
            "steps": len(rows[0]), "stages": rows[0][-1]["stages"] if rows[0] else {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/dp-gate")
    ap.add_argument("--chips", default="0,1")
    ap.add_argument("--ladder", default=None,
                    help="world sizes to measure, e.g. 1,2,4. Each takes the first w chips. "
                         "Default is the pair 1,len(chips)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-dp-gate")
    ap.add_argument("--data", default="synthetic", help="synthetic | sabdab")
    ap.add_argument("--split", default="train")
    ap.add_argument("--json", default=None, help="write the whole report here")
    args = ap.parse_args()
    chips = [int(c) for c in args.chips.split(",")]
    rungs = ([int(w) for w in args.ladder.split(",")] if args.ladder
             else sorted({1, len(chips)}))
    for w in rungs:
        if w > len(chips):
            print(f"rung {w} needs {w} chips and --chips names {len(chips)}")
            return 2
        if args.global_batch % (w * args.micro):
            print(f"rung {w}: global batch {args.global_batch} does not split into {w} ranks "
                  f"of whole {args.micro}-sample micro-batches. The global batch is the axis "
                  f"the recipe pins, so it is the world that moves, never the batch")
            return 2
    root = Path(args.out)

    results = {w: [] for w in rungs}
    for r in range(args.rounds):
        # Reversed on odd rounds so no rung systematically runs last, when the host has settled
        # and the compile cache is warmest.
        for w in (rungs if r % 2 == 0 else list(reversed(rungs))):
            results[w].append(arm(root, chips[:w], args, r))

    # The arm's cadence is its SLOWEST rank's, because every step ends at a rendezvous and the
    # world moves at the rank that arrives last. A mean over ranks would hide the skew that is
    # the whole question at four chips.
    def slowest(a, key):
        return max((v.get(key, 0.0) for v in a["attribution"].values()), default=0.0)

    hdr = (f"{'world':>5} {'cadence':>9} {'wall':>8} {'speedup':>8} {'eff':>7} {'data':>7} "
           f"{'outer':>7} {'hosttorch':>10} {'reduce':>8} {'wait':>7} {'MB':>7} "
           f"{'AICLK m/min':>12} {'agree':>6} {'clean':>6}")
    print("\n" + hdr)
    base = None
    lines = []
    rep = {"chips": chips, "rungs": rungs, "rounds": args.rounds, "steps": args.steps,
           "micro": args.micro, "global_batch": args.global_batch, "tokens": args.tokens,
           "blocks": args.blocks, "seed": args.seed, "data": args.data, "arms": {}}
    for w in rungs:
        arms = [a for a in results[w] if a["median"] is not None]
        if not arms:
            print(f"{w:>5}  no usable arm (rc {[a['rc'] for a in results[w]]})")
            continue
        cad = statistics.median([slowest(a, "cadence") for a in arms])
        med = statistics.median([a["median"] for a in arms])
        if base is None:
            base = cad
        clk = arms[-1]["aiclk"]
        clkstr = f"{clk.get('median')}/{clk.get('min')}" if clk.get("median") else "NONE"
        agree = all(a["digests_agree"] for a in arms)
        clean = all(a["clean"] for a in arms)

        def med_of(key):
            return round(statistics.median([slowest(a, key) for a in arms]), 4)

        host = round(statistics.median(
            [slowest(a, "losses") + slowest(a, "host_backward") for a in arms]), 4)
        row = {"world": w, "n": len(arms), "cadence_s": round(cad, 4), "wall_s": round(med, 4),
               "speedup": round(base / cad, 4), "efficiency": round((base / cad) / w, 4),
               "cadence_per_round": [round(slowest(a, "cadence"), 3) for a in arms],
               "wall_per_round": [round(a["median"], 3) for a in arms],
               "data_s": med_of("data"), "outer_s": med_of("outer"), "host_torch_s": host,
               "forward_s": med_of("forward"), "device_backward_s": med_of("device_backward"),
               "optimizer_s": med_of("optimizer"),
               "reduce_s": med_of("reduce_s"), "reduce_wait_s": med_of("reduce_wait_s"),
               "reduce_mb": med_of("reduce_mb"), "aiclk": clk,
               "nodes_per_round": [a["nodes"] for a in arms],
               "digests_agree": agree, "clean": clean,
               "attribution_per_rank": [a["attribution"] for a in arms]}
        rep["arms"][str(w)] = row
        lines.append(row)
        print(f"{w:>5} {cad:>9.3f} {med:>8.3f} {base / cad:>7.3f}x {100 * (base / cad) / w:>6.1f}% "
              f"{row['data_s']:>7.3f} {row['outer_s']:>7.3f} {host:>10.3f} "
              f"{row['reduce_s']:>8.3f} {row['reduce_wait_s']:>7.3f} {row['reduce_mb']:>7.2f} "
              f"{clkstr:>12} {'yes' if agree else 'NO':>6} {'yes' if clean else 'NO':>6}")
        for a in arms:
            for r, c in sorted(a["cotenancy"].items()):
                if not c.get("clean"):
                    print(f"        rank {r} nodes {c.get('nodes')}: "
                          f"{c.get('same_node_samples')}/{c.get('samples')} samples with a "
                          f"foreign holder of OUR node, "
                          f"{c.get('elsewhere_samples')}/{c.get('samples')} elsewhere")

    # NODES: asserted distinct per rank, read off each rank's own /proc/<pid>/fd by the
    # provenance recorder. TT_VISIBLE_DEVICES=N does not select /dev/tenstorrent/N, and at four
    # ranks that is four chances to put two ranks on one chip and still show a speedup.
    print("\nNODES: " + "; ".join(f"world {w} -> {[a['nodes'] for a in results[w]]}"
                                  for w in rungs if results[w]))
    bad_nodes = [(w, a.get("nodes")) for w in rungs for a in results[w]
                 if not a.get("nodes") or len(set(a["nodes"])) != w]
    if bad_nodes:
        print(f"FAIL: a rank set did not resolve to as many DISTINCT /dev/tenstorrent nodes as "
              f"it had ranks: {bad_nodes}. Two ranks on one chip still shows a speedup")
        return 1
    if lines:
        print(f"STAGES (widest arm, per rank, median settled step, seconds): "
              f"{lines[-1]['attribution_per_rank'][-1]}")
    bad = [ln["world"] for ln in lines if not ln["digests_agree"]]
    if bad:
        print(f"FAIL: master digests disagree across ranks on world {bad}; a throughput number "
              f"from diverged replicas is not scaling")
        return 1
    print("SYNC: 1 distinct master hash on every arm (" +
          ", ".join(f"world {ln['world']}" for ln in lines) +
          "), so the ratios above are one model trained faster, not N models trained apart")
    dirty = [ln["world"] for ln in lines if not ln["clean"]]
    if dirty:
        print(f"CONTENDED: world {dirty} ran beside a foreign device holder for part of the "
              f"measurement, so its step time is an UPPER bound and its speedup a LOWER one.")
    else:
        print("CLEAN: no foreign process held any Tenstorrent node on this host at any sample "
              "DURING any arm, so these medians are clean readings rather than upper bounds")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
