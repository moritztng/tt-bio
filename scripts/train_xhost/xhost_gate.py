#!/usr/bin/env python3
"""Data-parallel scaling for ABodyBuilder3 across CHIPS and across HOSTS, measured on this model.

    PYTHONPATH=$PWD python3 scripts/train_xhost/xhost_gate.py \
        --arms tt-quietbox:0 tt-quietbox:0,tt-quietbox:1 tt-quietbox:0,tt-quietbox2:1 \
        --steps 4 --rounds 1

Each arm is a comma-separated list of ``host:chip``, one entry per rank. Ranks on this host are
launched directly; ranks elsewhere are launched over ssh, in the foreground with `-n`, so a rank
cannot outlive the launcher that started it.

**The rank's ``--chips`` are logical dp indices, not physical chips.** The physical chip is set
per rank through ``TT_VISIBLE_DEVICES`` on the process that opens the device, because two hosts
can both offer a chip 0 and ``tt_bio/train/mesh.py`` refuses a repeated id on one axis. Which
node each rank actually held is then read back from its OWN ``/proc/<pid>/fd`` by the provenance
recorder, since ``TT_VISIBLE_DEVICES=N`` does not select ``/dev/tenstorrent/N``.

**Every host runs the same rendezvous protocol over its own directory.** The spec handed to each
rank names its local directory plus the other hosts as peers, and `tt_bio/train/xhost.py` mirrors
the files. Nothing about the reduce changes with the host count, which is the point.

AICLK is sampled DURING the run by each rank on its own host, so a two-host arm reports two
clocks and neither is taken on trust from the other box.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = socket.gethostname()


def ssh_dest(host: str, user: str) -> str:
    return f"{user}@{host}"


def rendezvous_for(host: str, hosts: list, base: str, user: str) -> str:
    peers = [ssh_dest(h, user) for h in dict.fromkeys(hosts) if h != host]
    return base + ("+" + ",".join(peers) if peers else "")


def rank_cmd(rank: int, world: int, chip: int, rv: str, out: str, repo: str, python: str,
             args) -> list:
    return [python, f"{repo}/scripts/abb3_port/repro.py",
            "--out", out, "--steps", str(args.steps),
            "--global-batch", str(args.global_batch), "--micro", str(args.micro),
            "--tokens", str(args.tokens), "--blocks", str(args.blocks),
            "--seed", str(args.seed), "--rank", str(rank), "--world", str(world),
            "--chips", ",".join(str(i) for i in range(world)),
            "--rendezvous", rv, "--checkpoint-minutes", "600", "--no-resume",
            "--data", "synthetic", "--torch-threads", str(args.torch_threads)]


def launch(rank: int, place: tuple, hosts: list, args, out_root: Path) -> subprocess.Popen:
    host, chip = place
    remote = host != LOCAL
    repo = args.remote_repo if remote else str(REPO)
    python = args.remote_python if remote else sys.executable
    out = f"{args.remote_out}/{out_root.name}" if remote else str(out_root)
    rv = rendezvous_for(host, hosts, args.rendezvous, args.user)
    env = {"TT_VISIBLE_DEVICES": str(chip), "TT_BIO_LEASE_CARDS": str(chip),
           "TT_BIO_LEASE_HOLDER": "worker:train-j-multihost", "PYTHONPATH": repo,
           "ABB3_RUN_ID": args.run_id}
    # Only when asked for. Torch's default thread count is what b3 measured 1.888x under, and
    # a cap set here would make this arm a different recipe from the one it is compared to.
    if args.torch_threads:
        env["OMP_NUM_THREADS"] = str(args.torch_threads)
    cmd = rank_cmd(rank, len(hosts), chip, rv, out, repo, python, args)
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fh = open(logs / f"rank{rank}.log", "a", buffering=1)
    if remote:
        # One quoted shell line, run in the FOREGROUND under ssh -n: a backgrounded remote
        # command outlives the channel and keeps the far card, and an ssh without -n competes
        # for this process's stdin.
        line = " ".join(f"{k}={v}" for k, v in env.items()) + " " + \
            " ".join(str(c) for c in cmd)
        full = ["ssh", "-n", "-o", "BatchMode=yes", ssh_dest(host, args.user),
                f"mkdir -p {out} && cd {repo} && {line}"]
        proc = subprocess.Popen(full, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    else:
        e = dict(os.environ)
        e.update(env)
        proc = subprocess.Popen(cmd, env=e, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=repo, start_new_session=True)
    proc._log, proc._place = fh, place
    return proc


def wipe(hosts: list, args) -> None:
    base, _, _ = args.rendezvous.partition("+")
    for h in dict.fromkeys(hosts):
        if h == LOCAL:
            shutil.rmtree(base, ignore_errors=True)
        else:
            subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", ssh_dest(h, args.user),
                            f"rm -rf {base}"], check=False)


def fetch(hosts: list, args, out_root: Path) -> None:
    """Bring every remote rank's history and provenance back to the launching host."""
    for h in dict.fromkeys(hosts):
        if h == LOCAL:
            continue
        src = f"{ssh_dest(h, args.user)}:{args.remote_out}/{out_root.name}/"
        subprocess.run(["rsync", "-a", "--include", "*.jsonl", "--include", "*.json",
                        "--include", "COMPLETE*", "--exclude", "*", src, f"{out_root}/"],
                       check=False)


def arm(places: list, args, root: Path, round_no: int) -> dict:
    hosts = [p[0] for p in places]
    world = len(places)
    tag = f"w{world}-{'x' if len(set(hosts)) > 1 else 'l'}-r{round_no}"
    out_root = root / tag
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    wipe(hosts, args)
    print(f"\n[xhost] {tag}: {places}", flush=True)
    t0 = time.monotonic()
    procs = [launch(r, p, hosts, args, out_root) for r, p in enumerate(places)]
    rcs = []
    try:
        for p in procs:
            rcs.append(p.wait(timeout=args.arm_timeout))
    except subprocess.TimeoutExpired:
        for p in procs:
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), 9)
        rcs.append(-1)
    seconds = time.monotonic() - t0
    fetch(hosts, args, out_root)
    rows, prov = {}, {}
    for r in range(world):
        f = out_root / f"history-rank{r}.jsonl"
        rows[r] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] \
            if f.exists() else []
        q = out_root / f"provenance-rank{r}.json"
        prov[r] = json.loads(q.read_text()) if q.exists() else {}
    # The slowest rank sets the step: every step ends at a rendezvous, so the arm's step time
    # is the per-step max across ranks, not rank 0's view of it.
    n = min((len(v) for v in rows.values()), default=0)
    per_step = [max(rows[r][i]["wall"] for r in rows) for i in range(n)]
    walls = per_step[1:]  # the first step carries a fresh process's warmup
    digests = {r: [x["digest"] for x in rows[r]] for r in rows}
    agree = len({tuple(v) for v in digests.values()}) == 1 if n else False
    distinct = len({d for v in digests.values() for d in v[:n]}) if n else 0
    nodes = {r: prov[r].get("device_nodes") for r in prov}
    clocks = {r: prov[r].get("aiclk", {}) for r in prov}
    waits = {r: (rows[r][-1].get("stages", {}) if rows[r] else {}) for r in rows}
    return {"tag": tag, "places": places, "rcs": rcs, "walls": walls,
            "first": per_step[0] if per_step else None,
            "median": statistics.median(walls) if walls else None,
            "steps": n, "seconds": seconds, "agree": agree, "distinct_digests": distinct,
            "nodes": nodes, "clocks": clocks, "stages": waits,
            "cotenancy": {r: prov[r].get("config", {}).get("cotenancy", {}) for r in prov}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="each arm is host:chip[,host:chip...] in rank order")
    ap.add_argument("--out", default="runs/xhost-gate")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-xhost")
    ap.add_argument("--user", default="ttuser")
    ap.add_argument("--remote-repo", default="/home/ttuser/xhost-j/tt-bio")
    ap.add_argument("--remote-python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    ap.add_argument("--remote-out", default="/home/ttuser/xhost-j/runs")
    ap.add_argument("--arm-timeout", type=float, default=1800.0)
    ap.add_argument("--run-id", default="xhostj")
    args = ap.parse_args()
    arms = [[(s.split(":")[0], int(s.split(":")[1])) for s in a.split(",")] for a in args.arms]
    root = Path(args.out)

    results = []
    for rd in range(args.rounds):
        for places in arms:
            results.append(arm(places, args, root, rd))

    base = None
    print(f"\n{'arm':>12} {'ranks':>6} {'median step s':>14} {'speedup':>8} {'eff':>7} "
          f"{'agree':>6} {'distinct':>9} {'rc':>10}")
    for a in results:
        w = len(a["places"])
        if a["median"] is None:
            print(f"{a['tag']:>12} {w:>6}  no usable arm (rc {a['rcs']})")
            continue
        if base is None:
            base = a["median"]
        print(f"{a['tag']:>12} {w:>6} {a['median']:>14.3f} {base / a['median']:>7.3f}x "
              f"{100 * (base / a['median']) / w:>6.1f}% {'yes' if a['agree'] else 'NO':>6} "
              f"{a['distinct_digests']:>9} {str(a['rcs']):>10}")
        for r, p in enumerate(a["places"]):
            c = a["clocks"].get(r, {})
            print(f"             rank {r} on {p[0]} chip {p[1]}: nodes {a['nodes'].get(r)}, "
                  f"AICLK {c.get('median')} med / {c.get('min')} min, {c.get('samples')} smp")
    allnodes = [(a["tag"], r, tuple(a["nodes"].get(r) or ())) for a in results
                for r in range(len(a["places"]))]
    print(f"\nNODES: {allnodes}")
    bad = [a["tag"] for a in results if a["median"] is not None and not a["agree"]]
    if bad:
        print(f"FAIL: master digests disagree across ranks on {bad}; a throughput number from "
              f"diverged replicas is not scaling")
        return 1
    print("PASS: every rank's master digest matched at every step on every arm, across the "
          "host boundary as well as within a host")
    print(json.dumps([{k: v for k, v in a.items() if k != 'stages'} for a in results],
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
