#!/usr/bin/env python3
"""The 1-chip vs 2-chip arm, interleaved, with both arms' clocks and cotenants recorded.

Interleaved and not all-A-then-all-B: on a shared host an all-A-then-all-B pair read +13.3 %
where the interleaved one read +5.2 %, so the ordering is the measurement and not a detail.
Each repeat is a fresh process per arm, which is what the launcher does anyway, so neither arm
is ever warm from the other and both pay their own kernel compile on the first step. That step
is excluded from every median by ``launcher.tick()`` returning ``None`` for it.

    python3 perf/train_d_dp/scale.py --cards 2,0 --repeats 3 --steps 20

Reports aggregate examples/s per arm, the speedup, the parallel efficiency, and the collective's
own cost beside the barrier wait. It does NOT average the two arms' clocks together: a number
whose AICLK differed between arms is not a comparison, so both are printed and a spread is
called out.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BENCH = HERE / "dpbench.py"


def cotenants():
    """Every other process on this host holding a /dev/tenstorrent node, named.

    A timed run on a shared box is only comparable with the neighbours written down beside it,
    and the neighbours are read off their own fds for the same reason the ranks' nodes are.
    """
    out = []
    me = os.getpid()
    for p in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(p.name)
            if pid == me:
                continue
            nodes = sorted({int(os.readlink(f).rsplit("/", 1)[1])
                            for f in p.glob("fd/*")
                            if "/dev/tenstorrent/" in (os.readlink(f) if f.is_symlink() else "")})
            if nodes:
                out.append({"pid": pid, "nodes": nodes,
                            "cmd": (p / "cmdline").read_bytes().decode(
                                errors="replace").replace("\0", " ")[:120]})
        except (OSError, ValueError, PermissionError):
            continue
    return out


def arm(cards, a, tag):
    env = dict(os.environ)
    env["TT_BIO_LEASE_CARDS"] = ",".join(str(c) for c in cards)
    env.setdefault("TT_BIO_LEASE_HOLDER", "worker:train-d-dp-launcher")
    if len(cards) == 1:
        env["TT_VISIBLE_DEVICES"] = str(cards[0])
    else:
        env.pop("TT_VISIBLE_DEVICES", None)
    cmd = [sys.executable, "-u", str(BENCH), "--cards", ",".join(str(c) for c in cards),
           "--steps", str(a.steps), "--tokens", str(a.tokens), "--channels", str(a.channels),
           "--blocks", str(a.blocks), "--examples", str(a.examples),
           "--lora-rank", str(a.lora_rank), "--out", a.out, "--tag", tag]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{tag} exited {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    res = json.loads((Path(a.out) / tag / "result.json").read_text())
    res["arm_wall_s"] = time.perf_counter() - t0
    print(f"  {tag}: world {res['world']} median step {res['median_step_s']:.4f} s, "
          f"{res['examples_per_s']:.4f} ex/s, nodes {res['nodes']}, "
          f"AICLK {(res['provenance'].get('aiclk') or {}).get('median')} MHz, "
          f"loadavg {res['loadavg'][0]:.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="2,0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--channels", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    cards = [int(c) for c in a.cards.split(",") if c != ""]

    before = cotenants()
    print(f"cotenants at start: {json.dumps(before)}", flush=True)
    runs = {1: [], 2: []}
    for i in range(a.repeats):
        print(f"repeat {i}", flush=True)
        # one chip first on the odd repeats, two first on the even ones, so neither arm
        # systematically lands on the same phase of a neighbour's job
        order = [(cards[:1], 1), (cards, len(cards))] if i % 2 == 0 else \
                [(cards, len(cards)), (cards[:1], 1)]
        for cs, w in order:
            runs[w].append(arm(cs, a, f"r{i}_w{w}_rank{a.lora_rank}"))
    after = cotenants()

    rep = {"cards": cards, "repeats": a.repeats, "steps": a.steps, "tokens": a.tokens,
           "channels": a.channels, "blocks": a.blocks, "lora_rank": a.lora_rank,
           "cotenants_before": before, "cotenants_after": after, "arms": {}}
    for w, rs in runs.items():
        if not rs:
            continue
        ex = [r["examples_per_s"] for r in rs]
        st = [r["median_step_s"] for r in rs]
        clk = [(r["provenance"].get("aiclk") or {}).get("median") for r in rs]
        rep["arms"][str(w)] = {
            "world": w, "n": len(rs),
            "examples_per_s_median": statistics.median(ex),
            "examples_per_s": ex,
            "median_step_s": statistics.median(st), "step_s": st,
            "aiclk_median": statistics.median([c for c in clk if c]) if any(clk) else None,
            "aiclk": clk,
            "nodes": rs[0]["nodes"], "sites": rs[0]["sites"],
            "distinct_master_sha": [r["distinct_master_sha"] for r in rs],
            "comm_bytes": rs[0].get("comm_bytes"),
            "median_transfer_s": rs[0].get("median_transfer_s"),
            "median_barrier_wait_s": rs[0].get("median_barrier_wait_s"),
            "loadavg": [r["loadavg"][0] for r in rs],
        }
    one, two = rep["arms"].get("1"), rep["arms"].get(str(len(cards)))
    if one and two:
        rep["speedup"] = two["examples_per_s_median"] / one["examples_per_s_median"]
        rep["efficiency"] = rep["speedup"] / two["world"]
        rep["ptxft_reference"] = {"speedup": 1.87, "efficiency": 0.935,
                                  "what": "perf/ptxft/dpscale.py, Protenix-v2 distogram LoRA, "
                                          "qb1 chips 1 and 3, 8 steps per rank"}
    out = Path(a.out) / f"scale_rank{a.lora_rank}.json"
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print()
    for w in sorted(rep["arms"], key=int):
        x = rep["arms"][w]
        print(f"world {x['world']}: {x['examples_per_s_median']:.4f} ex/s median over "
              f"{x['n']} runs, step {x['median_step_s']:.4f} s, nodes {x['nodes']}, "
              f"AICLK {x['aiclk_median']} MHz, loadavg {x['loadavg']}", flush=True)
    if "speedup" in rep:
        print(f"SPEEDUP {rep['speedup']:.4f}x on {two['world']} chips, efficiency "
              f"{100 * rep['efficiency']:.2f} % (ptxft: 1.87x / 93.5 %)", flush=True)
        print(f"all-reduce {two['comm_bytes'] / 1e6:.3f} MB in "
              f"{1e3 * (two['median_transfer_s'] or 0):.2f} ms, barrier wait "
              f"{1e3 * (two['median_barrier_wait_s'] or 0):.2f} ms, against a "
              f"{two['median_step_s']:.4f} s step", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
