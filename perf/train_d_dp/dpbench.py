#!/usr/bin/env python3
"""1-chip and 2-chip data-parallel scaling for tt_bio.train's launcher.

Run as a PROGRAM, because that is how the launcher works: a wide axis makes
``train.finetune`` hand the run to :func:`tt_bio.train.launcher.drive`, which re-executes this
file once per chip with ``TT_BIO_DP_RANK`` set. The ranks reach the same ``finetune`` call,
each takes its own shard, and the driver comes back with the aggregate. So nothing below
branches on rank: the file is written once and run ``world`` times, and that IS the test of the
launcher.

    python3 perf/train_d_dp/dpbench.py --cards 2            # one chip
    python3 perf/train_d_dp/dpbench.py --cards 2,0          # two chips, one process each

The global batch is the chip count, i.e. one example per rank, so a rank does the same work in
both arms and the aggregate is comparable to ``perf/ptxft/dpscale.py``'s 1.87x. That is the
caller pinning the global batch per arm, which is the only place it may come from -- deriving
it from the chip count inside the recipe is what ``tt_bio/train/sharding.py`` refuses.

Both arms interleave nothing and neither is warm from the other: they are separate processes by
construction, so the first step of each carries its own kernel compile and is dropped from every
median by ``launcher.tick()`` returning ``None`` for it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perf.train_d_dp import model as M          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="2", help="UMD ids on the dp axis, comma separated")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=8192)
    # plan() answers about the PROTENIX forward, whose replica is measured at 256 aa and whose
    # OOM is measured at 512. This trunk is not that forward, so the fit check is asked about
    # the shape it has a measurement for rather than about this one -- a fit answer keyed to a
    # token count that means something else would be a fabricated number.
    ap.add_argument("--plan-tokens", type=int, default=256)
    ap.add_argument("--channels", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--examples", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "perf" / "train_d_dp" / "out"))
    ap.add_argument("--tag", default=None)
    # The live negative control for the sync check. Offsets each rank's seed so the ranks
    # genuinely train different models, which is what the real defect did when `lora_factors`
    # seeded its adapter init from entropy. A harness flag, not a knob in the shipped code: a
    # gate nobody has seen fail is not known to be a gate.
    ap.add_argument("--break-sync", action="store_true",
                    help="perturb each rank's seed so the ranks diverge; the run must FAIL")
    a = ap.parse_args()

    from tt_bio import train
    from tt_bio.train import launcher
    from tt_bio.train.lora import LoraConfig
    from tt_bio.train.mesh import Mesh

    cards = [int(c) for c in a.cards.split(",") if c != ""]
    tag = a.tag or f"world{len(cards)}"
    out = Path(a.out) / tag
    forward, dataset = M.build(tokens=a.tokens, channels=a.channels, blocks=a.blocks,
                               examples=a.examples, seed=a.seed)
    t0 = time.perf_counter()
    run = train.finetune(
        forward, dataset,
        out_dir=str(out / "ckpt"),
        # One example per chip: the global batch is pinned per arm, not derived inside.
        global_batch=len(cards),
        steps=a.steps, objective="af3", weights={"mse": 1.0},
        mesh=Mesh({"dp": cards}), seed=a.seed + (launcher.rank() if a.break_sync else 0),
        lr=a.lr, warmup_steps=10,
        checkpoint_every=max(a.steps, 1), tokens=a.plan_tokens,
        lora=LoraConfig(rank=a.lora_rank))
    wall = time.perf_counter() - t0
    print(run, flush=True)

    if launcher.inside():
        return 0        # a rank reports through the launcher, not through this file
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "tag": tag, "cards": cards, "world": len(cards), "steps": a.steps,
        "tokens": a.tokens, "channels": a.channels, "blocks": a.blocks,
        "lora_rank": a.lora_rank, "global_batch": len(cards),
        "wall_s": wall,
        "sites": len(run.params) or len((run.dp or {}).get("params", [])),
        "history": run.history, "displacement": run.displacement,
        "provenance": run.provenance.as_dict(), "dp": run.dp,
        "lease_holder": os.environ.get("TT_BIO_LEASE_HOLDER"),
        "loadavg": os.getloadavg(),
    }
    if run.dp:
        d = run.dp
        # examples/s aggregated over the chips: one example per rank per step, so the whole
        # axis retires `world` examples per step.
        rec["examples_per_s"] = d["world"] / d["median_step_s"]
        rec["median_step_s"] = d["median_step_s"]
        rec["median_transfer_s"] = d["median_transfer_s"]
        rec["median_barrier_wait_s"] = d["median_barrier_wait_s"]
        rec["comm_bytes"] = d["comm_bytes"]
        rec["nodes"] = d["nodes"]
        rec["distinct_master_sha"] = d["distinct_master_sha"]
    else:
        s = [h["s"] for h in run.history if h.get("s") is not None]
        s.sort()
        med = s[len(s) // 2] if s else None
        rec["median_step_s"] = med
        rec["examples_per_s"] = (1.0 / med) if med else None
        rec["nodes"] = run.provenance.as_dict().get("device_nodes")
        rec["distinct_master_sha"] = 1
    (out / "result.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    clk = (rec["provenance"].get("aiclk") or {})
    print(f"\n{tag}: world {rec['world']} median step {rec['median_step_s']:.4f} s, "
          f"{rec['examples_per_s']:.4f} examples/s, nodes {rec['nodes']}, "
          f"{rec['distinct_master_sha']} distinct master sha, "
          f"AICLK {clk.get('median')} MHz median over {clk.get('samples')} during, "
          f"loadavg {os.getloadavg()[0]:.1f}", flush=True)
    print(f"wrote {out / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
