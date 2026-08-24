#!/usr/bin/env python3
"""p113 -- what is inside the token encoder's pairformer blocks, 91.486 ms/step.

E11.6 item 3. p111 put `pairformer[0] + [1]` at 91.486 ms/step, half the token encoder region and
the largest located-but-unscreened item on the board, and it timed them as one block each. This
splits them.

A `PairformerBlock` is three submodules and some residual bookkeeping:

    z_transition   Transition(c_z=128, n=4)  -> the H=512 pair Transition, already levered by
                                                _PAIR_TRANSITION_L1 (7.058 s/design landed)
    attn           PairformerAttention        -> the census's `pf attn`, 20 ops, 3.10 GB/step
    s_transition   Transition(c_s=384, n=4)

**This instrument wraps the real methods; it does not copy them.** p46 itemises by replacing
`run_device` with a hand-written duplicate, that duplicate drifted when `_CONCAT_ALIGNED` landed,
and it then reported a 25.584 ms/call op that had not shipped for weeks -- at the top of the
ranking it exists to produce (E11.2). So here `Transition.__call__` and
`PairformerAttention.__call__` are wrapped at the class level and only fire their timer for
instances a `PairformerBlock` has tagged. Whatever the block actually calls is what gets timed,
and the file cannot go stale.

Same two reading rules as p46. The sum of sync-bracketed submodules overshoots the block's own
wall, because each sync drains work that would otherwise overlap -- so the block wall is printed
beside the sum and the RANKING is the output, not the absolute. And the GB/s column would need a
measured roof, so it is not printed at all rather than printed against an asserted one.
"""
import collections
import json
import os
import pathlib
import statistics
import sys
import time

import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p113/pairformer.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
COLD = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
BLOCKS_PER_STEP = 4                     # 2 pairformer blocks x 2 recycles

TIMES = collections.defaultdict(list)
_orig_tr = M.Transition.__call__
_orig_attn = M.PairformerAttention.__call__
_orig_blk = M.PairformerBlock.__call__


def _wrap(orig):
    def w(self, *a, **k):
        tag = getattr(self, "_p113_tag", None)
        if tag is None:                          # a Transition somewhere else in the model
            return orig(self, *a, **k)
        dev = self.device
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = orig(self, *a, **k)
        ttnn.synchronize_device(dev)
        TIMES[tag].append((time.perf_counter() - t0) * 1e3)
        return out
    return w


M.Transition.__call__ = _wrap(_orig_tr)
M.PairformerAttention.__call__ = _wrap(_orig_attn)


def _blk(self, s, z):
    self.z_transition._p113_tag = "z_transition"
    self.s_transition._p113_tag = "s_transition"
    self.attn._p113_tag = "attn"
    dev = self.device
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out = _orig_blk(self, s, z)
    ttnn.synchronize_device(dev)
    TIMES["BLOCK"].append((time.perf_counter() - t0) * 1e3)
    return out


M.PairformerBlock.__call__ = _blk


def main():
    print("[p113] steps=%d cold=%d card=%s" % (STEPS, COLD, os.environ.get("TT_VISIBLE_DEVICES")),
          flush=True)
    out_dir = "/tmp/rfd3_p113"
    os.system("rm -rf %s" % out_dir)
    rfd3_design.run_design(json.loads(FIXTURE.read_text()), out_dir, checkpoint_dir=CKPT,
                           from_pdb=True, num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)

    n_blk = len(TIMES["BLOCK"])
    drop = COLD * BLOCKS_PER_STEP
    if n_blk <= drop:
        raise SystemExit("[p113] only %d block calls, need more than %d" % (n_blk, drop))
    # Drop the cold prefix. TokenInitializer runs its own pairformer stack once before the
    # diffusion loop, so the prefix also absorbs those.
    kept = {k: v[drop:] for k, v in TIMES.items()}
    n_kept_blk = len(kept["BLOCK"])
    steps_counted = n_kept_blk / BLOCKS_PER_STEP

    rows = {}
    for k, v in kept.items():
        med = statistics.median(v)
        rows[k] = dict(calls=len(v), ms_per_call=round(med, 4),
                       ms_per_step=round(sum(v) / steps_counted, 4),
                       s_per_design=round(sum(v) / steps_counted * 200 / 1000.0, 3))

    sub = sum(rows[k]["ms_per_step"] for k in ("z_transition", "attn", "s_transition")
              if k in rows)
    blk = rows["BLOCK"]["ms_per_step"]
    print("\n=== token encoder pairformer, %d block calls kept (%.1f steps) ==="
          % (n_kept_blk, steps_counted), flush=True)
    print("%-14s %6s %10s %10s %12s" % ("submodule", "calls", "ms/call", "ms/step", "s/design"),
          flush=True)
    for k in ("z_transition", "attn", "s_transition", "BLOCK"):
        if k not in rows:
            continue
        r = rows[k]
        print("%-14s %6d %10.4f %10.4f %12.3f"
              % (k, r["calls"], r["ms_per_call"], r["ms_per_step"], r["s_per_design"]), flush=True)
    print("%-14s %6s %10s %10.4f" % ("SUM of subs", "", "", sub), flush=True)
    print("%-14s %6s %10s %10.4f   oversync %.2fx"
          % ("BLOCK wall", "", "", blk, sub / blk if blk else float("nan")), flush=True)
    resid = blk - sub
    print("%-14s %6s %10s %10.4f   (residual adds/typecasts in the block)"
          % ("unattributed", "", "", resid), flush=True)
    print("\n[p113] p111 measured pairformer[0]+[1] at 91.486 ms/step; this run's BLOCK total is "
          "%.3f ms/step" % blk, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(steps=STEPS, cold=COLD, seed=SEED, fixture=str(FIXTURE),
                                   blocks_per_step=BLOCKS_PER_STEP,
                                   block_calls_total=n_blk, block_calls_kept=n_kept_blk,
                                   steps_counted=steps_counted, rows=rows,
                                   sum_of_subs_ms_per_step=round(sub, 4),
                                   block_wall_ms_per_step=round(blk, 4),
                                   oversync=round(sub / blk, 4) if blk else None,
                                   unattributed_ms_per_step=round(resid, 4),
                                   p111_pairformer_ms_per_step=91.486,
                                   card=os.environ.get("TT_VISIBLE_DEVICES"),
                                   host=os.uname().nodename), indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
