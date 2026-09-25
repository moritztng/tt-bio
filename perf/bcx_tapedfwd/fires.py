#!/usr/bin/env python3
"""Which fused forwards fire in BindCraft 2's TAPED gradient forward, counted at runtime.

BC2's round differs from OpenFold3's step in the one way that matters here. BC2 runs TWO trunk
forwards per round (`bindcraft/af2.py:139-143`): a stop-gradient recycle that is UNTAPED, and the
differentiated pass that is taped. So BC2 loses the fused routes on only one of its two forwards,
and the untaped one is the control -- in the same process, on the same shapes, minutes apart.

That makes a second question BC2's own, which OF3T does not have: the recycle is untaped, so it
SHOULD serve. At the campaign's own arm it does not. hPDL1 chain A is 115 residues and the
`step288` binder is 146, so the complex is 261 and BindCraft 2's own `length_bucket_size` 32 pads
it to 288 -- and 288 is the one hole in the fused route's serve sweep (`bcx-forward`'s
`serves.json`: served at 192, 224, 256, 320, 384; declined at 288 on `fill_preconditions`). A
decline there is not the tape's doing and is not by design.

So this harness answers both halves at both sizes: 288, the arm BC2 actually runs, and 256, a
size the route is known to serve, as the positive control that proves the instrument can see a
serve at all.

It does not count kernels, it counts CALL SITES. `tt_bio.ops.taping` is replaced by a function
that records its caller's frame and returns whatever the arm wants, so it sees every tape-gated
route in the engine -- 32 sites across 10 modules on this tree, of which only 14 are `generic_op`
kernels; the rest are L1 residency, an in-place pair add and a DRAM/L1 memory-config switch that
have nothing to do with `generic_op` and are not fixed by writing a backward.

Three arms, same inputs, same block counts, interleaved so drift lands on all three:

  A  untaped, `taping()` -> False   the inference forward, every lever live
  B  untaped, `taping()` -> True    EXACTLY the route set the taped forward takes, with none of
                                    the tape's own recording cost. A minus B is what the
                                    tape-gated levers cost, with no confound
  C  taped                          the real thing. C minus B is the tape's own overhead

A/B is the number this row owes; running B untaped is what separates "the levers are off" from
"recording a tape is slow", which a taped-vs-untaped A/B cannot do at all.

    fires.py --ns 288,256 --reps 3 --out out/fires_288.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "perf" / "bcx_forward"))
import forward as F                                                    # noqa: E402
from perf.bcx_stack.stack import Clock                                 # noqa: E402

#: Every `[served, declined]` counter a fused forward on BC2's trunk keeps. Read as a delta
#: around one forward, so a counter that is busy in arm A and zero in arm B is the bypass made
#: visible per route rather than in aggregate.
COUNTERS = [
    "tt_bio.triatt_sdpa.STATS", "tt_bio.triatt_sdpa.GATE_STATS",
    "tt_bio.triatt_qkv.STATS", "tt_bio.triatt_qkv.TAIL_STATS",
    "tt_bio.triatt_qkv.QKVG_STATS", "tt_bio.triatt_qkv.QKVGB_STATS",
    "tt_bio.reblock_permute.STATS", "tt_bio.reblock_permute.STATS_BACK",
    "tt_bio.reblock_permute.STATS_GATED",
    "tt_bio.trimul_tail.STATS", "tt_bio.trimul_tail.OUT_L1_STATS",
    "tt_bio.swiglu_fused.STATS", "tt_bio.mm_dualnoc.STATS",
    "tt_bio.softmax_generic.SSTATS", "tt_bio.softmax_generic.PVSTATS",
    "tt_bio.tenstorrent.TRIATT_FUSED_HIFI_STATS",
    # Every refusal LATCH, read as a set length. These are the caches the recorded suspicion is
    # about: if a taped call still poisons one, the arm that follows serves less. A latch that
    # GROWS during arm B or C is the defect `69a0a6bdc` fixed, recurring.
    "tt_bio.tenstorrent._TRIATT_HIFI_OVER_L1", "tt_bio.tenstorrent._SDPA_Q_CHUNK_OVER_L1",
    "tt_bio.tenstorrent._L1_OUT_REFUSED", "tt_bio.tenstorrent._BMM_CFG_REFUSED",
    "tt_bio.triatt_sdpa._PM_OVER_L1", "tt_bio.triatt_sdpa._GATE_OVER_L1",
]

#: Reject-reason dicts, so a decline says WHY and not just that it happened.
REJECT_DICTS = ["tt_bio.triatt_sdpa.REJECTS", "tt_bio.triatt_sdpa.GATE_REJECTS",
                "tt_bio.triatt_qkv.REJECTS", "tt_bio.reblock_permute.REJECTS",
                "tt_bio.trimul_tail.REJECTS", "tt_bio.swiglu_fused.REJECTS",
                "tt_bio.mm_dualnoc.REJECTS"]


def _get(attr):
    mod, _, name = attr.rpartition(".")
    m = sys.modules.get(mod)
    return getattr(m, name, None) if m is not None else None


def read_counters():
    """Snapshot every counter by name. Lists, int-dicts and sets only."""
    out = {}
    for attr in COUNTERS:
        v = _get(attr)
        if isinstance(v, list):
            out[attr] = list(v)
        elif isinstance(v, set):
            out[attr] = len(v)
        elif isinstance(v, dict) and all(isinstance(x, int) for x in v.values()):
            out[attr] = dict(v)
    return out


def read_rejects():
    out = {}
    for attr in REJECT_DICTS:
        v = _get(attr)
        if isinstance(v, dict):
            out[attr] = {str(k): int(n) for k, n in v.items()}
    return out


def _delta(after, before):
    """after - before, elementwise, dropping everything that did not move."""
    d = {}
    for k, a in after.items():
        b = before.get(k)
        if isinstance(a, list) and isinstance(b, list):
            v = [x - y for x, y in zip(a, b)]
            if any(v):
                d[k] = v
        elif isinstance(a, int) and isinstance(b, int):
            if a - b:
                d[k] = a - b
        elif isinstance(a, dict) and isinstance(b, dict):
            v = {kk: vv - b.get(kk, 0) for kk, vv in a.items() if vv - b.get(kk, 0)}
            if v:
                d[k] = v
    return d


class Gate:
    """Stands in for `tt_bio.ops.taping`, recording who asked and answering per the arm.

    `force` is None to tell the truth (arm C, where a real tape is open), False for arm A and
    True for arm B. Recording the CALLER's frame is what makes this see all 32 gate sites and
    not only the ones that keep a counter.
    """

    def __init__(self, real):
        self.real = real
        self.force = None
        self.sites = Counter()
        self.on = False

    def __call__(self):
        ans = self.real() if self.force is None else self.force
        if self.on:
            f = sys._getframe(1)
            # `triatt_qkv._taping` / `eltwise_fusion._taping` are one-line shims; charge the site
            # that actually gated, not the shim.
            if f.f_code.co_name == "_taping" and f.f_back is not None:
                f = f.f_back
            self.sites[f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"] += 1
        return ans


def arm_once(dev, lv, gate, m0, z0, ke, kv, arm):
    """One forward under one arm. Returns (span, gate sites, counter deltas, rejects).

    Arm C is the real tape, so the gate must tell the truth there and `force` stays None.
    `taped_step` with `backward=False` stops after the forward, which is this row's scope: the
    backward is `of3t-bwsurvey`'s and `bcx-bwdplan`'s.
    """
    gate.force = {"A": False, "B": True, "C": None}[arm]
    gate.sites.clear()
    c0, r0 = read_counters(), read_rejects()
    gate.on = True
    try:
        if arm == "C":
            span = F.taped_step(dev, lv, m0, z0, None, None, ke, kv,
                                ckpt=True, backward=False)["spans"][0]
        else:
            span = F.untaped_fwd(dev, m0, z0, ke, kv)
    finally:
        gate.on = False
        gate.force = None
    return (span, dict(gate.sites), _delta(read_counters(), c0),
            _delta(read_rejects(), r0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="288,256",
                    help="288 is BC2's own arm (115+146=261 -> bucket 32); 256 is the control")
    ap.add_argument("--arms", default="ABC")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ke", type=int, default=4, help="extra-MSA blocks")
    ap.add_argument("--kv", type=int, default=48, help="Evoformer blocks")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--params", default="/home/moritz/bcx_shipped/af2_params")
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "0"),
                    help="for the stamp's sysfs read; the grant itself is TT_BIO_LEASE_CARDS")
    ap.add_argument("--out", default="out/fires.json")
    args = ap.parse_args()

    import tt_bio.ops as ops
    gate = Gate(ops.taping)
    ops.taping = gate

    lv, dev, ref = F.open_all(args, "stack")
    dev.dm.set_triatt_fused(frozenset(["extra_msa", "evoformer"]))

    clock = Clock(0.25)          # samples this card's own sysfs node from __init__
    rows = []
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, _, _ = F.inputs(ref, n, args.seed)
        F.untaped_fwd(dev, m0, z0, args.ke, args.kv)          # warm: compile, allocate, cache
        per = {a: [] for a in args.arms}
        last = {}
        spans = {a: [] for a in args.arms}
        for _ in range(args.reps):
            for a in args.arms:                                # interleaved, so drift hits all
                span, sites, counters, rejects = arm_once(
                    dev, lv, gate, m0, z0, args.ke, args.kv, a)
                per[a].append(span[1] - span[0])
                spans[a].append(span)
                last[a] = {"sites": sites, "counters": counters, "rejects": rejects}
                print(f"n={n} arm {a}: {span[1] - span[0]:.4f} s", flush=True)
        for a in args.arms:                    # the clock INSIDE each arm's own windows
            last[a]["aiclk"] = clock.window(spans[a])
        row = {"n": n, "blocks": [args.ke, args.kv],
               "secs": {a: {"median": statistics.median(v), "all": v} for a, v in per.items()},
               "arms": last}
        if "A" in per and "B" in per:
            row["A_over_B"] = statistics.median(per["B"]) / statistics.median(per["A"])
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "arms"}), flush=True)
    clock.stop()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stamp": F.stamp(args, clock), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
