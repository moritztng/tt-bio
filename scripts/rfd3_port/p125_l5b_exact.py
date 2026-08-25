#!/usr/bin/env python3
"""p125 -- L5b's bit-exactness gate, and the decline path, across the size ladder.

Gate order is the one `state/rfd3-fusion-programme.md` §4 pre-committed: `torch.equal` at the
production shape comes first and everything after it is gated on it. This script runs only that
first gate, on synthetic operands at the production shape, plus the ladder rungs either side of
it -- including at least one rung where L5b must DECLINE, because a pass that only measures the
addressable rung never exercises the decline path.

Protocol, on pc card 0 (`pc-card0-512aa-fold-nondeterminism`): the card miscomputes some ttnn
matmuls at a low, location-keyed rate. So the SHIPPED path runs three times first and the three
results are compared to each other. If that control is not unanimous the card is miscomputing on
this run and no bit-exactness verdict from it means anything -- reported as `control_failed`, not
as an L5b result.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:... \
      PYTHONPATH=$PWD python3 scripts/rfd3_port/p125_l5b_exact.py [out.json] [rungs]
"""
import json
import os
import pathlib
import sys
import time

import torch
import ttnn

from tt_bio import softmax_generic as SG
from tt_bio.tenstorrent import attn_value_matmul, get_device

TILE = 32
HEADS = 4
HEAD_DIM = 32

# (name, rows, key width). R4 is the census fixture's padded atom axis; R3 is what p122 measured
# on this box; R2 and R1 are the rungs on the other side of the two size gates.
LADDER = [("R1", 2624, 2624), ("R2", 3712, 3712), ("R3", 4576, 4576), ("R4", 6051, 6080)]


def _ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def shipped(scores, vv, ckc):
    attn = SG.softmax_bf16(scores, ttnn.bfloat16)
    out = attn_value_matmul(attn, vv, ckc, ttnn.bfloat16)
    ttnn.deallocate(attn)
    return out


def run_rung(device, name, rows, key_w, ckc, seed):
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    x_t = torch.randn(1, HEADS, rows, key_w, generator=g, dtype=torch.float32) * 4.0
    v_t = torch.randn(1, HEADS, key_w, HEAD_DIM, generator=g, dtype=torch.bfloat16)
    scores = ttnn.from_torch(x_t, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    vv = ttnn.from_torch(v_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    del x_t, v_t

    verdict = SG.pv_classify(scores, vv, ttnn.bfloat16, ckc)
    rec = {"rung": name, "rows": rows, "key_width": key_w,
           "classify": {k: (str(v) if not isinstance(v, (bool, int, float, str)) else v)
                        for k, v in verdict.items()},
           "host_seconds_setup": round(time.time() - t0, 1)}

    # --- the control: the shipped path three times, compared to itself ---
    refs = []
    for _ in range(3):
        o = shipped(scores, vv, ckc)
        refs.append(ttnn.to_torch(o))
        ttnn.deallocate(o)
    control_ok = bool(torch.equal(refs[0], refs[1]) and torch.equal(refs[1], refs[2]))
    rec["control_reps"] = 3
    rec["control_unanimous"] = control_ok

    fused = SG.softmax_pv_fused(scores, vv, ttnn.bfloat16, ckc)
    rec["declined"] = fused is None
    if fused is None:
        # A decline is the correct outcome wherever `classify` says no; it is only a failure when
        # `classify` said yes.
        rec["ok"] = not verdict["ok"]
        rec["note"] = ("declined as classified" if not verdict["ok"]
                       else "classified addressable but declined anyway")
    else:
        f = ttnn.to_torch(fused)
        ttnn.deallocate(fused)
        eq = bool(torch.equal(f, refs[0]))
        d = (f.float() - refs[0].float()).abs()
        rec["torch_equal"] = eq
        rec["maxabs"] = float(d.max())
        rec["mismatched_elements"] = int((d != 0).sum())
        rec["elements"] = int(d.numel())
        rec["ok"] = eq and control_ok
        rec["note"] = ("bit-exact" if eq else
                       ("card control failed, verdict void" if not control_ok
                        else "NOT bit-exact"))
        del f
    for r in refs:
        del r
    ttnn.deallocate(scores)
    ttnn.deallocate(vv)
    rec["host_seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    out_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p125/l5b_exact.json")
    want = sys.argv[2].split(",") if len(sys.argv) > 2 else [r[0] for r in LADDER]
    SG.set_pv_enabled(True)
    # tt_bio's own device, not `ttnn.open_device`: `_attn_value_program_config` reads the L1
    # allocator through `tenstorrent.get_device()`, and a second mesh device in the same process
    # takes the context id out of range and leaves the close path throwing.
    device = get_device()
    ckc = _ckc()
    rungs = []
    try:
        for i, (name, rows, key_w) in enumerate(LADDER):
            if name not in want:
                continue
            r = run_rung(device, name, rows, key_w, ckc, seed=1000 + i)
            rungs.append(r)
            print("%-4s rows=%-5d keyW=%-5d addressable=%-5s declined=%-5s %s  (%.0fs)"
                  % (name, rows, key_w, r["classify"]["ok"], r["declined"],
                     r.get("note", ""), r["host_seconds"]), flush=True)
    finally:
        pass

    res = {"rungs": rungs,
           "all_ok": all(r["ok"] for r in rungs),
           "declines_seen": sum(1 for r in rungs if r["declined"]),
           "exact_seen": sum(1 for r in rungs if r.get("torch_equal")),
           "provisional_on": "pc-card0",
           "env": {k: os.environ.get(k) for k in
                   ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "RFD3_SOFTMAX_PV_FUSED")}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print("\nall_ok=%s  declines=%d  bit-exact=%d  ->  %s"
          % (res["all_ok"], res["declines_seen"], res["exact_seen"], out_path))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
