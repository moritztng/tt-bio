#!/usr/bin/env python3
"""The template pair stack's VJP graded at wave 10's COMPOSED configuration.

`perf/bcx_p10_stack/grad_compose.py` grades the `hifi` route. This is the same grader with the
three levers wave 10 stacks on top of it armed as well, because a gradient bar taken with the
backward lever off says nothing about a stack whose whole point is a different backward:

    --triatt-bw     `triatt_bw.FUSED`, the fused triangle-attention backward
    --rne-kernel    `AF2PairBlock.rne_kernel` plus the `rne_add` tape entry
    TT_BIO_MM_LAYOUT   a core grid for the plan-less batched matmuls

`vjp.py` is adopted unchanged, as it was there: float64 reference, bfloat16 CPU arm beside it as
the floor no correct device implementation can beat, device arm inside `--bar` times that floor.
Rewriting it would mean grading against a second reference.

Every lever's reach counter is printed and written beside the reading. An arm whose counter is
zero graded nothing, and on the VJP that is the likely failure: `triatt_bw` only fires under a
backward that actually enters `autograd.triangle_attention`.

    python3 perf/bcx_p10_stack2/grad_stack2.py --arm on  --out out_acc/grad_on
    python3 perf/bcx_p10_stack2/grad_stack2.py --arm off --out out_acc/grad_off
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["on", "off"], required=True,
                    help="`off` is bcx-p10-stack's hifi route, the anchor the campaign's "
                         "1.0421x was taken on. `on` is that route plus wave 10's three levers")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stack", default="template")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    on = a.arm == "on"

    # Both arms take the hifi route: it is the anchor, not a lever under test here.
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"
    # Resolved at import, so it has to be in the environment before tt_bio loads.
    os.environ["TT_BIO_TAPED_KERNELS"] = ",".join(
        ["tri_att_sdpa_hifi"] + (["rne_add"] if on else []))
    os.environ["TT_BIO_TRIATT_FUSED_HIFI"] = "1"
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1"
    os.environ["TT_BIO_TRIATT_BW_FUSED"] = "1" if on else "0"
    os.environ["TT_BIO_MM_LAYOUT"] = "1" if on else "0"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_tmplemb"))
    sys.argv = ["vjp.py", "--n", str(a.n), "--reps", str(a.reps), "--stack", a.stack,
                "--out", a.out]

    from tt_bio.af2 import AF2PairBlock
    AF2PairBlock.rne_kernel = on

    import vjp
    vjp.main()

    from tt_bio import mm_layout, reblock_permute, rne_add, taped_ttnn, tenstorrent, triatt_bw
    reach = {"arm": a.arm,
             "fused_hifi_stats": dict(tenstorrent.TRIATT_FUSED_HIFI_STATS),
             "triatt_bw_stats": dict(triatt_bw.STATS),
             "triatt_bw_fused": bool(triatt_bw.FUSED),
             "rne_add_stats": list(rne_add.STATS),
             "rne_kernel": AF2PairBlock.rne_kernel,
             "mm_layout": dict(mm_layout.reach()),
             "mm_layout_on": bool(mm_layout.MM_LAYOUT),
             "widen_add": bool(rne_add.WIDEN_ADD),
             "widen_reach": dict(rne_add.WIDEN_REACH),
             "taped_channel_move": bool(reblock_permute.TAPED_MOVE),
             "kernel_entry_stats": {k: list(v) for k, v in taped_ttnn.KERNEL_STATS.items()}}
    print("REACH " + json.dumps(reach))
    # vjp.py treats --out as a directory prefix, so the counters go beside its file.
    d = pathlib.Path(a.out)
    (d / "reach.json" if d.is_dir() else d.with_suffix(".reach.json")).write_text(
        json.dumps(reach, indent=1))


if __name__ == "__main__":
    main()
