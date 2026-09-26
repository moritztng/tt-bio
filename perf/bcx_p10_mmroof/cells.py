#!/usr/bin/env python3
"""bcx-p10-mmroof leg 1: cell E censused by shape AND plan, on the COMPOSED round.

Two things make this different from `perf/bcx_p10_mmlay/cells.py`, and both are the brief's.

**The key is wider.** `plankey.PlanOpTimer` writes the live program config's factory and field
values into the second key, so the 2316 calls `bcx-p10-mmlay` declined as "has a program config"
stop being one row and become the classes they actually are.

**The levers are ON.** `bcx-p10-calls`'s leg 1 census was taken with `--triatt-bw`, `--rne-kernel`
and `TT_BIO_MM_LAYOUT` all off, at a 9.525 s device column. The round this row is optimising is
the composed one, and `bcx-p10-tabwire` alone removed ~95 ttnn verbs of triangle-attention
backward -- 33 % of `matmul` by the census's own call-site column. A lever priced against the
stale census is pricing wave 10 a second time, so the arming here mirrors
`perf/bcx_round/run_round.py`'s, field for field, and the stamp records what was armed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_p10_mmroof import plankey as PK        # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_mmroof" / "out"


def arm(triatt_bw: int, rne_kernel: int, hifi: int, mm_layout: int) -> dict:
    """`run_round.py:160-186` verbatim, minus the pieces cell E already does itself.

    Cell E's own spec sets the hifi route through `SH.set_hifi`, so the only thing this adds
    on that axis is the taped-kernel registry entry `rne_add` needs to exist beside it.
    """
    kernels = [n for n, on in (("tri_att_sdpa_hifi", hifi), ("rne_add", rne_kernel)) if on]
    os.environ["TT_BIO_TAPED_KERNELS"] = ",".join(kernels)
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1" if hifi else "0"
    os.environ["TT_BIO_MM_LAYOUT"] = str(int(mm_layout))

    from tt_bio.af2 import AF2PairBlock
    from tt_bio import tenstorrent as _tn
    from tt_bio import triatt_bw as _tbw
    from tt_bio import mm_layout as _mm
    AF2PairBlock.rne_kernel = bool(rne_kernel)
    _tn._TRIATT_FUSED_HIFI = bool(hifi)
    _tbw.FUSED = bool(triatt_bw)
    # `mm_layout.MM_LAYOUT` is resolved at import, like `_TRIATT_FUSED_HIFI`, so the env var
    # above is not enough on its own once tt_bio is imported. Set the module attribute too and
    # read it back into the stamp, so an arm that armed nothing says so.
    _mm.MM_LAYOUT = bool(mm_layout)
    if triatt_bw and not hifi:
        raise SystemExit("--triatt-bw needs the hifi route to be reachable; cell E carries it")
    return {"triatt_bw_fused": bool(_tbw.FUSED), "rne_kernel": bool(AF2PairBlock.rne_kernel),
            "triatt_hifi": bool(_tn._TRIATT_FUSED_HIFI), "mm_layout": bool(_mm.MM_LAYOUT),
            "taped_kernels": os.environ["TT_BIO_TAPED_KERNELS"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--cells", default="E")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="cells_mmroof_E.json")
    ap.add_argument("--verb-records", action="store_true", default=True)
    ap.add_argument("--triatt-bw", dest="triatt_bw", type=int, default=1)
    ap.add_argument("--rne-kernel", dest="rne_kernel", type=int, default=1)
    ap.add_argument("--hifi", type=int, default=1)
    ap.add_argument("--mm-layout", dest="mm_layout", type=int, default=1)
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)

    armed = arm(args.triatt_bw, args.rne_kernel, args.hifi, args.mm_layout)
    print("ARMED %s" % json.dumps(armed), flush=True)

    SH.D.OpTimer = PK.PlanOpTimer             # the whole diff, as in bcx-p10-mmlay
    SH.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    SH.cmd_cells(args)

    # The reach counters, at the end of the run rather than never: an arm whose levers served
    # nothing measured the unarmed round and has to say so beside its own table.
    from tt_bio import mm_layout as _mm
    from tt_bio import triatt_bw as _tbw
    path = OUT / args.out
    blob = json.loads(path.read_text())
    blob["armed"] = armed
    blob["reach"] = {"mm_layout": _mm.reach(), "triatt_bw": dict(_tbw.STATS)}
    path.write_text(json.dumps(blob, indent=1, default=str))
    print("REACH %s" % json.dumps(blob["reach"]), flush=True)


if __name__ == "__main__":
    main()
