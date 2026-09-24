"""Where esmfold2's fold time goes with the rows view off and on: every `ops.linear` call
sync-timed and bucketed by operand shape, beside the fold's wall.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=0,3 python perf/bcx_oplin/diag_esmfold2.py out.json

A synchronize before and after each call serialises the pipeline, so the absolute seconds are
not the served fold's; the two arms pay the same distortion, which is what makes them comparable.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn

import tt_bio.ops as ops
from common import Clock, arm


def main(out):
    import tt_baseline as B
    one_fold, meta, _ = B.build_fold("esmfold2", Path(tempfile.mkdtemp(prefix="oplin-msa-")),
                                     ROOT / "perf/size512/fixtures/cdk2x2_512.yaml",
                                     ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    inner = ops._via2d
    rows = {}

    def timed(x, fn, kw=None):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        y = ops_via(x, fn, kw)
        ttnn.synchronize_device(dev)
        k = str([int(d) for d in x.shape])
        r = rows[arm_name].setdefault(k, [0, 0.0])
        r[0] += 1; r[1] += time.perf_counter() - t0
        return y

    res = {"meta": {k: meta.get(k) for k in ("hardware", "card_type", "grid")}}
    one_fold()  # cold, arm on
    for arm_name, on in (("off", False), ("on", True), ("off2", False), ("on2", True)):
        rows[arm_name] = {}
        with arm(on):
            ops_via = ops._via2d
            ops._via2d = timed
            try:
                with Clock() as clk:
                    t, _m = one_fold()
            finally:
                ops._via2d = ops_via
        tot = sum(v[1] for v in rows[arm_name].values())
        res[arm_name] = dict(wall_s=round(t, 3), linear_s=round(tot, 3), aiclk=clk.stats(),
                             top=sorted(([k, v[0], round(v[1], 4)] for k, v in rows[arm_name].items()),
                                        key=lambda r: -r[2])[:10])
        print(arm_name, json.dumps(res[arm_name]), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
