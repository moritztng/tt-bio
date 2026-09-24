"""esmfold2's outer-product-mean projection timed in place, on the fold's own tensors.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=0,3 python perf/bcx_oplin/diag_inplace.py out.json

`ops.linear` is wrapped for one cold fold. At the first call with a [1,32,512,1024] operand
both arms run 20 times on the live tensors (sync-timed), and the arguments are recorded.
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
from common import arm

OUT = {}


def main(out):
    import tt_baseline as B
    one_fold, meta, _ = B.build_fold("esmfold2", Path(tempfile.mkdtemp(prefix="oplin-msa-")),
                                     ROOT / "perf/size512/fixtures/cdk2x2_512.yaml",
                                     ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    orig = ops.linear

    def spy(x, w, bias=None, **kw):
        if not OUT and [int(d) for d in x.shape] == [1, 32, 512, 1024]:
            desc = lambda t: None if t is None else dict(
                shape=[int(d) for d in t.shape], padded=[int(d) for d in t.padded_shape],
                dtype=str(t.dtype), layout=str(t.layout),
                buffer=str(t.memory_config().buffer_type),
                mem=str(t.memory_config().memory_layout))
            OUT["x"], OUT["w"], OUT["bias"] = desc(x), desc(w), desc(bias)
            OUT["kw"] = {k: str(v) for k, v in kw.items()}
            for name, on in (("off", False), ("on", True), ("off2", False), ("on2", True)):
                with arm(on):
                    orig(x, w, bias=bias, **kw); ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(20):
                        orig(x, w, bias=bias, **kw)
                    ttnn.synchronize_device(dev)
                    OUT[name + "_us"] = round((time.perf_counter() - t0) / 20 * 1e6, 1)
            print(json.dumps(OUT), flush=True)
        return orig(x, w, bias=bias, **kw)

    ops.linear = spy
    try:
        with arm(True):
            one_fold()
    finally:
        ops.linear = orig
    Path(out).write_text(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
