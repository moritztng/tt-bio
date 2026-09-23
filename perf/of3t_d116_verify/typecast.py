#!/usr/bin/env python3
"""Why TT_BIO_SOFTMAX_CKC is bit-exactly inert at the trunk's crop-64 shape, and at which
width it stops being.

`_fp32_softmax_tail` upcasts bf16 scores to fp32, runs `ttnn.softmax`, and typecasts the
result straight back to bf16. The compute kernel config improves the fp32 softmax; the
typecast then quantises the improvement. This measures the improvement on both sides of that
typecast, at three widths, so "inert" is a property with a boundary rather than one reading.
"""
import json
import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio                                                    # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT)

import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402
from tt_bio.autograd import precise_config                                  # noqa: E402
from tt_bio.tenstorrent import get_device                                   # noqa: E402


def rel(a, b):
    return float(torch.linalg.vector_norm((a - b).double())
                 / torch.linalg.vector_norm(b.double()))


def main():
    dev = get_device()
    rows = []
    for shape in ((64, 4, 64, 64), (1, 16, 384, 384), (1, 4, 1024, 1024)):
        for seed in (20260921, 7):
            torch.manual_seed(seed)
            x = (torch.randn(*shape) * 3.0).to(torch.bfloat16)       # bf16 scores, as the tail gets
            x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            p = torch.softmax(x.double(), dim=-1)
            r = {"shape": list(shape), "seed": seed}
            for nm, cfg in (("none", None), ("precise_config()", precise_config())):
                kw = {} if cfg is None else {"compute_kernel_config": cfg}
                y_tt = ttnn.softmax(x_tt, dim=-1, **kw)
                y32 = ttnn.to_torch(y_tt).double()
                ybf = ttnn.to_torch(ttnn.typecast(y_tt, ttnn.bfloat16)).double()
                r[nm] = {"fp32_out_rel_l2": rel(y32, p), "after_bf16_typecast_rel_l2": rel(ybf, p)}
                r[nm + "_bf16_bits"] = ttnn.to_torch(ttnn.typecast(y_tt, ttnn.bfloat16))
                ttnn.deallocate(y_tt)
            a, b = r.pop("none_bf16_bits"), r.pop("precise_config()_bf16_bits")
            r["bf16_outputs_bit_identical"] = bool(torch.equal(a, b))
            r["gain_before_typecast"] = (r["none"]["fp32_out_rel_l2"]
                                         / r["precise_config()"]["fp32_out_rel_l2"])
            r["gain_after_typecast"] = (r["none"]["after_bf16_typecast_rel_l2"]
                                        / r["precise_config()"]["after_bf16_typecast_rel_l2"])
            rows.append(r)
            print("%-20s seed=%-9d gain fp32 %6.2fx -> after bf16 typecast %5.2fx   "
                  "bf16 bit-identical %s" % (shape, seed, r["gain_before_typecast"],
                                             r["gain_after_typecast"],
                                             r["bf16_outputs_bit_identical"]), flush=True)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "what": __doc__.strip().splitlines()[0], "cases": rows}
    with open("perf/of3t_d116_verify/TYPECAST.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
