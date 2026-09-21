#!/usr/bin/env python3
"""Why TT_BIO_SOFTMAX_CKC is bit-exactly inert on the trunks 288 fp32 softmax calls.

The flag passes the CALLERs compute kernel config. On the trunk that config is
HiFi4 + fp32_dest_acc_en + packer_l1_acc with math_approx_mode=True, recorded at the call by a
ttnn.softmax spy. precise_config() differs from it in exactly one field. This isolates which
field the accuracy is in, at the trunks own shape, against a float64 softmax on the same values.
"""
import json, os, sys, time
sys.path.insert(0, os.getcwd())
import torch, ttnn
from tt_bio.autograd import precise_config
from tt_bio.tenstorrent import get_device

def rel(a, b):
    return float(torch.linalg.vector_norm((a - b).double()) / torch.linalg.vector_norm(b.double()))

def cfg(fid, approx, fp32acc, packer):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx,
                                            fp32_dest_acc_en=fp32acc, packer_l1_acc=packer)

H4 = ttnn.MathFidelity.HiFi4
ARMS = [("none (op default)", None),
        ("trunk caller: HiFi4, approx=True, fp32acc, packer", cfg(H4, True, True, True)),
        ("same but math_approx_mode=False", cfg(H4, False, True, True)),
        ("precise_config()", precise_config())]

dev = get_device()
out = []
for shape in ((64, 4, 64, 64), (1, 16, 384, 384)):
    for seed in (20260921, 7):
        torch.manual_seed(seed)
        x = (torch.randn(*shape) * 3.0).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        p = torch.softmax(x.double(), dim=-1)
        base, row = None, {"shape": list(shape), "seed": seed, "arms": {}}
        for nm, c in ARMS:
            kw = {} if c is None else {"compute_kernel_config": c}
            y = ttnn.to_torch(ttnn.softmax(xt, dim=-1, **kw))
            if base is None:
                base = y
            row["arms"][nm] = {"rel_l2_vs_float64": rel(y.double(), p),
                               "bit_identical_to_no_config": bool(torch.equal(y, base))}
            print("%-18s seed=%-9d %-52s rel %.4e  bit-identical to no-config %s"
                  % (shape, seed, nm, row["arms"][nm]["rel_l2_vs_float64"],
                     row["arms"][nm]["bit_identical_to_no_config"]), flush=True)
        out.append(row)
json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "what": __doc__.strip().splitlines()[0], "cases": out},
          open("perf/of3t_d116_verify/APPROX.json", "w"), indent=1)
