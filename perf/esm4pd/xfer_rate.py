#!/usr/bin/env python3
'''L-H screen: is a device->host crossing link-bound or host-untilize bound?

The boundary census of the predecessor measured 268.4 MB (fp32) crossings at 0.153 s each.
That is 0.877 GB/s of actual bf16 link bytes against a x16 PCIe link, so >90 % of the
crossing cannot be the link. `TorchWrapper._to_torch` is `ttnn.to_torch(x).to(float32)`:
a host-side untilize of a TILE_LAYOUT tensor on one thread, then a second full host pass to
widen. This times the three candidate spellings at the real pair shape.

Bit-exactness is checked, not assumed: untilize is a pure layout change and bf16 -> fp32 is
lossless in both directions, so every arm must be torch.equal to the shipped one.
'''
import json, os, statistics as st, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T

REPS = 5
dev = T.get_device()
g = dev.compute_with_storage_grid_size()
res = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
       'grid': [g.x, g.y], 'reps': REPS, 'shapes': []}

def timeit(fn, n=REPS):
    fn(); ttnn.synchronize_device(dev)
    v = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t)
    return st.median(v), out

for shape in ([1, 512, 512, 256], [1, 512, 512, 128]):
    x = ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), device=dev,
                        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    mb_bf16 = 2 * torch.tensor(shape).prod().item() / 1e6

    def shipped():
        return torch.Tensor(ttnn.to_torch(x)).to(torch.float32)

    def dev_untilize():
        r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out = torch.Tensor(ttnn.to_torch(r)).to(torch.float32)
        ttnn.deallocate(r)
        return out

    def bf16_only():
        return ttnn.to_torch(x)

    rows = {}
    for name, fn in (('shipped', shipped), ('dev_untilize', dev_untilize), ('bf16_only', bf16_only)):
        s, out = timeit(fn)
        rows[name] = {'s': round(s, 5), 'gbs_link_bf16': round(mb_bf16 / 1e3 / s, 2)}
        if name == 'shipped':
            ref = out
        elif name == 'dev_untilize':
            rows[name]['torch_equal_vs_shipped'] = bool(torch.equal(out, ref))
    rows['speedup_dev_untilize'] = round(rows['shipped']['s'] / rows['dev_untilize']['s'], 3)
    res['shapes'].append({'shape': shape, 'mb_bf16': round(mb_bf16, 1), 'arms': rows})
    print(shape, json.dumps(rows), flush=True)
    ttnn.deallocate(x)

Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
