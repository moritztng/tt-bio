"""How many copies of its input does the monolithic host float64 softmax hold live?

NOT the attribution -- that is of3t-restep's phase-tagged RSS profile, and the retained tape is
an independent term. This bounds ONE term: the peak-to-tensor ratio of
`torch.softmax(x.double(), dim)` as `autograd.host_f64_softmax_values` writes it on main. The
ratio is dimensionless, so it is measured at a quarter of the production rows and applies at
[384,4,384,384]. No device is opened and no engine file is read. pc only.
"""
import json
import resource

import torch

ROWS, HEADS, N = 96, 4, 384
FULL = 384 * HEADS * N * N


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


x = torch.empty(ROWS, HEADS, N, N, dtype=torch.float32).uniform_(-3, 3)
base = peak_mib()

# Exactly what host_f64_softmax_values does, minus the ttnn crossings at either end.
y = torch.softmax(x.double(), dim=-1)
mono_peak = peak_mib()

yr = y.clone()
del y

xc = x.clone()
yc = torch.empty(ROWS, HEADS, N, N, dtype=torch.float64)
before_chunk = peak_mib()
for i in range(0, ROWS, 8):
    yc[i:i + 8] = torch.softmax(xc[i:i + 8].double(), dim=-1)
chunk_peak = peak_mib()

t_mib = x.numel() * 8 / 2 ** 20
print(json.dumps({
    "input_elems": x.numel(),
    "fp64_tensor_mib": round(t_mib, 1),
    "rss_base_mib": round(base, 1),
    "rss_after_monolithic_mib": round(mono_peak, 1),
    "monolithic_growth_mib": round(mono_peak - base, 1),
    "monolithic_growth_over_fp64_tensor": round((mono_peak - base) / t_mib, 2),
    "chunked_rows8_growth_mib": round(chunk_peak - before_chunk, 1),
    "bit_identical": bool(torch.equal(yr, yc)),
    "full_shape_elems": FULL,
    "full_fp64_gib": round(FULL * 8 / 2 ** 30, 3),
    "note": ("ru_maxrss is a process high-water mark, so chunked_rows8_growth_mib is measured "
             "against the mark the monolithic arm already set and floors at 0 rather than "
             "reporting its own peak. The finding is monolithic_growth_over_fp64_tensor."),
}, indent=1))
