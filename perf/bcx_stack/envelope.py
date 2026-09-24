#!/usr/bin/env python3
"""torch bf16 and fp32 against float64 on the whole 4 + 48 stack, per seed, CPU only.

The same logits, readout and draws as `stack.py whole --seed s` (and `afgrad stack`), so a
device arm\x27s distance to float64 can be read against torch\x27s own bf16 distance on that seed.
"""
import json, pathlib, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402

torch.set_num_threads(16)
_, ref = A.load_models(A.DEFAULT_PARAMS, device_arm=False)
n, ke, kv = 128, 4, 48
out = {}
for seed in [int(s) for s in sys.argv[1].split(",")]:
    torch.manual_seed(seed)
    ridx = torch.arange(n)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5
    g = {}
    for arm in ("f64", "bf16", "f32"):
        lg = logits.clone().to(torch.float64 if arm == "f64" else torch.float32).requires_grad_(True)
        m, z = A.embed(ref[arm], lg, ridx)
        m, z = A.ref_stack(ref[arm], m, z, ke, kv)
        ((wm * m.double()).sum() + (wz * z.double()).sum()).backward()
        g[arm] = lg.grad.double()
    out[seed] = {"f64_grad_norm": float(g["f64"].norm()),
                 "torch_bf16_vs_f64": A.cmp(g["bf16"], g["f64"]),
                 "torch_f32_vs_f64": A.cmp(g["f32"], g["f64"])}
    print(seed, json.dumps(out[seed]), flush=True)
(ROOT / "perf/bcx_stack/envelope_n128_e4_v48.json").write_text(json.dumps(out, indent=1))
