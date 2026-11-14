"""Whole edm_device rollout (trace + staging) with stub conditioning/denoiser vs the host traced path."""
import os, sys
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import math, torch, ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import openfold3_sample_diffusion as SD  # noqa: E402

dev = T.get_device(trace="diffusion")
try:
    g = torch.Generator().manual_seed(0)
    n, NP, sd, ss, steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4100, 4128, 16.0, 1.5, 20
    act = ttnn.bfloat16 if "bf16" in sys.argv else ttnn.float32
    m = (torch.rand(n, generator=g) > 0.05).float()
    x0 = torch.randn(1, n, 3, generator=g) * 160
    ts = [160.0 * (0.8 ** i) for i in range(steps)]; ct = ts[1:] + [0.0]
    rots = [[torch.linalg.qr(torch.randn(3, 3, generator=g))[0] for _ in range(steps)]]
    trans = [[torch.randn(3, generator=g) for _ in range(steps)]]
    noise = [[torch.randn(n, 3, generator=g) * 0.1 * ts[i] for i in range(steps)]]

    class DC:
        def single(self, a, b, emb, c):
            return ttnn.typecast(ttnn.multiply(emb, 1.0), act)
    class DM:
        def __call__(self, *a, samples=None, **k):
            # fake denoiser: rl_update = 0.5 * rl[:n_atom]
            return [ttnn.typecast(ttnn.multiply(ttnn.slice(rl, [0, 0, 0], [1, n, 3]), 0.5), act)
                    for _si, rl, _x, _t in samples]
    obj = SD.OF3SampleDiffusion.__new__(SD.OF3SampleDiffusion)
    obj._act_dtype, obj.dc, obj.dm, obj.device, obj.sigma_data = act, DC(), DM(), dev, sd
    obj.fourier_w, obj.fourier_b = torch.randn(256), torch.randn(256)
    args = (x0.clone(), m, rots, trans, noise, ts, ct, ss, None, (), {}, (None, None, None), n, NP, 32, 32)
    host = obj._rollout_traced(*args)
    T._LEVERS = set(T._LEVERS) | {"edm_device"} if not isinstance(T._LEVERS, frozenset) else frozenset(T._LEVERS | {"edm_device"})
    assert T.lever("edm_device")
    if "eager" in sys.argv:
        def _no(*a, **k):
            raise RuntimeError("probe: no trace")
        ttnn.begin_trace_capture = _no
        ttnn.release_trace = lambda *a, **k: None
        ttnn.end_trace_capture = lambda *a, **k: None
    devr = obj._rollout_edm_device(*args)
    print("host range", float(host.abs().max()), "dev range", float(devr.abs().max()),
          "max_abs", float((host - devr).abs().max()), flush=True)
finally:
    T.cleanup()
