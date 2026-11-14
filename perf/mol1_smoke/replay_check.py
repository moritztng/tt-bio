"""CPU check of edm_sample's replayed draw stream against an upstream Mol-1 trajectory (mol1-ref goldens).
The denoiser is replaced by upstream's own recorded x_denoised, so every x_noisy the sampler builds must equal the
one upstream recorded at that step: this pins the draw order, the rotation convention and the schedule.
Usage: python perf/mol1_smoke/replay_check.py <seed_dir with sampler.pt + draws.pt> [chunk]"""
import sys
from types import SimpleNamespace

import torch

from tt_bio.protenix import edm_sample

d, chunk = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
sam = torch.load(f"{d}/sampler.pt", weights_only=False)
draws = torch.load(f"{d}/draws.pt", weights_only=False)
starts = [i for i, (t, _v) in enumerate(draws) if t == "init"]
n_step = len(sam["noise_schedule"]) - 1
calls = sam["calls"][chunk * n_step:(chunk + 1) * n_step]
stream = iter(draws[starts[chunk]:])
n_atoms = draws[starts[chunk]][1].reshape(-1, 3).shape[0]
seen = []


def denoise(x_noisy, t, _cond):
    k = len(seen)
    seen.append((x_noisy.reshape(-1, 3).double(), float(t.reshape(-1)[0])))
    return calls[k]["x_denoised"].float().reshape(x_noisy.shape)


x = edm_sample(SimpleNamespace(denoise=denoise), {}, n_atoms, n_step=n_step, draws=stream)
worst, wt = 0.0, 0.0
for k, (xn, th) in enumerate(seen):
    ref = calls[k]["x_noisy"].reshape(-1, 3).double()
    rel = float((xn - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    worst = max(worst, rel)
    wt = max(wt, abs(th - float(torch.as_tensor(calls[k]["t_hat"]).reshape(-1)[0])) / max(th, 1e-12))
print(f"steps {len(seen)}/{n_step}  max rel |x_noisy - upstream| {worst:.2e}  max rel t_hat {wt:.2e}")
