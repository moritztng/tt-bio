"""Run perf/tfg_acc/run.py with every guided sampler step recorded.

TFG_TRACE=<dir>: per guided fold, <dir>/<n>.pt holds the step inputs/outputs ([steps, M, N, 3] fp32 x_noisy, x0, x)
and the schedule, plus feats.pt (the guidance features) once per fold. Outputs are unchanged: the hook only copies.
usage: TFG_TRACE=DIR python trace.py <run.py args>
"""
import os, runpy, sys, torch
from tt_bio.tfg import guidance as G

DIR = os.environ["TFG_TRACE"]; os.makedirs(DIR, exist_ok=True)
_step = G.Guidance.step
rec = {}


def step(self, x_noisy, x0, *, t_hat, sigma_t, eta, step, n_step):
    x = _step(self, x_noisy, x0, t_hat=t_hat, sigma_t=sigma_t, eta=eta, step=step, n_step=n_step)
    if step == 0:
        rec.clear(); rec.update(x_noisy=[], x0=[], x=[], t_hat=[], sigma_t=[], eta=[])
    for k, v in dict(x_noisy=x_noisy, x0=x0, x=x).items():
        rec[k].append(v.detach().float().cpu().clone())
    rec["t_hat"].append(float(t_hat)); rec["sigma_t"].append(float(sigma_t)); rec["eta"].append(float(eta))
    if step == n_step - 1:
        n = len([f for f in os.listdir(DIR) if f.endswith("_traj.pt")])
        torch.save({k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in rec.items()},
                   f"{DIR}/{n}_traj.pt")
        torch.save(self.feats, f"{DIR}/{n}_feats.pt")
        print(f"TFG_TRACE wrote {DIR}/{n}_traj.pt", flush=True)
    return x


G.Guidance.step = step
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
