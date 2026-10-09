"""Where a guided OpenDDE fold spends its diffusion time: device denoiser vs host guidance, per fold.

Runs tfg-accuracy's fold harness (perf/tfg_acc/run.py, same arguments) with timers around the sampler:
denoise (the device denoiser incl. its host<->device copies), and the guidance host work split into the TFG engine
(projection + 20 gradient steps on x0), the rigid early pass on x0 (steps 100..187) and the rigid late pass on x
(steps 190..199). One JSON line per sampler call goes to OUT/split.jsonl. Run seeds 101,102 so the second seed's
unconstrained fold is warm (the first one pays the program cache).

    python perf/tfg_kernel/split.py --panel P --target 1a14 --out RUN/1a14 --chip 3 --seeds 101,102
"""
import json
import runpy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
out = Path(sys.argv[sys.argv.index("--out") + 1]).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

from tt_bio import protenix  # noqa: E402
from tt_bio.tfg import engine, guidance  # noqa: E402

acc = {}


def timed(key, fn):
    def run(*args, **kw):
        t = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t
    return run


protenix.denoise_in_chunks = timed("denoise", protenix.denoise_in_chunks)
engine.TFGEngine.update = timed("engine_incl_early", engine.TFGEngine.update)
guidance.Guidance._x0_hook = timed("early", guidance.Guidance._x0_hook)
guidance.Guidance.step = timed("guidance", guidance.Guidance.step)
_edm = protenix.edm_sample


def edm_sample(*args, guidance=None, **kw):
    acc.clear()
    t = time.perf_counter()
    x = _edm(*args, guidance=guidance, **kw)
    wall = time.perf_counter() - t
    g = acc.get("guidance", 0.0)
    rec = dict(guided=guidance is not None, mode=getattr(guidance, "mode", None), atoms=int(x.shape[-2]),
               samples=int(x.shape[0]), threads=torch.get_num_threads(), sampler_s=round(wall, 2),
               denoise_s=round(acc.get("denoise", 0.0), 2), guidance_s=round(g, 2),
               engine_s=round(acc.get("engine_incl_early", 0.0) - acc.get("early", 0.0), 2),
               early_s=round(acc.get("early", 0.0), 2),
               late_s=round(g - acc.get("engine_incl_early", 0.0), 2), t_unix=time.time())
    with (out / "split.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("SPLIT " + json.dumps(rec), flush=True)
    return x


protenix.edm_sample = edm_sample
sys.argv[0] = str(REPO / "perf/tfg_acc/run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
