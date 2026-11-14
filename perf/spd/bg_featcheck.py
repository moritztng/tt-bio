"""bg_featcheck.py DESIGN_ARGS... : featurize the designs in-process and in bg_featproc's spawned worker from the same
numpy/random state, handed to the worker after it starts as predict.py does, on CPU (no device opens), and check every
batch is identical, with the time per design both ways.

    python perf/spd/bg_featcheck.py design perf/dsfix/fixtures/bg_R3.yaml --model boltzgen --steps design \
        --num_designs 4 --out_dir OUT --seed 0
"""
import multiprocessing.synchronize, random, runpy, sys, time
import numpy as np, torch
import ttnn
def _no_device(*a, **k): raise RuntimeError("featcheck must not open a device")
for _n in ("open_device", "open_mesh_device", "CreateDevice", "CreateDevices", "GetNumAvailableDevices",
           "GetNumPCIeDevices", "get_device_ids", "get_num_devices"):
    if hasattr(ttnn, _n): setattr(ttnn, _n, _no_device)
import tt_bio.runtime
# One "device" keeps the CLI in-process; with several it fans shards out to subprocesses this guard does not reach.
tt_bio.runtime.detect_tenstorrent_devices = lambda *a, **k: [0]
from tt_bio.boltzgen.task.predict import predict as P


def _same(a, b, path="batch"):
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys(): return [path + " keys"]
        return [d for k in a for d in _same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b): return [path + " len"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _same(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray): return [] if np.array_equal(a, b) else [path]
    try: return [] if a == b else [path]
    except Exception: return []


def run(self, config=None, run_prediction=True):
    self.data.num_workers = 0
    P._pickle_mol_props()
    # A named semaphore does not survive logind's RemoveIPC on a Galaxy, so the worker path must not create one.
    def _no_semlock(*a, **k): raise RuntimeError("bg_featproc created a named semaphore")
    multiprocessing.synchronize.SemLock.__init__ = _no_semlock
    pool = P._FeatureProcess(self.data)
    np.random.rand(7); random.random()   # stands in for the model load drawing between pool start and first design
    rng, proc = (np.random.get_state(), random.getstate()), []
    while True:
        t = time.perf_counter()
        b = pool.submit(None if proc else rng).result(); dt = time.perf_counter() - t
        if b is None: break
        proc.append(b); print(f"FEAT proc design {len(proc) - 1} {dt:.3f} s", flush=True)
    pool.__exit__()
    np.random.set_state(rng[0]); random.setstate(rng[1])
    items, bad = iter(self.data.predict_dataloader()), 0
    for i, pb in enumerate(proc):
        t = time.perf_counter(); b = next(items); dt = time.perf_counter() - t
        d = _same(b, pb); bad += bool(d)
        print(f"FEAT thread design {i} {dt:.3f} s {'IDENTICAL' if not d else 'DIFFERS ' + ' '.join(d[:8])}", flush=True)
    print(f"FEAT {'PASS' if not bad and next(items, None) is None else 'FAIL'} {len(proc)} designs", flush=True)
    raise SystemExit(0)


P.Predict.run = run
if __name__ == "__main__":   # a spawned featurize worker re-imports this file
    sys.argv = ["tt-bio"] + sys.argv[1:]
    runpy.run_module("tt_bio.main", run_name="__main__")
