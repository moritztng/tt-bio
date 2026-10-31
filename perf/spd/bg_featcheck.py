"""bg_featcheck.py DESIGN_ARGS... : featurize the designs in-process and in bg_featproc's spawned worker from the same
numpy/random state on CPU (no device opens) and check every batch is identical, with the time per design both ways.

    python perf/spd/bg_featcheck.py design perf/dsfix/fixtures/bg_R3.yaml --model boltzgen --steps design \
        --num_designs 4 --out_dir OUT --seed 0
"""
import multiprocessing, pickle, random, runpy, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
import ttnn
def _no_device(*a, **k): raise RuntimeError("featcheck must not open a device")
for _n in ("open_device", "open_mesh_device", "CreateDevice", "CreateDevices", "GetNumAvailableDevices",
           "GetNumPCIeDevices", "get_device_ids", "get_num_devices"):
    if hasattr(ttnn, _n): setattr(ttnn, _n, _no_device)
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
    rng = (np.random.get_state(), random.getstate())
    P._pickle_mol_props()
    pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=P._featurize_open,
                               initargs=(pickle.dumps(self.data), rng))
    proc = []
    while True:
        t = time.perf_counter(); b = pickle.loads(pool.submit(P._featurize_next).result()); dt = time.perf_counter() - t
        if b is None: break
        proc.append(b); print(f"FEAT proc design {len(proc) - 1} {dt:.3f} s", flush=True)
    pool.shutdown()
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
