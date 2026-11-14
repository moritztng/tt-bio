"""bg_featprof.py DESIGN_ARGS... : time the BoltzGen featurizer per design on CPU (no device opens), with a cProfile
of one warm design, whether the data module pickles (a spawned featurize worker needs it) and the cost of shipping a featurized batch across a process (plain pickle size, dumps, loads).

    python perf/spd/bg_featprof.py design perf/dsfix/fixtures/bg_R3.yaml --model boltzgen --steps design \
        --num_designs 4 --out_dir OUT --seed 0
"""
import cProfile, pickle, pstats, runpy, sys, time
import ttnn
def _no_device(*a, **k): raise RuntimeError("featprof must not open a device")
for _n in ("open_device", "open_mesh_device", "CreateDevice", "CreateDevices", "GetNumAvailableDevices",
           "GetNumPCIeDevices", "get_device_ids", "get_num_devices"):
    if hasattr(ttnn, _n): setattr(ttnn, _n, _no_device)
from tt_bio.boltzgen.task.predict import predict as P


def run(self, config=None, run_prediction=True):
    self.data.num_workers = 0
    t = time.perf_counter()
    try:
        s = pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"FEAT datamodule pickles {len(s)/1e6:.1f} MB in {time.perf_counter() - t:.3f} s", flush=True)
    except Exception as e:
        print(f"FEAT datamodule does not pickle: {type(e).__name__}: {e}", flush=True)
    items = iter(self.data.predict_dataloader())
    prof = cProfile.Profile()
    for i in range(10**9):
        t = time.perf_counter()
        if i == 1: prof.enable()
        b = next(items, None)
        if i == 1: prof.disable()
        if b is None: break
        dt = time.perf_counter() - t
        t = time.perf_counter(); s = pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL); dd = time.perf_counter() - t
        t = time.perf_counter(); pickle.loads(s); dl = time.perf_counter() - t
        print(f"FEAT design {i} featurize {dt:.3f} s pickle {len(s)/1e6:.1f} MB dumps {dd:.3f} loads {dl:.3f}", flush=True)
    pstats.Stats(prof).sort_stats("cumulative").print_stats(45)
    raise SystemExit(0)


P.Predict.run = run
if __name__ == "__main__":   # a spawned featurize worker re-imports this file
    sys.argv = ["tt-bio"] + sys.argv[1:]
    runpy.run_module("tt_bio.main", run_name="__main__")
