"""rf3_stages.py BENCH_ARGS... : perf/spd/bench.py --model rf3 with a device-synced wall clock on every RF3 stage.

The Tracy census of the 512 aa page fold on a p150a reads 22.0 s of device kernels and 3.6 s of gaps against a
46.0 s warm fold, so about 20 s is spent off the device. This splits the fold into the stages RF3.predict runs
(host input build, trunk and each recycle, distogram, diffusion prepare, each diffusion step, confidence) and the
worker's own work around it, so the off-device time has an owner. Each stage ends with a synchronize, which costs a
little pipelining across stage boundaries; the totals are therefore an upper bound on each stage.

One line per fold on stdout: `RF3STAGES {json}`, printed when the worker's `_predict_rf3_one` returns. Worker-side
host stages (featurize, MSA lookup, conformers, confidence summary, structure write) are wall time without a sync.

    python perf/spd/rf3_stages.py perf/spd/bench.py --out OUT --chip 0 --model rf3 --inputs s512 --data DATA \\
        --samples 1 --seed 0 --warm 2
"""
import collections
import json
import runpy
import sys
import time

import ttnn

from tt_bio import tenstorrent as T
from tt_bio import worker
from tt_bio._vendor.atomworks.ml.transforms import af3_reference_molecule as af3_ref
from tt_bio.rf3 import confidence as rf3_confidence, featurize as rf3_featurize, host as rf3_host, \
    model as rf3_model, sampler as rf3_sampler

ACC = collections.defaultdict(float)
CNT = collections.Counter()
DEPTH = [0]


def _sync():
    try:
        ttnn.synchronize_device(T.get_device())
    except Exception:
        pass


def timed(owner, name, label, static=False):
    orig = owner.__dict__[name]
    fn = orig.__func__ if isinstance(orig, (classmethod, staticmethod)) else orig

    def wrapper(*a, **kw):
        _sync()
        t0 = time.monotonic()
        DEPTH[0] += 1
        try:
            return fn(*a, **kw)
        finally:
            DEPTH[0] -= 1
            _sync()
            ACC[label] += time.monotonic() - t0
            CNT[label] += 1
            if label == "rf3_one":
                print("RF3STAGES " + json.dumps({k: [round(v, 3), CNT[k]] for k, v in ACC.items()}), flush=True)
                ACC.clear()
                CNT.clear()

    if isinstance(orig, classmethod):
        setattr(owner, name, classmethod(wrapper))
    elif isinstance(orig, staticmethod):
        setattr(owner, name, staticmethod(wrapper))
    else:
        setattr(owner, name, wrapper)


timed(rf3_model.RF3, "predict", "predict")
timed(rf3_host.HostInputs, "build", "host_inputs")
timed(rf3_model.RF3, "trunk", "trunk")
timed(rf3_model.Recycler, "__call__", "recycle")
timed(rf3_model.DiffusionModule, "prepare", "diffusion_prepare")
timed(rf3_model.DiffusionModule, "__call__", "diffusion_step")
timed(rf3_sampler.DiffusionSampler, "sample", "diffusion_total")
timed(rf3_model.RF3, "confidence", "confidence")
for mod, fn in ((rf3_featurize, "featurize"), (worker, "search_msas"), (worker, "_write_atom_array_structure"),
                (worker, "check_capabilities"), (rf3_confidence, "summary"), (rf3_confidence, "atomwise_plddt"),
                (af3_ref, "ccd_code_to_rdkit_with_conformers")):
    timed(mod, fn, fn)
timed(worker._WorkerState, "_predict_rf3_one", "rf3_one")

bench = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(bench, run_name="__main__")
