"""Unprofiled wall per stage of build_openfold3_features (cProfile inflates the python-heavy loops 2-4x).

python perf/of3_host/stages.py perf/size512/fixtures/cdk2x2_512.yaml [--reps 5]
Wraps the module-level callees of tt_bio.openfold3_data in timers, then runs featprof's driver.
"""
import collections, functools, sys, time

import tt_bio.openfold3_data as d

T = collections.defaultdict(list)
for name in [n for n in dir(d) if n.startswith(("featurize_", "_get_structure", "make_openfold3_msa"))]:
    fn = getattr(d, name)
    if callable(fn):
        def wrap(fn, name=name):
            @functools.wraps(fn)
            def w(*a, **k):
                t = time.perf_counter(); r = fn(*a, **k); T[name].append(time.perf_counter() - t); return r
            return w
        setattr(d, name, wrap(fn))
sys.argv = [sys.argv[0]] + sys.argv[1:]
import runpy
try:
    runpy.run_path(__file__.replace("stages.py", "featprof.py"), run_name="__main__")
finally:
    for k, v in sorted(T.items(), key=lambda kv: -sum(kv[1][1:])):
        print(f"{k:45s} warm " + " ".join(f"{x:.3f}" for x in v[1:]))
