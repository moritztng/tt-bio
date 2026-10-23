"""Write stand-in CIFs for several seeds with whichever tt_bio is on PYTHONPATH; time each write.
    python wcmp.py <spec> <outdir> <seeds> <reps>"""
import sys, time, importlib.util
from pathlib import Path
spec, out, nseeds, reps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
from tt_bio import rfd4
wp = importlib.util.spec_from_file_location("wp", Path(rfd4.__file__).parents[2] / "perf/rfd4/host3/write_prof.py")
m = importlib.util.module_from_spec(wp); wp.loader.exec_module(m)
designs = rfd4.prepare_examples(spec, out / "inputs")
features, atom_array = rfd4.featurize(designs[0].cif, seed=0)
ts = []
for s in range(nseeds):
    o = m.stand_in(features, atom_array, seed=s)
    for r in range(reps):
        t = time.perf_counter()
        rfd4.write_design(o, features, atom_array, out / f"s{s}r{r}.cif")
        ts.append(time.perf_counter() - t)
ts.sort()
print(f"{spec.split('/')[-1]} writes {len(ts)} median {ts[len(ts)//2]:.3f} min {ts[0]:.3f} s", flush=True)
