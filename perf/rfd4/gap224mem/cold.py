"""Card-free: time inside one forked writer, first write vs second, to find what a fresh writer pays."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
from tt_bio import rfd4
from tt_bio.rfd4 import io as IO
from test_rfd4_write_designs import _stand_in
from concurrent.futures import ProcessPoolExecutor

from coldjob import job
def _unused():
    t0 = time.perf_counter()
    if prof:
        import cProfile, pstats, io
        pr = cProfile.Profile(); pr.enable()
    IO.write_design(row, mask, aa, p)
    if prof:
        pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25); print(s.getvalue()[:6000])
    return time.perf_counter() - t0

out = Path(__file__).parent / "runs" / f"cold-{os.getpid()}"
ex = rfd4.prepare_examples(sys.argv[2], out / "inputs")[0]
f, aa = rfd4.featurize(ex.cif, seed=1000)
o = _stand_in(f, 2)
IO.start_writer(2); time.sleep(8)
row = IO._Row(*(getattr(o, k)[0:1].clone() for k in IO._Row._fields))
mask = {"padding_mask": f["padding_mask"].clone()}
with IO._main_hidden():
    pool = ProcessPoolExecutor(1, mp_context=IO._context())
    t0 = time.perf_counter(); fu = pool.submit(job, row, mask, aa, out / "a.cif", False); r1 = fu.result(); e1 = time.perf_counter() - t0
    t0 = time.perf_counter(); r2 = pool.submit(job, row, mask, aa, out / "b.cif", False).result(); e2 = time.perf_counter() - t0
print("first in-worker", r1, "wall", round(e1, 3), "| second", r2, "wall", round(e2, 3))
