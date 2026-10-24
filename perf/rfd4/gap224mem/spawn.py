"""Card-free: what starting n writers from the warmed forkserver costs (each task blocks, so all n start)."""
import sys, time
sys.path.insert(0, sys.argv[1])
from concurrent.futures import ProcessPoolExecutor
from tt_bio.rfd4 import io as IO
IO.start_writer(); time.sleep(10)
for n in (1, 4, 8):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(n, mp_context=IO._context()) as pool:
        with IO._main_hidden():
            fs = [pool.submit(time.sleep, 0.5) for _ in range(n)]
        t1 = time.perf_counter()
        [f.result() for f in fs]
        t2 = time.perf_counter()
    print(n, "all started", round(t1 - t0, 3), "all done (0.5 s each)", round(t2 - t0, 3), flush=True)
