"""Time a bare device open. Nothing else -- no weights, no fold."""
import sys, time
t0 = time.perf_counter()
import torch; torch.set_grad_enabled(False)
t_imp = time.perf_counter() - t0
t1 = time.perf_counter()
from tt_bio.tenstorrent import get_device, arch_name, COMPUTE_GRID_MAIN
t_imp2 = time.perf_counter() - t1
t2 = time.perf_counter()
get_device(trace_region_size=0)
t_open = time.perf_counter() - t2
print(f"torch import {t_imp:.1f}s | tt_bio import {t_imp2:.1f}s | get_device {t_open:.1f}s | "
      f"arch {arch_name()} grid {tuple(COMPUTE_GRID_MAIN)}", flush=True)
