"""Blackhole neutrality of the wh-perf-esmfold2 knobs, proved on the CPU before any card is booked.

Both levers are supposed to be unreadable on a >= 110-core grid. That is a claim about control flow,
so it is checkable without a device: set every knob to its most aggressive value, apply each grid in
turn, and compare what the shipped gate functions return.

No device is opened. `_apply_grid_thresholds` calls `ttnn.get_max_worker_l1_unreserved_size()`,
which boots the cluster, so it is stubbed to the per-core L1 the audit MEASURED on the Galaxy
(1,466,080 B) -- the same substitution `perf/wh-baseline/gate_dump.py` makes, and it is what makes
this runnable on a busy host without contending for a card.
"""
import sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/wh-perf-esmfold2")
import ttnn
ttnn.get_max_worker_l1_unreserved_size = lambda *a, **k: 1466080
import tt_bio.tenstorrent as T

T.set_fast_mode(True)
SIZES = (298, 320, 512, 1024)
rows = []
for grid in [(13, 10), (11, 10), (8, 9)]:
    T.COMPUTE_GRID_MAIN = grid
    T._apply_grid_thresholds(grid)
    gates = lambda: {L: (T._trimul_l1_max_seq(), T._trimul_chunk_size(L, 256)) for L in SIZES}
    off = gates()
    T.set_small_grid_trimul_l1_max_seq(320)
    T.set_small_grid_trimul_budget_scale(2.0)
    on = gates()
    T.set_small_grid_trimul_l1_max_seq(0)
    T.set_small_grid_trimul_budget_scale(1.0)
    rows.append((grid, T._IS_SMALL_GRID, off == on))
    print(f"grid {grid} cores={grid[0]*grid[1]} small={T._IS_SMALL_GRID} "
          f"identical_with_knobs_maxed={off == on}")
    print(f"   shipped  {off}")
    print(f"   knobs on {on}")
assert all(same for g, small, same in rows if not small), "a Blackhole grid saw a knob -- NOT neutral"
assert any(not same for g, small, same in rows if small), "the small grid saw nothing -- inert everywhere"
print("\nNEUTRAL on every >= 110-core grid; live only on 8x9.")
