#!/usr/bin/env python3
"""storeWeightedSums screen: predict the realizable device time before building anything.

Every input is a measurement someone already took; nothing here is a new number.
  shape       perf/relion-kernel-diff2-fine/results/shape_per_iteration.json  (22,260 live calls)
  kernel rate state/relion-kernel-coarse-projection.md 8.15  (420.0 ns/pair, 1 core, bit-exact)
  roofs       projprobe/b0_roofs.json                        (qb1 card 0, 130 cores)
  CPU cost    state/relion-end-to-end.md 4/5 + relion-kernel-diff2-fine pass 5 (RELION's own Timer)
"""
import json

CORES = 130                # p150 compute grid the coarse kernel was scaled over
NS_PAIR_GATHERED = 420.0   # measured, zero-fill hoisted, bit-exact (coarse 8.15)
NS_PAIR_SKIPPED  = 81.3    # ablation: everything except reads/copies/zero-fill (coarse 8.15)
NS_PAIR_OPT      = 250.0   # coarse 8.17 optimistic row: x-parity relayout + 2nd gather RISC
NS_READ          = 22.9    # 183.0 ns/pair over 8 reads (coarse 8.15)
NS_CMP_PER_TRANS = 11.0/9  # coarse compare, 11 ns/pair at 9 translations, 1 accumulator
RMW128_GBS       = 80.1    # MEASURED, b0_roofs.json rmw/128B/be16
INSIDE           = None    # per-iteration below

# --- shape, measured over a whole live refinement ---------------------------------
SHAPE = {   # iter: (calls, O_mean, image_size, maxR)
    13: (4452, 16.733153638814017, 19404,  98),
    14: (4452, 52.681042228212040, 19404,  98),
    15: (4452, 49.877807726864330, 19404,  98),
    16: (4452, 48.846361185983830, 19404,  98),
    17: (4452, 47.796945193171610, 33024, 128),
}
O_COARSE = 186.0   # coarse dumps, state/relion-kernel-coarse-projection.md 2.1 (174/180/204)
T_FINE   = 36
DENSITY  = 0.15047587228128828   # significant (o,t) fraction, shape_report.json

import math
def inside_frac(image_size, maxR):
    # half-space Fourier grid: x in [0,maxR], y in [-maxR,maxR); disc of radius maxR
    return (math.pi * maxR * maxR / 2.0) / image_size

tot = {"sws_raw": 0.0, "sws_in": 0.0, "coarse_raw": 0.0, "coarse_in": 0.0}
for it, (calls, O, isz, maxR) in SHAPE.items():
    f = inside_frac(isz, maxR)
    tot["sws_raw"]    += calls * O        * isz
    tot["sws_in"]     += calls * O        * isz * f
    tot["coarse_raw"] += calls * O_COARSE * isz
    tot["coarse_in"]  += calls * O_COARSE * isz * f

def dev_seconds(raw, ins, ns_gath, ns_cmp_pair):
    """core-ns -> wall seconds on one p150."""
    core_ns = ins * ns_gath + (raw - ins) * NS_PAIR_SKIPPED + raw * ns_cmp_pair
    return core_ns / CORES / 1e9

print("inside-radius fraction, box196 / box256: %.4f / %.4f"
      % (inside_frac(19404, 98), inside_frac(33024, 128)))
print("(orientation,pixel) pairs per refinement")
print("  coarse : %.3e raw, %.3e inside radius" % (tot["coarse_raw"], tot["coarse_in"]))
print("  fine / wavg / backproject : %.3e raw, %.3e inside radius" % (tot["sws_raw"], tot["sws_in"]))

# --- CPU baselines, RELION's own Timer -------------------------------------------
CPU = {"coarse": 681.3, "fine": 61.9, "sws": 80.1, "bproj": 41.9, "wavg": 38.2}
WALL = 922.19
HOST_RESIDUE = 51.0

print("\n-- calibration: does this model reproduce the coarse leg's own extrapolation? --")
c = dev_seconds(tot["coarse_raw"], tot["coarse_in"], NS_PAIR_GATHERED, 9*NS_CMP_PER_TRANS)
print("  coarse, current kernel : %6.1f s  (%.1f s/iteration; coarse 8.17 said ~50)" % (c, c/5))
c_opt = dev_seconds(tot["coarse_raw"], tot["coarse_in"], NS_PAIR_OPT, 9*NS_CMP_PER_TRANS)
print("  coarse, optimistic     : %6.1f s  (%.1f s/iteration; coarse 8.17 said ~30)" % (c_opt, c_opt/5))
print("  RELION CPU coarse      : %6.1f s  -> device/CPU %.3f, speedup %.2fx"
      % (CPU["coarse"], c/CPU["coarse"], CPU["coarse"]/c))

print("\n-- SCREEN 1: the wavg half (dense, no scatter, it is a PROJECTION) --")
# wavg per (o,p,t): 3 accumulators (parts, XA, AA) against the coarse compare's 1
for label, ns, cmp_trans in (("current kernel, dense 36 trans", NS_PAIR_GATHERED, T_FINE*3),
                             ("optimistic kernel, dense",       NS_PAIR_OPT,      T_FINE*3),
                             ("optimistic + density-sparse",    NS_PAIR_OPT,      T_FINE*3*DENSITY)):
    w = dev_seconds(tot["sws_raw"], tot["sws_in"], ns, cmp_trans*NS_CMP_PER_TRANS)
    print("  %-32s %6.1f s   vs RELION CPU %.1f s -> %.2fx %s"
          % (label, w, CPU["wavg"], w/CPU["wavg"], "SLOWER" if w > CPU["wavg"] else "faster"))
wavg_dev = dev_seconds(tot["sws_raw"], tot["sws_in"], NS_PAIR_GATHERED, T_FINE*3*NS_CMP_PER_TRANS)
wavg_opt = dev_seconds(tot["sws_raw"], tot["sws_in"], NS_PAIR_OPT, T_FINE*3*DENSITY*NS_CMP_PER_TRANS)

print("\n-- SCREEN 2: the backproject half, against the MEASURED roofs --")
corner_touches = tot["sws_in"] * 8
print("  corner touches: %.3e   (x3 volumes = %.3e fp32 read-modify-writes)"
      % (corner_touches, corner_touches*3))
# 8 corners x 3 volumes = 24 RMW in 12 contiguous 8 B runs (2 x-adjacent floats)
runs = tot["sws_in"] * 12
bytes_rmw = runs * 128 * 2          # 128 B granularity, read + write
t_dram = bytes_rmw / (RMW128_GBS * 1e9)
print("  (a) DRAM read-modify-write, measured 128 B rmw roof %.1f GB/s : %7.1f s -> %.1fx CPU"
      % (RMW128_GBS, t_dram, t_dram/CPU["bproj"]))
# L1-resident destination (47.5 MB of volume across 195 MB of grid L1), NoC issue-bound
NS_NOC = 37.0
t_l1 = runs * 2 * NS_NOC / CORES / 1e9
print("  (b) destination L1-resident, NoC issue-bound at %.1f ns/xact       : %7.1f s -> %.1fx CPU"
      % (NS_NOC, t_l1, t_l1/CPU["bproj"]))
# destination-stationary gather: same bipartite graph, read side instead of write side
t_ds = dev_seconds(tot["sws_raw"], tot["sws_in"], NS_PAIR_GATHERED, 0)
t_ds_opt = dev_seconds(tot["sws_raw"], tot["sws_in"], NS_PAIR_OPT, 0)
print("  (c) destination-stationary GATHER, forward kernel's own rate      : %7.1f s -> %.1fx CPU"
      % (t_ds, t_ds/CPU["bproj"]))
print("      same, optimistic kernel                                       : %7.1f s -> %.1fx CPU"
      % (t_ds_opt, t_ds_opt/CPU["bproj"]))

print("\n-- the device/CPU ratio, against §5's assumed 0.0688 --")
sws_dev  = wavg_dev + t_l1
sws_opt  = wavg_opt + t_ds_opt
print("  storeWeightedSums, best available route : %.1f s / %.1f s = %.2f  (%.0fx §5's 0.0688)"
      % (sws_opt, CPU["sws"], sws_opt/CPU["sws"], (sws_opt/CPU["sws"])/0.0688))
print("  storeWeightedSums, current primitives   : %.1f s / %.1f s = %.2f  (%.0fx §5's 0.0688)"
      % (sws_dev, CPU["sws"], sws_dev/CPU["sws"], (sws_dev/CPU["sws"])/0.0688))

print("\n-- the refinement wall, revised (device-kernel arithmetic only, bridge excluded) --")
fine_dev = dev_seconds(tot["sws_raw"], tot["sws_in"], NS_PAIR_GATHERED, T_FINE*NS_CMP_PER_TRANS)
rows = [
    ("nothing on device (MEASURED)",           WALL, 0.0),
    ("coarse",                                 WALL-CPU["coarse"]+c, 0),
    ("coarse+fine",                            WALL-CPU["coarse"]-CPU["fine"]+c+fine_dev, 0),
    ("coarse+fine+sws, best route",            WALL-CPU["coarse"]-CPU["fine"]-CPU["sws"]+c+fine_dev+sws_opt, 0),
    ("coarse+fine+sws, current primitives",    WALL-CPU["coarse"]-CPU["fine"]-CPU["sws"]+c+fine_dev+sws_dev, 0),
]
for name, w, _ in rows:
    print("  %-38s %7.1f s   %.2fx" % (name, w, WALL/w))
print("  %-38s %7.1f s   %.2fx" % ("(§5 predicted for coarse+fine+sws)", 154.7, WALL/154.7))
print("\n  fine on device: %.1f s vs RELION CPU %.1f s -> %.2fx" % (fine_dev, CPU["fine"], fine_dev/CPU["fine"]))

print("\n-- the ceiling, which no kernel can beat --")
print("  a FREE storeWeightedSums: %.2f s -> %.3fx on the refinement" % (WALL-CPU["sws"], WALL/(WALL-CPU["sws"])))

print("\n-- RELION's own two CPU kernels, cost per (orientation,pixel) pair --")
for name, thread_s, pairs in (("coarse", 6265.5, tot["coarse_raw"]/4),
                              ("wavg",   340.2,  tot["sws_raw"]/4),
                              ("fine",   556.3,  tot["sws_raw"]/4)):
    print("  %-7s %8.1f thread-s/rank / %.3e pairs = %6.1f ns/pair" % (name, thread_s, pairs, thread_s/pairs*1e9))
