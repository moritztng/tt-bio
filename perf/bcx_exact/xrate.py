#!/usr/bin/env python3
"""Is the float64 floor measuring the instrument, or measuring a loud qb2?

ROUNDFLOOR.json timed the instrument's arithmetic on qb2 at loadavg 12-23 on 16 cores. This
re-times the same arithmetic on pc, which is quiet (loadavg ~1.5 on 12 cores) and is the host
of3t-bwattrib measured its real OF3T backward on, so both comparisons are same-host.

of3t-bwattrib, measured on pc over a real device-attached OF3T backward:
    host_f64_softmax..bw  111 calls  250.35 s   -> 2.2554 s per call, PCIe round trip INCLUDED
    226,492,416 elements per call, which is exactly 384*4*384*384 -- the same (n,4,n,n) form as
    AF2's triangle attention, at n=384
    1.81 GB per exact softmax, 391.4 GB over 216 calls of both legs, 178 s at 2.2 GB/s measured
    -> 0.8241 s per call of PCIe, so ~1.4313 s per call of host arithmetic, 6.32 ns/element

Softmax at these sizes is bandwidth bound, so ns/element is comparable across n. pc is a 30 GB
box and of3t-exactscope already hit a global OOM on it, so n=384 (1.81 GB per float64 tensor,
~11 GB peak in the backward) is deliberately NOT run here. n=224 and n=288 match the shapes
ROUNDFLOOR.json used, which is what makes the qb2/pc comparison exact.
"""
import json, os, statistics as st, sys, time
import torch

OF3T_ARITH_NS_PER_ELEMENT = 1.4313 / 226492416 * 1e9      # 6.32
OF3T_TOTAL_NS_PER_ELEMENT = 2.2554 / 226492416 * 1e9      # 9.96, round trip included
# qb2 medians from ROUNDFLOOR.json, threads=8, same expressions, same shapes.
QB2 = {224: {"fwd": 0.187414, "bwd": 0.400090}, 288: {"fwd": 0.398245, "bwd": 0.653029}}


def t_fwd(shape, reps):
    x = torch.randn(*shape, dtype=torch.float32); ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y64 = torch.softmax(x.double(), -1); y = y64.float()
        ts.append(time.perf_counter() - t0); del y64, y
    return ts


def t_bw(shape, reps):
    y64 = torch.softmax(torch.randn(*shape, dtype=torch.float32).double(), -1)
    g = torch.randn(*shape, dtype=torch.float32); ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g64 = g.double(); dx = y64 * (g64 - (g64 * y64).sum(-1, keepdim=True)); d = dx.float()
        ts.append(time.perf_counter() - t0); del g64, dx, d
    return ts


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    torch.set_num_threads(threads)
    t_fwd((64, 4, 64, 64), 1)
    rows = []
    for n in (224, 288):
        sh = (n, 4, n, n); el = n * 4 * n * n
        f, b = st.median(t_fwd(sh, reps)), st.median(t_bw(sh, reps))
        rows.append({"n": n, "elements": el,
                     "pc_fwd_s": round(f, 5), "pc_bwd_s": round(b, 5),
                     "pc_bwd_ns_per_element": round(b / el * 1e9, 3),
                     "qb2_bwd_s": QB2[n]["bwd"],
                     "qb2_bwd_ns_per_element": round(QB2[n]["bwd"] / el * 1e9, 3),
                     "qb2_over_pc": round(QB2[n]["bwd"] / b, 3),
                     "pc_over_of3t_arith": round((b / el * 1e9) / OF3T_ARITH_NS_PER_ELEMENT, 3),
                     "loadavg1": round(os.getloadavg()[0], 2)})
    out = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
           "opened_a_device": False, "threads": threads, "reps": reps, "nproc": os.cpu_count(),
           "torch": torch.__version__, "loadavg": os.getloadavg(),
           "of3t_arith_ns_per_element": round(OF3T_ARITH_NS_PER_ELEMENT, 3),
           "of3t_total_ns_per_element_round_trip_included": round(OF3T_TOTAL_NS_PER_ELEMENT, 3),
           "rows": rows}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
