"""Bare per-program dispatch floor t_d: 2000 back-to-back ttnn.add on a single [1,1,32,32] tile,
one synchronize at the end, wall / 2000. A lower bound on true per-program overhead."""
import json, statistics, time
import ttnn

dev = ttnn.open_device(device_id=0)
try:
    a = ttnn.from_torch(__import__("torch").zeros(1, 1, 32, 32), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    N = 2000
    reps = []
    for r in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(N):
            b = ttnn.add(a, a)
        ttnn.synchronize_device(dev)
        reps.append((time.perf_counter() - t0) / N * 1e6)
    warm = reps[1:]
    out = {"t_d_us_reps_all": [round(x, 3) for x in reps],
           "t_d_us_warm": [round(x, 3) for x in warm],
           "t_d_us_median": round(statistics.median(warm), 3),
           "t_d_us_min": round(min(warm), 3), "t_d_us_max": round(max(warm), 3),
           "n_warm": len(warm), "n_ops_per_rep": N,
           "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "unknown"}
    print(json.dumps(out, indent=2))
    open("perf/dsfix/results/td.json", "w").write(json.dumps(out, indent=2) + "\n")
finally:
    ttnn.close_device(dev)
