"""Smoke test: does ttnn's device profiler expose per-program core_count on this build?"""
import os
import ttnn

d = ttnn.open_device(device_id=0)
print("compute grid:", d.compute_with_storage_grid_size())
a = ttnn.from_torch(__import__("torch").randn(1, 1, 512, 512), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=d)
b = ttnn.matmul(a, a)
c = ttnn.add(b, a)
ttnn.synchronize_device(d)
ttnn.ReadDeviceProfiler(d)
data = ttnn.profiler.get_all_programs_perf_data()
print("chips:", list(data.keys()))
for chip, progs in data.items():
    for p in progs:
        rs = p.program_analyses_results
        durs = {k: (v.duration if hasattr(v, "duration") else v) for k, v in (rs.items() if hasattr(rs, "items") else enumerate(rs))}
        print(f"  rid={p.program_execution_uid.runtime_id} cores={p.core_count}/{p.num_available_cores} res={durs}")
ttnn.close_device(d)
