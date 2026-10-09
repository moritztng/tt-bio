"""Table of run_silu.sh output: per arm, exactness and min-over-rounds op times with the AICLK seen.

    python perf/spd_swiglu/silu_table.py OUTDIR
"""
import json
import sys
from pathlib import Path

runs = {}
for p in sorted(Path(sys.argv[1]).glob("*.r*.json")):
    r = json.loads(p.read_text())
    runs.setdefault(r["arm"], []).append(r)
for arm, rs in runs.items():
    r0 = rs[0]
    print(f"## {arm}  host {r0['host']} chip {r0['chip']} {r0['arch']} rounds {len(rs)} root {r0.get('runtime_root')}")
    if "exact" in r0:
        e = r0["exact"]
        same = all(r.get("exact") == e for r in rs)
        print(f"exact: max_rel {e['max_rel']:.3g} at {e['at']}, p99.9 {e['p999_rel']:.3g}, bf16 flips "
              f"{e['bf16_differs']:.3g}, finite {e['finite']}, same every round {same}, digests {e.get('digest')} "
              f"{e.get('digest_fp32_in')}")
    for name in r0.get("ops", {}):
        cells = []
        for kind in ("fused", "matmul", "unfused"):
            v = [r["ops"][name][kind] for r in rs]
            clk = sorted({c["clock"]["median"] for c in v if c["clock"]})
            meds = ", ".join("%.1f" % x["us_med"] for x in v)
            cells.append(f"{kind} {min(x['us_min'] for x in v):.1f} us (med {meds}; MHz {clk})")
        print(f"{name} {r0['ops'][name]['shape']}: " + " | ".join(cells))
    for name in r0.get("transition", {}):
        v = [r["transition"][name] for r in rs]
        print(f"transition {name}: {min(x['ms_min'] for x in v):.2f} ms (med {', '.join(str(x['ms_med']) for x in v)}), "
              f"rel_rms {v[0]['rel_rms']:.4g}, max_abs {v[0]['max_abs']:.3g}")
