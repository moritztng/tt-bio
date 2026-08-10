#!/usr/bin/env python3
"""Census every `batched_matmul` call a real fold issues, and whether the chooser applies.

The gate `batch * m_tiles < cores` declines the config below 130 blocks. This says how much of a
real fold's traffic lands in that declined set, which is the whole value of relaxing it.

Usage: census.py <out.json> -- <tt_bio.main argv...>
"""
import atexit, collections, json, sys
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix  # noqa: F401  -- imports batched_matmul into its own namespace

out_path = sys.argv[1]
argv = sys.argv[sys.argv.index("--") + 1:]

orig = T.batched_matmul
counts = collections.Counter()


def wrapped(a, b, compute_kernel_config=None, dtype=None):
    sa, sb = tuple(int(d) for d in a.shape), tuple(int(d) for d in b.shape)
    applies = False
    try:
        if (T._BATCHED_MATMUL_ON and len(sa) >= 4 and len(sa) == len(sb) and sa[:-2] == sb[:-2]
                and a.dtype == b.dtype and T._dram_interleaved(a) and T._dram_interleaved(b)):
            batch = 1
            for d in sa[:-2]:
                batch *= d
            applies = T._batched_matmul_config(
                batch, -(-sa[-2] // 32), -(-sa[-1] // 32), -(-sb[-1] // 32),
                4 if a.dtype == ttnn.float32 else 2) is not None
    except Exception:                                                          # noqa: BLE001
        pass
    counts[(sa, sb, str(a.dtype), applies)] += 1
    return orig(a, b, compute_kernel_config=compute_kernel_config, dtype=dtype)


patched = []
for name, mod in list(sys.modules.items()):
    if getattr(mod, "batched_matmul", None) is orig:
        setattr(mod, "batched_matmul", wrapped)
        patched.append(name)
print("census patched:", patched, flush=True)


@atexit.register
def dump():
    rows = [{"in0": list(k[0]), "in1": list(k[1]), "dtype": k[2], "applies": k[3], "calls": v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    json.dump({"argv": argv, "rows": rows,
               "calls_applied": sum(r["calls"] for r in rows if r["applies"]),
               "calls_declined": sum(r["calls"] for r in rows if not r["applies"])},
              open(out_path, "w"), indent=2)
    print(f"\ncensus -> {out_path}", flush=True)
    for r in rows:
        print(f"  {'APPLIES' if r['applies'] else 'DECLINED':8s} {r['calls']:6d} x "
              f"{r['in0']} @ {r['in1']} {r['dtype']}", flush=True)


sys.argv = ["tt_bio.main"] + argv
import runpy
runpy.run_module("tt_bio.main", run_name="__main__")
