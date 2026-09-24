#!/usr/bin/env python3
"""bcx-forward: render the census json as a table, each op against a roof for its own class."""
import json
import re
import sys

CLASS = [("matmul", r"^(matmul|linear)\b"), ("layout", r"^(permute|transpose|reshape|concat|clone|"
          r"to_layout|pad|slice|chunk|squeeze|unsqueeze|copy|tilize|untilize|to_memory_config|"
          r"reallocate|experimental\.nlp)"), ("softmax", r"^(softmax|transformer\.)"),
         ("layernorm", r"^layer_norm"), ("reduction", r"^(sum|mean|max|min|prod)\b"),
         ("eltwise", r"")]


def klass(name):
    for k, pat in CLASS:
        if re.match(pat, name):
            return k
    return "eltwise"


def main(path, top=25):
    b = json.load(open(path))
    r = b["roofs"]
    copy_roof = r["clone_4096_bf16"]["gbytes_s"] * 1e9
    dram_roof = max(r["clone_4096_bf16"]["gbytes_s"], r.get("add3_4096_bf16", r["add_4096_bf16"])
                    ["gbytes_s"]) * 1e9
    mm_roof = r["matmul_4096_bf16"]["tflops"] * 1e12
    print(f"roofs: matmul {mm_roof/1e12:.1f} TFLOP/s  copy {copy_roof/1e9:.0f} GB/s  "
          f"dram {dram_roof/1e9:.0f} GB/s")
    for st, d in b["blocks"].items():
        wall = d["wall_unwrapped"]["median"]
        syn, enq = d["synced"]["rows"], d["enqueue"]["rows"]
        tot = sum(v["t"] for v in syn.values())
        print(f"\n==== {st}  wall {wall*1e3:.1f} ms  enqueue_sum {d['enqueue']['sum_t']*1e3:.1f} ms "
              f"synced_sum {tot*1e3:.1f} ms  calls {d['enqueue']['calls']}  "
              f"loadavg {d['loadavg'][0]:.1f}  aiclk {d['aiclk'].get('median')}")
        by = {}
        for k, v in syn.items():
            c = klass(k.split(" ")[0])
            a = by.setdefault(c, {"t": 0.0, "e": 0.0, "calls": 0})
            a["t"] += v["t"]
            a["e"] += enq.get(k, {"t": 0})["t"]
            a["calls"] += v["calls"]
        for c, a in sorted(by.items(), key=lambda kv: -kv[1]["t"]):
            print(f"   class {c:10} {a['t']*1e3:7.2f} ms  {a['t']/tot*100:5.1f}%  "
                  f"enqueue {a['e']*1e3:6.2f} ms  calls {a['calls']}")
        print(f"   {'op / in-shapes':62} {'n':>3} {'syn':>6} {'enq':>6} {'dev':>6} "
              f"{'sh':>5} {'cum':>5} {'MB':>6} {'roof':>6}")
        cum = 0.0
        for k, v in list(syn.items())[:top]:
            e = enq.get(k, {"t": 0})["t"]
            cum += v["t"]
            dev = max(v["t"] - e, 1e-9)
            mb = (v["in_bytes"] + v["out_bytes"]) / 1e6
            c = klass(k.split(" ")[0])
            roof = copy_roof if c == "layout" else dram_roof
            u = mb * 1e6 / dev / roof * 100
            print(f"   {k[:62]:62} {v['calls']:3d} {v['t']*1e3:6.2f} {e*1e3:6.2f} {dev*1e3:6.2f} "
                  f"{v['t']/tot*100:4.1f}% {cum/tot*100:4.1f}% {mb:6.1f} {u:5.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
