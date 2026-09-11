#!/usr/bin/env python3
"""Buffer-deduped DRAM floor per captured call, and the two arms side by side.

The published instrument (`perf/bioir_roofline/fold_bytes_512.py`) dedupes DRAM reads by
TENSOR ID. ttnn hands a reshape / unsqueeze / slice a fresh tensor id over the SAME DRAM
buffer, so a metadata view is charged a full DRAM read it never performs. Deduping by
buffer instead is the compulsory-traffic floor, and it is the convention BioIR`s 54.8 MB
per diffusion layer was counted in (weights once, bias once, each intermediate once).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itemize import itemize                                                   # noqa: E402


def counts(call):
    nodes = call["nodes"]
    seen, pub_r, pub_w = set(), 0, 0
    for n in nodes:
        t, p = n.get("node_type"), (n.get("params") or {})
        if t == "tensor":
            tid = p.get("tensor_id")
            if tid in seen:
                continue
            seen.add(tid)
            if "DRAM" in str(p.get("buffer_type", "")):
                pub_r += int(p.get("size", 0) or 0)
        elif t == "buffer_allocate" and str(p.get("type")) == "DRAM":
            pub_w += int(p.get("size", 0) or 0)
    ops, rows = itemize(call)
    dram = [r for r in rows if r["kind"] == "DRAM"]
    pre = sum(r["size"] for r in dram if r["alloc_op_i"] is None)
    alloc = sum(r["size"] for r in dram if r["alloc_op_i"] is not None)
    return {"pub": (pub_r + pub_w) / 1e6, "pub_r": pub_r / 1e6, "pub_w": pub_w / 1e6,
            "floor": (pre + alloc) / 1e6, "pre": pre / 1e6, "alloc": alloc / 1e6,
            "n_ops": len(ops), "n_buf": len(dram),
            "big": sorted(((r["size"] / 1e6, r["alloc_op"] or "(pre-existing)") for r in dram),
                          reverse=True)[:8]}


def main():
    sig = sys.argv[1] if len(sys.argv) > 1 else "512x768"
    print("%-26s %9s %9s %7s | %9s %9s %9s | %4s %4s"
          % ("arm/shape", "pub_read", "pub_write", "pub_tot", "pre-exist", "intermed", "FLOOR",
             "ops", "buf"))
    for path in sys.argv[2:]:
        d = json.load(open(path))
        for c in d["calls"]:
            if sig not in c["sig"]:
                continue
            x = counts(c)
            print("%-26s %9.3f %9.3f %7.1f | %9.3f %9.3f %9.3f | %4d %4d"
                  % (Path(path).stem, x["pub_r"], x["pub_w"], x["pub"],
                     x["pre"], x["alloc"], x["floor"], x["n_ops"], x["n_buf"]))
            print("    biggest buffers MB: " + ", ".join("%.3f %s" % b for b in x["big"]))


if __name__ == "__main__":
    main()
