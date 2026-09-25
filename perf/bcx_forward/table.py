#!/usr/bin/env python3
"""bcx-forward: the census json as a table sorted to 90 % of the block, each op against the roof
for its OWN class, measured in the same process (`forward.roofs`).

Per-op device time is the synced reading minus the unsynced enqueue reading for the same op, so
it keeps each op's sync latency; on a loaded host that understates small ops' utilisation. A
matmul sits at whichever of its FLOP time and its byte time is larger."""
import json
import re
import sys

CLASSES = [("matmul", r"^(matmul|linear|experimental\.minimal_matmul)\b"),
           ("layout", r"^(permute|transpose|reshape|concat|clone|to_layout|pad|slice|chunk|squeeze|"
                      r"unsqueeze|copy|tilize|untilize|to_memory_config|reallocate|typecast|"
                      r"experimental\.nlp|experimental\.view|view)"),
           ("softmax", r"^(softmax|transformer\.)"), ("layernorm", r"^layer_norm"),
           ("reduction", r"^(sum|mean|max|min|prod)\b")]
ROOF_OF = {"layout": "copy", "eltwise": "eltwise", "softmax": "softmax", "layernorm": "layernorm",
           "reduction": "reduction", "matmul": "eltwise"}


def klass(op):
    for k, pat in CLASSES:
        if re.match(pat, op):
            return k
    return "eltwise"


def matmul_flops(inshapes):
    sh = [list(map(int, m.split(","))) for m in re.findall(r"\[([\d, ]+)\]", inshapes)]
    if len(sh) < 2:
        return 0
    a, b = sh[0], sh[1]
    m, k = a[-2], a[-1]
    n = b[-1]
    batch = 1
    for d in a[:-2]:
        batch *= d
    return 2 * batch * m * k * n


def main(path, cover=0.90):
    b = json.load(open(path))
    r = b["roofs"]
    gbs = {k: v["gbytes_s"] * 1e9 for k, v in r.items()}
    mm = r["matmul"]["tflops"] * 1e12
    print("roofs: " + ", ".join(f"{k} {v['tflops']:.1f} TFLOP/s" if v["tflops"] else
                                f"{k} {v['gbytes_s']:.0f} GB/s" for k, v in r.items()))
    for st, d in b["blocks"].items():
        syn, enq = d["synced"]["rows"], d["enqueue"]["rows"]
        tot = sum(v["t"] for v in syn.values())
        print(f"\n{st}: wall {d['wall_unwrapped']['median']*1e3:.1f} ms, {d['enqueue']['calls']} "
              f"calls, enqueue {d['enqueue']['sum_t']*1e3:.1f} ms, synced sum {tot*1e3:.1f} ms, "
              f"loadavg {d['loadavg'][0]:.1f}")
        print("| # | op | in shapes | calls | ms | share | cum | class | vs own roof |")
        print("|---|---|---|---|---|---|---|---|---|")
        cum = 0.0
        for i, (key, v) in enumerate(syn.items(), 1):
            op, ins = key.split(" ", 1)[0], key.split(" ", 1)[1].rsplit(" ", 1)[0]
            c = klass(op)
            dev = max(v["t"] - enq.get(key, {"t": 0})["t"], 1e-9)
            byt = v["in_bytes"] + v["out_bytes"]
            ideal = byt / gbs[ROOF_OF[c]]
            if c == "matmul":
                ideal = max(ideal, matmul_flops(ins) * v["calls"] / mm)
            cum += v["t"]
            u = ideal / dev * 100
            if dev < 20e-6 * v["calls"]:
                note = "sync latency, no kernel time resolved"
            elif u > 100:
                note = f"{u:.0f} %: above the DRAM roof, operand L1-resident or in place"
            else:
                note = f"{u:.0f} %"
            print(f"| {i} | {op} | {ins[:60]} | {v['calls']} | {v['t']*1e3:.2f} | "
                  f"{v['t']/tot*100:.1f} % | {cum/tot*100:.1f} % | {c} | {note} |")
            if cum / tot >= cover:
                break


if __name__ == "__main__":
    main(sys.argv[1])
