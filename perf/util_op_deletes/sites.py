"""Attribute every device program of one 512 aa PairformerLayer to its source call site.

The `b2z-kernel-cycle-census` row measured 272 dispatched programs of one settled block on
qb2 card 0 under Tracy, but its rows carry no shapes and no call site: it ranks op CODES.
This joins them. `perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz` is a ttnn.graph capture of
the same block with the owning `file:line:function` and every operand shape, so aligning the two
op-class sequences carries the per-call kernel time onto the call site.

The alignment is not assumed, it is checked: the two sequences agree on all 272 census rows,
with 9 WH-only layout rows (views that BH resolves without a program) dropped. Nothing else
differs, which is what makes the join usable at all.

    python3 perf/util_op_deletes/sites.py --census <block_census.json> --trace <trace_512.json>
"""
import argparse, difflib, json, sys
from collections import defaultdict

# ttnn python entry point -> the device op class the census names it by.
CLS = {
    "ttnn.layer_norm": "LayerNorm", "ttnn.linear": "Matmul", "ttnn.matmul": "Matmul",
    "ttnn.multiply_": "BinaryNg", "ttnn.multiply": "BinaryNg", "ttnn.add_": "BinaryNg",
    "ttnn.add": "BinaryNg", "ttnn.subtract": "BinaryNg", "ttnn.generic_op": "GenericOp",
    "ttnn.concat": "Concat", "ttnn.softmax": "Softmax",
    "ttnn.experimental.nlp_create_qkv_heads": "NlpCreateHeads",
    "ttnn.transpose": "Layout", "ttnn.permute": "Layout", "ttnn.reshape": "Layout",
    "ttnn.unsqueeze": "Layout", "ttnn.slice": "Slice", "ttnn.chunk": "Slice",
}
BH = {
    "TransposeDeviceOperation": "Layout", "PermuteDeviceOperation": "Layout",
    "ReshapeViewDeviceOperation": "Layout", "SliceDeviceOperation": "Slice",
    "BinaryNgDeviceOperation": "BinaryNg", "MatmulDeviceOperation": "Matmul",
    "LayerNormDeviceOperation": "LayerNorm", "GenericOpDeviceOperation": "GenericOp",
    "ConcatDeviceOperation": "Concat", "SoftmaxDeviceOperation": "Softmax",
    "NlpCreateHeadsDeviceOperation": "NlpCreateHeads",
}


def dispatching(trace):
    """The trace rows that become a device program, ttnn.chunk expanded to one row per slice."""
    out = []
    for r in trace["rows"]:
        if r["op"] not in CLS:
            continue
        for _ in range(len(r["out_s"]) if r["op"] == "ttnn.chunk" else 1):
            out.append(r)
    return out


def align(wh, bh):
    """Index map census row -> trace row, plus the trace rows the census has no program for."""
    sm = difflib.SequenceMatcher(None, [CLS[r["op"]] for r in wh],
                                 [BH[o["op"]] for o in bh], autojunk=False)
    pairs, dropped = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs.update({j1 + k: i1 + k for k in range(i2 - i1)})
        else:
            dropped.append((tag, [wh[i]["owner"] for i in range(i1, i2)],
                            [bh[j]["op"] for j in range(j1, j2)]))
    return pairs, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--per-op", action="store_true", help="one line per dispatched program")
    ap.add_argument("--blocks", type=int, default=280, help="pairformer calls per 512 aa fold")
    a = ap.parse_args()

    bh = json.load(open(a.census))["ops"]
    wh = dispatching(json.load(open(a.trace)))
    pairs, dropped = align(wh, bh)
    if len(pairs) != len(bh):
        print("REFUSED: %d of %d census rows aligned" % (len(pairs), len(bh)))
        return 1
    print("aligned %d/%d census rows; %d trace rows carry no BH program" % (
        len(pairs), len(bh), sum(len(d[1]) for d in dropped)))

    sites = defaultdict(lambda: [0, 0.0, set()])
    for o in bh:
        w = wh[pairs[o["i"]]]
        key = (BH[o["op"]], w["owner"], w["op"])
        sites[key][0] += 1
        sites[key][1] += o["kernel_ns"] / 1e6
        sites[key][2].add("%s -> %s" % (",".join(w["in_s"][:2]) or "-", ",".join(w["out_s"][:1])))
        if a.per_op:
            print("%3d %-30s %8.2f us  %-14s %s" % (
                o["i"], o["op"], o["kernel_ns"] / 1e3, w["op"], w["owner"]))

    print("\n%-14s %-44s %4s %10s %9s  %s" % (
        "class", "call site", "n", "ms/block", "s/fold", "shape"))
    for (cls, owner, op), (n, ms, shapes) in sorted(sites.items(), key=lambda kv: -kv[1][1]):
        print("%-14s %-44s %4d %10.4f %9.4f  %s" % (
            cls, owner, n, ms, ms * a.blocks / 1e3, sorted(shapes)[0]))
    tot = sum(v[1] for v in sites.values())
    print("\ntotal %.4f ms/block, %.3f s/fold at %d calls" % (tot, tot * a.blocks / 1e3, a.blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
