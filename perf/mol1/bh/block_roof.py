"""Every op of one fenced Pairformer block against its own roofline, from an opgap.py prof report.

    python perf/mol1/bh/block_roof.py --ops <ops_perf_results.csv> --regions <regions.json> --block pf48/b10/c320 \
        [--peak 145] [--dram 512] [--json out.json]

Per op: the kernel (a generic op is named by its compute kernel's directory), kernel us, cores, the FLOPs and the
DRAM bytes the op cannot avoid, achieved TFLOP/s and GB/s, and its roof: max(FLOPs / peak, bytes / DRAM bandwidth).
FLOPs count 2 x MAC of the matmul-class work only (as roof.py does). Bytes are the op's necessary traffic: the
tensors it must read and write once, at the element size of its first input; a kernel that re-reads an operand or
reads an aliased output pays more than this, which is exactly the lost time the table is for. Consecutive calls of
one kernel that split one module's channels or hidden width (trimul T1 groups, transition hidden chunks) share the
module's work equally. The pair width c and N come from the op's first pair-shaped input; heads are c // 32.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opgap_table import cut, load  # noqa: E402

ELEM = {"BFLOAT16": 2.0, "BFLOAT8_B": 1088 / 1024, "FLOAT32": 4.0, "BFLOAT4_B": 576 / 1024, "UINT32": 4.0}


def dims(row, pre, k):
    out = []
    for d in "WZYX":
        v = row.get(f"{pre}_{k}_{d}_PAD[LOGICAL]")
        if not v:
            return None
        out.append(int(re.sub(r"\[.*\]", "", v)))
    return out


def tensors(row, pre, n):
    ts = []
    for k in range(n):
        s = dims(row, pre, k)
        if s is None:
            break
        ts.append((s, ELEM.get(row.get(f"{pre}_{k}_DATATYPE", ""), 2.0),
                   "L1" in row.get(f"{pre}_{k}_MEMORY", "")))
    return ts


def kernel(row):
    src = re.findall(r"'([^']+)'", row.get("COMPUTE KERNEL SOURCE", "") or "")
    code = row["OP CODE"].replace("DeviceOperation", "")
    if code == "GenericOp":
        if not src:
            return "generic(no compute)"
        p = src[0].split("/kernels/")[-1] if "/kernels/" in src[0] else src[0]
        return p.split("/")[0]
    return code


def pair_nc(ins):
    """N and c of the block's pair tensor: the first [1, N, N, c] or [1, c, N, N] input."""
    for s, _, _ in ins:
        if s[0] == 1 and s[1] == s[2] and s[1] >= 64:
            return s[1], s[3]
        if s[0] == 1 and s[2] == s[3] and s[2] >= 64:
            return s[2], s[1]
        if s[0] >= 64 and 0 <= s[2] - s[0] < 64:  # [N, H, N, 32] q/k/v, keys padded to a tile multiple
            return s[0], s[1] * s[3]
    return None, None


def model(name, row, k_of, n_of):
    """(FLOPs, necessary DRAM bytes, note) of one op; k_of/n_of: this op is call n_of of k_of in its run."""
    ins, outs = tensors(row, "INPUT", 12), tensors(row, "OUTPUT", 4)
    e = ins[0][1] if ins else 2.0
    N, c = pair_nc(ins)
    if name == "Matmul":
        (a, ea, la), (b, eb, lb) = ins[0], ins[1]
        M, K, Nn = a[-2], a[-1], b[-1]
        batch = max(a[0] * a[1], 1)
        o = outs[0]
        by = (0 if la else M * K * batch * ea) + (0 if lb else K * Nn * b[0] * b[1] * eb) \
            + (0 if o[2] else o[0][0] * o[0][1] * o[0][2] * o[0][3] * o[1])
        return 2.0 * batch * M * K * Nn, by, ""
    if N is None:
        size = lambda t: 0 if t[2] else t[0][0] * t[0][1] * t[0][2] * t[0][3] * t[1]
        return 0.0, size(ins[0]) + sum(size(o) for o in outs) if ins else 0.0, ""
    z = N * N * c * e
    H = c // 32
    if name == "trimul_in_t1":     # LN, a/b projections + gates, the output gate: 5 c^2 per pair; reads z, writes a^T b^T g
        return 2.0 * N * N * 5 * c * c / k_of, z + 3 * z / k_of, f"T1 group {n_of + 1}/{k_of}"
    if name in ("bmm_res", "trimul_t2", "trimul_t2cm", "trimul_t2m"):
        return 2.0 * N ** 3 * c, 3 * z, "einsum"
    if name == "permute_back_pk":
        return 0.0, 2 * z, "x^T -> x"
    if name == "trimul_tail_res":  # LN(x) W_out * g + z
        return 2.0 * N * N * c * c, 4 * z, "tail"
    if name == "triatt_ln":        # LN, q k v g projections + pair bias
        if c < 64:
            return 2.0 * N * N * ins[1][0][3] * H, z, "bias only"
        return 2.0 * N * N * c * (4 * c + H), 5 * z + N * N * H * e, "LN+qkvg+bias"
    if name == "triatt_sdpa":
        return 4.0 * N ** 3 * c, 4 * z + N * N * H * e, "QK^T, PV"
    if name == "triatt_tail":      # gated out-proj + residual
        return 2.0 * N * N * c * c, 4 * z, "o-proj+gate+resid"
    if name == "pair_swiglu_ws":   # fc1, fc2: c -> 4c, fc3: 4c -> c, hidden split in k chunks
        return 2.0 * N * N * 12 * c * c / k_of, (2 * z if k_of == 1 else z + (z if n_of == 0 else 2 * z)), \
            f"hidden chunk {n_of + 1}/{k_of}"
    if name == "LayerNorm":
        return 0.0, 2 * z, "LN"
    if name in ("Permute", "Transpose", "BinaryNg", "Typecast", "Reduce", "Unary", "generic(no compute)"):
        size = lambda t: 0 if t[2] else t[0][0] * t[0][1] * t[0][2] * t[0][3] * t[1]
        return 0.0, size(ins[0]) + sum(size(o) for o in outs), "glue"
    size = lambda t: 0 if t[2] else t[0][0] * t[0][1] * t[0][2] * t[0][3] * t[1]
    return 0.0, size(ins[0]) + sum(size(o) for o in outs), ""


def runs(names):
    """For each op, (k, n): length of the run of identical kernel names it sits in, and its index there."""
    out, i = [], 0
    while i < len(names):
        j = i
        while j < len(names) and names[j] == names[i]:
            j += 1
        out += [(j - i, n) for n in range(j - i)]
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--block", required=True, help="region label prefix; the last match (warm, last cycle) is used")
    ap.add_argument("--peak", type=float, default=145.0, help="bf16 TFLOP/s of the part")
    ap.add_argument("--dram", type=float, default=512.0, help="DRAM GB/s of the part")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    regs = [r for r in cut(load(a.ops), json.load(open(a.regions))) if r["label"].startswith(a.block)]
    if not regs:
        raise SystemExit(f"no region {a.block}")
    reg = regs[-1]
    ops = reg["ops"]
    names = [kernel(o) for o in ops]
    # The single track rides at the end of a Pairformer block: from its first [1, 1, S, c_s] op on.
    rec, tot = [], {"k": 0.0, "roof": 0.0, "gap": 0.0, "fl": 0.0}
    print(f"### {reg['label']}: {len(ops)} ops\n")
    print("| # | kernel | note | input 0 | cores | kernel us | gap after us | GFLOP | MB | TFLOP/s | GB/s | roof us | x roof |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (o, nm, (k, n)) in enumerate(zip(ops, names, runs(names))):
        fl, by, note = model(nm, o, k, n)
        t = o["_k"] / 1e3
        gap = (ops[i + 1]["_gap"] if i + 1 < len(ops) else reg["next_gap"]) / 1e3
        roof = max(fl / (a.peak * 1e6), by / (a.dram * 1e3))
        r = dict(i=i, kernel=nm, note=note, in0=o["_in0"], cores=o["_cores"], us=t, gap_us=gap, gflop=fl / 1e9,
                 mb=by / 1e6, tflops=fl / t / 1e6 if t else 0, gbs=by / t / 1e3 if t else 0, roof_us=roof)
        rec.append(r)
        tot["k"] += t
        tot["roof"] += roof
        tot["gap"] += gap
        tot["fl"] += fl
        print(f"| {i} | {nm} | {note} | {o['_in0']} | {o['_cores']} | {t:.1f} | {gap:.1f} | {fl / 1e9:.1f} | "
              f"{by / 1e6:.0f} | {r['tflops']:.1f} | {r['gbs']:.0f} | {roof:.1f} | {t / roof if roof else 0:.2f} |")
    print(f"\nkernel {tot['k'] / 1e3:.2f} ms, gaps {tot['gap'] / 1e3:.2f} ms, roof {tot['roof'] / 1e3:.2f} ms, "
          f"{tot['fl'] / 1e12:.3f} TFLOP -> {tot['fl'] / tot['k'] / 1e6:.1f} TFLOP/s on kernel time")
    if a.json:
        json.dump({"label": reg["label"], "ops": rec, "total": tot}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
