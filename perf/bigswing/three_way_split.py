#!/usr/bin/env python3
"""The three-way split of a Pairformer block: arithmetic, bytes, and neither.

§106 step 2. Reads a `pf_block_ops.py` census and prices every row against the roof its own
writer can reach, never one generic number. The point of the split is the "neither" bucket:
op-boundary cost that only reducing the op count can delete, which is the hard ceiling on the
whole fusion thesis.

Per row, from the padded shapes the census already records:
  read bytes  = sum over inputs, write bytes = output
  FLOP        = 2 * prod(out) * K for matmul-class rows, 4*B*H*S*S*D for SDPA, else 0
  arith floor = FLOP / compute roof          (the rate this card has shown on a real trimul matmul)
  byte floor  = max(read/read_roof, write/writer_roof)   -- not (r+w)/one_roof, because the read
                and write paths are separately measured on this card and the binding one is the max
  bucket      = arithmetic-bound if arith_floor/t >= THRESH, bandwidth-bound if byte_floor/t >= THRESH,
                else neither

Every roof is a measured number with its source named in ROOFS. Usage:
    python3 three_way_split.py --census ops_pv2_512_fast_qb2c0.json --out split_512_fast.json
"""
import argparse
import json
from collections import defaultdict
from math import prod

# --- roofs, all measured on this card. Source for each is the state doc / knowledge file named. ---
ROOFS = {
    # compute
    "compute_trimul_pc": 40.40e12,   # measured, tri_matmul at 512 aa on qb2 with its program config (§62)
    "compute_l1_nt64": 93.47e12,     # measured, K=256 nt=64 L1 output (z-transition-chunk pass 2)
    # writers
    "w_matmul_dram": 168.5e9,        # matmul writer, DRAM out (protenix-trunk--trimul.md corr #3)
    "w_unary_dram": 263.6e9,         # unary writer, DRAM out (ibid.)
    "w_l1": 1152e9,                  # L1 clone (§72 step 3)
    "r_dram": 277.6e9,               # DRAM read (§80)
}
THRESH = 0.60

DTYPE_B = {
    "BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0,
    "UINT16": 2.0, "UINT8": 1.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625,
}
MM_OPS = {"matmul", "linear", "minimal_matmul"}
SDPA_OPS = {"scaled_dot_product_attention"}


def owner_map(path):
    """line -> "Class.func" for tenstorrent.py, so a census row can be attributed to its body.

    The census records call sites as line numbers only. Attribution has to come from the file the
    census was taken against, which is why this parses it rather than hard-coding line ranges: the
    numbers moved between the 320 and 512 censuses and a stale range silently mis-attributes.
    """
    import ast
    src = open(path).read()
    tree = ast.parse(src)
    out = []

    def walk(node, cls=None):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                walk(ch, ch.name)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((ch.lineno, getattr(ch, "end_lineno", ch.lineno), cls, ch.name))
                walk(ch, cls)
            else:
                walk(ch, cls)
    walk(tree)
    return out


def owner_of(owners, line):
    best = None
    for s, e, cls, fn in owners:
        if s <= line <= e and (best is None or s > best[0]):
            best = (s, cls, fn)
    return (best[1], best[2]) if best else (None, None)


BODY = {"TriangleMultiplication": "trimul", "TriangleAttention": "tri_att"}


def body_of(owners, chain):
    """Innermost frame that names a real submodule. `Module` is the shared helper base, skip it."""
    names = []
    for fr in chain:
        try:
            line = int(str(fr).rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        cls, fn = owner_of(owners, line)
        if cls and cls != "Module":
            names.append(cls)
    for n in names:
        if n in BODY:
            return BODY[n]
    return names[0] if names else "other"


def tb(d):
    """Bytes of one tensor descriptor, from the padded shape the census recorded."""
    if not d or not isinstance(d, dict) or not d.get("shape"):
        return 0.0
    return prod(d["shape"]) * DTYPE_B.get(str(d.get("dtype", "BFLOAT16")).upper(), 2.0)


def row_flop(r):
    """FLOP for a row, 0 for anything that does no arithmetic worth counting."""
    op, ins, out = r["op"], r.get("in") or [], r.get("out")
    if not out or not out.get("shape"):
        return 0.0
    if op in MM_OPS:
        if not ins or not ins[0].get("shape"):
            return 0.0
        k = ins[0]["shape"][-1]
        return 2.0 * prod(out["shape"]) * k
    if op in SDPA_OPS:
        # q (B,H,S,D), k (B,H,S,D): QK^T then PV, both 2*B*H*S*S*D
        q = ins[0]["shape"] if ins and ins[0].get("shape") else None
        k = ins[1]["shape"] if len(ins) > 1 and ins[1].get("shape") else None
        if not q or not k or len(q) < 2 or len(k) < 2:
            return 0.0
        b, s_q, d = prod(q[:-2]), q[-2], q[-1]
        s_k = k[-2]
        return 4.0 * b * s_q * s_k * d
    return 0.0


def writer_roof(r):
    """The roof this row's own writer can reach, picked by op class and output buffer type."""
    out = r.get("out") or {}
    if str(out.get("buf", "DRAM")).upper() == "L1":
        return ROOFS["w_l1"], "L1"
    return (ROOFS["w_matmul_dram"], "mm/DRAM") if r["op"] in MM_OPS | SDPA_OPS \
        else (ROOFS["w_unary_dram"], "un/DRAM")


def read_roof_bytes(r):
    """Read bytes, split by where they come from. L1-resident reads are not DRAM traffic."""
    dram = l1 = 0.0
    for d in (r.get("in") or []):
        b = tb(d)
        if str((d or {}).get("buf", "DRAM")).upper() == "L1":
            l1 += b
        else:
            dram += b
    return dram, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--invocations", type=int, default=480,
                    help="block invocations per fold: 48 blocks x 10 recycles (protenix.py:1982/2006)")
    ap.add_argument("--source", default="tt_bio/tenstorrent.py",
                    help="the file the census was taken against; line numbers are resolved here")
    ap.add_argument("--compute-roof", default="compute_trimul_pc",
                    choices=["compute_trimul_pc", "compute_l1_nt64"])
    a = ap.parse_args()

    c = json.load(open(a.census))
    recs = c["records"]
    croof = ROOFS[a.compute_roof]
    owners = owner_map(a.source)

    errs = [r for r in recs if r.get("error")]
    good = [r for r in recs if not r.get("error") and r.get("s")]

    buckets = defaultdict(float)          # bucket -> seconds/block
    bybucket_ops = defaultdict(lambda: defaultdict(float))
    per_body = defaultdict(lambda: defaultdict(float))
    tot = dict(t=0.0, flop=0.0, rd=0.0, rl1=0.0, w=0.0,
               arith_floor=0.0, byte_floor=0.0, row_floor=0.0)
    rows_out = []

    for r in good:
        t = float(r["s"])
        f = row_flop(r)
        rd, rl1 = read_roof_bytes(r)
        wb = tb(r.get("out"))
        wroof, wtag = writer_roof(r)
        af = f / croof
        bf = max(rd / ROOFS["r_dram"], wb / wroof, rl1 / ROOFS["w_l1"])
        ae, be = af / t, bf / t
        bucket = "arithmetic" if ae >= THRESH else ("bandwidth" if be >= THRESH else "neither")

        ch = r.get("chain") or []
        if not isinstance(ch, list):
            ch = [ch]
        body = body_of(owners, [r.get("site")] + list(ch))

        buckets[bucket] += t
        bybucket_ops[bucket][r["op"]] += t
        per_body[body][bucket] += t
        per_body[body]["total"] += t
        for k, v in (("t", t), ("flop", f), ("rd", rd), ("rl1", rl1), ("w", wb),
                     ("arith_floor", af), ("byte_floor", bf), ("row_floor", max(af, bf))):
            tot[k] += v
        rows_out.append(dict(i=r.get("i"), op=r["op"], site=r.get("site"), body=body,
                             s=t, flop=f, rd_MB=rd / 1e6, rl1_MB=rl1 / 1e6, w_MB=wb / 1e6,
                             tflops=(f / t / 1e12) if f else 0.0,
                             wroof=wtag, arith_eff=ae, byte_eff=be, bucket=bucket))

    inv = a.invocations
    wall = c.get("block_wall_s") or 0.0
    out = dict(
        census=a.census, model=c.get("model"), n=c.get("n"), fast=c.get("fast"),
        loadavg=c.get("loadavg"), compute_roof=a.compute_roof, compute_roof_tflops=croof / 1e12,
        thresh=THRESH, invocations=inv,
        n_ops=len(recs), n_error=len(errs), n_timed=len(good),
        error_ops=sorted({r["op"] for r in errs}),
        block_wall_s=wall, census_sum_s=tot["t"],
        coverage_pct=(100.0 * tot["t"] / wall) if wall else None,
        fold_s_from_census=tot["t"] * inv, fold_s_from_wall=wall * inv,
        buckets_ms_block={k: 1e3 * v for k, v in buckets.items()},
        buckets_s_fold={k: v * inv for k, v in buckets.items()},
        buckets_pct={k: 100.0 * v / tot["t"] for k, v in buckets.items()} if tot["t"] else {},
        per_body_s_fold={b: {k: v * inv for k, v in d.items()} for b, d in per_body.items()},
        top_ops_by_bucket={k: sorted(((o, 1e3 * s) for o, s in d.items()),
                                     key=lambda x: -x[1])[:8] for k, d in bybucket_ops.items()},
        totals=dict(
            flop_block=tot["flop"], flop_fold=tot["flop"] * inv,
            tflops_achieved=tot["flop"] / tot["t"] / 1e12 if tot["t"] else 0.0,
            dram_read_MB_block=tot["rd"] / 1e6, l1_read_MB_block=tot["rl1"] / 1e6,
            write_MB_block=tot["w"] / 1e6,
            dram_read_GB_fold=tot["rd"] * inv / 1e9, write_GB_fold=tot["w"] * inv / 1e9,
            arith_floor_s_fold=tot["arith_floor"] * inv,
            byte_floor_s_fold=tot["byte_floor"] * inv,
            row_floor_s_fold=tot["row_floor"] * inv,
        ),
        rows=sorted(rows_out, key=lambda r: -r["s"]),
    )
    json.dump(out, open(a.out, "w"), indent=1)

    p = out
    print(f"{p['model']} n={p['n']} fast={p['fast']} load={p['loadavg']} roof={p['compute_roof_tflops']:.2f} TF/s")
    print(f"ops {p['n_timed']} timed, {p['n_error']} error {p['error_ops']}")
    print(f"census sum {1e3*p['census_sum_s']:.3f} ms/block vs block wall {1e3*p['block_wall_s']:.3f} ms"
          f"  -> coverage {p['coverage_pct']:.1f} %")
    print(f"fold: census {p['fold_s_from_census']:.3f} s, wall {p['fold_s_from_wall']:.3f} s "
          f"({p['invocations']} invocations)")
    print(f"achieved {p['totals']['tflops_achieved']:.2f} TFLOP/s on {p['totals']['flop_fold']:.4e} FLOP/fold")
    print(f"traffic/fold: {p['totals']['dram_read_GB_fold']:.1f} GB DRAM read, "
          f"{p['totals']['write_GB_fold']:.1f} GB write, {p['totals']['l1_read_MB_block']:.1f} MB/block L1 read")
    print("\n  bucket        ms/block     s/fold      %")
    for k in ("arithmetic", "bandwidth", "neither"):
        print(f"  {k:<12}{p['buckets_ms_block'].get(k,0):>10.3f}{p['buckets_s_fold'].get(k,0):>11.3f}"
              f"{p['buckets_pct'].get(k,0):>8.1f}")
    print(f"\nper-row floors s/fold: arithmetic {p['totals']['arith_floor_s_fold']:.3f}, "
          f"bytes {p['totals']['byte_floor_s_fold']:.3f}, max-per-row {p['totals']['row_floor_s_fold']:.3f}")
    print("\nper body s/fold:")
    for b, d in sorted(p["per_body_s_fold"].items(), key=lambda x: -x[1].get("total", 0)):
        print(f"  {b:<8} total {d.get('total',0):>8.3f}  arith {d.get('arithmetic',0):>8.3f}"
              f"  bw {d.get('bandwidth',0):>8.3f}  neither {d.get('neither',0):>8.3f}")
    print("\ntop 12 rows by time:")
    for r in p["rows"][:12]:
        print(f"  {1e3*r['s']:>8.3f} ms {r['op']:<28} {r['bucket']:<11} {r['tflops']:>6.2f} TF/s "
              f" ae {r['arith_eff']:.2f} be {r['byte_eff']:.2f}  {r['site']}")


if __name__ == "__main__":
    main()
