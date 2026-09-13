"""Split the census's single `GenericOp` row into the programs it actually is.

Host only, opens no device. `ttnn.generic_op` reports one op code for every hand-written kernel in
tt-bio, so a per-op-code table cannot tell the trimul channel move from the TriAtt SDPA. The
in/out shape signature can. Run on any ops CSV:

    python3 genericop_split.py perf/b2z2_byte_floor/art/ops_perf_blocksum_qb2c0.csv.gz
"""
import csv, gzip, json, statistics as st, sys
from collections import defaultdict


def sig(r, cols, pre):
    out = []
    for i in range(12):
        k = f"{pre}_{i}_X_PAD[LOGICAL]"
        if k not in cols or not r.get(k):
            break
        out.append("x".join(r[f"{pre}_{i}_{a}_PAD[LOGICAL]"].split("[")[0] for a in "WZYX"))
    return tuple(out)


def name(s):
    n = len(s)
    if s[0] == "1x512x512x512":
        return "reblock_permute_gated (trimul gate + channel move)"
    if s[0] == "1x128x512x512":
        return "reblock_permute_back (channel move home)"
    if s[0] == "512x4x512x32":
        return "triatt_sdpa" if n == 6 else "TriAtt out-proj [512,4,512,32]@[128,128]"
    if n > 1 and s[1] == "1x1x128x512":
        return "trimul fused in-proj [.,128]@[128,512]"
    if n > 1 and s[1] == "1x1x128x384":
        return "TriAtt qkv-proj [.,128]@[128,384]"
    if n > 1 and s[1] == "1x1x128x128":
        return "TriAtt gate-proj [.,128]@[128,128]"
    return "unclassified " + "|".join(s)


def main(path):
    rd = csv.DictReader(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    cols = rd.fieldnames
    rows = [r for r in rd if r["OP CODE"].startswith("GenericOp")]
    g = defaultdict(list)
    for r in rows:
        g[name(sig(r, cols, "INPUT"))].append(r)
    F = lambda r, k: float(r[k])
    tot = sum(F(r, "DEVICE KERNEL DURATION [ns]") for r in rows)
    print(f"{len(rows)} GenericOp rows, {tot/1e6:.3f} ms device time\n")
    hdr = (f"{'program':52s} {'n':>3s} {'mean us':>8s} {'%gen':>5s} "
           f"{'T0':>5s} {'T1':>5s} {'T2':>5s} {'in/T0':>6s} {'out/T2':>6s}")
    print(hdr + "\n" + "-" * len(hdr))
    res = {}
    for k in sorted(g, key=lambda k: -sum(F(r, "DEVICE KERNEL DURATION [ns]") for r in g[k])):
        v = g[k]
        cc = float(v[0]["CORE COUNT"])
        m = lambda c: st.mean(F(r, c) for r in v)
        dur = m("DEVICE KERNEL DURATION [ns]")
        t0, t1, t2 = (m(f"DEVICE TRISC{i} KERNEL DURATION [ns]") for i in (0, 1, 2))
        wf = m("DEVICE COMPUTE CB WAIT FRONT [ns]") / cc
        rb = m("DEVICE COMPUTE CB RESERVE BACK [ns]") / cc
        share = sum(F(r, "DEVICE KERNEL DURATION [ns]") for r in v) / tot
        print(f"{k:52s} {len(v):3d} {dur/1000:8.1f} {100*share:5.1f} "
              f"{100*t0/dur:5.1f} {100*t1/dur:5.1f} {100*t2/dur:5.1f} "
              f"{100*wf/t0:6.1f} {100*rb/t2:6.1f}")
        res[k] = {"n": len(v), "mean_us": dur / 1000, "share": share, "cores": cc,
                  "in_over_trisc0": wf / t0, "out_over_trisc2": rb / t2}
    w = sum(r["share"] * r["in_over_trisc0"] for r in res.values())
    b0 = sum(1 for r in rows
             if F(r, "DEVICE COMPUTE CB WAIT FRONT [ns]") / float(r["CORE COUNT"])
             > F(r, "DEVICE TRISC0 KERNEL DURATION [ns]"))
    b2 = sum(1 for r in rows
             if F(r, "DEVICE COMPUTE CB RESERVE BACK [ns]") / float(r["CORE COUNT"])
             > F(r, "DEVICE TRISC2 KERNEL DURATION [ns]"))
    print(f"\ntime-weighted input stall {100*w:.1f} %  "
          f"(FINDINGS.md gives 56.6 % for GenericOp on this capture)")
    print(f"counter bound check: wait_front>TRISC0 {b0}/{len(rows)}, "
          f"reserve_back>TRISC2 {b2}/{len(rows)}")
    json.dump({"capture": path, "weighted_input_stall": w, "programs": res},
              open(__file__.replace(".py", ".json"), "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "perf/b2z2_byte_floor/art/ops_perf_blocksum_qb2c0.csv.gz")
