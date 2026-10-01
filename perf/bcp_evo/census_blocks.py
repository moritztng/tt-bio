#!/usr/bin/env python3
"""Evoformer census by sub-block: calls, card seconds and DRAM bytes a round, forward and backward.

    census_blocks.py <seq.tsv.gz from seq_dump.py> [--json out.json]

Input is bcp-roofline's device-profiled N=1 round (qb2 p300c card 0, AICLK 1350, 288 tokens,
4 identical rounds, main 1ee324e01), dumped op by op by seq_dump.py. The Evoformer program is
periodic: 158 ops a block in an untaped forward seam (7,584 = 48 x 158), 646 in the backward seam
(31,007 = 48 x 646 - 1), and every op's position in its period maps to one sub-block. The ranges
below were read off one block of each (ops named, shapes checked); a range edge can only be off by
small [2,288,256]-sized ops, which carry no bytes that matter.

The profile predates lnbw (bcp-device), so each composed layer-norm backward (a 19-op Reduce/DIV/
SUB/MUL chain) is replaced, as a model, by one call at lnbw's measured op-level ratio (6.3x on the
pair, 4.7x on the MSA, bcp-device lnbw_bench) and 3 passes of its input's bytes (x, dy in; dx out).

Per round: forward = 2 x the mean 7,584-op seam (two Evoformer passes a round, untaped; the two
7,728-op seams are the same program plus 3 small ops); backward = the mean of the three 31,007-op
seams (the 30,386-op first seam is round 1's variant).
"""
import argparse
import collections
import csv
import gzip
import json

FWD = [(0, 7, "mask"), (8, 40, "opm"), (41, 61, "msa_row"), (62, 77, "msa_col"),
       (78, 81, "msa_trans"), (82, 99, "tri_mul_out"), (100, 117, "tri_mul_in"),
       (118, 132, "tri_att_start"), (133, 135, "pair_T"), (136, 149, "tri_att_end"),
       (150, 152, "pair_T"), (153, 153, "tri_att_end"), (154, 157, "pair_trans")]
BWD = [(0, 7, "mask"), (8, 40, "opm"), (41, 61, "msa_row"), (62, 77, "msa_col"),
       (78, 82, "msa_trans"), (83, 100, "tri_mul_out"), (101, 118, "tri_mul_in"),
       (119, 133, "tri_att_start"), (134, 136, "pair_T"), (137, 150, "tri_att_end"),
       (151, 153, "pair_T"), (154, 154, "tri_att_end"), (155, 159, "pair_trans")]
BWD_B = [(160, 183, "pair_trans"), (184, 184, "tri_att_end"), (185, 187, "pair_T"),
         (188, 234, "tri_att_end"), (235, 237, "pair_T"), (238, 286, "tri_att_start"),
         (287, 359, "tri_mul_in"), (360, 433, "tri_mul_out"), (434, 458, "msa_trans"),
         (459, 515, "msa_col"), (516, 568, "msa_row"), (569, 618, "opm"),
         (619, 645, "msa_row")]
LN_CHAIN = ["Reduce", "DIV", "SUB", "MUL", "Reduce", "DIV", "ADD", "Unary", "MUL", "MUL",
            "Reduce", "DIV", "MUL", "Reduce", "DIV", "SUB", "MUL", "SUB", "MUL"]
ORDER = ["opm", "msa_row", "msa_col", "msa_trans", "tri_mul_out", "tri_mul_in", "tri_att_start",
         "tri_att_end", "pair_T", "pair_trans", "mask"]


def short(fam):
    f = fam.replace("DeviceOperation", "").replace("BinaryNg:", "")
    return "Unary" if f.startswith("Unary:") else f


def label(table, i):
    for lo, hi, name in table:
        if lo <= i <= hi:
            return name
    raise ValueError(i)


def tally(seam, period, tables, phase_of):
    out = collections.defaultdict(collections.Counter)
    rows = list(seam)
    i = 0
    while i < len(rows):
        r = rows[i]
        off = i % period
        sub = label(tables, off)
        ph = phase_of(off)
        fams = [short(x["fam"]) for x in rows[i:i + len(LN_CHAIN)]]
        if ph == "bwd" and fams == LN_CHAIN:
            chain = rows[i:i + len(LN_CHAIN)]
            k = sum(int(x["kern_ns"]) for x in chain) * 1e-9
            b = sum(int(x["dram_b"]) for x in chain)
            pair = "288x288x128" in chain[-1]["outs"]
            xbytes = int(chain[-1]["dram_b"]) / 3          # MUL a x b -> out: three equal tensors
            out[(sub, ph)].update(calls=len(LN_CHAIN), kern=k, dram=b,
                                  calls_l=1, kern_l=k / (6.3 if pair else 4.7), dram_l=3 * xbytes)
            out[(sub, ph)]["lnbw_chains"] += 1
            i += len(LN_CHAIN)
            continue
        k, b = int(r["kern_ns"]) * 1e-9, int(r["dram_b"])
        out[(sub, ph)].update(calls=1, kern=k, dram=b, calls_l=1, kern_l=k, dram_l=b)
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--json")
    a = ap.parse_args()
    seams = collections.OrderedDict()
    for r in csv.DictReader(gzip.open(a.seq, "rt"), delimiter="\t"):
        if r["name"] in ("evo_fwd", "evo_bwd"):
            seams.setdefault(r["seam"], []).append(r)
    fwd = [v for v in seams.values() if len(v) == 7584]
    bwd = [v for v in seams.values() if len(v) == 31007]
    tot = collections.defaultdict(collections.Counter)
    for s in fwd:
        for key, c in tally(s, 158, FWD, lambda o: "fwd").items():
            for f, v in c.items():
                tot[key][f] += v * 2 / len(fwd)
    for s in bwd:
        for key, c in tally(s, 646, BWD + BWD_B,
                            lambda o: "rcp" if o < 160 else "bwd").items():
            for f, v in c.items():
                tot[key][f] += v / len(bwd)
    print("Evoformer, 48 blocks, per round (N=1). fwd = 2 untaped passes, rcp = the backward's "
          "checkpoint recompute, bwd = the VJP.\n'_l' columns: with lnbw's chains replaced "
          "(modelled, see docstring).\n")
    hdr = (f"{'sub-block':14s} {'ph':3s} {'calls':>6s} {'kern s':>7s} {'GB':>7s} | "
           f"{'calls_l':>7s} {'kern_l':>7s} {'GB_l':>7s} {'GB/s_l':>6s}")
    print(hdr)
    rows = []
    for sub in ORDER:
        for ph in ("fwd", "rcp", "bwd"):
            c = tot.get((sub, ph))
            if not c:
                continue
            rows.append({"sub": sub, "phase": ph, **{k: float(v) for k, v in c.items()}})
            print(f"{sub:14s} {ph:3s} {c['calls']:6.0f} {c['kern']:7.3f} {c['dram'] / 1e9:7.1f} | "
                  f"{c['calls_l']:7.0f} {c['kern_l']:7.3f} {c['dram_l'] / 1e9:7.1f} "
                  f"{c['dram_l'] / 1e9 / c['kern_l'] if c['kern_l'] else 0:6.0f}")
    print()
    print(f"{'sub-block, all phases':22s} {'calls_l':>7s} {'kern_l s':>8s} {'GB_l':>7s} {'share':>6s}")
    T = sum(r["kern_l"] for r in rows)
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        agg[r["sub"]].update(calls_l=r["calls_l"], kern_l=r["kern_l"], dram_l=r["dram_l"])
    for sub, c in sorted(agg.items(), key=lambda x: -x[1]["dram_l"]):
        print(f"{sub:22s} {c['calls_l']:7.0f} {c['kern_l']:8.3f} {c['dram_l'] / 1e9:7.1f} "
              f"{c['kern_l'] / T * 100:5.1f}%")
    Tc = sum(r["calls_l"] for r in rows)
    Tb = sum(r["dram_l"] for r in rows)
    print(f"{'Evoformer total':22s} {Tc:7.0f} {T:8.3f} {Tb / 1e9:7.1f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
