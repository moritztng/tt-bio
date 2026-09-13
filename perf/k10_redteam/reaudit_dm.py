#!/usr/bin/env python3
"""Independent re-reduction of k10-instrument's committed DM-zone JSON.

Host only, opens no device. Inputs are the raw artifacts on wk/k10-instrument:
  perf/k10_instrument/{starved,starved2,compute,compute2,block,block2}_dm.json

k10-instrument's state doc reports every reader figure from NCRISC alone. These
files also carry BRISC. Re-reducing both threads changes three of its arguments.
Nothing here re-runs a device; it is arithmetic on numbers that row measured.

Usage: reaudit_dm.py <dir-with-the-json>
"""
import json
import sys


def frac(d, key="ALL"):
    if key in d:
        return d[key]["frac"], d[key]
    return d["op_codes"][key]["frac"], d["op_codes"][key]


def line(tag, f):
    # producer-side = DM thread blocked waiting for data to arrive from DRAM/NoC
    prod = f["br_noc_read"] + f["nc_noc_read"]
    # consumer-side, as k10-instrument defines the zones: DM blocked BY the compute
    # cluster, either on a CB it already filled or waiting for compute output.
    cons_rb = f["br_reserve_back"] + f["nc_reserve_back"]
    cons_wf = f["br_wait_front"] + f["nc_wait_front"]
    return dict(
        tag=tag,
        nc_prod=f["nc_noc_read"],
        nc_cons=f["nc_reserve_back"] + f["nc_wait_front"],
        dm_prod=prod,
        dm_cons=cons_rb + cons_wf,
        dm_cons_rb=cons_rb,
        dm_cons_wf=cons_wf,
        cpu_wf=f["cpu_wait_front"],
        trisc0=f["trisc0"],
        resid=f["br_unaccounted"] + f["nc_unaccounted"],
    )


def main(d):
    st, _ = frac(json.load(open(f"{d}/starved2_dm.json")))
    cb, _ = frac(json.load(open(f"{d}/compute2_dm.json")))
    blk = json.load(open(f"{d}/block2_dm.json"))
    gen, genrow = frac(blk, "GenericOpDeviceOperation")

    rows = [line("starved ttnn.add", st), line("compute-bound matmul", cb),
            line("TARGET fused GenericOp", gen)]

    print("A. CONSUMER-SIDE SEPARATION, NCRISC-only (as published) vs both DM threads")
    print(f"{'case':<26}{'NC cons':>9}{'DM cons':>9}{'DM rsv-back':>13}{'DM wait-front':>15}")
    for r in rows:
        print(f"{r['tag']:<26}{r['nc_cons']:>8.1%}{r['dm_cons']:>9.1%}"
              f"{r['dm_cons_rb']:>13.1%}{r['dm_cons_wf']:>15.1%}")
    s, c, g = rows
    print(f"  published separation  target/starved on NC consumer : "
          f"{g['nc_cons'] / s['nc_cons']:.1f}x")
    print(f"  same ratio on BOTH DM threads                       : "
          f"{g['dm_cons'] / s['dm_cons']:.2f}x")
    print(f"  on reserve-back alone (the zone the starved control  ")
    print(f"  actually holds near zero)                           : "
          f"{g['dm_cons_rb'] / s['dm_cons_rb']:.1f}x   "
          f"(compute-bound control reads {c['dm_cons_rb']:.1%})")

    print("\nB. PRODUCER-SIDE SEPARATION (the axis the 39 % bound rests on)")
    print(f"{'case':<26}{'NC noc-read':>12}{'DM noc-read':>13}")
    for r in rows:
        print(f"{r['tag']:<26}{r['nc_prod']:>11.1%}{r['dm_prod']:>13.1%}")
    print(f"  starved/compute-bound separation, both threads : "
          f"{s['dm_prod'] / c['dm_prod']:.0f}x")

    print("\nC. THE 1:1 CONVERSION ASSUMPTION, CALIBRATED ON THE STARVED CONTROL")
    print("   The starved arm ran at 98.9 % of the measured DRAM roof, so essentially")
    print("   all of its compute input stall is memory-caused. Its own ratio is the")
    print("   empirical ns-of-compute-stall per ns-of-reader-DRAM-wait.")
    k = s["cpu_wf"] / s["dm_prod"]
    print(f"   starved: compute wait-front {s['cpu_wf']:.2%} / reader DRAM "
          f"{s['dm_prod']:.2%} = {k:.3f}")
    b39 = g["dm_prod"] / g["cpu_wf"]
    print(f"   published bound  reader DRAM / compute stall = "
          f"{g['dm_prod']:.2%} / {g['cpu_wf']:.2%} = {b39:.1%}")
    print(f"   same bound at the calibrated conversion      = {b39 * k:.1%}")

    print("\nD. 'ISSUING' IS A RESIDUAL, NOT A MEASUREMENT")
    print("   dm_report.py reports whatever is left of a thread's residency after the")
    print("   five zones as 'issuing'. Op classes where that residual is implausibly")
    print("   large for address generation alone:")
    for op, v in sorted(blk["op_codes"].items(), key=lambda kv: -kv[1]["total_span_ms"]):
        f = v["frac"]
        for t in ("br", "nc"):
            if f[f"{t}_unaccounted"] > 0.35:
                print(f"     {op:<32} {t.upper()} residual {f[f'{t}_unaccounted']:>6.1%}"
                      f"  (span {v['total_span_ms']:.2f} ms)")

    print("\nE. THE COMPUTE-STALL COLUMN IS NORMALISED TO SPAN, NOT TO TRISC0")
    print("   Where TRISC0 residency is well below 1 the two differ, and the published")
    print("   table ranks ops on the span version.")
    print(f"{'op':<32}{'TRISC0 res':>11}{'of span':>9}{'of TRISC0':>11}")
    for op, v in sorted(blk["op_codes"].items(), key=lambda kv: -kv[1]["total_span_ms"]):
        f = v["frac"]
        if f["trisc0"] < 0.01:
            continue
        print(f"{op:<32}{f['trisc0']:>10.1%}{f['cpu_wait_front']:>9.1%}"
              f"{f['cpu_wait_front'] / f['trisc0']:>11.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")


def census_and_cashout(d):
    """F: re-derive the census split from the instrument's block capture.
    G: the one end-to-end cash-out where both a prediction and a measurement exist."""
    blk = json.load(open(f"{d}/block2_dm.json"))
    blk1 = json.load(open(f"{d}/block_dm.json"))
    print("\nF. THE CENSUS SPLIT, RE-DERIVED FROM A DIFFERENT CAPTURE ON A DIFFERENT BUILD")
    print("   census (cb_split.py, fenced block median, stock-ish b2z build):")
    print("     wait-front 57.0 %, reserve-back 9.7 %, in:out 5.9:1")
    for nm, b in (("block2 (5-zone)", blk), ("block (3-zone)", blk1)):
        f = b["ALL"]["frac"]
        sp_i, sp_o = f["cpu_wait_front"], f["cpu_reserve_back"]
        th_i, th_o = sp_i / f["trisc0"], sp_o / f["trisc2"]
        print(f"   {nm:<18} of span     {sp_i:.1%} / {sp_o:.1%} = {sp_i / sp_o:.2f}:1")
        print(f"   {'':<18} of own thread {th_i:.1%} / {th_o:.1%} = {th_i / th_o:.2f}:1")
    f = blk["ALL"]["frac"]
    th_i = f["cpu_wait_front"] / f["trisc0"]
    print(f"   gap on the headline stall: {th_i:.1%} vs census 57.0 % = "
          f"{abs(th_i - 0.570) / 0.570:.1%}")
    print("   error bar on 5.9:1 is set by POPULATION, not by normalisation:")
    print("     normalisation (span vs own thread), same capture : "
          f"{abs(f['cpu_wait_front'] / f['cpu_reserve_back'] - th_i / (f['cpu_reserve_back'] / f['trisc2'])) / (th_i / (f['cpu_reserve_back'] / f['trisc2'])):.1%}")
    print("     population (fenced block 5.88 vs whole capture 4.86, "
          "k10-orchestrator's own numbers): 17 %")

    print("\nG. DOES AN ATTRIBUTED SECOND ACTUALLY CASH? the one matched pair that exists")
    print("   k10-orchestrator's prior: 'attributed stall seconds have so far returned")
    print("   about 12 % of their face value' (util-op-deletes 1.015x against a 1.062x")
    print("   bound over the whole BinaryNg + Transpose classes).")
    print("   But 1.015x is not a deletion of those classes. It is a CHAIN of two")
    print("   deletions, and only D4 has an op-bench prediction to compare against:")
    base, ratio, pred = 18.5509, 1.00613, 0.1545
    got = base * (1 - 1 / ratio)
    print(f"     D4 op bench, production shape, qb2 card 1 : {pred:.4f} s/fold predicted")
    print(f"     D4 end to end, 6 paired rounds, base median {base:.4f} s at {ratio}x")
    print(f"                                               : {got:.4f} s/fold banked")
    print(f"     cash-out rate                             : {got / pred:.0%}, not 12 %")
