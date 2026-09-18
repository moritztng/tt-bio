#!/usr/bin/env python3
"""Price TT_BIO_APB_CONCAT_HEADS from measured per-class launch costs, before spending a board pair.

The flag's premise is a launch count: the shipped head re-assembly takes four launches, the flag
takes one (`tenstorrent.py:1438`). This prices both arms against
`perf/roof_launch/launch_trace_qb2c1.json` -- per-class fixed cost and per-tile slope measured
under TRACE REPLAY on qb2 p300c, the same part and grid the fold of record runs on -- and against
the shapes the ops actually run at, read out of the captured graph of the token-DiT
AttentionPairBias at 512 aa (`perf/roof_quiet/capture/captures`), not assumed.

Two things the count argument leaves out, and together they decide the sign:

  * `nlp_concat_heads` has the HIGHEST fixed cost in the whole measured ladder, 14.57 us, which is
    2.3-3.2x slice, transpose and permute. "One launch" is not one cheap launch.
  * the one-op path still needs a reshape, and on a 33 % WIDER tensor, because it keeps the pad
    lanes. `reshape`'s slope is 0.042 us/tile, 6-14x steeper than every other class here, so the
    flag pays MORE for its reshape than the shipped path pays for its own.

Then the cost side the comment does name: the gate and the output projection run at
n_heads*padded_head_dim, 1024 instead of 768.

Run anywhere, no device: it reads two committed artifacts.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCH = REPO / "perf" / "roof_launch" / "launch_trace_qb2c1.json"

# Shapes read out of cap_AttentionPairBias__1x512x768,1x16x512x512.json.gz, the token-DiT site at
# 512 aa: S=512, n_heads=16, head_dim=48, padded_head_dim=64 (head_dim_padding = -48 % 32 = 16).
S, H, D, DP, C = 512, 16, 48, 64, 768
TILE = 32


def tiles(*dims):
    """Tile count: the last two axes are tiled, the rest are batch. A partial tile still costs one."""
    import math
    b = 1
    for d in dims[:-2]:
        b *= d
    return b * math.ceil(dims[-2] / TILE) * math.ceil(dims[-1] / TILE)


def main() -> int:
    fits = json.loads(LAUNCH.read_text())["fits"]

    def cost(cls, n_tiles):
        f = fits[cls]
        return f["launch_floor_us"] + f["slope_us_per_tile"] * n_tiles, n_tiles

    # ---- shipped: slice, permute, reshape, permute. Every one of them allocates a device tensor in
    # the capture, so all four are programs, not views.
    shipped = [
        ("slice      o[..., :48]", "slice", tiles(1, H, S, DP)),
        ("permute    (0,1,3,2)", "transpose", tiles(1, H, S, D)),
        ("reshape    (1,768,512)", "reshape", tiles(1, H * D, S)),
        ("permute    (0,2,1)", "permute", tiles(1, H * D, S)),
    ]
    # ---- flag: nlp_concat_heads, then a reshape to (b, seq, n_heads*padded_head_dim)
    flag = [
        ("nlp_concat_heads", "nlp_concat_heads", tiles(1, H, S, DP)),
        ("reshape    (1,512,1024)", "reshape", tiles(1, S, H * DP)),
    ]

    out = {"artifact": str(LAUNCH.relative_to(REPO)),
           "shapes_from": "perf/roof_quiet/capture/captures/"
                          "cap_AttentionPairBias__1x512x768,1x16x512x512.json.gz",
           "site": {"S": S, "n_heads": H, "head_dim": D, "padded_head_dim": DP, "c": C},
           "arms": {}}
    for name, rows in (("shipped", shipped), ("flag", flag)):
        tot = 0.0
        out["arms"][name] = {"ops": []}
        print(f"--- {name}")
        for label, cls, nt in rows:
            us, nt = cost(cls, nt)
            tot += us
            out["arms"][name]["ops"].append({"op": label, "class": cls, "tiles": nt,
                                             "us": round(us, 3)})
            print(f"    {label:26s} {cls:18s} {nt:5d} tiles  {us:7.2f} us")
        out["arms"][name]["total_us"] = round(tot, 3)
        print(f"    {'':26s} {'':18s} {'':5s}        {tot:7.2f} us total")

    saved = out["arms"]["shipped"]["total_us"] - out["arms"]["flag"]["total_us"]

    # ---- cost side: the gate and proj_o run at n_heads*padded_head_dim.
    # proj_g's OUTPUT axis widens (relaned axis=-1), proj_o's INPUT axis widens (axis=0).
    add_flops = 2 * S * C * (H * DP - H * D) * 2          # both matmuls, 2 FLOP per MAC
    # gate elementwise over the wider tensor: two reads and a write of the extra lanes, plus the
    # extra weight columns/rows both matmuls must read. bf16, 2 B.
    add_bytes = S * (H * DP - H * D) * 2 * 3 + C * (H * DP - H * D) * 2 * 2
    # rates measured in-fold by this campaign: the add class at 423.6 GB/s, and a matmul of this
    # shape well below cube peak. Both stated as a band, because the band is what is honest.
    for tflops in (60.0, 115.0):
        for gbs in (423.6,):
            arith_us = add_flops / (tflops * 1e12) * 1e6
            traf_us = add_bytes / (gbs * 1e9) * 1e6
            net = saved - arith_us - traf_us
            row = {"assumed_tflops": tflops, "assumed_gbs": gbs,
                   "reassembly_saved_us": round(saved, 3),
                   "added_arith_us": round(arith_us, 3), "added_traffic_us": round(traf_us, 3),
                   "net_us_per_call": round(net, 3),
                   "net_s_at_5064_calls": round(net * 5064 / 1e6, 4)}
            out.setdefault("net", []).append(row)
            print(f"net @ {tflops:5.0f} TFLOP/s: reassembly {saved:+.2f} us, "
                  f"arith {-arith_us:+.2f} us, traffic {-traf_us:+.2f} us "
                  f"=> {net:+.2f} us/call, {net * 5064 / 1e6:+.4f} s at 5064 served calls")

    # ---- the sign has to survive the one assumption that could be wrong: that the flag's reshape
    # is a program and not a view. Charge it zero on BOTH arms and re-read the sign.
    sh_noview = sum(o["us"] for o in out["arms"]["shipped"]["ops"] if o["class"] != "reshape")
    fl_noview = sum(o["us"] for o in out["arms"]["flag"]["ops"] if o["class"] != "reshape")
    saved_noview = sh_noview - fl_noview
    out["reshape_free_control"] = {
        "shipped_us": round(sh_noview, 3), "flag_us": round(fl_noview, 3),
        "reassembly_saved_us": round(saved_noview, 3),
        "net_us_per_call_at_115_tflops": round(
            saved_noview - add_flops / 115e12 * 1e6 - add_bytes / 423.6e9 * 1e6, 3)}
    print(f"\nreshape-free control (both reshapes charged zero): saved {saved_noview:+.2f} us/call, "
          f"net {out['reshape_free_control']['net_us_per_call_at_115_tflops']:+.2f} us/call")
    out["added"] = {"flops_per_call": add_flops, "bytes_per_call": add_bytes}
    dest = REPO / "perf" / "c14_land" / "apb_launch_price.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
