#!/usr/bin/env python3
"""bcx-p10-devmap: turn the per-verb record into the attribution table and the roofline.

  device_i = synced_i - enqueue_i - lambda * calls_i

`synced` times each verb with a `synchronize_device` inside the timed region, `enqueue` times
the same verbs with nothing after them, and `lambda` is a sync on an already-idle device. The
subtraction is what leaves the card's own seconds; the enqueue column IS the ttnn dispatch
loop, so the same run answers leg 1 and leg 3.

The round applies the per-block numbers the way the round runs them: 2 taped forwards and 1
backward per round, 48 Evoformer blocks and 4 extra-MSA blocks.
"""
import argparse
import collections
import json
import pathlib

FWD_PER_ROUND = 2
BWD_PER_ROUND = 1
BLOCKS = {"evo": 48, "extra": 4}

#: `bcx-p10-resident`'s arm B, the same public configuration on qb2 at AICLK 1350.
ANCHOR = {"evo": 11.577, "extra": 1.054, "total": 12.796}


def _split(tag):
    stack, block, direction, family = tag.split("|")
    return stack, int(block), direction, family


def per_block(blob, drop_first_block=True):
    """{(stack, dir, family): {synced, enqueue, calls, read, written}} per ONE block."""
    lam = blob["sync_floor_s"]["median"]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in blob["records"]:
        if rec["K"] != max(blob["ks"]):
            continue
        k = rec["K"]
        per = collections.defaultdict(lambda: [0.0, 0, 0.0, 0.0])
        for tag, w in rec["wall"].items():
            stack, block, direction, family = _split(tag)
            if stack == "-":                      # the tape engine's own verbs, not a block
                key = (rec["stack"], direction, "tape_engine")
                scale = 1.0 / k
            else:
                if drop_first_block and block == 0:
                    continue
                key = (stack, direction, family)
                scale = 1.0 / (k - 1 if drop_first_block else k)
            e = per[key]
            e[0] += w * scale
            e[1] += rec["calls"][tag] * scale
            e[2] += rec["read"][tag] * scale
            e[3] += rec["written"][tag] * scale
        for key, (w, c, r, wr) in per.items():
            acc[key][rec["mode"]].append((w, c, r, wr))
    out = {}
    for key, modes in acc.items():
        if "sync" not in modes:
            continue
        sy = sorted(w for w, _, _, _ in modes["sync"])
        fr = sorted(w for w, _, _, _ in modes.get("free", [(0, 0, 0, 0)]))
        calls = sum(c for _, c, _, _ in modes["sync"]) / len(modes["sync"])
        read = sum(r for _, _, r, _ in modes["sync"]) / len(modes["sync"])
        wrote = sum(w for _, _, _, w in modes["sync"]) / len(modes["sync"])
        synced = sy[len(sy) // 2]
        enqueue = fr[len(fr) // 2]
        out[key] = {"synced": synced, "enqueue": enqueue, "calls": calls,
                    "device": max(synced - enqueue - lam * calls, 0.0),
                    "read": read, "written": wrote}
    return out


def to_round(pb):
    """Per-block per-direction seconds -> seconds per ROUND, over all 52 blocks."""
    rows = []
    for (stack, direction, family), v in pb.items():
        mult = BLOCKS[stack] * (FWD_PER_ROUND if direction == "fwd" else BWD_PER_ROUND)
        rows.append({"stack": stack, "dir": direction, "family": family,
                     "blocks": BLOCKS[stack], "mult": mult,
                     "device_s": v["device"] * mult,
                     "dispatch_s": v["enqueue"] * mult,
                     "synced_s": v["synced"] * mult,
                     "calls": v["calls"] * mult,
                     "read_GB": v["read"] * mult / 1e9,
                     "written_GB": v["written"] * mult / 1e9,
                     "per_block_device_ms": v["device"] * 1e3})
    rows.sort(key=lambda r: -r["device_s"])
    return rows


def roofline(rows, blob, args):
    """Per family: bytes, analytic FLOPs, intensity, achieved against both roofs."""
    flops = blob["flops_fwd_analytic_padded"]
    out = []
    for r in rows:
        f = flops.get(r["family"])
        if f is None:
            fl = None
        else:
            # a matmul's backward is two matmuls of the same size
            fl = f * r["mult"] * (1.0 if r["dir"] == "fwd" else 2.0)
        byts = (r["read_GB"] + r["written_GB"]) * 1e9
        rec = dict(r)
        rec["flop"] = fl
        rec["intensity"] = (fl / byts) if fl and byts else None
        rec["GBs"] = byts / r["device_s"] / 1e9 if r["device_s"] > 0 else None
        rec["TFLOPs"] = fl / r["device_s"] / 1e12 if fl and r["device_s"] > 0 else None
        rec["pct_dram_roof"] = 100 * rec["GBs"] / args.dram if rec["GBs"] else None
        rec["pct_compute_roof"] = (100 * rec["TFLOPs"] / args.tflops) if rec["TFLOPs"] else None
        rec["binds"] = None
        if rec["intensity"] is not None:
            rec["binds"] = "dram" if rec["intensity"] < args.ridge else "compute"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--dram", type=float, default=435.2, help="achievable DRAM GB/s")
    ap.add_argument("--tflops", type=float, default=100.55, help="HiFi4 bf16 TFLOP/s")
    ap.add_argument("--ridge", type=float, default=231.0, help="FLOP/byte machine balance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    blob = json.load(open(args.blob))
    args.ridge = args.tflops * 1e12 / (args.dram * 1e9)
    pb = per_block(blob)
    rows = roofline(to_round(pb), blob, args)

    tot = collections.Counter()
    for r in rows:
        tot[(r["stack"], "device")] += r["device_s"]
        tot[(r["stack"], "dispatch")] += r["dispatch_s"]
    print("roofs: DRAM %.1f GB/s  compute %.2f TFLOP/s  ridge %.1f FLOP/byte"
          % (args.dram, args.tflops, args.ridge))
    print("%-6s %-4s %-16s %9s %9s %8s %9s %8s %7s %7s %s"
          % ("stack", "dir", "family", "dev_s", "disp_s", "read_GB", "GB/s", "TFLOP/s",
             "FLOP/B", "%roof", "binds"))
    for r in rows:
        print("%-6s %-4s %-16s %9.3f %9.3f %8.2f %9.1f %8.2f %7s %7s %s"
              % (r["stack"], r["dir"], r["family"], r["device_s"], r["dispatch_s"],
                 r["read_GB"], r["GBs"] or 0, r["TFLOPs"] or 0,
                 "%.1f" % r["intensity"] if r["intensity"] else "-",
                 "%.1f" % (r["pct_dram_roof"] or 0), r["binds"] or "-"))
    for stack in ("evo", "extra"):
        d, p = tot[(stack, "device")], tot[(stack, "dispatch")]
        a = ANCHOR[stack]
        print("%-6s device %7.3f s  dispatch %7.3f s   anchor %6.3f s   ratio %.3f"
              % (stack, d, p, a, d / a))
    d = sum(tot[(s, "device")] for s in ("evo", "extra"))
    p = sum(tot[(s, "dispatch")] for s in ("evo", "extra"))
    print("TOTAL  device %7.3f s  dispatch %7.3f s   anchor %6.3f s   ratio %.3f"
          % (d, p, ANCHOR["total"], d / ANCHOR["total"]))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"roofs": {"dram_GBs": args.dram, "tflops": args.tflops, "ridge": args.ridge},
             "anchor": ANCHOR, "rows": rows,
             "totals": {"device_s": d, "dispatch_s": p}}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
