#!/usr/bin/env python3
"""Render the Wormhole ladder out of the rung files, so the table in the state doc is generated.

    table.py ~/b2p-wh/out/timed1 [more dirs...]

Reads every `*/rung.json` under each directory and prints one markdown row per rung, keyed on the
axis the Evoformer seam actually ran. Nothing here is typed by hand, which is the point: `bgx`
mislabelled a whole ladder by a bucket because the table was keyed on arithmetic.

Seconds per round DROP round 1 -- it compiles -- and every reading carries the AICLK sampled at
1 Hz inside that round's own window. A rung run with `--footprint` reports no timing, because
`get_memory_view` drains the pipeline; its columns are the resident peak instead.
"""
import argparse
import json
import pathlib
import sys


def rows(dirs):
    out = []
    for d in dirs:
        for path in sorted(pathlib.Path(d).expanduser().glob("*/rung.json")):
            try:
                out.append((path, json.loads(path.read_text())))
            except ValueError as exc:
                print(f"# {path}: {exc}", file=sys.stderr)
    return out


def per_round(rung):
    """Seconds per round with round 1 dropped, and the clock over the rounds that remain."""
    rs = [r for r in rung.get("per_round", []) if r["round"] > 1]
    if not rs:
        return None, None, None, None
    secs = sorted(r["seconds"] for r in rs)
    med = secs[len(secs) // 2] if len(secs) % 2 else (secs[len(secs) // 2 - 1]
                                                      + secs[len(secs) // 2]) / 2
    clocks = [r["aiclk_med"] for r in rs if r.get("aiclk_med")]
    lo = min((r["aiclk_min"] for r in rs if r.get("aiclk_min")), default=None)
    return med, (secs[0], secs[-1]), (min(clocks, default=None), max(clocks, default=None)), lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", dest="out", default=None)
    args = ap.parse_args()

    got = rows(args.dirs)
    print("| axis | target | ch | binder | mode | s/round (median) | spread | AICLK in-round | "
          "load1 | resident peak GB | free at peak GB | largest free block | auto pre/open | "
          "fused fwd served/declined | fused bw served/declined | rounds | outcome |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    table = []
    for path, r in sorted(got, key=lambda kv: (kv[1].get("evoformer_axis") or 0,
                                               kv[1].get("footprint_probe"))):
        med, spread, clk, clk_lo = per_round(r)
        stats = r.get("lever_stats") or {}
        fwd = stats.get("triatt_fused_hifi") if isinstance(stats, dict) else None
        bw = stats.get("triatt_bw") if isinstance(stats, dict) else None
        err = r.get("error") or ""
        outcome = ("completes" if "StopAfterRounds" in str(err) or not err
                   else "REFUSES/dies: " + str(err)[:60])
        mode = "footprint" if r.get("footprint_probe") else "timed"
        row = {"axis": r.get("evoformer_axis"), "target": r.get("target"),
               "chains": r.get("target_chains"), "binder": r.get("binder"), "mode": mode,
               "s_round_median": med, "s_round_spread": spread,
               "aiclk_med_range": clk, "aiclk_min": clk_lo,
               "load1": (r.get("aiclk_run") or {}).get("load1"),
               "resident_peak_gb": r.get("resident_peak_gb"),
               "free_at_peak_gb": r.get("free_at_peak_gb"),
               "largest_free_block_at_peak_mb": r.get("largest_free_block_at_peak_mb"),
               "device_free_min_gb": (round(r["device_free_min"] / 1e9, 3)
                                      if r.get("device_free_min") else None),
               "dram_total_gb": (round(r["dram_total_bytes"] / 1e9, 3)
                                 if r.get("dram_total_bytes") else None),
               "auto_before_open": (r.get("auto_before_open") or {}).get("count"),
               "auto_card_open": (r.get("auto_card_open") or {}).get("count"),
               "auto_card_open_why": (r.get("auto_card_open") or {}).get("why"),
               "auto_priced_axis_short_by": r.get("auto_priced_axis_short_by"),
               "fused_fwd": fwd, "fused_bw": bw, "rounds_done": r.get("rounds_done"),
               "complex_residues": r.get("complex_residues"), "error": err,
               "outcome": outcome, "dir": str(path.parent)}
        table.append(row)
        fmt = lambda v, p="{}": "-" if v is None else p.format(v)   # noqa: E731
        print("| {axis} | {target} | {ch} | {binder} | {mode} | {med} | {spread} | {clk} | "
              "{load} | {peak} | {free} | {blk} | {ab}/{ao} | {fwd} | {bw} | {n} | {out} |".format(
                  axis=fmt(row["axis"]), target=row["target"], ch=fmt(row["chains"]),
                  binder=row["binder"], mode=mode, med=fmt(med, "{:.2f}"),
                  spread="-" if not spread else f"{spread[0]:.2f}-{spread[1]:.2f}",
                  clk="-" if not clk or clk[0] is None else
                      (f"{clk[0]}" if clk[0] == clk[1] else f"{clk[0]}-{clk[1]}"),
                  load=fmt(row["load1"]), peak=fmt(row["resident_peak_gb"]),
                  free=fmt(row["free_at_peak_gb"]),
                  blk=fmt(row["largest_free_block_at_peak_mb"], "{} MB"),
                  ab=fmt(row["auto_before_open"]), ao=fmt(row["auto_card_open"]),
                  fwd="-" if not fwd else f"{fwd.get('served')}/{fwd.get('declined')}",
                  bw="-" if not bw else f"{bw.get('served')}/{bw.get('declined')}",
                  n=fmt(row["rounds_done"]), out=outcome))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
