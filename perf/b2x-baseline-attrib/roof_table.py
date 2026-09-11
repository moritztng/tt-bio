#!/usr/bin/env python3
"""Every named phase of the 512 aa fold against BOTH measured roofs, ranked by roof deficit.

Bytes are recounted from this task's own 26 ttnn.graph captures with the corrected rule from
``perf/b2x_difflayer/real_traffic.py`` (wk/b2x-diffusion-layer-bytes): dedupe on BUFFER ADDRESS,
charge views nothing, charge an in-place op a read and a write. The published counter deduped on
tensor id and charged a full DRAM read for every metadata view, so every per-call byte figure in
the campaign's budget -- including the ones this task published earlier today -- is 1.1-1.8x high.

The counter is not vendored. It is read out of the sibling branch, so there is exactly one copy of
the rule in the repo and this table cannot drift from it.

FLOP/call comes from ``perf/bioir_roofline/flops_bytes_512.json``: each phase run once on CPU under
torch's FlopCounterMode, not a hand formula. Time/call is this task's own bracketed fold, median
over that unit's calls, with a device sync on both sides of every bracket -- so each ms/call is an
upper bound and every achieved rate below is a floor.

The diagnostic is the PAIR of roof fractions, not either one:

    bandwidth high, compute low -> bandwidth-bound. Delete bytes.
    bandwidth low,  compute high -> compute-bound. Fidelity and tile shape, not bytes.
    both low                    -> latency or dispatch bound. Neither bytes nor FLOP help.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

STREAM_ROOF = 429.9e9          # GB/s, 8192^2 read+write, roofs_p300c_qb2_card2.json
COMPUTE_ROOF = 85.96e12        # dense bf16 HiFi4 at N=8192, same file
TARGET_FRACTION = 0.80         # "what the phase gives back at 80 % of roof"

COUNTER_BRANCH = "origin/wk/b2x-diffusion-layer-bytes"
COUNTER_FILES = ("perf/b2x_difflayer/itemize.py", "perf/b2x_difflayer/real_traffic.py")

# capture signature -> (phase label, FLOP/call from flops_bytes_512.json)
FLOPS = {
    "PairformerLayer|1x512x384,1x512x512x128": ("pairformer block", 502796386304),
    "MSALayer|1x512x512x128,1x1024x512x64": ("MSA block", 595167543296),
    "DiffusionTransformerLayer|1x512x768,1x512x768": ("token DiT layer", 12280922112),
    "DiffusionTransformerLayer|1x224x32x128,1x224x32x128": ("atom transformer layer", 4580179968),
    # one denoiser step = 24 token layers + 6 atom transformer layers
    "DiffusionModule|": ("diffusion step (whole denoiser)", 24 * 12280922112 + 6 * 4580179968),
}


def load_counter():
    """Import the sibling's counter without copying it into this branch."""
    d = ROOT / "perf" / "b2x_difflayer"
    if not (d / "real_traffic.py").is_file():
        d = Path(tempfile.mkdtemp(prefix="b2xcounter-"))
        for f in COUNTER_FILES:
            (d / Path(f).name).write_bytes(subprocess.run(
                ["git", "show", f"{COUNTER_BRANCH}:{f}"], cwd=ROOT,
                capture_output=True, check=True).stdout)
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("real_traffic", d / "real_traffic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "attrib_512_qb2c1.json")
    ap.add_argument("--captures", type=Path, default=HERE / "captures")
    ap.add_argument("--out-json", type=Path, default=HERE / "roof_deficit_512_qb2c1.json")
    ap.add_argument("--out-md", type=Path, default=HERE / "ROOF_DEFICIT.md")
    a = ap.parse_args()

    rt = load_counter()
    run = json.loads(a.run.read_text())
    # Op counts come from this task's own walker, not the counter's. ttnn.graph under fast
    # runtime mode drops `function_end` for some device operations, so a big capture is
    # unbalanced (the pairformer block: 1413 starts, 1331 ends). The counter's stack-based
    # span walker then never returns to depth zero and reports 4 top-level ops where there
    # are 428. Its BYTE total is unaffected -- it agrees with the sibling's independent
    # card-2 capture to 0.005 % on all three shared phases -- but its op count is not usable
    # on these captures.
    attribution = json.loads((HERE / "attribution_512_qb2c1.json").read_text())
    ops_by_sig = {k: v["ttnn_ops"] for k, v in attribution["per_capture"].items()}
    fold_s = run["baseline_summary"]["plain_median_s"]
    device_s = run["baseline_summary"]["device_s"]

    # calls/fold and median ms/call, per capture signature, out of the bracketed fold
    attrib = run["attrib"]
    per_sig = {s: {"calls": v["calls"], "ms": v["median_ms"]}
               for s, v in attrib["sigs"].items()}

    rows = []
    for cap in sorted(a.captures.glob("cap_*.json.gz")):
        sig = cap.name[len("cap_"):-len(".json.gz")].replace("__", "|")
        nodes = json.load(gzip.open(cap, "rt"))
        c = rt.counts({"nodes": nodes})
        m = per_sig.get(sig)
        if m is None:
            continue
        real_B = c["real_MB"] * 1e6
        n_ops = ops_by_sig.get(sig, c["n_ops"])
        ms = m["ms"]
        calls = m["calls"]
        s_fold = calls * ms / 1e3
        gbps = real_B / (ms / 1e3)
        label, flops = FLOPS.get(sig, (sig, None))
        row = {
            "sig": sig, "phase": label, "calls": calls, "ms_per_call": round(ms, 4),
            "s_per_fold": round(s_fold, 3),
            "real_MB_per_call": round(c["real_MB"], 3),
            "published_MB_per_call": None,
            "floor_MB_per_call": round(c["floor_MB"], 3),
            "once_MB_per_call": round(c["once_MB"], 3),
            "TB_per_fold": round(calls * real_B / 1e12, 4),
            "ops_per_call": n_ops,
            "kB_per_op": round(real_B / n_ops / 1e3, 1),
            "ops_per_fold": calls * n_ops,
            "us_per_op": round(1e3 * ms / n_ops, 1),
            "achieved_GBps": round(gbps / 1e9, 1),
            "pct_stream_roof": round(100 * gbps / STREAM_ROOF, 1),
            "GFLOP_per_call": round(flops / 1e9, 2) if flops else None,
            "achieved_TFLOPs": round(flops / (ms / 1e3) / 1e12, 2) if flops else None,
            "pct_compute_roof": round(100 * flops / (ms / 1e3) / COMPUTE_ROOF, 1) if flops else None,
        }
        # what the phase gives back if it ran at TARGET_FRACTION of the streaming roof
        at_target = calls * real_B / (TARGET_FRACTION * STREAM_ROOF)
        row["s_at_80pct_roof"] = round(at_target, 3)
        row["roof_deficit_s"] = round(s_fold - at_target, 3)
        rows.append(row)

    known = [r for r in rows if r["sig"] in FLOPS]
    known.sort(key=lambda r: -r["roof_deficit_s"])
    rows.sort(key=lambda r: -r["roof_deficit_s"])

    # fold closure on the four top-level phases (no double counting: these four are disjoint)
    TOP = ["PairformerLayer|1x512x384,1x512x512x128", "MSALayer|1x512x512x128,1x1024x512x64",
           "DiffusionModule|"]
    by_sig = {r["sig"]: r for r in rows}
    top_TB = sum(by_sig[s]["TB_per_fold"] for s in TOP if s in by_sig)
    top_s = sum(by_sig[s]["s_per_fold"] for s in TOP if s in by_sig)

    bytes_explained_s = top_TB * 1e12 / STREAM_ROOF
    flops_explained_s = 206.705568776192e12 / COMPUTE_ROOF
    probe_path = HERE / "host_dispatch_512_qb2c1.json"
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None

    summary = {
        "fold_s": fold_s, "device_s": device_s,
        "stream_roof_GBps": STREAM_ROOF / 1e9, "compute_roof_TFLOPs": COMPUTE_ROOF / 1e12,
        "top_level_TB_per_fold": round(top_TB, 4),
        "top_level_s_per_fold": round(top_s, 3),
        "top_level_share_of_fold_pct": round(100 * top_s / fold_s, 1),
        "fold_achieved_GBps_on_device_s": round(top_TB * 1e12 / device_s / 1e9, 1),
        "fold_pct_stream_roof": round(100 * top_TB * 1e12 / device_s / STREAM_ROOF, 1),
        "fold_TFLOP": 206.705568776192,
        "fold_achieved_TFLOPs": round(206.705568776192 / device_s, 2),
        "fold_pct_compute_roof": round(100 * 206.705568776192e12 / device_s / COMPUTE_ROOF, 1),
        "s_explained_by_bytes_at_roof": round(bytes_explained_s, 3),
        "s_of_phase_time_not_explained_by_bytes": round(top_s - bytes_explained_s, 3),
        "s_explained_by_flops_at_roof": round(flops_explained_s, 3),
        "fold_s_if_every_phase_byte_deleted": round(fold_s - bytes_explained_s, 3),
        "ceiling_x_if_every_phase_byte_deleted": round(fold_s / (fold_s - bytes_explained_s), 3),
        "counter": f"real_traffic.py from {COUNTER_BRANCH}",
    }
    if probe:
        warm = [f for f in probe["folds"] if f["arm"].startswith("warm")]
        oc = probe["op_census"]
        summary["dispatch"] = {
            "probe_fold_s": round(sum(f["wall_s"] for f in warm) / len(warm), 3),
            "main_thread_cpu_s": round(sum(f["main_thread_cpu_s"] for f in warm) / len(warm), 3),
            "main_thread_cpu_pct_of_wall": round(
                sum(f["main_thread_cpu_pct"] for f in warm) / len(warm), 1),
            "n_ttnn_calls_per_fold": oc["n_ttnn_calls"],
            "s_inside_ttnn_calls": oc["s_inside_ttnn_calls"],
            "mean_us_per_ttnn_call": oc["us_per_call_wall"],
            "loadavg_during_probe": warm[0]["loadavg"],
        }
    out = {"summary": summary, "rows": rows}
    a.out_json.write_text(json.dumps(out, indent=1))

    L = ["# Boltz-2 512 aa: every named phase against both measured roofs",
         "",
         f"Card 1 of qb2 (p300c), ttnn 0.68.0, fold {fold_s:.3f} s (device {device_s:.3f} s). Bytes "
         f"recounted from this task's own captures with the corrected buffer-address rule "
         f"(`real_traffic.py`, {COUNTER_BRANCH}); FLOP/call from `flops_bytes_512.json`; ms/call is "
         "the median over that unit's calls in the bracketed fold, which syncs on both sides of "
         "every bracket, so every rate here is a floor.",
         "",
         f"Roofs, measured on this part: streaming **{STREAM_ROOF/1e9:.1f} GB/s**, dense bf16 HiFi4 "
         f"**{COMPUTE_ROOF/1e12:.2f} TFLOP/s**.",
         "",
         "| phase | calls | ms/call | MB/call | GB/s | % stream roof | GFLOP/call | % compute roof "
         "| ops/call | ops/fold | kB/op | us/op | s/fold | s at 80 % roof | **deficit s** |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in known:
        L.append("| {phase} | {calls} | {ms_per_call:.3f} | {real_MB_per_call:.1f} | "
                 "{achieved_GBps} | {pct_stream_roof} % | {GFLOP_per_call} | {pct_compute_roof} % | "
                 "{ops_per_call} | {ops_per_fold} | {kB_per_op} | {us_per_op} | {s_per_fold} | "
                 "{s_at_80pct_roof} | **{roof_deficit_s}** |".format(**r))
    L += ["", "## Every captured unit, same counter, ranked the same way", "",
          "| unit | calls | ms/call | MB/call | published MB/call | GB/s | % roof | ops/call | "
          "kB/op | s/fold | deficit s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['sig']}` | {r['calls']} | {r['ms_per_call']:.3f} | "
                 f"{r['real_MB_per_call']:.1f} | | {r['achieved_GBps']} | {r['pct_stream_roof']} % | "
                 f"{r['ops_per_call']} | {r['kB_per_op']} | {r['s_per_fold']} | "
                 f"{r['roof_deficit_s']} |")
    L += ["", "## What the pair of roofs says", "",
          f"Every phase sits at 36-43 % of the streaming roof and 3-14 % of the compute roof. "
          f"Nothing is bandwidth-bound and nothing is compute-bound, so the diagnostic's third "
          f"row is the one that fires: latency or dispatch. Of {top_s:.3f} s of phase time, "
          f"{bytes_explained_s:.3f} s is explained by moving those bytes at the streaming roof and "
          f"{top_s - bytes_explained_s:.3f} s is not. The whole fold's 206.71 TFLOP is "
          f"{flops_explained_s:.2f} s at the compute roof.",
          "",
          f"**Delete every byte of every named phase and the fold is still "
          f"{fold_s - bytes_explained_s:.3f} s = "
          f"{fold_s / (fold_s - bytes_explained_s):.3f}x.**", ""]
    if probe:
        d = summary["dispatch"]
        L += ["## Where the rest goes: the host", "",
              f"One fold issues **{d['n_ttnn_calls_per_fold']:,} ttnn calls**. Measured on the "
              f"calling thread with `time.thread_time()`, which counts only the CPU that thread "
              f"burns: **{d['main_thread_cpu_s']:.3f} s of main-thread CPU in a "
              f"{d['probe_fold_s']:.3f} s fold, {d['main_thread_cpu_pct_of_wall']:.1f} %**. "
              f"{d['s_inside_ttnn_calls']:.3f} s of the wall is spent inside ttnn entry points, "
              f"{d['mean_us_per_ttnn_call']:.1f} us per call on average.",
              "",
              "The host is issuing ops for essentially the whole fold while the device runs at "
              "37 % of one roof and 10 % of the other. Op count, not bytes, is what that buys "
              "back. (Probe ran co-tenanted at loadavg "
              f"{', '.join(d['loadavg_during_probe'])}, so its wall is ~9 % above the benchlocked "
              "23.841 s; the ratio is the result, not the wall.)", ""]
    L += ["", "## Summary", "", "```", json.dumps(summary, indent=1), "```"]
    a.out_md.write_text("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    for r in known:
        print("%-32s %5d x %8.3f ms  %8.1f MB  %5.1f %% stream  %5.1f %% compute  deficit %6.3f s"
              % (r["phase"], r["calls"], r["ms_per_call"], r["real_MB_per_call"],
                 r["pct_stream_roof"], r["pct_compute_roof"] or -1, r["roof_deficit_s"]))
    print("WROTE", a.out_json, a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
