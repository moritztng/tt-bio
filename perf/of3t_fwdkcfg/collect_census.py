#!/usr/bin/env python3
"""Fold the per-process probe dumps into one per-site census.

One dump per process, and the fold runs in a child, so the parent's file is a genuine
zero-call dump of a process that did no model work. Merging on call count rather than taking
the last file is what keeps that from reading as "this site is never reached".
"""
import json
import pathlib
import socket
import subprocess
import sys


def clk():
    try:
        out = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                             text=True, timeout=60).stdout
        d = json.loads(out)
        return [b.get("telemetry", {}).get("aiclk") for b in d.get("device_info", [])]
    except Exception:
        return None


def main():
    work, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    models = {}
    for mdir in sorted(p for p in work.iterdir() if p.is_dir()):
        per = {"site_softmax": {}, "other_softmax_ops": {}, "processes": 0}
        for f in sorted(mdir.glob("probe_*.json")):
            d = json.loads(f.read_text())
            per["processes"] += 1
            if not d["site_softmax"] and not d["ttnn_softmax"]:
                continue        # a process that ran no model work; the parent is always one
            per["site_softmax"].update(d["site_softmax"])
            for op, sites in d["ttnn_softmax"].items():
                per["other_softmax_ops"].setdefault(op, {}).update(sites)
            per["owner_modules"] = d["owner_modules"]
            per["fold_pid"] = d["pid"]
            per["host_f64_stats"] = d["stats"]
        models[mdir.name] = per
    rep = {"what": "softmax construction sites a shipped inference fold reaches, by dtype",
           "host": socket.gethostname(), "card": 0, "board": "p300c",
           "fixture": "perf/size512/fixtures/cdk2x2_128.yaml (298 aa, 128-token crop)",
           "fold": "predict --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0",
           "aiclk_after": clk(), "models": models}
    out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print("wrote", out)
    for m, v in models.items():
        for k, s in sorted(v["site_softmax"].items()):
            print("  %-10s %-50s %4d calls  %-9s %s  ckc=%s"
                  % (m, k, s["calls"], s["dtype"].split(".")[-1], s["shape"], s["ckc"]))


if __name__ == "__main__":
    main()
