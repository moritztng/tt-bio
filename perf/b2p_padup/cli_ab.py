#!/usr/bin/env python3
"""Pad-up A/B through a model's own CLI: one `tt_bio.main predict` process per leg, arms alternated.

`TT_BIO_TRIATT_HIFI_PAD_UP` is read at import, so each arm is its own process; `site/` is put on
PYTHONPATH so every leg dumps `TRIATT_FUSED_HIFI_STATS` at exit. AICLK is sampled during each leg.
Wall time includes model load, so it is reported beside the CLI's own fold timing when present.

  cli_ab.py --yaml examples/hsa_no_msa.yaml --model boltz2 --arms off,on,off,on --out f.json \
      -- --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --single_sequence
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="off,on,off,on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    import clocksample
    rest = [x for x in a.rest if x != "--"]
    work = a.out.with_suffix("")
    res = {"yaml": a.yaml, "model": a.model, "args": rest, "seed": a.seed,
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "legs": []}
    for i, arm in enumerate(a.arms.split(",")):
        leg_dir = work / f"{arm}_{i}"
        leg_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TT_BIO_TRIATT_HIFI_PAD_UP={"off": "0", "on": "2"}[arm],
                   B2P_PADUP_DUMP=str(leg_dir / "stats.json"),
                   PYTHONPATH=f"{Path(__file__).parent / 'site'}:{ROOT}")
        cmd = [sys.executable, "-m", "tt_bio.main", "predict", a.yaml, "--model", a.model,
               "--out_dir", str(leg_dir), "--seed", str(a.seed), "--override", *rest]
        t0 = time.perf_counter()
        with clocksample.during(period=1.0) as clk, open(leg_dir / "log.txt", "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - t0
        cifs = sorted(leg_dir.glob("**/*.cif"))
        h = hashlib.sha256()
        for f in cifs:
            h.update(f.read_bytes())
        dumps = [json.loads(f.read_text()) for f in sorted(leg_dir.glob("stats.json.*"))]
        ran = [d for d in dumps if d.get("stats") and any(d["stats"].values())]
        stats = {"processes": len(dumps), "tenstorrent_imported": sum(d["tenstorrent_imported"] for d in dumps),
                 "stats": ran[0]["stats"] if len(ran) == 1 else [d["stats"] for d in ran] or
                          [d.get("stats") for d in dumps if d["tenstorrent_imported"]],
                 "padded": [d.get("padded") for d in ran], "pad_up_tiles": sorted({d.get("pad_up_tiles") for d in dumps if d["tenstorrent_imported"]}, key=str)}
        leg = {"i": i, "arm": arm, "rc": rc, "wall_s": round(wall, 3), "aiclk": clk.summary(),
               "clock_line": clk.line(0), "cifs": [str(c.relative_to(work)) for c in cifs],
               "digest": h.hexdigest(), "dump": stats}
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"leg {i} {arm} rc={rc} {wall:.1f}s {leg['digest'][:12]} {leg['clock_line']} "
              f"stats={stats and stats.get('stats')}", flush=True)


if __name__ == "__main__":
    main()
