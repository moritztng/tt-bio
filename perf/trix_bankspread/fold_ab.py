#!/usr/bin/env python3
"""Fold A/B of the channel move's bank spread, both arms in one process.

The op-level ladder says four streams is 1.1079x on `reblock_permute` at 512 aa. This campaign's
record is four op-level levers reaching the fold at 25x-to-infinite error with two flipping sign,
so only this number counts.

ONE PROCESS FOR BOTH ARMS, unlike `perf/c14_bfp8/fold_ab.py`. `CT_STREAM` is read in
`_ct_stream()` on every `_prepare`, and it is part of the descriptor cache key through the kernel
compile-time args, so flipping the module global between folds rebuilds the descriptor and cannot
mix arms. The model is 64 pairformer blocks and loads once instead of six times, which buys the
extra fold reps that a 0.6 %-sized prediction needs.

PROTOCOL
  * One warmup fold, discarded.
  * Arms strictly alternated base, on, base, on... An all-base-then-all-on order reads a drift as
    a lever (`op-ab-must-interleave-arms-compile-warmup-bias`).
  * The A/A floor is the spread of the base reps against each other, read exactly the way base is
    read against on.
  * Both arms PIN the flag. Neither relies on the shipped default, so the arms cannot silently
    become the same arm.
  * `reblock_permute.STATS` per fold, so a null cannot be a lever that never fired.
  * AICLK sampled from sysfs at 5 Hz DURING every fold, in a thread that opens no device. The fold
    releases the GIL in ttnn, so the sampler keeps up; the sample count per fold is reported and a
    fold with too few samples is visible rather than silently averaged.
  * The CIF digest and plDDT per fold, so the parity claim is not only an op-level `torch.equal`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"


def _helpers():
    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ClockSampler(threading.Thread):
    ROOT = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.aiclk: list[int] = []
        self.power: list[int] = []
        self._clk = self.ROOT / f"tenstorrent!{card}" / "tt_aiclk"
        if not self._clk.exists():
            cand = next(iter(self.ROOT.glob(f"*{card}/tt_aiclk")), None)
            self._clk = cand if cand else self._clk
        self._pw = next(iter(self._clk.parent.glob("device/hwmon/hwmon*/power1_input")), None)

    def run(self):
        while not self.stop.wait(0.2):
            try:
                v = int(self._clk.read_text().strip())
                if v < 3000:
                    self.aiclk.append(v)
            except (OSError, ValueError):
                pass
            if self._pw is not None:
                try:
                    self.power.append(int(self._pw.read_text().strip()))
                except (OSError, ValueError):
                    pass

    def take(self):
        a, w = self.aiclk[:], self.power[:]
        if not a:
            return {"aiclk_n": 0}
        return {"aiclk_min": min(a), "aiclk_max": max(a),
                "aiclk_median": int(st.median(a)), "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="512")
    ap.add_argument("--reps", type=int, default=4, help="folds per arm")
    ap.add_argument("--card", default="2")
    ap.add_argument("--base-stream", default="1", help="CT_STREAM for the control (shipped walk)")
    ap.add_argument("--on-stream", default="4", help="CT_STREAM for the lever")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import reblock_permute as rbp
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree "
        "(memory parity-gate-scores-installed-package-not-checkout)")

    H = _helpers()
    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="trix-bankspread-", dir=str(REPO / "perf" / "trix_bankspread")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{a.size}.yaml",
                (FIX / f"cdk2x2_{a.size}.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("trix-bankspread-ab", cfg)
    model_load_s = round(time.perf_counter() - t0, 3)

    g = dev.compute_with_storage_grid_size()
    target = FIX / f"cdk2x2_{a.size}.yaml"
    out: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "card": a.card, "size": a.size,
        "model_load_s": model_load_s, "grid": [g.x, g.y],
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "arms": {"base": a.base_stream, "on": a.on_stream},
        "protocol": {"recycling_steps": cfg["recycling_steps"],
                     "sampling_steps": cfg["sampling_steps"],
                     "diffusion_samples": cfg["diffusion_samples"], "seed": cfg["seed"]},
        "folds": [],
    }
    outp = Path(a.out)

    def one(arm: str, stream: str, tag: str) -> dict:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        rbp._CT_STREAM_PIN, rbp._CT_BUFS_PIN = stream, "2"
        rbp._CACHE.clear()
        rbp.STATS[0] = rbp.STATS[1] = 0
        state.pfn = None
        cs = ClockSampler(a.card)
        cs.start()
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cs.stop.set()
        clk = cs.take()
        cifs = sorted(struct_dir.glob("*.cif"))
        Ct = 8  # C = 256 at 512 aa; reported from STATS as a cross-check, not assumed
        row = {
            "arm": arm, "tag": tag, "ct_stream": stream, "fold_s": round(wall, 3),
            "clock": clk, "reblock_served_declined": list(rbp.STATS),
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest() if cifs else None,
            "loadavg1": round(os.getloadavg()[0], 2),
        }
        del Ct
        print(f"  {arm:4s} S={stream} {tag:7s} {wall:7.3f}s  rbp={rbp.STATS[0]}  "
              f"clk {clk.get('aiclk_min')}-{clk.get('aiclk_max')} (n={clk.get('aiclk_n')})  "
              f"load {row['loadavg1']}", flush=True)
        return row

    w = one("base", a.base_stream, "warmup")
    w["discarded"] = True
    out["warmup"] = w
    outp.write_text(json.dumps(out, indent=1))

    for i in range(a.reps):
        for arm, stream in (("base", a.base_stream), ("on", a.on_stream)):
            out["folds"].append(one(arm, stream, str(i)))
            outp.write_text(json.dumps(out, indent=1))

    def secs(arm):
        return [f["fold_s"] for f in out["folds"] if f["arm"] == arm]

    b, o = secs("base"), secs("on")
    # A/A floor read the same way the lever is: every base rep against every other base rep.
    aa = [max(x, y) / min(x, y) for i, x in enumerate(b) for y in b[i + 1:]] or [None]
    clocks = [f["clock"] for f in out["folds"] if f.get("clock", {}).get("aiclk_n")]
    digests = {arm: sorted({f["cif_sha256"] for f in out["folds"] if f["arm"] == arm})
               for arm in ("base", "on")}
    out["summary"] = {
        "base_median_s": round(st.median(b), 3), "on_median_s": round(st.median(o), 3),
        "base_all_s": b, "on_all_s": o,
        "delta_s": round(st.median(b) - st.median(o), 4),
        "ratio_base_over_on": round(st.median(b) / st.median(o), 5),
        "aa_floor_ratio_max": round(max(x for x in aa if x), 5) if aa[0] else None,
        "aa_floor_pct": round((max(x for x in aa if x) - 1) * 100, 4) if aa[0] else None,
        "reblock_calls_per_fold": {arm: sorted({f["reblock_served_declined"][0]
                                                for f in out["folds"] if f["arm"] == arm})
                                   for arm in ("base", "on")},
        "plddt": {arm: sorted({f["plddt"] for f in out["folds"] if f["arm"] == arm})
                  for arm in ("base", "on")},
        "cif_digests": digests,
        "cif_digest_identical_across_arms": digests["base"] == digests["on"],
        "clock_min": min((c["aiclk_min"] for c in clocks), default=None),
        "clock_max": max((c["aiclk_max"] for c in clocks), default=None),
        "clock_median": int(st.median([c["aiclk_median"] for c in clocks])) if clocks else None,
        "clock_samples": sum(c["aiclk_n"] for c in clocks),
        "loadavg1_range": [min(f["loadavg1"] for f in out["folds"]),
                           max(f["loadavg1"] for f in out["folds"])],
        "verdict_rule": "a ratio inside aa_floor_ratio_max is a NULL, not a win",
    }
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outp.write_text(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    s = out["summary"]
    print(f"\n{a.size} aa  base {s['base_median_s']:.3f}s  on {s['on_median_s']:.3f}s  "
          f"delta {s['delta_s']:+.4f}s  ratio {s['ratio_base_over_on']:.5f}  "
          f"A/A {s['aa_floor_pct']:.3f}%  clk {s['clock_min']}-{s['clock_max']} "
          f"(median {s['clock_median']}, {s['clock_samples']} samples)", flush=True)
    print(f"digests identical across arms: {s['cif_digest_identical_across_arms']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
