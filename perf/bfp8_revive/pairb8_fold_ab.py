#!/usr/bin/env python3
"""Price TT_BIO_PAIR_B8 -- full-track bfp8 on the Boltz-2 pair track -- at 298 and 512 aa.

This row revives the lever that `b2x-bfp8-pair-track` recorded on 2026-09-11 as 0.949x and
1.4965 A. The accuracy half of that rejection no longer applies (state/bfp8-accuracy-policy.md,
Moritz 2026-09-18: for bfp8 accuracy is a reported cost, not a gate). The SPEED half was measured
on a 26.078 s fold on a co-tenanted card, with the refusal gates of the time still in place; the
fold is 14.588 s now and the gates have moved, so the number is re-taken rather than inherited.

Protocol and harness taken from perf/c14_bfp8/fold_ab.py; see its docstring, quoted below, for
why each rule is there. Added here: the full lever census is read in-process after every fold, so
which program served and which declined is recorded for the SAME folds that produced the times
instead of costing a separate pair of runs.

Taken from `c14-land-tail`'s `perf/c14_land/apb_fold_ab.py` with this row's flag
added to FLAGS and the scratch directories repointed at this worktree. Its protocol
is the reason it was reused rather than rewritten, and it is quoted unchanged below.
The lever it was written for was TT_BIO_APB_CONCAT_HEADS:

The lever collapses the token head re-assembly from four launches to one by keeping the pad lanes
(`tenstorrent.py:1446`). The firing census (`perf/c14_land/firing.json`) established that it serves
864 of 864 calls per fold at BOTH sizes, so unlike most of the default-off tail it is reachable.
What it has never had is a fold-level price.

WHY THIS NEEDS ONE PROCESS PER ARM, unlike every other flag A/B in this tree. The flag is consumed
in `AttentionPairBias.__init__` (`tenstorrent.py:7753`) and re-lanes proj_g and proj_o to
n_heads*padded_head_dim there (`:7756`). Flipping the module global after the model is built would
run padded activation lanes against unpadded weights: a WRONG answer, not a slow one. So each arm
is its own process with the env var set before import, and the A/A floor comes from independent
base processes rather than from two folds inside one.

PROTOCOL
  * Blocks of `--folds` timed folds, arms interleaved block by block: base, apb, base, apb, ...
    An all-base-then-all-apb order reads a drift as a lever (memory
    `op-ab-must-interleave-arms-compile-warmup-bias`).
  * One warmup fold per process, discarded: program cache and JIT are a ~13x spread cold to warm.
  * The A/A floor is free and comes from the same data: base block 1 vs base block 2 vs base
    block 3, compared exactly the way base is compared to apb. A ratio inside that floor is a null.
  * AICLK pinned by the caller (ARC 0x33, `perf/c10_bare_baseline/force_aiclk.py`) and sampled from
    sysfs at 5 Hz DURING every fold. A fold whose during-sampled clock is not what the caller
    pinned is reported and excluded, not quietly averaged in: on this part the clock SETS the fold
    time.
  * Accuracy comes along for free: every fold's CIF digest and plDDT are recorded, and the CIFs of
    the closing fold per arm are kept so the RMSD arm can be scored without re-folding.

RUN IT UNDER benchlock AND ONLY ON A CLEAN BOARD PAIR (`pair_idle.py --card N`). A p300c's two
chips share a board power budget, which benchlock cannot see.

    ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
      env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      python3 perf/bfp8_revive/pairb8_fold_ab.py --sizes 512 --blocks 3 --folds 2 \
        --card 2 --out perf/bfp8_revive/pairb8_ab_512.json \
        --cifdir perf/bfp8_revive/cifs
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
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"

# flag -> the tenstorrent module global it sets, asserted in the worker so a typo cannot
# silently produce two identical arms. Both are settable in the environment before import.
FLAGS = {
    "TT_BIO_APB_CONCAT_HEADS": "_APB_CONCAT_HEADS",
    "TT_BIO_SDPA_BAND_DIV_K": "_SDPA_BAND_DIV_K",
    "TT_BIO_TRIATT_BIAS_B8": "_TRIATT_BIAS_B8",
    "TT_BIO_TRIATT_B8": "_TRIATT_B8",
    "TT_BIO_PAIR_B8": "_PAIR_B8",
}


def _census():
    """[served, declined] per lever for every module this process imported.

    `scripts/lever_census._snapshot_process` is the same reader the standalone census chain uses,
    so the two are comparable. Counters are cumulative over the process, so the caller diffs.
    """
    import importlib.util
    p = REPO / "scripts" / "lever_census.py"
    spec = importlib.util.spec_from_file_location("_lever_census", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _snap(lc):
    out = {}
    for flag, row in lc._snapshot_process().items():
        s, d = row.get("served"), row.get("declined")
        if s is None and d is None:
            continue
        out[flag] = [s or 0, d or 0]
    return out


def _delta(a, b):
    """Per-lever [served, declined] over one fold, plus the resolved flag values."""
    keys = sorted(set(a) | set(b))
    out = {}
    for k in keys:
        x, y = a.get(k, [0, 0]), b.get(k, [0, 0])
        d = [y[0] - x[0], y[1] - x[1]]
        if d != [0, 0]:
            out[k] = d
    return out


def _helpers():
    """Reuse the b2x-flag-levers Boltz-2 config rather than re-deriving 40 lines of it."""
    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ClockSampler(threading.Thread):
    """Card AICLK and power from sysfs at 5 Hz. Opens no device, so it cannot be the contention."""

    ROOT = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.card = f"tenstorrent!{card}" if "!" not in str(card) else str(card)
        self.stop = threading.Event()
        self.aiclk: list[int] = []
        self.power: list[int] = []
        base = self.ROOT / str(card)
        self._clk = base / "tt_aiclk"
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
        self.aiclk.clear()
        self.power.clear()
        if not a:
            return {"aiclk_n": 0}
        return {"aiclk_min": min(a), "aiclk_max": max(a),
                "aiclk_mean": round(sum(a) / len(a), 1), "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


# --------------------------------------------------------------------------- worker (one arm)
def worker(args) -> int:
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree "
        "(memory parity-gate-scores-installed-package-not-checkout)")

    attr = FLAGS[args.flag]
    want = args.arm == "on"
    got = bool(getattr(TT, attr, False))
    assert got == want, (
        f"arm={args.arm} wants {attr}={want} but the module imported {got}. The flag must be set "
        "in the ENVIRONMENT before import. For APB in particular a post-load flip would run "
        "padded lanes against unpadded weights, which is wrong rather than slow.")

    H = _helpers()
    lc = _census()
    dev = get_device()
    out: dict = {"arm": args.arm, "size": args.size, "folds": []}

    work = Path(tempfile.mkdtemp(prefix="pairb8-", dir=str(REPO / "perf" / "bfp8_revive")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{args.size}.yaml",
                (FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c14-apb-ab", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)

    g = dev.compute_with_storage_grid_size()
    out["env"] = {
        "host": socket.gethostname(), "grid": [g.x, g.y],
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "flag": args.flag, "attr": attr,
        "env_flag": os.environ.get(args.flag),
        "module_flag": got,
        "counter_name": "TRIMUL_MM_TRANSPOSE_STATS branch mix + full lever census delta",
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "protocol": {"recycling_steps": cfg["recycling_steps"],
                     "sampling_steps": cfg["sampling_steps"],
                     "diffusion_samples": cfg["diffusion_samples"], "seed": cfg["seed"]},
    }

    target = FIX / f"cdk2x2_{args.size}.yaml"

    def one(tag: str) -> dict:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        TT.TRIMUL_MM_TRANSPOSE_STATS.clear()
        pre = _snap(lc)
        state.pfn = None
        cs = ClockSampler(args.card)
        cs.start()
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cs.stop.set()
        clk = cs.take()
        cifs = sorted(struct_dir.glob("*.cif"))
        row = {
            "tag": tag, "fold_s": round(wall, 3), "clock": clk,
            "trimul_branch": {"/".join(k): v
                              for k, v in TT.TRIMUL_MM_TRANSPOSE_STATS.items()},
            "census_delta": _delta(pre, _snap(lc)),
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest() if cifs else None,
            "loadavg1": round(os.getloadavg()[0], 2),
        }
        if args.cifdir and tag != "warmup":
            d = Path(args.cifdir) / f"{args.size}_{args.arm}_{args.block}_{tag}"  # scorer reads the arm from the tag
            d.mkdir(parents=True, exist_ok=True)
            if cifs:
                shutil.copy2(cifs[0], d / cifs[0].name)
        return row

    w = one("warmup")
    w["discarded"] = True
    out["warmup"] = w
    for i in range(args.folds):
        out["folds"].append(one(str(i)))
        Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    med = st.median([f["fold_s"] for f in out["folds"]])
    print(f"    {args.arm:4s} block{args.block} {args.size}aa median {med:7.3f}s "
          f"branches={out['folds'][0]['trimul_branch']} "
          f"clk={out['folds'][0]['clock'].get('aiclk_min')}-"
          f"{out['folds'][0]['clock'].get('aiclk_max')}", flush=True)
    return 0


# --------------------------------------------------------------------------- driver
def driver(args) -> int:
    out: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "card": args.card,
        "note": "One process per arm: TT_BIO_PAIR_B8 is read at import and decides the dtype "
                "every pair-scale allocation is built with, so it cannot be flipped in process. "
                "Arms interleaved block by block; the A/A floor is the spread across the base "
                "blocks, compared the same way base is compared to on.",
        "blocks": [],
    }
    outp = Path(args.out)
    # this row's own directory, not the sibling's: `perf/c14_land` does not exist in this
    # worktree, and a job rooted in another slug's tree gets its files deleted by fleet
    # hygiene when that slug concludes.
    tmpdir = Path(tempfile.mkdtemp(prefix="pairb8-driver-",
                                   dir=str(REPO / "perf" / "bfp8_revive")))

    def save():
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=1))

    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        print(f"[{size} aa]", flush=True)
        for b in range(args.blocks):
            for arm in ("base", "on"):
                jf = tmpdir / f"{size}_{arm}_{b}.json"
                env = dict(os.environ)
                if arm == "on":
                    env[args.flag] = "1"
                else:
                    env.pop(args.flag, None)
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--arm", arm, "--size", size, "--folds", str(args.folds),
                       "--flag", args.flag,
                       "--block", str(b), "--card", str(args.card), "--out", str(jf)]
                if args.cifdir:
                    cmd += ["--cifdir", str(args.cifdir)]
                r = subprocess.run(cmd, env=env)
                row = {"size": size, "arm": arm, "block": b, "returncode": r.returncode}
                if jf.exists():
                    row["result"] = json.loads(jf.read_text())
                out["blocks"].append(row)
                save()

    # ---- summary: medians per arm, ratio, and the A/A floor from the base blocks -----------
    summary = {}
    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        def folds(arm):
            v = []
            for row in out["blocks"]:
                if row["size"] == size and row["arm"] == arm and row.get("result"):
                    v.append([f["fold_s"] for f in row["result"]["folds"]])
            return v
        base_blocks, apb_blocks = folds("base"), folds("on")
        flat = lambda bs: [x for b in bs for x in b]
        if not flat(base_blocks) or not flat(apb_blocks):
            summary[size] = {"error": "an arm produced no fold"}
            continue
        mb, ma = st.median(flat(base_blocks)), st.median(flat(apb_blocks))
        # A/A floor: every base block median against every other, as a ratio
        bm = [st.median(b) for b in base_blocks]
        aa = [max(x, y) / min(x, y) for i, x in enumerate(bm) for y in bm[i + 1:]] or [None]
        clocks = [f["clock"] for row in out["blocks"] if row["size"] == size and row.get("result")
                  for f in row["result"]["folds"] if f.get("clock", {}).get("aiclk_n")]
        summary[size] = {
            "base_median_s": round(mb, 3), "on_median_s": round(ma, 3),
            "base_n": len(flat(base_blocks)), "on_n": len(flat(apb_blocks)),
            "delta_s": round(mb - ma, 4),
            "ratio_base_over_on": round(mb / ma, 5),
            "aa_floor_ratio_max": round(max(a for a in aa if a), 5) if aa[0] else None,
            "base_block_medians": [round(x, 3) for x in bm],
            "on_block_medians": [round(st.median(b), 3) for b in apb_blocks],
            "clock_min": min((c["aiclk_min"] for c in clocks), default=None),
            "clock_max": max((c["aiclk_max"] for c in clocks), default=None),
            "clock_samples": sum(c["aiclk_n"] for c in clocks),
            "verdict_rule": "a ratio inside aa_floor_ratio_max is a NULL, not a win",
        }
    out["summary"] = summary
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    shutil.rmtree(tmpdir, ignore_errors=True)
    for size, s in summary.items():
        if "error" in s:
            print(f"{size} aa: {s['error']}", flush=True)
            continue
        print(f"{size} aa  base {s['base_median_s']:7.3f}s  on {s['on_median_s']:7.3f}s  "
              f"delta {s['delta_s']:+.4f}s  ratio {s['ratio_base_over_on']:.5f}  "
              f"A/A floor {s['aa_floor_ratio_max']}  clk {s['clock_min']}-{s['clock_max']} "
              f"({s['clock_samples']} samples)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--card", default="0")
    ap.add_argument("--cifdir", default=None)
    # worker-only
    ap.add_argument("--arm", choices=["base", "on"])
    ap.add_argument("--flag", default="TT_BIO_PAIR_B8", choices=sorted(FLAGS))
    ap.add_argument("--size")
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()
    return worker(args) if args.arm else driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
