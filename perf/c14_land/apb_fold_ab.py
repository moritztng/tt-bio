#!/usr/bin/env python3
"""Price TT_BIO_APB_CONCAT_HEADS on the Boltz-2 fold, at 298 and 512 aa.

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
      python3 perf/c14_land/apb_fold_ab.py --sizes 298,512 --blocks 3 --folds 3 \
        --card 3 --out perf/c14_land/apb_ab.json --cifdir perf/c14_land/apb_cifs
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
    # c14-matmul-ceiling's lever, +0.047 s at the fold, accuracy already discharged on BOTH
    # Boltz-2 and RF3 (CA-lDDT against the experimental structure, A/A exactly 0.000000).
    # Three of that row's sessions failed their own A/A test because a 0.047 s effect sits at
    # this box's paired noise floor and a release gate started mid-session. n and a per-arm
    # guard are the whole fix, and this harness has the guard. Run --blocks 6 --folds 3.
    "TT_BIO_MM_SHORT_M_BW": "_MM_SHORT_M_BW",
    "TT_BIO_SDPA_BAND_DIV_K": "_SDPA_BAND_DIV_K",
    "TT_BIO_TRIATT_BIAS_B8": "_TRIATT_BIAS_B8",
    # TT_BIO_ADALN_MEMO_EAGER was here. Its module global came off this branch with the
    # hit-driven AdaLN retain that pass 6 measured as a net loss, so the entry pointed at a name
    # this tree no longer defines. The worker asserts the global, so selecting it would have
    # failed at the arm rather than silently producing two identical arms -- but a FLAGS entry
    # that cannot resolve is a trap, and the check that found it now runs over the whole map.
}


def _helpers():
    """Reuse b2z_ttnn's config rather than re-deriving 40 lines of it.

    It is the same config b2x-flag-levers builds, with one difference that matters here: its
    `build_cfg` takes the model, so this harness is not boltz-2-only. That is what lets the AdaLN
    memo be measured on RF3, which is the caller its mechanism predicts the whole effect on.
    """
    p = REPO / "perf" / "b2z_ttnn" / "stack_fold.py"
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
    dev = get_device()
    out: dict = {"arm": args.arm, "size": args.size, "model": args.model, "folds": []}

    work = Path(tempfile.mkdtemp(prefix="c14-apb-", dir=str(REPO / "perf" / "c14_land")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{args.size}.yaml",
                (FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir, args.model)
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
        "flag": args.flag, "attr": attr, "model": args.model,
        "env_flag": os.environ.get(args.flag),
        "module_flag": got,
        "counter_name": "APB_CONCAT_HEADS_STATS (meaningful for the APB flag only)",
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "protocol": {"recycling_steps": cfg["recycling_steps"],
                     "sampling_steps": cfg["sampling_steps"],
                     "diffusion_samples": cfg["diffusion_samples"], "seed": cfg["seed"]},
    }

    target = FIX / f"cdk2x2_{args.size}.yaml"

    def one(tag: str) -> dict:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        stats = getattr(TT, "APB_CONCAT_HEADS_STATS", [0, 0])
        stats[0] = stats[1] = 0
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
            "apb_served_declined": list(stats),
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
          f"apb_counter={out['folds'][0]['apb_served_declined']} "
          f"clk={out['folds'][0]['clock'].get('aiclk_min')}-"
          f"{out['folds'][0]['clock'].get('aiclk_max')}", flush=True)
    return 0


# --------------------------------------------------------------------------- preflight
def preflight(card, guard=None):
    """Refuse the session unless BOTH couplings are clear: the board-pair sibling and the host.

    The harness enforces this itself rather than trusting its caller. On 2026-09-18
    c14-matmul-ceiling took a fold A/B with the sibling idle, benchlock held and the clock forced
    1350-1350 during every fold, and had to retract the number: the release gate was folding on the
    other BOARD, and same-host PSU, DRAM and PCIe put the base median 0.37 s above the number of
    record with a 4.35 % base spread. benchlock does not serialise the gate, and the box read
    loadavg 1.63 against benchlock's 2.00 ceiling while a fold child burned 100 % CPU, so neither
    the lock nor the load ceiling is the guard. Exit 75 is benchlock's own "retry later, do not
    measure anyway".
    """
    if guard is not None:
        r = subprocess.run([sys.executable,
                            str(Path(__file__).resolve().parent / "pair_channel_quiet.py"),
                            "--card", str(card)] + list(guard),
                           capture_output=True, text=True)
        return r.returncode == 0, ["pair_channel: %s" % l for l in
                                   (r.stdout + r.stderr).strip().splitlines()]
    g = Path(__file__).resolve().parents[1] / "c12_orchestrator" / "pair_guard"
    lines, ok = [], True
    for chk, extra in (("pair_idle.py", ["--card", str(card)]), ("host_quiet.py", [])):
        r = subprocess.run([sys.executable, str(g / chk)] + extra,
                           capture_output=True, text=True)
        lines += ["%s: %s" % (chk[:-3], l)
                  for l in (r.stdout + r.stderr).strip().splitlines()]
        if r.returncode != 0:
            ok = False
    return ok, lines


def wait_admissible(card, cap_s: float, tag: str, guard=None) -> dict:
    """`preflight`, again, before EVERY arm process -- and block until it passes.

    Running it once at launch protects the first second of a 25-minute session and nothing after
    it. That is not a hypothetical: three `c14-matmul-ceiling` sessions failed their own A/A test
    because a release gate started AFTER their launch check passed, on the other board of the same
    host. Waiting beats aborting -- the arms stay interleaved and a transient neighbour costs
    minutes instead of the session -- but it never proceeds quietly: past `cap_s` the run fails,
    because a contaminated number costs more than a missing one.

    Expect a wait between arms even on an empty box. `host_quiet`'s loadavg ceiling cannot tell
    this session's own just-exited arm from a neighbour, and loadavg decays over about a minute,
    so a clean session self-throttles ~90 s per arm. That is a real cost and it is the safe
    direction to be wrong in; the settle also lets the board's power budget recover before the
    next arm's first fold.
    """
    t0 = time.time()
    ok, lines = preflight(card, guard)
    while not ok:
        waited = time.time() - t0
        if waited > cap_s:
            raise SystemExit("NOT ADMISSIBLE after %.0fs at %s: %s. Refusing to time a "
                             "contaminated session." % (waited, tag, " | ".join(lines)))
        print("  %s: waiting %.0fs of %.0fs -- %s" % (tag, waited, cap_s, " | ".join(lines)),
              flush=True)
        time.sleep(30)
        ok, lines = preflight(card, guard)
    return {"admissible": True, "checks": lines, "waited_s": round(time.time() - t0, 1),
            "loadavg1": round(os.getloadavg()[0], 2)}


def guard_args(args):
    """None -> pair_idle + host_quiet (the default, unchanged). A list -> pair_channel_quiet.

    `host_quiet` fails on ANY busy device fd holder on the host and on loadavg over 2.00, so a
    multi-day training campaign on the other board pair makes it unsatisfiable rather than
    protective. `--guard pair_channel` keeps the board-pair rule hard and reports the host channel
    instead of merging it, which admits a STATIONARY neighbour on the other pair. It also narrows
    the claim: see pair_channel_quiet.py's header. Absolute seconds from such a session are not a
    fold time of record; only the paired ratio is, and the A/A floor decides whether even that is.
    """
    if args.guard != "pair_channel":
        return None
    return ["--maxload", str(args.maxload), "--drift", str(args.drift),
            "--settle", str(args.settle)]


# --------------------------------------------------------------------------- driver
def driver(args) -> int:
    out: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "card": args.card,
        "note": "One process per arm: TT_BIO_APB_CONCAT_HEADS is consumed at model construction "
                "and re-lanes proj_g/proj_o, so it cannot be flipped in process. Arms interleaved "
                "block by block; the A/A floor is the spread across the base blocks, compared the "
                "same way base is compared to apb.",
        "blocks": [],
    }
    outp = Path(args.out)
    ok, pf = preflight(args.card, guard_args(args))
    out["preflight"] = {"admissible": ok, "checks": pf}
    for l in pf:
        print(l, flush=True)
    if not ok and not args.force_contended:
        outp.parent.mkdir(parents=True, exist_ok=True)
        out["aborted"] = "preflight refused: not an admissible timing read"
        outp.write_text(json.dumps(out, indent=1))
        print("REFUSED (exit 75): wait for a clean pair AND a quiet host, or DEFER.", flush=True)
        return 75
    if not ok:
        out["CONTAMINATED"] = ("--force-contended was passed: these seconds are NOT a number of "
                               "record and must not be quoted")
    tmpdir = Path(tempfile.mkdtemp(prefix="c14-apb-driver-", dir=str(REPO / "perf" / "c14_land")))

    def save():
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=1))

    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        print(f"[{size} aa]", flush=True)
        for b in range(args.blocks):
            # --bracket puts base at BOTH ends of every block, which is what makes the A/A floor
            # structurally matched to the A/B comparison. Without it a 12-block session yields 12
            # A/B pairs but only 6 A/A pairs, so its floor is estimated at half the n of the effect
            # it has to beat, and a marginal lever is refused for a reason that is the design's
            # rather than the lever's. c14-matmul-ceiling's f3 bracketed and is the cleanest session
            # this box has produced.
            for pos, arm in enumerate(("base", "on", "base") if args.bracket else ("base", "on")):
                jf = tmpdir / f"{size}_{arm}_{b}_{pos}.json"
                env = dict(os.environ)
                if arm == "on":
                    env[args.flag] = "1"
                else:
                    env.pop(args.flag, None)
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--arm", arm, "--size", size, "--folds", str(args.folds),
                       "--flag", args.flag, "--model", args.model,
                       "--block", str(b), "--card", str(args.card), "--out", str(jf)]
                if args.cifdir:
                    cmd += ["--cifdir", str(args.cifdir)]
                tag = f"{size} aa block {b} {arm}[{pos}]"
                guard = ({"skipped": "--force-contended"} if args.force_contended
                         else wait_admissible(args.card, args.quiet_wait, tag,
                                              guard_args(args)))
                r = subprocess.run(cmd, env=env)
                row = {"size": size, "arm": arm, "block": b, "pos": pos,
                       "returncode": r.returncode, "guard_before": guard}
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
    # An arm that produced no fold is a FAILED session, not a finished one. It returned 0 on
    # 2026-09-18 and the caller read "HARNESS EXIT 0" off a run where every fold had died in
    # ModuleNotFoundError, so the failure looked like a completed measurement.
    failed = [sz for sz, v in summary.items() if "error" in v]
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
    if failed:
        print(f"FAILED: no fold at {', '.join(failed)} aa. This is not a measurement.", flush=True)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--card", default="0")
    ap.add_argument("--quiet-wait", type=float, default=2400.0, dest="quiet_wait",
                    help="seconds to wait for an admissible box BEFORE EACH ARM, not once at "
                         "launch; past it the run fails rather than timing a loud box")
    ap.add_argument("--cifdir", default=None)
    # Deliberately awkward to reach: a contended session produces a void, and pass 1 already spent
    # three of them. It exists only so a screen can be taken knowingly, stamped CONTAMINATED.
    ap.add_argument("--force-contended", action="store_true")
    ap.add_argument("--guard", choices=["host_quiet", "pair_channel"], default="host_quiet",
                    help="host_quiet (default, unchanged) or pair_channel, which keeps the "
                         "board-pair rule hard and admits a stationary neighbour on the other pair")
    ap.add_argument("--bracket", action="store_true",
                    help="run base,on,base per block instead of base,on, so the A/A floor has the "
                         "same n and the same arm separation as the A/B delta")
    ap.add_argument("--maxload", type=float, default=8.0)
    ap.add_argument("--drift", type=float, default=1.5)
    ap.add_argument("--settle", type=float, default=60.0)
    # worker-only
    ap.add_argument("--arm", choices=["base", "on"])
    ap.add_argument("--flag", default="TT_BIO_APB_CONCAT_HEADS", choices=sorted(FLAGS))
    ap.add_argument("--model", default="boltz2",
                    help="boltz2 is the 512 aa number of record; rf3 is where the AdaLN memo's "
                         "mechanism predicts its effect")
    ap.add_argument("--size")
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()
    return worker(args) if args.arm else driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
