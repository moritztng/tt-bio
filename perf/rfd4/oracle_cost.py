"""What one scoring-oracle call costs, at the protocol RFD4's beam search actually asks for.

RFD4's beam search scores every branched candidate with a structure-prediction oracle, so the
design throughput the 3.00x gate measures depends on oracle throughput as much as on RFD4's own.
The breakeven is sharp: with the shipped beam recipe, one oracle fold worth more than 75 RFD4
trunk passes makes the search oracle-bound rather than RFD4-bound (`tt_bio/rfd4/search.py`,
`state/rfd4/BEAM.md`).

The numbers that exist today do not answer it. `docs/size_ladder_baseline.d` folds at **6**
diffusion steps; upstream's oracle runs AF3 at `--num_recycles=5 --num_diffusion_samples=1` over
AF3's own 200-step rollout (`rfproteina/af3/af3_apptainer.py:18,141,143`). So the ladder's 9.3 s
at 256 tokens is a loose lower bound on an oracle call, not an estimate of one.

This harness measures both, on one card, interleaved, so the matched-protocol number comes with
its own anchor against the cells already in the tree rather than against another box:

    python3 perf/rfd4/oracle_cost.py --card 0 --models openfold3 --rungs 256,512 --reps 3

It opens no device itself: every fold is a `tt-bio predict` subprocess, which is the path the
service uses and the only one that carries the mesh descriptor. The clock is sampled DURING each
fold by `perf/clocksample.py`, because on Blackhole the AICLK sets the fold time and a reading
taken before the launch describes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perf import clocksample  # noqa: E402

#: Upstream's oracle protocol, as the AF3 container is invoked for a search checkpoint.
ORACLE = {"recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 1}
#: The protocol every `docs/size_ladder_baseline.d` cell was recorded at, for the anchor arm.
LADDER = {"sampling_steps": 6, "diffusion_samples": 1}

#: Models that fold the shared apo fixture, and the fixture family they use.
COFOLD_FIXTURE = ROOT / "perf" / "size512" / "fixtures" / "cdk2x2_{rung}.yaml"
#: AF2-IG takes a complex plus the binder's sequence, so it brings its own rungs
#: (built by perf/af2igcov/fixture.py: a real target chain, a real 80-residue binder chain,
#: an interface that is nonsense on purpose -- this asks what a fold costs, not whether it is right).
AF2IG_FIXTURE = ROOT / "perf" / "rfd4" / "inputs" / "af2ig_{rung}" / "input.yaml"


def fixture(model: str, rung: int) -> Path:
    if model == "af2ig":
        path = Path(str(AF2IG_FIXTURE).format(rung=rung))
        if path.exists():
            # The YAML carries an absolute path to its PDB, which is wrong in any checkout but
            # the one that built it. Repoint it at this tree before the CLI copies it to scratch.
            text = path.read_text()
            fixed = re.sub(r"(?m)^(\s*file:\s*).*$", r"\g<1>" + str(path.parent / "complex.pdb"),
                           text)
            if fixed != text:
                path.write_text(fixed)
        return path
    return Path(str(COFOLD_FIXTURE).format(rung=rung))


def fold_command(model: str, rung: int, protocol: str, out_dir: Path) -> list[str]:
    """One `tt-bio predict` invocation at the chosen protocol."""
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fixture(model, rung)),
           "--model", model, "--out_dir", str(out_dir), "--seed", "0"]
    if model == "af2ig":
        # No MSA, no diffusion, no seed effect: the whole job is the trunk, so neither
        # protocol's sampling settings apply. Both arms run AF2-IG's shipped 3 recycles,
        # which is also what upstream's AF2 oracle leaves at its default.
        return cmd
    cmd += ["--single_sequence"]
    for flag, value in (ORACLE if protocol == "oracle" else LADDER).items():
        cmd += [f"--{flag}", str(value)]
    return cmd


def fold_seconds(model: str, rung: int, out_dir: Path) -> float | None:
    """The fold's own seconds, as the size ladder reads them: results.json, not the wall."""
    from tt_bio.main import predict_results_dir_name

    results = out_dir / predict_results_dir_name(model, fixture(model, rung).stem) / "results.json"
    if not results.exists():
        return None
    rows = json.loads(results.read_text())
    times = [r["runtime_s"] for r in rows
             if r.get("status") == "ok" and r.get("runtime_s") is not None]
    return max(times) if times else None


def one_rep(model: str, rung: int, protocol: str, card: str, work: Path,
            timeout_s: int) -> dict:
    out_dir = work / f"{model}_{rung}_{protocol}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TT_VISIBLE_DEVICES": card,
           "TT_BIO_LEASE_CARDS": os.environ.get("TT_BIO_LEASE_CARDS", card)}
    log = out_dir / "fold.log"
    t0 = time.monotonic()
    with open(log, "w") as fp, clocksample.during(period=2.0) as clk:
        try:
            rc = subprocess.run(fold_command(model, rung, protocol, out_dir), cwd=ROOT, env=env,
                                stdout=fp, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    wall = time.monotonic() - t0
    rep = {"wall_s": round(wall, 3), "rc": rc, "timed_out": timed_out,
           "aiclk": clk.summary().get(0), "load": None, "runtime_s": None}
    if clk.load:
        rep["load"] = round(statistics.median(clk.load), 3)
    if rc == 0:
        rep["runtime_s"] = fold_seconds(model, rung, out_dir)
    else:
        rep["log"] = str(log)
    return rep


def run(args) -> int:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    board = (subprocess.run(["bash", "-lc", "cat /proc/device-tree/model 2>/dev/null"],
                            capture_output=True, text=True).stdout.strip() or None)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)

    for model in args.models.split(","):
        for rung in [int(r) for r in args.rungs.split(",")]:
            path = fixture(model, rung)
            if not path.exists():
                print(f"SKIP {model} {rung}: no fixture at {path}")
                continue
            # AF2-IG has no diffusion rollout, so both protocols are the same fold: run it once.
            protocols = ["oracle"] if model == "af2ig" else args.protocols.split(",")
            for protocol in protocols:
                reps = [one_rep(model, rung, protocol, args.card, work, args.timeout)
                        for _ in range(args.reps)]
                times = [r["runtime_s"] for r in reps if r["runtime_s"] is not None]
                clocks = [r["aiclk"]["median"] for r in reps if r.get("aiclk")]
                row = {
                    "model": model, "rung_tokens": rung, "protocol": protocol,
                    "settings": {} if model == "af2ig" else (ORACLE if protocol == "oracle"
                                                             else LADDER),
                    "fixture": str(path), "card": args.card, "host": os.uname().nodename,
                    "board": board, "commit": commit, "reps": reps,
                    "runtime_s_median": round(statistics.median(times), 3) if times else None,
                    # The clock each number was taken at. A row whose aiclk is null, or whose
                    # median sits below ~1200 MHz on Blackhole, is an artifact and says so
                    # rather than being reported as a slower model.
                    "aiclk_median_per_rep": clocks or None,
                    "clock_ok": bool(clocks) and min(clocks) >= args.clock_floor,
                    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with out_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{model:>10} {rung:>5} tok {protocol:>6}: "
                      f"{row['runtime_s_median']} s  clocks {clocks}  "
                      f"{'OK' if row['clock_ok'] else 'CLOCK SUSPECT -- not a measurement'}")
    print(f"\nrows appended to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", required=True, help="the leased card, as TT_VISIBLE_DEVICES")
    ap.add_argument("--models", default="openfold3,af2ig")
    ap.add_argument("--rungs", default="256,512")
    ap.add_argument("--protocols", default="oracle,ladder",
                    help="oracle = upstream's 5 recycles / 200 steps; ladder = the 6-step "
                         "protocol every size_ladder_baseline cell was recorded at")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--clock-floor", type=int, default=1200,
                    help="Blackhole: below this the card was throttled and the row is an artifact")
    ap.add_argument("--work", default=str(ROOT / "perf" / "rfd4" / "work"))
    ap.add_argument("--out", default=str(ROOT / "perf" / "rfd4" / "results" / "oracle_cost.jsonl"))
    ap.add_argument("--print-commands", action="store_true",
                    help="print the fold commands and exit, opening nothing")
    args = ap.parse_args()
    if args.print_commands:
        for model in args.models.split(","):
            for rung in [int(r) for r in args.rungs.split(",")]:
                for protocol in (["oracle"] if model == "af2ig" else args.protocols.split(",")):
                    print(" ".join(fold_command(model, rung, protocol, Path("<out>"))))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
