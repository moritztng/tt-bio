#!/usr/bin/env python3
"""How far the forward softmax config moves a fold, in Angstrom, beside the seed floor.

A digest is a yes/no. The bar this campaign is held to is structural: on the 512 aa cell the
kill bar is 0.60 A and re-running with a different seed moves the structure 1.84 A, so a lever
is judged against the variation already accepted, not against zero. This measures both on the
same target in the same run:

    lever   ca_rmsd(off seed 0, on seed 0)      what the config does
    floor   ca_rmsd(off seed 0, off seed 1)     what a seed does

Structures, not digests, so an identical digest is reported as 0.000 A and a moved one carries
its size. Kabsch over Ca atoms through `perf/of3t_rankunify/ca_rmsd.py`, the same maths the
OpenFold3 fold gate uses.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "of3t_rankunify"))

from perf import clocksample                                                # noqa: E402


def fold(py, model, fixture, out, seed, lever_env, lever_value, card, extra=()):
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": "worker:of3t-fwdkcfg", "PYTHONPATH": str(ROOT)})
    env.pop(lever_env, None)
    if lever_value:
        env[lever_env] = lever_value
    cmd = [py, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--diffusion_samples", "1", "--seed", str(seed), "--out_dir", str(out)]
    cmd += list(extra)
    t = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode:
        print((r.stdout + r.stderr)[-2500:], flush=True)
        raise SystemExit("fold failed: %s seed=%s lever=%r" % (model, seed, lever_value))
    cifs = sorted(out.rglob("*.cif"))
    if not cifs:
        raise SystemExit("fold wrote no .cif: %s seed=%s" % (model, seed))
    return cifs[0], round(time.time() - t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--card", default="0")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lever-env", default="TT_BIO_SOFTMAX_PRECISE_AB")
    ap.add_argument("--kill-bar", type=float, default=0.60)
    ap.add_argument("--case", action="append", required=True,
                    help="model=token, the token the site resolves under")
    ap.add_argument("--fold-arg", action="append", default=[],
                    help="extra argument for the fold, repeatable. The cell has to be one where "
                         "the SEED FLOOR is small: at 6 sampling steps with --single_sequence a "
                         "128-token target's two seeds land 106 A apart, and a floor that size "
                         "passes anything")
    a = ap.parse_args()

    if os.environ.get("TT_VISIBLE_DEVICES") != a.card:
        raise SystemExit("export TT_VISIBLE_DEVICES=%s here too: the AICLK sampler reads "
                         "tt-smi index 0" % a.card)

    from ca_rmsd import ca_rmsd
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    rows = {}
    with clocksample.during(period=5.0) as clk:
        for case in a.case:
            model, _, token = case.partition("=")
            arms = {}
            for name, seed, lever in (("off_s0", 0, ""), ("on_s0", 0, token),
                                      ("off_s1", 1, "")):
                cif, secs = fold(a.python, model, Path(a.fixture), wd / (model + "_" + name),
                                 seed, a.lever_env, lever, a.card, a.fold_arg)
                arms[name] = {"cif": str(cif), "seconds": secs,
                              "sha256": hashlib.sha256(cif.read_bytes()).hexdigest()}
                print("  %s %s %.2fs" % (model, name, secs), flush=True)
            lever_a = ca_rmsd(arms["on_s0"]["cif"], arms["off_s0"]["cif"])
            floor_a = ca_rmsd(arms["off_s1"]["cif"], arms["off_s0"]["cif"])
            rows[model] = {
                "token": token, "arms": arms,
                "lever_ca_rmsd_A": round(lever_a, 4),
                "seed_floor_ca_rmsd_A": round(floor_a, 4),
                "kill_bar_A": a.kill_bar,
                "digest_moved": arms["on_s0"]["sha256"] != arms["off_s0"]["sha256"],
                "lever_over_seed_floor": (round(lever_a / floor_a, 4) if floor_a else None),
                "verdict": ("INSIDE THE BAR: %.4f A against a %.2f A kill bar and a %.4f A "
                            "seed floor" % (lever_a, a.kill_bar, floor_a)
                            if lever_a <= a.kill_bar else
                            "OVER THE BAR: %.4f A against %.2f A" % (lever_a, a.kill_bar)),
            }
            print("  %s lever %.4f A | seed floor %.4f A | bar %.2f A"
                  % (model, lever_a, floor_a, a.kill_bar), flush=True)

    rep = {"what": "Angstrom move of the forward softmax config, beside the seed floor",
           "host": socket.gethostname(), "card": a.card, "board": "p300c",
           "host_card": "%s card %s" % (socket.gethostname(), a.card),
           "fixture": a.fixture, "lever_env": a.lever_env,
           "clock_aiclk_during": clk.summary(), "clock_line": clk.line(0),
           "models": rows}
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(clk.line(0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
