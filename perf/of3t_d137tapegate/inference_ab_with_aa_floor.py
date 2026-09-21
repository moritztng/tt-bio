#!/usr/bin/env python3
"""Does the tape gate make a shipped inference fold slower? A/B against an A/A floor.

Moritz, 2026-09-21: "make sure regular inference is not changed to softmax fp64, not made
slower. cause it was already in a good state. we did this only for training. i dont want to see
regression in inference." `inference_blast_radius.py` answered the first half by digest. This
answers the second, and it re-takes the digests in the same run because the arms are the same
folds.

Three arms per model, all on one card, same fixture and seed:

  base   the tree immediately before the gate commit -- the path present and ungated
  off    this tree, selector unset: what a user gets
  on     this tree, TT_BIO_HOST_F64_SOFTMAX_AB=all -- the flag a person can set

Every arm runs TWICE and the six folds are interleaved, direction alternating between rounds.
The A/A floor is |rep1 - rep2| within an arm and it is reported BEFORE the A/B, because a fold's
run-to-run spread is the only thing that makes a fold-to-fold difference readable. An A/B inside
the floor is not a small regression, it is no reading at all.

Two digest properties, and the second is the one the gate added:
  base == off   the path present and off changes nothing (this is D63's claim, re-taken)
  base == on    the FLAG changes nothing either, because the gate refuses it without a tape.
                Before the gate this was false by measurement: `accf5df02` recorded the on arm
                moving the digest on all three models.

AICLK is sampled DURING each fold, not before it. On Blackhole the clock sets the fold time, so
an arm whose clock sat low is an artifact and is reported as one rather than as a regression.
"""
import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perf import clocksample                                        # noqa: E402

ARMS = ("base", "off", "on")


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(py, tree: Path, model: str, fixture: Path, out: Path, env_extra: dict, card: str,
         stats: bool):
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": os.environ.get(
                    "TT_BIO_LEASE_HOLDER", "worker:of3t-d137-tapegate"),
                "PYTHONPATH": str(tree)})
    env.pop("TT_BIO_HOST_F64_SOFTMAX_AB", None)
    env.update(env_extra)
    if stats:
        # Read the softmax census off the fold itself, without touching the engine: a
        # sitecustomize on the path registers an atexit hook. That turns the per-call number
        # from `gate_call_cost.py` into a fold-level bound instead of a ratio with no scale.
        sc = out.parent / "_sitecustomize"
        sc.mkdir(parents=True, exist_ok=True)
        (sc / "sitecustomize.py").write_text(
            "import atexit, json\n"
            "def _dump():\n"
            "    try:\n"
            "        from tt_bio.tenstorrent import HOST_F64_SOFTMAX_STATS as s\n"
            "        print('SOFTMAX_CENSUS ' + json.dumps(dict(s)), flush=True)\n"
            "    except Exception:\n"
            "        pass\n"
            "atexit.register(_dump)\n")
        env["PYTHONPATH"] = str(sc) + os.pathsep + env["PYTHONPATH"]
    cmd = [py, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--single_sequence", "--sampling_steps", "6", "--diffusion_samples", "1",
           "--seed", "0", "--out_dir", str(out)]
    with clocksample.during() as clk:
        t = time.time()
        r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
        secs = time.time() - t
    census = {}
    for line in r.stdout.splitlines():
        if line.startswith("SOFTMAX_CENSUS "):
            census = json.loads(line.split(" ", 1)[1])
    cifs = sorted(out.rglob("*.cif"))
    return {"rc": r.returncode, "seconds": round(secs, 2),
            "clock": clk.summary().get(0), "clock_line": clk.line(),
            "softmax_census": census,
            "cifs": {c.name: digest(c) for c in cifs},
            "tail": (r.stdout + r.stderr)[-3000:] if r.returncode else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--base-tree", required=True,
                    help="a checkout of 6d7f32dc0, the commit before the gate landed")
    ap.add_argument("--tree", required=True, help="this worktree")
    ap.add_argument("--python", required=True, help="the interpreter with ttnn and torch")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--reps", type=int, default=2, help="folds per arm; 2 is the A/A floor")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if os.environ.get("TT_VISIBLE_DEVICES") != a.card:
        # clocksample reads tt-smi index 0, and tt-smi maps index 0 to the granted chip only
        # when TT_VISIBLE_DEVICES is exported HERE. Setting it on the fold subprocess alone
        # leaves the sampler on UMD 0, so on a four-card host every AICLK in the artifact
        # belongs to a chip that ran none of the work. That reads as a clocked measurement.
        return die("export TT_VISIBLE_DEVICES=%s for this process too, not just the fold: the "
                   "AICLK sampler reads tt-smi index 0 and would otherwise clock UMD 0 "
                   "(got %r)" % (a.card, os.environ.get("TT_VISIBLE_DEVICES")))

    base, tree = Path(a.base_tree), Path(a.tree)
    if not (base / "tt_bio/tenstorrent.py").is_file():
        return die("--base-tree has no tt_bio/tenstorrent.py: %s" % base)
    if "host_softmax_hook" in (base / "tt_bio/tenstorrent.py").read_text():
        return die("--base-tree ALREADY HAS THE GATE, so base-vs-off would be an A/A wearing "
                   "an A/B's label. Check out 6d7f32dc0, not a branch tip: wk/of3t took the "
                   "gate commit in on 2026-09-21 and stopped being the 'before' tree.")
    if "host_softmax_hook" not in (tree / "tt_bio/tenstorrent.py").read_text():
        return die("--tree does not have the gate; there is nothing to measure")

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    plan = {"base": (base, {}), "off": (tree, {}),
            "on": (tree, {"TT_BIO_HOST_F64_SOFTMAX_AB": "all"})}
    res = {k: [] for k in ARMS}
    for r in range(a.reps):
        for name in (ARMS if r % 2 == 0 else ARMS[::-1]):
            t_, env = plan[name]
            out = wd / ("out_%s_%s_r%d" % (a.model, name, r))
            f = fold(a.python, t_, a.model, Path(a.fixture), out, env, a.card,
                     stats=(name != "base"))
            res[name].append(f)
            print("[%s r%d] rc=%d %.2fs %s | %s" % (name, r, f["rc"], f["seconds"],
                                                    f["clock_line"], f["cifs"]), flush=True)
            if f["rc"]:
                print(f["tail"], flush=True)
                return die("arm %s rep %d failed" % (name, r))
            if not f["cifs"]:
                # A fold that wrote nothing exits 0 on some inputs, and two empty digest maps
                # compare equal, so without this the whole report reads FREE on zero folds.
                print(f["tail"], flush=True)
                return die("arm %s rep %d produced no .cif, so there is nothing to compare"
                           % (name, r))

    secs = {k: [f["seconds"] for f in v] for k, v in res.items()}
    floors = {k: round(max(v) - min(v), 2) for k, v in secs.items()}
    floor = max(floors.values())
    med = {k: round(statistics.median(v), 2) for k, v in secs.items()}
    ab = {"off_minus_base": round(med["off"] - med["base"], 2),
          "on_minus_off": round(med["on"] - med["off"], 2)}

    digs = {k: [f["cifs"] for f in v] for k, v in res.items()}
    stable = all(all(d == v[0] for d in v) and bool(v[0]) for v in digs.values())
    base_eq_off = digs["base"][0] == digs["off"][0]
    base_eq_on = digs["base"][0] == digs["on"][0]

    clocks = {k: [f["clock"] for f in v] for k, v in res.items()}
    lows = [k for k, v in clocks.items()
            if any((c or {}).get("median", 0) < 1200 for c in v)]

    slower = [k for k, d in ab.items() if d > floor]
    rep = {"model": a.model, "fixture": a.fixture,
           "host": socket.gethostname(), "card": a.card,
           "host_card": "%s card %s" % (socket.gethostname(), a.card), "reps": a.reps,
           "seconds": secs, "median_seconds": med,
           "AA_floor_seconds_per_arm": floors, "AA_floor_seconds": floor,
           "AB_seconds": ab, "readable_above_the_floor": {k: abs(v) > floor
                                                          for k, v in ab.items()},
           "clock": clocks, "arms_below_1200MHz": lows,
           "softmax_calls_per_fold": {k: (v[0]["softmax_census"] or {}).get("declined")
                                      for k, v in res.items() if k != "base"},
           "digests": digs, "digest_stable_within_arm": stable,
           "base_equals_off": base_eq_off, "base_equals_on": base_eq_on,
           "arms": res}
    rep["verdict"] = (
        "ARTIFACT: %s ran below 1200 MHz, so these timings are the clock's, not the gate's"
        % ", ".join(lows) if lows else
        "FAIL: digests are not stable within an arm, so nothing here is comparable"
        if not stable else
        "FAIL: the gate changed the fold output (base==off %s, base==on %s)"
        % (base_eq_off, base_eq_on) if not (base_eq_off and base_eq_on) else
        "REGRESSION: %s exceeds the %.2f s A/A floor" % (", ".join(slower), floor) if slower else
        "FREE: output byte-identical across all three arms, and every A/B (%s) is inside the "
        "%.2f s A/A floor" % (ab, floor))
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0 if rep["verdict"].startswith("FREE") else 3


def die(msg):
    print("REFUSING: " + msg, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
