"""Known-answer control for score_mmshort.py. Three synthetic sessions, one per decision path."""
import json, random, subprocess, sys, tempfile, os
S = "/home/ttuser/.coworker/wt/c14-land-tail/perf/c14_land/score_mmshort.py"

def session(n, effect, noise, drift, seed):
    random.seed(seed)
    rows = []
    for b in range(n):
        lvl = 14.60 + drift * b
        for pos, arm in enumerate(("base", "on", "base")):
            off = -effect if arm == "on" else 0.0
            folds = [{"fold_s": lvl + off + random.gauss(0, noise)} for _ in range(5)]
            rows.append({"size": "512", "arm": arm, "block": b, "pos": pos,
                         "returncode": 0, "result": {"folds": folds}})
    return {"blocks": rows}

cases = [
    ("clear win, tight box   (effect 0.047, per-fold noise 0.05)", 12, 0.047, 0.05, 0.0, 1, 0),
    ("no effect, tight box   (effect 0.000, per-fold noise 0.05)", 12, 0.000, 0.05, 0.0, 2, 1),
    ("real effect, loud box  (effect 0.047, per-fold noise 0.30)", 12, 0.047, 0.30, 0.0, 3, 1),
]
bad = 0
for name, n, eff, noise, drift, seed, want in cases:
    f = tempfile.mktemp(suffix=".json")
    json.dump(session(n, eff, noise, drift, seed), open(f, "w"))
    r = subprocess.run([sys.executable, S, f, "--size", "512"], capture_output=True, text=True)
    os.unlink(f)
    ok = "OK" if r.returncode == want else "WRONG"
    if r.returncode != want:
        bad += 1
    print("%-58s want rc=%d got rc=%d  %s" % (name, want, r.returncode, ok))
    print("    " + r.stdout.strip().replace("\n", "\n    "))
print("CONTROL " + ("PASS" if not bad else "FAIL"))
sys.exit(1 if bad else 0)
