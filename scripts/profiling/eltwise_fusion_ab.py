"""Paired A/B of the three eltwise/norm fusion gates on the committed Blackhole cell.

Protocol, per model:
  * Three arms per rep in the order OFF, ON, OFF -- so every ON is bracketed by an OFF
    and arm order cannot charge one arm the warmup or a load ramp (memory
    op-ab-must-interleave-arms-compile-warmup-bias).
  * The two OFF runs of a rep are an A/A pair by construction: identical env, identical
    tree. Their spread IS the floor, measured in the same session as the A/B rather than
    asserted from an earlier one, which is the only thing that survives co-tenant load
    arriving mid-run (memory benchlock-one-shot-check-blind-to-mid-run-contention).
  * The ratio is reported against the mean of a rep's own two OFF runs, and a win is
    only claimed if it clears the measured A/A floor.

Each arm is a fresh subprocess: the gates are read at import, so one process cannot
hold two arms.
"""
import argparse, json, os, re, statistics, subprocess, sys, time

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
GATES = ("TT_BIO_FUSE_SCALE_ADD", "TT_BIO_FUSE_MASK_ADD", "TT_BIO_FUSE_NORM_RESIDUAL")
_VAL = re.compile(r"^\[(\S+)\]\s+([0-9.]+)\s+(\S+)", re.M)


def run(model, on, gates=GATES):
    env = {**os.environ, "PYTHONPATH": WT,
           **{g: ("1" if on else "0") for g in gates}}
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(WT, "scripts/perf_regression.py"),
                        "--model", model, "--allow-contended"],
                       cwd=WT, env=env, capture_output=True, text=True, timeout=1800)
    # perf_regression writes the gate table to stdout and the per-model value line to
    # stderr, so both streams have to be searched.
    m = _VAL.search(p.stderr) or _VAL.search(p.stdout)
    if not m:
        raise RuntimeError(f"{model} on={on} produced no value:\n{p.stdout[-2500:]}\n{p.stderr[-1500:]}")
    return float(m.group(2)), m.group(3), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--gates", default=",".join(GATES),
                    help="comma-separated subset to flip (the rest stay at their default)")
    ap.add_argument("--out", default=os.path.join(WT, "artifacts/eltwise_ab.json"))
    a = ap.parse_args()
    gates = tuple(g for g in a.gates.split(",") if g)
    results = {}
    for model in a.models.split(","):
        offs, ons = [], []
        unit = None
        for rep in range(a.reps):
            try:
                for label, on, bucket in (("off", False, offs), ("on", True, ons),
                                          ("off", False, offs)):
                    v, unit, dt = run(model, on, gates)
                    bucket.append(v)
                    print(f"  {model} rep{rep} {label:3s} {v:.6f} {unit} ({dt:.0f}s)", flush=True)
            except Exception as e:
                print(f"  {model} rep{rep} FAILED: {e}", flush=True)
                results[model] = dict(status="ERROR", err=str(e)[:2000])
                break
        if model in results:
            continue
        if not ons:
            continue
        # A/A floor: the per-rep spread of the two identical OFF runs
        pairs = [(offs[2 * i], offs[2 * i + 1]) for i in range(len(offs) // 2)]
        aa = [abs(x - y) / ((x + y) / 2) for x, y in pairs]
        base = [(x + y) / 2 for x, y in pairs]
        ratios = [ons[i] / base[i] for i in range(min(len(ons), len(base)))]
        results[model] = dict(
            status="OK", unit=unit, off=offs, on=ons,
            off_median=statistics.median(offs), on_median=statistics.median(ons),
            aa_floor_pct=[100 * v for v in aa],
            aa_floor_worst_pct=100 * max(aa) if aa else None,
            ratio_per_rep=ratios,
            ratio_median=statistics.median(ratios) if ratios else None,
            gain_pct=100 * (statistics.median(ratios) - 1) if ratios else None)
        r = results[model]
        verdict = ("above floor" if r["gain_pct"] is not None
                   and abs(r["gain_pct"]) > r["aa_floor_worst_pct"] else "INSIDE FLOOR")
        print(f"{model}: off {r['off_median']:.6f} -> on {r['on_median']:.6f} {unit} "
              f"= {r['ratio_median']:.5f}x ({r['gain_pct']:+.2f}%), "
              f"A/A floor worst {r['aa_floor_worst_pct']:.2f}% -> {verdict}", flush=True)
    meta = dict(gates=list(gates), reps=a.reps, host=os.uname().nodename,
                card=os.environ.get("TT_VISIBLE_DEVICES"), when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(a.out, "w") as fh:
        json.dump(dict(meta=meta, results=results), fh, indent=1)
    print("wrote " + a.out)


if __name__ == "__main__":
    main()
