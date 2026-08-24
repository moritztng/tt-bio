#!/usr/bin/env python3
"""p107 -- the fold A/B for the block-sparse atom attention arm.

p103 predicts +3.787 s/design and p105 says +3.744 out of sample, but both are a cost model over
measured parts. This is the measurement. p106 already established the arm is correct and
two-sided; this one only has to be trustworthy about time, so it obeys the three rules this
lineage learned the hard way:

* **Interleaved.** A round is off(1 design) then on(1 design), and every delta is taken inside a
  round. All-A-then-all-B on a box whose load ramped turned a 1.226 s/design win into a 3.382
  s/design loss once already, with every arm assertion passing.
* **Load sampled per fold, and a round dropped whole.** A benchlock acquisition is not evidence
  the fold was clean: it gates on load once, at acquire, and the co-tenants that matter here do
  not take the lock at all. A round with one clean fold and one loaded fold is worth less than no
  round, so it is discarded rather than averaged.
* **One warm round first**, so each arm's program set is compiled before it is timed. The on arm
  compiles one program set per bucket plus the dense fallback, and reading that as run time is
  exactly the trap that made a batching screen report ~19 s/design of regression.

Every fold checks its own digest and its own arm counters, so a drifting or silent arm is caught
in the fold that produced it rather than inferred from the shape of the totals.
"""
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import block_sparse as BS                               # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p107/ab.json")
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
LOAD_BAR = float(os.environ.get("P107_LOAD_BAR", "3.0"))
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SHIPPED_DIGEST = "5295e526ebd0b757"
PREDICTED = 3.744          # p105, out of sample
# Clean card-1 baseline for this fixture. Co-tenancy inflates both arms roughly multiplicatively,
# so a raw delta taken on a loaded box overstates the prize by the same factor the absolute is
# inflated by. The fraction (off - on) / off is what survives that, and scaling it by a baseline
# measured on a quiet box is what turns it back into s/design.
CLEAN_BASELINE = float(os.environ.get("P107_CLEAN_BASELINE", "94.087"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def load1():
    return float(pathlib.Path("/proc/loadavg").read_text().split()[0])


def fold(label, on, r):
    out_dir = "/tmp/rfd3_p107_%s_%d" % (label, r)
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    BS.STATS[0] = BS.STATS[1] = BS.STATS[2] = 0
    was = BS.set_enabled(on)
    l0 = load1()
    try:
        rfd3_design.run_design(json.loads(FIXTURE.read_text()), out_dir, checkpoint_dir=CKPT,
                               from_pdb=True, num_timesteps=STEPS, seed=SEED, num_designs=1,
                               batch_size=1, verbose=False)
    finally:
        BS.set_enabled(was)
    l1 = load1()
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    blocked, fallback, shipped = BS.STATS
    # the on arm must have taken every atom call; the off arm none of them
    arm_ok = (blocked > 0 and shipped == 0) if on else (blocked == 0 and fallback == 0)
    dig = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else "NO CIF"
    dig_ok = (dig != SHIPPED_DIGEST) if on else (dig == SHIPPED_DIGEST)
    row = dict(arm=label, round=r, warm=(r == 0), s_per_design=round(sum(WALLS), 3),
               n_cifs=len(cifs), digest=dig, blocked=blocked, fallback=fallback,
               shipped=shipped, load_before=l0, load_after=l1,
               load_max=round(max(l0, l1), 2), arm_verified=arm_ok, digest_ok=dig_ok,
               load_clean=max(l0, l1) <= LOAD_BAR)
    print("[p107] r%d %-3s %9.3f s/design  blk=%-5d fb=%-5d ship=%-5d  load<=%5.2f  %s %s %s  %s"
          % (r, label, row["s_per_design"], blocked, fallback, shipped, row["load_max"],
             "ARM-OK" if arm_ok else "ARM-WRONG", "DIG-OK" if dig_ok else "DIG-WRONG",
             "CLEAN" if row["load_clean"] else "LOADED-VOID", dig), flush=True)
    return row


def main():
    print("[p107] rounds=%d (+1 warm) steps=%d seed=%d card=%s load bar %.1f  Q=%d buckets=%s"
          % (ROUNDS, STEPS, SEED, os.environ.get("TT_VISIBLE_DEVICES"), LOAD_BAR,
             BS.config()[0], BS.config()[1]), flush=True)
    print("[p107] predicted from the cost model: %+.3f s/design (p105, out of sample)"
          % PREDICTED, flush=True)
    rows = []
    for r in range(0, ROUNDS + 1):
        for label, on in (("off", False), ("on", True)):
            rows.append(fold(label, on, r))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Dump before any summary arithmetic: this harness's predecessor lost 40 minutes of card time
    # to a format-string bug in its summary, every fold already folded.
    OUT.write_text(json.dumps(dict(rounds=ROUNDS, steps=STEPS, seed=SEED, load_bar=LOAD_BAR,
                                   q_block=BS.config()[0], buckets=list(BS.config()[1]),
                                   predicted_s_per_design=PREDICTED,
                                   card=os.environ.get("TT_VISIBLE_DEVICES"),
                                   host=os.uname().nodename, rows=rows), indent=2) + "\n")

    timed = [x for x in rows if not x["warm"]]
    bad = [x for x in rows if not (x["arm_verified"] and x["digest_ok"])]
    print("\n" + "=" * 74, flush=True)
    if bad:
        print("[p107] %d fold(s) failed an arm or digest check -- the deltas below are void"
              % len(bad), flush=True)
    deltas = []
    for r in range(1, ROUNDS + 1):
        got = {x["arm"]: x for x in timed if x["round"] == r}
        if len(got) != 2:
            continue
        if not all(x["load_clean"] for x in got.values()):
            print("round %d dropped: load %s" % (r, {k: v["load_max"] for k, v in got.items()}),
                  flush=True)
            continue
        o, n = got["off"]["s_per_design"], got["on"]["s_per_design"]
        frac = (o - n) / o
        deltas.append(dict(round=r, off=o, on=n, raw_delta=round(o - n, 3),
                           frac=round(frac, 5),
                           scaled_prize=round(frac * CLEAN_BASELINE, 3)))
        print("round %d  off %8.3f  on %8.3f  raw %+7.3f  frac %+.4f  -> %+7.3f s/design "
              "at the %.3f clean baseline"
              % (r, o, n, o - n, frac, frac * CLEAN_BASELINE, CLEAN_BASELINE), flush=True)

    summary = dict(n_rounds_used=len(deltas), rounds=deltas, bad=len(bad),
                   clean_baseline=CLEAN_BASELINE)
    if deltas:
        sc = [d["scaled_prize"] for d in deltas]
        med = statistics.median(sc)
        spread = (max(sc) - min(sc)) / abs(med) * 100 if med else float("nan")
        summary.update(median_scaled_prize=round(med, 3), spread_pct=round(spread, 2),
                       median_raw_delta=round(statistics.median(
                           [d["raw_delta"] for d in deltas]), 3),
                       predicted=PREDICTED, ratio_to_prediction=round(med / PREDICTED, 3))
        print("\n[p107] median %+.3f s/design over n=%d rounds, round-to-round spread %.2f %%"
              % (med, len(sc), spread), flush=True)
        print("[p107] cost model predicted %+.3f -- measured/predicted = %.3f"
              % (PREDICTED, med / PREDICTED), flush=True)
        if spread > 25.0:
            print("[p107] SPREAD TOO WIDE -- treat this as void and re-run on a quiet box",
                  flush=True)
    else:
        print("\n[p107] NO CLEAN ROUND -- the box never went quiet. Re-run, do not re-derive.",
              flush=True)
    d = json.loads(OUT.read_text())
    d["summary"] = summary
    OUT.write_text(json.dumps(d, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
