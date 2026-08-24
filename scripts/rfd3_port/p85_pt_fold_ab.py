#!/usr/bin/env python3
"""p85 -- `_PAIR_TRANSITION_L1` fold A/B on the CURRENT tree, and the baseline re-verify.

Two jobs in one hold.

1. Settle the calibration question p84 raised badly. p84 measured the lever at 20.816 s/design
   isolated and compared it to the 6.974 s/design fold delta that landed it -- but that fold A/B
   ran on a tree whose baseline was 116.122 s/design, and today's is 94.262. Comparing a current
   isolated number against a fold delta from an older tree is not a calibration pair, so p84's
   "isolated over-prices 3x" is not established. The clean pair already in the tree says the
   opposite at the atom site: perf/p75/softmax_fold_ab.json predicted -4.523 and measured -4.527,
   a ratio of 1.001, same tree and same fixture. This run produces the pair-Transition equivalent.

2. Re-verify 94.262. The `on` arm IS the shipped default, so its median is the baseline every
   ratio in this task is quoted against, and no pass in this lineage has measured it. The A/A
   control is the two consecutive `on` arms this opens with, which is also the contamination
   detector.

Arms `on,on,off,off,on`: A/A on the default first, then the lever off twice, then one more
default to catch drift across the hold.

    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p3 \
      PYTHONPATH=$PWD python3 -u scripts/rfd3_port/p85_pt_fold_ab.py \
          perf/p85/pt_fold_ab.json 200 on,on,off,off,on
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p85/pt_fold_ab.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "on,on,off,off,on").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
BASELINE_OF_RECORD = 94.262      # perf/p75/softmax_fold_ab.json, on_median_s, card 1
DIGEST_OF_RECORD = "5295e526ebd0b757"
ISOLATED_DELTA_S = 20.816        # p84, 7.8423 + 18.1771 ms/call x 4 calls each x 200 steps
OLD_FOLD_DELTA_S = 6.974         # the A/B that landed it, on a 116.122 s/design tree
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "3"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(specs, out_dir):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return WALLS[0], dig, len(cifs)


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p85] steps=%d card=%d arms=%s" % (STEPS, CARD, ",".join(ARMS)), flush=True)
    print("[p85] baseline of record %.3f s/design (digest %s)"
          % (BASELINE_OF_RECORD, DIGEST_OF_RECORD), flush=True)
    print("[p85] the lever off-minus-on: %.3f isolated (p84) vs %.3f fold on the 116.122 tree"
          % (ISOLATED_DELTA_S, OLD_FOLD_DELTA_S), flush=True)

    M._PAIR_TRANSITION_L1 = True
    s, dig, n = fold(specs, "/tmp/rfd3_p85_warm")
    print("[p85] warmup fold %.3f s (%s), discarded" % (s, dig[:20]), flush=True)
    warm = round(s, 3)

    rows = []
    for i, arm in enumerate(ARMS):
        M._PAIR_TRANSITION_L1 = (arm == "on")
        s, dig, n = fold(specs, "/tmp/rfd3_p85_%d" % i)
        rows.append({"arm": arm, "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n})
        print("[p85] rep%d %-3s %9.3f s  %d cif  %s" % (i, arm, s, n, dig[:20]), flush=True)

    def med(name):
        v = sorted(r["sampler_s"] for r in rows if r["arm"] == name)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    on = med("on")
    off = med("off")
    aa = [r["sampler_s"] for r in rows if r["arm"] == "on"][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a}) for a in set(ARMS)}

    print("\nA/A control (the two consecutive on arms): %s -> spread %s s (%s %%)"
          % (aa, None if aa_spread is None else round(aa_spread, 3),
             None if aa_spread is None else round(100.0 * aa_spread / aa[0], 3)), flush=True)
    print("on  (shipped default) median %9.3f s  [%s, %s]  n=%d" % on, flush=True)
    if off[0] is not None:
        print("off (lever disabled)  median %9.3f s  [%s, %s]  n=%d" % off, flush=True)
        delta = off[0] - on[0]
        print("\nlever worth, MEASURED IN THE FOLD  : %+.3f s/design (%.4gx)"
              % (delta, off[0] / on[0]), flush=True)
        print("  p84 isolated said                : %+.3f s/design" % ISOLATED_DELTA_S, flush=True)
        print("  the A/B that landed it said      : %+.3f s/design (on a 116.122 tree)"
              % OLD_FOLD_DELTA_S, flush=True)
        print("  isolated / fold                  :  %.3fx" % (ISOLATED_DELTA_S / delta),
              flush=True)
    print("\nbaseline re-verify: on median %.3f vs %.3f of record  (%+.3f s, %+.2f %%)"
          % (on[0], BASELINE_OF_RECORD, on[0] - BASELINE_OF_RECORD,
             100.0 * (on[0] - BASELINE_OF_RECORD) / BASELINE_OF_RECORD), flush=True)
    print("digests %s  (record %s)" % (digs, DIGEST_OF_RECORD), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "seed": SEED, "arms": ARMS,
        "flag": "RFD3_PAIR_TRANSITION_L1", "discarded_warmup_s": warm,
        "on_median_s": on[0], "on_min_s": on[1], "on_max_s": on[2], "on_n": on[3],
        "off_median_s": off[0], "off_min_s": off[1], "off_max_s": off[2], "off_n": off[3],
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "fold_delta_s": None if off[0] is None else round(off[0] - on[0], 3),
        "isolated_delta_s": ISOLATED_DELTA_S, "old_fold_delta_s": OLD_FOLD_DELTA_S,
        "isolated_over_fold": None if off[0] is None
        else round(ISOLATED_DELTA_S / (off[0] - on[0]), 3),
        "baseline_of_record_s": BASELINE_OF_RECORD,
        "baseline_delta_s": round(on[0] - BASELINE_OF_RECORD, 3),
        "digests": digs, "digest_of_record": DIGEST_OF_RECORD,
        "host": os.uname().nodename, "card": CARD,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
