#!/usr/bin/env python3
"""p91 -- the collapsed process_z in the fold, and the token encoder's isolated/fold factor.

Two results from one run.

1. The prize. E4.3 is the rule this lineage now works under: an isolated per-call delta is a
   CEILING, never a prize, because `_PAIR_TRANSITION_L1` measures 20.816 s/design isolated and
   6.974 in the fold. p90 measures the collapsed process_z at 3.895 s/design isolated. Only a fold
   A/B says what it is worth.

2. The token encoder's calibration factor, which the board has never had. E6.9 carries the whole
   encoder item at "1.4 - 4.0 s/design, unknown factor" for exactly this reason. The known factors
   are 1.00 at the atom attention (p75), 1.86 at the DiT (p88) and 2.95 at the pair Transition
   (p85) -- all three different, so a global one would be wrong everywhere. This is the first
   fold/isolated pair inside the token encoder that is not a pair Transition.

Arm assertion (E6.7): both branches of the run_device fork count themselves in model.PZSTATS, so
the on arm must show shipped == 0 and the off arm collapsed == 0 at the same total. No guessed
expected count -- the off arm measures what the total is.
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p91/encoder_fold_ab.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "off,off,on,on").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
ISOLATED_DELTA_S = 3.895          # p90: 30.397 -> 10.923 ms/step over both recycles x 200 steps
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(out_dir, specs):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    M.PZSTATS[0] = M.PZSTATS[1] = 0
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return sum(WALLS), dig, len(cifs), len(WALLS), M.PZSTATS[0], M.PZSTATS[1]


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p91] steps=%d card=%d arms=%s  flag default=%s"
          % (STEPS, CARD, ",".join(ARMS), M._PROCESS_Z_COLLAPSE), flush=True)
    print("[p91] isolated ceiling %+.3f s/design (p90, 30.397 -> 10.923 ms/step)"
          % ISOLATED_DELTA_S, flush=True)

    # One discarded fold per DISTINCT arm, not one for the run: the collapsed path issues a
    # different program set, so the first `on` rep would otherwise carry its compile. That is the
    # trap that read ~77 s of compile as ~19 s/design of batching regression in pass 11.
    rows, warmed = [], set()
    for i, arm in enumerate(ARMS):
        M.set_process_z_collapse(arm == "on")
        if arm not in warmed:
            warmed.add(arm)
            w, dig, n, b, pc, ps = fold("/tmp/rfd3_p91_warm_%s" % arm, specs)
            print("[p91] warmup %-3s %8.3f s  %d cif  %d batch  collapsed=%d shipped=%d  DISCARDED"
                  % (arm, w, n, b, pc, ps), flush=True)
        w, dig, n, b, pc, ps = fold("/tmp/rfd3_p91_%d" % i, specs)
        ok = (ps == 0 and pc > 0) if arm == "on" else (pc == 0 and ps > 0)
        print("[p91] rep%d %-3s %9.3f s  %d cif  %d batch  collapsed=%-5d shipped=%-5d %s  %s"
              % (i, arm, w, n, b, pc, ps, "OK" if ok else "ARM WRONG", dig[:20]), flush=True)
        rows.append(dict(arm=arm, rep=i, s_per_design=round(w, 3), n_cifs=n, batches=b,
                         collapsed_calls=pc, shipped_calls=ps, arm_verified=ok,
                         cif_sha256_16=dig))
    M.set_process_z_collapse(False)

    tot_on = {r["collapsed_calls"] for r in rows if r["arm"] == "on"}
    tot_off = {r["shipped_calls"] for r in rows if r["arm"] == "off"}
    same_total = tot_on == tot_off
    print("\nencoder calls per design: on arm %s  off arm %s  same=%s"
          % (sorted(tot_on), sorted(tot_off), same_total), flush=True)

    def med(a):
        v = sorted(r["s_per_design"] for r in rows if r["arm"] == a)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    off, on = med("off"), med("on")
    aa = [r["s_per_design"] for r in rows if r["arm"] == ARMS[0]][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a}) for a in set(ARMS)}
    verified = all(r["arm_verified"] for r in rows)

    print("A/A control (two consecutive %s arms): %s -> %s s (%s %%)"
          % (ARMS[0], aa, None if aa_spread is None else round(aa_spread, 3),
             None if aa_spread is None else round(100.0 * aa_spread / aa[0], 3)), flush=True)
    print("off (shipped concat+slice+rms_norm+linear) median %9.3f  [%s, %s] n=%d" % off, flush=True)
    print("on  (collapsed, 3 ops)                    median %9.3f  [%s, %s] n=%d" % on, flush=True)
    ratio = None
    if on[0] and off[0]:
        d = off[0] - on[0]
        ratio = ISOLATED_DELTA_S / d if d else None
        print("\nfold delta            %+.3f s/design" % d, flush=True)
        print("isolated ceiling      %+.3f s/design" % ISOLATED_DELTA_S, flush=True)
        print("token encoder isolated / fold  %s" % (None if ratio is None else round(ratio, 3)),
              flush=True)
        print("known factors: atom 1.00 (p75), DiT 1.86 (p88), pair Transition 2.95 (p85)",
              flush=True)
        if aa_spread:
            print("delta / A-A spread    %.1fx" % (abs(d) / aa_spread), flush=True)
    print("\narms all verified: %s\ndigests %s" % (verified, digs), flush=True)
    print("NOT bit-exact by construction (one fp32 accumulation becomes two bf16-rounded halves),"
          " so the digests are EXPECTED to differ between arms; both are quoted.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "arms": ARMS, "seed": SEED,
        "flag": "RFD3_PROCESS_Z_COLLAPSE", "arms_all_verified": verified,
        "encoder_calls_on": sorted(tot_on), "encoder_calls_off": sorted(tot_off),
        "same_total_calls": same_total,
        "off_median_s": off[0], "on_median_s": on[0],
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "fold_delta_s": None if not (on[0] and off[0]) else round(off[0] - on[0], 3),
        "isolated_delta_s": ISOLATED_DELTA_S,
        "encoder_isolated_over_fold": None if ratio is None else round(ratio, 3),
        "digests": digs, "host": os.uname().nodename, "card": CARD,
        "bit_exact": False,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
