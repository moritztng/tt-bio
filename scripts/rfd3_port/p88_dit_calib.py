#!/usr/bin/env python3
"""p88 -- the DiT site's isolated/fold calibration, from the L2 dense-bias kernel.

Why this is the highest-value screen left. The board's largest unmeasured item is head_dim
48 -> 64 at 5.80 s/design, and that figure is an ISOLATED number at the DiT. E5.2 showed the
isolated/fold correction factor is per-site -- 1.00 at the atom attention, 2.95 at the pair
Transition -- so 5.80 could be 5.80 or it could be ~2, and nothing in this lineage can say which
because the DiT has no calibration pair.

`RFD3_DENSE_BIAS_FUSED` (L2, `dense_fused_scores_bias_fp32`) is the clean candidate: it is a
landed, default-on, bit-exact DiT lever with an isolated screen already on disk. From
perf/p72/dense_kernel_probe.json at the page fixture's I=685 / n_key=704 row,

    shipped 0.5093 ms/call -> fused 0.2038, delta 0.3055 ms/call
    36 calls per step (the census's [token DiT] add-scores-plus-bias line) x 200 steps
    ISOLATED PREDICTION: 2.200 s/design

Measure it in the fold and the ratio is the DiT's factor.

Arm assertion, per E6.7's lesson: rfd3_bias.DSTATS[0] counts every call the fused kernel actually
serves, so the on arm must show 36*200 = 7200 and the off arm exactly 0. A silent arm is what made
p86 wrong; every rep below carries its own counter.
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio import rfd3_bias                                             # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p88/dit_calib.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "on,on,off,off").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
ISOLATED_DELTA_S = 2.200         # 0.3055 ms/call x 36 calls x 200 steps
EXPECT_FUSED_CALLS = 36 * STEPS
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
    rfd3_bias.DSTATS[0] = 0
    rfd3_bias.DSTATS[1] = 0
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return sum(WALLS), dig, len(cifs), rfd3_bias.DSTATS[0], rfd3_bias.DSTATS[1]


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p88] steps=%d card=%d arms=%s" % (STEPS, CARD, ",".join(ARMS)), flush=True)
    print("[p88] isolated prediction %+.3f s/design (p72, 0.5093 -> 0.2038 ms/call x %d calls)"
          % (ISOLATED_DELTA_S, 36), flush=True)
    print("[p88] arm assertion: fused-call counter must be %d on, 0 off"
          % EXPECT_FUSED_CALLS, flush=True)

    rfd3_bias.set_dense_enabled(True)
    w, dig, n, f, ie = fold(specs, "/tmp/rfd3_p88_warm")
    print("[p88] warmup %8.3f s  %d cif  fused=%d  DISCARDED" % (w, n, f), flush=True)

    rows = []
    for i, arm in enumerate(ARMS):
        rfd3_bias.set_dense_enabled(arm == "on")
        w, dig, n, f, ie = fold(specs, "/tmp/rfd3_p88_%d" % i)
        want = EXPECT_FUSED_CALLS if arm == "on" else 0
        ok = (f == want)
        print("[p88] rep%d %-3s %9.3f s  %d cif  fused=%-6d %s  %s"
              % (i, arm, w, n, f, "OK" if ok else "ARM WRONG (want %d)" % want, dig[:20]),
              flush=True)
        rows.append(dict(arm=arm, rep=i, s_per_design=round(w, 3), n_cifs=n,
                         fused_calls=f, fused_expected=want, arm_verified=ok,
                         ineligible=ie, cif_sha256_16=dig))
    rfd3_bias.set_dense_enabled(True)

    def med(a):
        v = sorted(r["s_per_design"] for r in rows if r["arm"] == a)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    on, off = med("on"), med("off")
    aa = [r["s_per_design"] for r in rows if r["arm"] == "on"][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a}) for a in set(ARMS)}
    verified = all(r["arm_verified"] for r in rows)

    print("\nA/A control (two consecutive on arms): %s -> %s s (%s %%)"
          % (aa, None if aa_spread is None else round(aa_spread, 3),
             None if aa_spread is None else round(100.0 * aa_spread / aa[0], 3)), flush=True)
    print("on  (fused, shipped default) median %9.3f  [%s, %s] n=%d" % on, flush=True)
    print("off (three ops)              median %9.3f  [%s, %s] n=%d" % off, flush=True)
    ratio = None
    if on[0] and off[0]:
        d = off[0] - on[0]
        ratio = ISOLATED_DELTA_S / d if d else None
        print("\nfold delta            %+.3f s/design" % d, flush=True)
        print("isolated prediction   %+.3f s/design" % ISOLATED_DELTA_S, flush=True)
        print("DiT isolated / fold    %s" % (None if ratio is None else round(ratio, 3)),
              flush=True)
        print("\nfor comparison: atom site 1.00 (p75), pair Transition 2.95 (p85)", flush=True)
        if ratio:
            print("=> head_dim's isolated 5.80 s/design is really ~%.2f s/design in the fold"
                  % (5.80 / ratio), flush=True)
    print("arms all verified: %s   digests %s" % (verified, digs), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "arms": ARMS, "seed": SEED,
        "flag": "RFD3_DENSE_BIAS_FUSED", "arms_all_verified": verified,
        "on_median_s": on[0], "off_median_s": off[0],
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "fold_delta_s": None if not (on[0] and off[0]) else round(off[0] - on[0], 3),
        "isolated_delta_s": ISOLATED_DELTA_S,
        "dit_isolated_over_fold": None if ratio is None else round(ratio, 3),
        "head_dim_isolated_s": 5.80,
        "head_dim_fold_implied_s": None if not ratio else round(5.80 / ratio, 3),
        "digests": digs, "host": os.uname().nodename, "card": CARD,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
