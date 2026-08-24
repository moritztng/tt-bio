#!/usr/bin/env python3
"""p87 -- REAL b=2 at the page fixture, with the effective batch verified rather than assumed.

p86 is wrong and this replaces it. At L=6051 design.py clamps

    effective_batch = min(batch_size, 512,
                          _BATCH_ATOM_PAIR_BUDGET // L**2,   ->  8*3359^2 // 6051^2 = 2
                          _BATCH_SPEED_CAP if L > 2952)      ->  1

so `batch_size=2` runs as TWO SEQUENTIAL batches of one. p86 summed the per-batch walls and
divided by the CIF count, so its "b=2 wins 1.135 s/design" is per-design fixed-cost amortisation
across two sequential designs, not batching at all.

The verification p86 lacked, and the reason it was fooled: RFD3Sampler.sample is called ONCE PER
BATCH, so len(WALLS) is the batch count. num_designs=2 with a real batch of 2 calls it once;
clamped to 1 it calls it twice. That is an airtight check on the arm and it is asserted here.

_BATCH_SPEED_CAP exists for a measured reason. design.py's own table, end to end at 200 steps:

    atoms      b=1       b=2      b=4      b=8
     2299   24.971         -   22.253   21.885   b=8 wins 1.141x
     2952   36.625         -        -   34.108   b=8 wins 1.074x
     3844   59.967    64.890   59.975        -   b=1 wins 1.082x
     6051  144.044   167.189        -        -   b=1 wins 1.161x

Batching stops paying between 2952 and 3844 atoms. This run re-tests the 6051 row on today's
tree (that table was taken when b=1 was 144.044; it is now ~97), because p2 asked for the cap to
be re-decided only behind a measured b=2 number and the L1 chunk fix has landed since.

Mechanism to expect: the pair Transition's L1 chunk shrinks as the batch grows -- chunk_h is 64
at b=1, 47 at b=2, 11 at b=8 -- so batching buys amortisation and pays for it in op count, and
this model is per-op bound at these shapes.
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p87/real_batch.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
L_ATOMS = 6051
BASELINE = 96.523          # perf/p85/pt_fold_ab.json, this card, this tree
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def clamp(batch_size, cap):
    """design.py's own expression, so the prediction is the code and not a paraphrase."""
    return min(batch_size, rfd3_design._BATCH_DESIGN_CEILING,
               max(1, rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L_ATOMS * L_ATOMS)),
               cap if L_ATOMS > rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS else batch_size)


def fold(specs, out_dir, num_designs, batch_size):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=num_designs,
                           batch_size=batch_size, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    # len(WALLS) is the number of batches the sampler actually ran.
    return sum(WALLS), dig, len(cifs), len(WALLS)


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p87] L=%d  atom-pair budget admits %d  speed cap %d (binds above %d atoms)"
          % (L_ATOMS, rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L_ATOMS * L_ATOMS),
             rfd3_design._BATCH_SPEED_CAP, rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS), flush=True)
    print("[p87] shipped clamp at batch_size=2 -> effective %d ; at 8 -> %d"
          % (clamp(2, rfd3_design._BATCH_SPEED_CAP), clamp(8, rfd3_design._BATCH_SPEED_CAP)),
          flush=True)
    print("[p87] chunk_h: b=1 %d, b=2 %d, b=4 %d, b=8 %d"
          % tuple(M._pair_transition_chunk_h(b, 704, 512, 685) for b in (1, 2, 4, 8)), flush=True)

    # arm = (label, num_designs, batch_size, speed_cap, expected_batches)
    ARMS = [
        ("b1", 1, 1, 1, 1),
        ("b2_real", 2, 2, 2, 1),      # cap raised to 2, so one batch of two
    ]
    rows, per = [], {}
    for label, nd, bs, cap, exp_batches in ARMS:
        rfd3_design._BATCH_SPEED_CAP = cap
        eff = clamp(bs, cap)
        print("\n=== arm %s: num_designs=%d batch_size=%d cap=%d -> effective_batch=%d, "
              "expect %d sampler call(s) ===" % (label, nd, bs, cap, eff, exp_batches), flush=True)
        try:
            w, dig, n, nb = fold(specs, "/tmp/rfd3_p87_warm_%s" % label, nd, bs)
            print("  warmup %8.3f s  %d cif  %d batch(es)  DISCARDED" % (w, n, nb), flush=True)
        except Exception as e:
            print("  warmup FAILED: %s" % str(e)[:220], flush=True)
            rows.append(dict(arm=label, stage="warmup", exc=str(e)[:500]))
            per[label] = None
            continue
        got = []
        for r in range(REPS):
            try:
                w, dig, n, nb = fold(specs, "/tmp/rfd3_p87_%s_%d" % (label, r), nd, bs)
                ok = (nb == exp_batches)
                sd = w / max(1, n)
                got.append(sd)
                print("  rep%d %8.3f s wall  %d cif  %d batch(es) %s  %8.3f s/design  %s"
                      % (r, w, n, nb, "OK" if ok else "ARM WRONG", sd, dig[:20]), flush=True)
                rows.append(dict(arm=label, rep=r, wall_s=round(w, 3), n_cifs=n, batches=nb,
                                 batches_expected=exp_batches, arm_verified=ok,
                                 s_per_design=round(sd, 3), cif_sha256_16=dig,
                                 effective_batch=eff))
            except Exception as e:
                print("  rep%d FAILED: %s" % (r, str(e)[:220]), flush=True)
                rows.append(dict(arm=label, rep=r, exc=str(e)[:500]))
        per[label] = (statistics.median(got), min(got), max(got), len(got)) if got else None
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"rows": rows, "partial": True}, indent=2) + "\n")
    rfd3_design._BATCH_SPEED_CAP = 1

    print("\n%-10s %14s %10s %10s %5s" % ("arm", "s/design med", "min", "max", "n"), flush=True)
    for label, *_ in ARMS:
        if per.get(label):
            print("%-10s %14.3f %10.3f %10.3f %5d" % (label, *per[label]), flush=True)
        else:
            print("%-10s %14s" % (label, "FAILED"), flush=True)

    verdict = None
    if per.get("b1") and per.get("b2_real"):
        aa = per["b1"][2] - per["b1"][1]
        d = per["b2_real"][0] - per["b1"][0]
        print("\nb=1 own-rep spread (A/A floor) : %.3f s (%.2f %%)"
              % (aa, 100.0 * aa / per["b1"][0]), flush=True)
        print("real b=2 minus b=1             : %+.3f s/design (%.4gx)"
              % (d, per["b2_real"][0] / per["b1"][0]), flush=True)
        verdict = ("inside the A/A floor" if abs(d) <= aa
                   else ("b=1 WINS %.3f s/design -- the cap is right" % d if d > 0
                         else "b=2 WINS %.3f s/design -- the cap should be revisited" % -d))
        print("verdict: %s" % verdict, flush=True)
        print("design.py's table said b=1 wins by 1.161x at 6051 atoms (144.044 vs 167.189)",
              flush=True)

    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "reps": REPS, "L_atoms": L_ATOMS,
        "atom_pair_budget_admits": rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L_ATOMS * L_ATOMS),
        "shipped_speed_cap": 1, "shipped_effective_batch_at_8": clamp(8, 1),
        "per_arm_s_per_design": {k: (per[k][0] if per.get(k) else None) for k, *_ in ARMS},
        "per_arm_min": {k: (per[k][1] if per.get(k) else None) for k, *_ in ARMS},
        "per_arm_max": {k: (per[k][2] if per.get(k) else None) for k, *_ in ARMS},
        "baseline_of_record_s": BASELINE, "verdict": verdict,
        "designpy_table_6051": {"b1": 144.044, "b2": 167.189},
        "host": os.uname().nodename, "card": CARD, "partial": False,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
