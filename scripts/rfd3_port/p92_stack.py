#!/usr/bin/env python3
"""p92 -- do the two prizes this task has actually MEASURED add up?

The brief's rule is "measure the stack, never add the deltas", and seven passes in, this lineage
has never tested it. Two levers are now fold-measured on the same host:

    process_z collapse   -3.208 s/design   p91, card 1, today, A/A 0.144 %
    batching b=2         -1.226 s/design   E6.3 / p87, card 3

If they add, the pair is worth 4.434. There is a specific reason to doubt it: batching amortises
per-op cost, and the collapse DELETES three of the six ops at the site with the largest per-call
byte traffic in the token encoder. Both levers are being paid out of the same per-op budget, which
is exactly the shape of `perf-mechanism-label-expires-when-lever-removes-its-traffic`.

Every arm carries both assertions this lineage has learned to demand. RFD3Sampler.sample runs once
per batch, so len(WALLS) IS the batch count (E6.2 -- p86 was fooled by two sequential batches of
one looking like a batch of two). And model.PZSTATS counts both branches of the process_z fork, so
the collapsed arm must show shipped == 0 at the same total the shipped arm reports (E7.2).

b=2 needs _BATCH_SPEED_CAP monkeypatched to 2, exactly as p87 did: the shipped clamp is
min(batch_size, 512, _BATCH_ATOM_PAIR_BUDGET // L^2 = 2, _BATCH_SPEED_CAP = 1) = 1 at 6051 atoms.
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p92/stack.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
L_ATOMS = 6051
PZ_ALONE = 3.208           # perf/p91, card 1, this tree
B2_ALONE = 1.226           # perf/p87, card 3
BASE_P91 = 94.087          # perf/p91 off-arm median, card 1
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def clamp(batch_size, cap):
    return min(batch_size, rfd3_design._BATCH_DESIGN_CEILING,
               rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L_ATOMS * L_ATOMS),
               cap if L_ATOMS > rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS else batch_size)


def fold(out_dir, specs, num_designs, batch_size):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    M.PZSTATS[0] = M.PZSTATS[1] = 0
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=num_designs,
                           batch_size=batch_size, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return sum(WALLS), dig, len(cifs), len(WALLS), M.PZSTATS[0], M.PZSTATS[1]


# label, num_designs, batch_size, speed_cap, collapse, expected_batches, reps
ARMS = [
    ("base",  1, 1, 1, False, 1, 1),
    ("b2",    2, 2, 2, False, 1, 2),
    ("both",  2, 2, 2, True,  1, 2),
]


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p92] steps=%d card=%d  shipped clamp at batch_size=2 -> effective %d"
          % (STEPS, CARD, clamp(2, rfd3_design._BATCH_SPEED_CAP)), flush=True)
    print("[p92] single-lever fold deltas: process_z %.3f (p91, this card), b=2 %.3f (p87, card 3)"
          "  -> additive prediction %.3f" % (PZ_ALONE, B2_ALONE, PZ_ALONE + B2_ALONE), flush=True)
    rows = []
    try:
        for label, nd, bs, cap, coll, want_b, reps in ARMS:
            rfd3_design._BATCH_SPEED_CAP = cap
            M.set_process_z_collapse(coll)
            eff = clamp(bs, cap)
            print("\n=== arm %s: num_designs=%d batch_size=%d cap=%d -> effective_batch=%d,"
                  " collapse=%s ===" % (label, nd, bs, cap, eff, coll), flush=True)
            for r in range(reps + 1):
                w, dig, n, nb, pc, ps = fold("/tmp/rfd3_p92_%s_%d" % (label, r), specs, nd, bs)
                spd = w / max(1, n)
                batch_ok = (nb == want_b and n == nd)
                pz_ok = (ps == 0 and pc > 0) if coll else (pc == 0 and ps > 0)
                tag = "WARMUP-DISCARD" if r == 0 else "rep%d" % r
                print("[p92] %-14s %-5s %9.3f s wall  %d cif  %d batch  %9.3f s/design"
                      "  collapsed=%-5d shipped=%-5d  %s  %s"
                      % (tag, label, w, n, nb, spd, pc, ps,
                         "OK" if (batch_ok and pz_ok) else "ARM WRONG", dig[:20]), flush=True)
                if r:
                    rows.append(dict(arm=label, rep=r, wall_s=round(w, 3), n_cifs=n, batches=nb,
                                     s_per_design=round(spd, 3), collapsed_calls=pc,
                                     shipped_calls=ps, arm_verified=bool(batch_ok and pz_ok),
                                     cif_sha256_16=dig))
    finally:
        rfd3_design._BATCH_SPEED_CAP = 1
        M.set_process_z_collapse(False)

    def med(a):
        v = sorted(r["s_per_design"] for r in rows if r["arm"] == a)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    base, b2, both = med("base"), med("b2"), med("both")
    verified = all(r["arm_verified"] for r in rows)
    print("\n" + "=" * 78, flush=True)
    for lab, m in (("base  (b=1, shipped)", base), ("b2    (b=2, shipped process_z)", b2),
                   ("both  (b=2 + collapse)", both)):
        print("%-32s median %9.3f  [%s, %s] n=%d" % ((lab,) + m), flush=True)
    print("\ncross-process drift control: p91's off-arm median was %.3f, this arm %s"
          % (BASE_P91, None if base[0] is None else round(base[0], 3)), flush=True)
    out = dict(rows=rows, arms_all_verified=verified, num_timesteps=STEPS, seed=SEED,
               card=CARD, host=os.uname().nodename, base_p91=BASE_P91,
               pz_alone=PZ_ALONE, b2_alone=B2_ALONE,
               base_median_s=base[0], b2_median_s=b2[0], both_median_s=both[0])
    if base[0] and b2[0] and both[0]:
        d_b2 = base[0] - b2[0]
        d_both = base[0] - both[0]
        d_pz_given_b2 = b2[0] - both[0]
        add = PZ_ALONE + d_b2
        print("\nb=2 alone, this card        %+.3f s/design  (card 3 gave %+.3f)" % (d_b2, B2_ALONE),
              flush=True)
        print("stack, measured             %+.3f s/design" % d_both, flush=True)
        print("stack, if the deltas added  %+.3f s/design  (p91's %.3f + this card's b=2 %.3f)"
              % (add, PZ_ALONE, d_b2), flush=True)
        print("ADDITIVITY                  %.3f of the sum  (%+.3f s/design %s)"
              % (d_both / add if add else float("nan"), d_both - add,
                 "LOST to overlap" if d_both < add else "gained"), flush=True)
        print("process_z's delta with b=2 already on: %+.3f, against %+.3f alone (%.2fx)"
              % (d_pz_given_b2, PZ_ALONE,
                 d_pz_given_b2 / PZ_ALONE if PZ_ALONE else float("nan")), flush=True)
        out.update(delta_b2_s=round(d_b2, 3), delta_stack_s=round(d_both, 3),
                   delta_pz_given_b2_s=round(d_pz_given_b2, 3),
                   additive_prediction_s=round(add, 3),
                   additivity_fraction=round(d_both / add, 4) if add else None)
    print("\narms all verified: %s" % verified, flush=True)
    print("digests: %s" % {r["arm"]: r["cif_sha256_16"][:41] for r in rows}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
