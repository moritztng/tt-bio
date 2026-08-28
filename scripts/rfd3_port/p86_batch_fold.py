#!/usr/bin/env python3
"""p86 -- SUPERSEDED BY p87_real_batch.py. ITS b=2 ARM NEVER BATCHED. Do not quote it.

At L=6051 design.py clamps effective_batch to
min(batch_size, 512, 8*3359**2 // 6051**2 = 2, _BATCH_SPEED_CAP = 1) = 1, so this script's
`batch_size=2` ran as TWO SEQUENTIAL batches of one. It summed the per-batch walls and divided by
the CIF count, so its "b=2 wins 1.135 s/design" is per-design fixed-cost amortisation across two
sequential designs and not batching at all. p87 raises the cap, asserts the batch count via
len(WALLS) (the sampler runs once per batch), and measures real b=2 at 94.201 s/design against
b=1 at 95.427.

--- original header below, kept for the record ---

p86 -- b=1 vs b=2 at the page fixture, on the fixed L1 chunk budget. Fold level.

Lever A has been closed three times and reopened once. p3 pass 1 root-caused the last closure:
`_pair_transition_chunk_h` sized its L1-resident chunk from width and hidden only, so at b=2 each
resident was 2x the budgeted footprint and the second one threw -- the exact byte figure in
perf/p76/batch_r4_qb2.log. `c32ba398` divides by the batch. This is the first b=2 measurement at
the page fixture on a build that fits.

The trap that burned pass 11 is compile-in-the-median: ttnn keys its program cache by shape, so
the b=2 arm compiles inside its first measured rep and ~77 s of compile once read as ~19 s/design
of batching penalty. Every batch shape therefore gets its OWN discarded warmup fold here, and the
per-shape warmup time is reported so a reader can see the compile that was excluded.

s/design is wall / num_designs, so the arms are directly comparable.

    env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p3 \
      PYTHONPATH=$PWD python3 -u scripts/rfd3_port/p86_batch_fold.py perf/p86/batch_fold.json 200 2
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p86/batch_fold.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
BATCHES = [1, 2]
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


def fold(specs, out_dir, batch):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=batch,
                           batch_size=batch, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return sum(WALLS), dig, len(cifs)


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p86] steps=%d card=%d batches=%s reps=%d  chunk_h at b=1/2: %d/%d"
          % (STEPS, CARD, BATCHES, REPS,
             M._pair_transition_chunk_h(704, 512, 685),
             M._pair_transition_chunk_h(704, 512, 685)), flush=True)
    print("[p86] baseline of record on this card %.3f s/design" % BASELINE, flush=True)

    rows, per = [], {}
    for b in BATCHES:
        try:
            w, dig, n = fold(specs, "/tmp/rfd3_p86_warm_b%d" % b, b)
            print("[p86] b=%d warmup %8.3f s (%d cif, %s) DISCARDED -- holds the compile"
                  % (b, w, n, dig[:20]), flush=True)
            rows.append(dict(batch=b, arm="warmup", wall_s=round(w, 3), n_cifs=n,
                             cif_sha256_16=dig, discarded=True))
        except Exception as e:
            print("[p86] b=%d warmup FAILED: %s" % (b, str(e)[:200]), flush=True)
            rows.append(dict(batch=b, arm="warmup", exc=str(e)[:400]))
            per[b] = None
            continue
        got = []
        for r in range(REPS):
            try:
                w, dig, n = fold(specs, "/tmp/rfd3_p86_b%d_%d" % (b, r), b)
                sd = w / max(1, n)
                got.append(sd)
                print("[p86] b=%d rep%d %8.3f s wall  %d cif  %8.3f s/design  %s"
                      % (b, r, w, n, sd, dig[:20]), flush=True)
                rows.append(dict(batch=b, arm="rep", rep=r, wall_s=round(w, 3), n_cifs=n,
                                 s_per_design=round(sd, 3), cif_sha256_16=dig))
            except Exception as e:
                print("[p86] b=%d rep%d FAILED: %s" % (b, r, str(e)[:200]), flush=True)
                rows.append(dict(batch=b, arm="rep", rep=r, exc=str(e)[:400]))
        per[b] = (statistics.median(got), min(got), max(got), len(got)) if got else None

    print("\n%-8s %14s %10s %10s %5s" % ("batch", "s/design med", "min", "max", "n"), flush=True)
    for b in BATCHES:
        if per.get(b):
            print("%-8d %14.3f %10.3f %10.3f %5d" % (b, *per[b]), flush=True)
        else:
            print("%-8d %14s" % (b, "FAILED"), flush=True)

    verdict = None
    if per.get(1) and per.get(2):
        aa = per[1][2] - per[1][1]
        delta = per[1][0] - per[2][0]
        print("\nb=1 spread (its own reps, the A/A floor) : %.3f s (%.2f %%)"
              % (aa, 100.0 * aa / per[1][0]), flush=True)
        print("b=2 minus b=1                            : %+.3f s/design (%.4gx)"
              % (-delta, per[1][0] / per[2][0]), flush=True)
        verdict = ("inside the A/A floor -- no effect resolvable"
                   if abs(delta) <= aa else
                   ("b=2 WINS %.3f s/design" % delta if delta > 0
                    else "b=2 LOSES %.3f s/design" % -delta))
        print("verdict: %s" % verdict, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "seed": SEED, "batches": BATCHES, "reps": REPS,
        "per_batch_s_per_design": {str(b): (per[b][0] if per.get(b) else None) for b in BATCHES},
        "per_batch_min": {str(b): (per[b][1] if per.get(b) else None) for b in BATCHES},
        "per_batch_max": {str(b): (per[b][2] if per.get(b) else None) for b in BATCHES},
        "baseline_of_record_s": BASELINE, "verdict": verdict,
        "host": os.uname().nodename, "card": CARD,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
