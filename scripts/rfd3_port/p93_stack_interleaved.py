#!/usr/bin/env python3
"""p93 -- do the two fold-measured prizes add up? Interleaved, with the load recorded.

p92 asked the same question and its timing was void: it ran all-base, then all-b2, then all-both
on a box whose load average went 0.13 -> 21 during the run, so the arms are ordered by contamination
rather than by lever. Within-arm spread came out 5.9 % (b2) and 8.2 % (both) against this lineage's
0.006-0.144 % A/A bar, and b=2 read +3.382 s/design SLOWER than base where E6.3 measured it 1.226
FASTER. Nothing in that table is a number.

Two changes make this harness say so itself instead of needing the inference:

1. **Interleave.** One round is base(1 design), b2(2), both(2). Deltas are taken WITHIN a round, so
   a load ramp lands in the round-to-round spread instead of in the lever. This is the playbook's
   rule 3, which p92 broke.
2. **Record /proc/loadavg around every fold** and void any fold above LOAD_BAR. benchlock only
   checks the load once, at acquire; a co-tenant that starts two minutes later is invisible to it
   and p92 had no evidence of its own contamination in its artifact.

The levers, each fold-measured alone on this host:

    process_z collapse   -3.208 s/design   p91, card 1
    batching b=2         -1.226 s/design   p87, card 3

There is a specific reason to doubt they add: batching amortises per-op cost, and the collapse
deletes three of the six ops at the site with the largest per-call byte traffic in the token
encoder. Both are paid out of the same per-op budget.

Arm assertions, both of them the two-sided form this lineage arrived at:
`RFD3Sampler.sample` runs once per batch, so len(WALLS) IS the batch count (E6.2); and
`model.PZSTATS` counts both branches of the process_z fork, so the collapsed arm must report
shipped == 0 at the same total the shipped arm reports (E7.2).
"""
import hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p93/stack.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2          # timed rounds, after one warm round
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
L_ATOMS = 6051
LOAD_BAR = 3.0             # max 1-min loadavg tolerated around a fold; benchlock's gate is 2.0
PZ_ALONE = 3.208           # perf/p91, card 1, this tree
B2_ALONE = 1.226           # perf/p87, card 3
BASE_P91 = 94.087          # perf/p91 off-arm median, card 1, 200 steps
DIG_SHIPPED = "5295e526ebd0b757"    # batch-1 digest of record, design 1
DIG_COLLAPSED = "f66aebd19d25caac"  # p91's collapsed digest, design 1
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

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


def clamp(batch_size, cap):
    return min(batch_size, rfd3_design._BATCH_DESIGN_CEILING,
               rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L_ATOMS * L_ATOMS),
               cap if L_ATOMS > rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS else batch_size)


# label, num_designs, batch_size, speed_cap, collapse
ARMS = [("base", 1, 1, 1, False),
        ("b2",   2, 2, 2, False),
        ("both", 2, 2, 2, True)]


def fold(label, r, specs, num_designs, batch_size):
    out_dir = "/tmp/rfd3_p93_%s_%d" % (label, r)
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    M.PZSTATS[0] = M.PZSTATS[1] = 0
    l0 = load1()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=num_designs,
                           batch_size=batch_size, verbose=False)
    l1 = load1()
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return dict(wall_s=round(sum(WALLS), 3), n_cifs=len(cifs), batches=len(WALLS),
                collapsed_calls=M.PZSTATS[0], shipped_calls=M.PZSTATS[1],
                cif_sha256_16=dig, load_before=l0, load_after=l1,
                load_max=round(max(l0, l1), 2))


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p93] steps=%d card=%d rounds=%d (+1 warm)  shipped clamp at batch_size=2 -> %d"
          % (STEPS, CARD, ROUNDS, clamp(2, rfd3_design._BATCH_SPEED_CAP)), flush=True)
    print("[p93] single-lever folds: process_z %.3f (p91), b=2 %.3f (p87) -> additive %.3f"
          % (PZ_ALONE, B2_ALONE, PZ_ALONE + B2_ALONE), flush=True)
    rows = []
    try:
        for r in range(ROUNDS + 1):
            print("\n=== round %d%s  loadavg %.2f ==="
                  % (r, " (WARM, discarded)" if r == 0 else "", load1()), flush=True)
            for label, nd, bs, cap, coll in ARMS:
                rfd3_design._BATCH_SPEED_CAP = cap
                M.set_process_z_collapse(coll)
                res = fold(label, r, specs, nd, bs)
                spd = res["wall_s"] / max(1, res["n_cifs"])
                arm_ok = (res["batches"] == 1 and res["n_cifs"] == nd and
                          ((res["shipped_calls"] == 0 and res["collapsed_calls"] > 0) if coll else
                           (res["collapsed_calls"] == 0 and res["shipped_calls"] > 0)))
                want = DIG_COLLAPSED if coll else DIG_SHIPPED
                dig_ok = res["cif_sha256_16"].split("|")[0] == want
                clean = res["load_max"] <= LOAD_BAR
                print("[p93] r%d %-5s %9.3f s/design  %d cif %d batch  coll=%-4d ship=%-4d"
                      "  load<=%5.2f  %s %s %s  %s"
                      % (r, label, spd, res["n_cifs"], res["batches"], res["collapsed_calls"],
                         res["shipped_calls"], res["load_max"],
                         "ARM-OK" if arm_ok else "ARM-WRONG",
                         "DIG-OK" if dig_ok else "DIG-DRIFT",
                         "CLEAN" if clean else "LOADED-VOID",
                         res["cif_sha256_16"][:37]), flush=True)
                res.update(arm=label, round=r, s_per_design=round(spd, 3), arm_verified=arm_ok,
                           digest_matches_record=dig_ok, load_clean=clean, warm=(r == 0))
                rows.append(res)
    finally:
        rfd3_design._BATCH_SPEED_CAP = 1
        M.set_process_z_collapse(False)

    timed = [x for x in rows if not x["warm"]]
    clean = [x for x in timed if x["load_clean"]]
    print("\n" + "=" * 78, flush=True)
    print("timed folds %d, of which load-clean %d (bar: 1-min loadavg <= %.1f)"
          % (len(timed), len(clean), LOAD_BAR), flush=True)
    print("arms all verified: %s   digests all match record: %s"
          % (all(x["arm_verified"] for x in rows),
             all(x["digest_matches_record"] for x in rows)), flush=True)

    out = dict(rows=rows, num_timesteps=STEPS, rounds=ROUNDS, seed=SEED, card=CARD,
               host=os.uname().nodename, load_bar=LOAD_BAR, base_p91_200step=BASE_P91,
               pz_alone=PZ_ALONE, b2_alone=B2_ALONE,
               arms_all_verified=all(x["arm_verified"] for x in rows),
               digests_all_match=all(x["digest_matches_record"] for x in rows),
               n_timed=len(timed), n_clean=len(clean))

    # Dump the raw rows before any summary arithmetic runs. p93's first run lost 40 min
    # of card time to a format-string bug in the summary, every fold already folded.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    # Per-round deltas: the point of interleaving. A round with only some arms clean is dropped
    # whole, because a delta across a load boundary is exactly what p92 produced.
    per_round = []
    for r in range(1, ROUNDS + 1):
        got = {x["arm"]: x for x in clean if x["round"] == r}
        if len(got) != len(ARMS):
            print("round %d dropped: only %s clean" % (r, sorted(got)), flush=True)
            continue
        b, t2, bo = (got["base"]["s_per_design"], got["b2"]["s_per_design"],
                     got["both"]["s_per_design"])
        per_round.append(dict(round=r, base=b, b2=t2, both=bo,
                              d_b2=round(b - t2, 3), d_stack=round(b - bo, 3),
                              d_pz_given_b2=round(t2 - bo, 3)))
        print("round %d  base %8.3f  b2 %8.3f (%+.3f)  both %8.3f (%+.3f)  pz|b2 %+.3f"
              % (r, b, t2, t2 - b, bo, bo - b, bo - t2), flush=True)
    out["per_round"] = per_round

    if per_round:
        med = lambda k: statistics.median(x[k] for x in per_round)          # noqa: E731
        d_b2, d_stack, d_pz_b2 = med("d_b2"), med("d_stack"), med("d_pz_given_b2")
        base_med = statistics.median(x["base"] for x in per_round)
        spread = (max(x["base"] for x in per_round) - min(x["base"] for x in per_round))
        add = PZ_ALONE + d_b2
        print("\nbase median            %9.3f s/design   (p91 gave %.3f, round spread %.3f = %.3f %%)"
              % (base_med, BASE_P91, spread, 100 * spread / base_med), flush=True)
        print("b=2 alone, this run    %+9.3f s/design   (p87 card 3 gave %+.3f)"
              % (d_b2, B2_ALONE), flush=True)
        print("stack, measured        %+9.3f s/design" % d_stack, flush=True)
        print("stack, if additive     %+9.3f s/design   (p91's %.3f + this run's b=2 %.3f)"
              % (add, PZ_ALONE, d_b2), flush=True)
        print("ADDITIVITY             %9.3f of the sum  (%+.3f s/design %s)"
              % (d_stack / add if add else float("nan"), d_stack - add,
                 "lost to overlap" if d_stack < add else "gained"), flush=True)
        print("process_z given b=2    %+9.3f  against %+.3f alone (%.2fx)"
              % (d_pz_b2, PZ_ALONE, d_pz_b2 / PZ_ALONE), flush=True)
        out.update(base_median_s=round(base_med, 3), base_round_spread_s=round(spread, 3),
                   delta_b2_s=round(d_b2, 3), delta_stack_s=round(d_stack, 3),
                   delta_pz_given_b2_s=round(d_pz_b2, 3),
                   additive_prediction_s=round(add, 3),
                   additivity_fraction=round(d_stack / add, 4) if add else None)
    else:
        print("\nNO CLEAN ROUND -- no verdict. Re-run on a quiet box.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
