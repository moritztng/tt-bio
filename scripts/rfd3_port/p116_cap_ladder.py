#!/usr/bin/env python3
"""p116 -- re-run design.py's `_BATCH_SPEED_CAP` table on today's tree, end to end.

Why this exists. The cap is justified by a table in its own comment, measured end to end at
200 timesteps:

    atoms      b=1       b=2      b=4      b=8
     2299   24.971         -   22.253   21.885   b=8 wins 1.141x
     2952   36.625         -        -   34.108   b=8 wins 1.074x
     3844   59.967    64.890   59.975        -   b=1 wins 1.082x
     6051  144.044   167.189        -        -   b=1 wins 1.161x

That table was taken when b=1 at 6051 atoms cost 144.044 s/design. It is now ~95. The L6a/L6b
bias kernels, the pair-Transition L1 residency, the tile-aligned concat and the matmul
calibration all landed after it, and p3 re-ran the 6051 row and got the opposite sign. Any
comment that quotes absolute seconds carries its own expiry date.

Descends from p87_real_batch.py, which is the harness that is airtight about the arm it ran, and
keeps the two load-bearing properties that record earned:

  * ``len(WALLS) == expected_batches`` asserted on every rep. RFD3Sampler.sample is called once
    per batch, so that is the only honest check that the arm ran the batch it claims. p3 had to
    retract a whole result for want of it.
  * a discarded warmup per arm. Each batch shape compiles its own programs, so an unwarmed arm
    measures the compiler.

What is new here, and why:

  1. ``--spec``/``--out``/``--reps`` on argv, so one script serves every rung of the ladder.
  2. No hardcoded atom count. L is read from the host featurizer and printed; the arms are
     derived from it. A hardcoded L in a size ladder is a wrong-variable gate waiting to happen.
  3. Every arm folds the SAME number of designs, ``N = max(arms)``, with the same per-design
     seeds. So design index k is the same draw in every arm, and the per-index digests are
     directly comparable.
  4. Per-design digests, not design 0's. A batch that corrupts design 1 passes a design-0 check.
  5. loadavg recorded per fold. This box is shared and no run lock is taken (the host lock gates
     on load once at acquire then proceeds anyway, the co-tenants that matter do not take it, and
     it is host-scoped with no fairness so a waiter starves). What stands in for it: the delta is
     taken inside one process, the loadavg is in the artifact, and the b=1 arm's own max-min
     spread is the noise floor -- a rung whose b=1 spread exceeds 1 % is void, not slow.

The timed quantity is ``sum(WALLS)``, the sampler wall only. Host featurisation, TokenInitializer
and CIF writing are all outside it, on purpose: the cap decides sampler batching and nothing
else, and including a per-fold fixed cost would hand the batched arm a free amortisation that has
nothing to do with batching (that is the exact error p86 made).
"""
import argparse, hashlib, json, os, pathlib, statistics, sys, time
import torch                                                             # noqa: F401
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402
from tt_bio.rfd3.input import InputSpecification                          # noqa: E402

CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
CARD = int(os.environ.get("TT_VISIBLE_DEVICES") or -1)   # empty = host-only check, not card 0

WALLS = []
CALLS = []
_sample = RFD3Sampler.sample


class _CallTimer:
    """Times every ``diffusion_module`` call. ``sample`` makes one per step (two with CFG), so a
    fold yields 200-400 samples of the same quantity instead of the one that ``sum(WALLS)`` is.

    That matters because this box is shared and cannot be quieted. Contention is *one-sided* --
    a co-tenant can only ever make a call slower, never faster -- so a low quantile over the
    fold's calls recovers the uncontended cost, while a sum over the whole fold integrates every
    stall it met. The R2 rung measured a 54.6 % b=1 A/A spread at loadavg 2.7-20.2 against a 1 %
    void threshold; the spread is the box, not the arm, and no number of 200-step reps fixes it.

    This does NOT move the pre-registered rule (state/rfd3-b8-to-4x-p4.md 3.3), which stays on
    the arm medians of ``sum(WALLS)``. It is the readout for the case that rule voids.
    """
    __slots__ = ("m",)

    def __init__(self, m):
        self.m = m

    def __getattr__(self, k):
        return getattr(self.m, k)

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        r = self.m(*a, **k)
        CALLS.append(time.perf_counter() - t0)
        return r


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, _CallTimer(dm), n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def q10(xs):
    """The 10th percentile, nearest-rank. No interpolation, so it is always an observed call."""
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, int(0.10 * len(ys))))]


def clamp(batch_size, cap, L):
    """design.py's own function, so the prediction is the code and not a paraphrase. `cap` is
    passed for readability at the call sites; the module global is what the function reads, and
    the arms set it, so they must agree."""
    assert cap == rfd3_design._BATCH_SPEED_CAP, (cap, rfd3_design._BATCH_SPEED_CAP)
    return rfd3_design.effective_design_batch(batch_size, L)


def host_atom_count(specs):
    """L, from the host featurizer alone -- no device, no assumption. run_design takes L from
    the TokenInitializer's Q_L_init.shape[0]; coord0 is f['motif_pos'] at (L, 3), so the
    featurizer already knows it."""
    from tt_bio.rfd3.featurize import featurize
    Ls = {}
    for spec_id, d in specs.items():
        spec = InputSpecification.from_dict(d)
        f = featurize(spec.input, spec)
        Ls[spec_id] = int(f["motif_pos"].shape[0])
    return Ls


def fold(specs, out_dir, num_designs, batch_size, steps):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    CALLS.clear()
    la = os.getloadavg()[0]
    res = rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                                 num_timesteps=steps, seed=SEED, num_designs=num_designs,
                                 batch_size=batch_size, verbose=False)
    digs = {}
    for r in sorted(res, key=lambda r: (r.spec_id, r.design_idx)):
        p = pathlib.Path(r.out_path)
        digs["%s#%d" % (r.spec_id, r.design_idx)] = (
            hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "NO CIF")
    n_atoms = sorted({r.n_atoms for r in res})
    # len(WALLS) is the number of batches the sampler actually ran.
    calls = list(CALLS)
    # calls/design is fixed per arm: a batch-b module call carries b designs, so it falls as 1/b.
    # q10 x calls/design is therefore the per-design cost with the box's stalls taken out.
    cs = dict(n=len(calls), q10=q10(calls) if calls else None,
              med=statistics.median(calls) if calls else None,
              covered=(sum(calls) / sum(WALLS)) if WALLS and sum(WALLS) else None)
    cs["robust_s_per_design"] = (cs["q10"] * cs["n"] / max(1, len(res))) if calls else None
    return (sum(WALLS), digs, len(res), len(WALLS), n_atoms, max(la, os.getloadavg()[0]), cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warm-steps", type=int, default=0,
                    help="timesteps for the discarded warmup (0 = same as --steps). Every step "
                         "runs the same shapes, so a short warmup compiles the same programs; "
                         "and the q10 readout is immune to a stray slow call anyway.")
    ap.add_argument("--max-arms", type=int, default=4)
    ap.add_argument("--arms", default="", help="comma list, overrides the derived ladder")
    args = ap.parse_args()
    OUT = pathlib.Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    specs = json.loads(pathlib.Path(args.spec).read_text())

    t0 = time.perf_counter()
    Ls = host_atom_count(specs)
    L = max(Ls.values())
    print("[p116] spec %s -> atoms %s (featurised host-side in %.1f s)"
          % (args.spec, Ls, time.perf_counter() - t0), flush=True)
    budget = max(1, rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L * L))
    print("[p116] L=%d  atom-pair budget admits %d  shipped cap %d (binds above %d atoms -> %s)"
          % (L, budget, rfd3_design._BATCH_SPEED_CAP,
             rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS,
             "BINDS" if L > rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS else "does not bind"),
          flush=True)
    print("[p116] shipped clamp at batch_size=8 -> effective %d"
          % clamp(8, rfd3_design._BATCH_SPEED_CAP, L), flush=True)

    if args.arms:
        arms = [int(x) for x in args.arms.split(",")]
    else:
        arms = [1] + [b for b in (2, 4, 6) if b <= budget]
    arms = arms[:args.max_arms]
    N = max(arms)
    print("[p116] arms %s, %d designs per fold in every arm, %d reps + 1 discarded warmup"
          % (arms, N, args.reps), flush=True)
    if L <= rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS:
        print("[p116] WARNING: the cap does not bind at this size. This rung cannot inform it.",
              flush=True)

    rows, per, digests, per_round, per_robust = [], {}, {}, {}, {}

    # Which arms the shipped clamp actually admits at this L. Resolved before any fold, so every
    # round below runs the same arm set.
    plan_arms = []
    for b in arms:
        label = "b%d" % b
        rfd3_design._BATCH_SPEED_CAP = b
        eff = clamp(b, b, L)
        exp_batches = (N + eff - 1) // eff
        if eff != b:
            print("=== arm %s SKIPPED: the clamp refuses it (effective %d != %d) ==="
                  % (label, eff, b), flush=True)
            rows.append(dict(arm=label, stage="skipped", effective_batch=eff, requested=b))
            per[label] = None
            continue
        print("=== arm %s: num_designs=%d batch_size=%d cap=%d -> effective_batch=%d, "
              "expect %d sampler call(s) ===" % (label, N, b, b, eff, exp_batches), flush=True)
        plan_arms.append([b, label, eff, exp_batches])

    # Warm every arm before any arm is timed. Each batch shape compiles its own programs, so an
    # unwarmed arm measures the compiler.
    for a in list(plan_arms):
        b, label, eff, exp_batches = a
        rfd3_design._BATCH_SPEED_CAP = b
        try:
            ws = args.warm_steps or args.steps
            w, dg, n, nb, na, la, cs = fold(specs, "/tmp/rfd3_p116_warm_%s" % label, N, b, ws)
            print("  warmup %-4s %8.3f s  %d step  %d design  %d batch(es)  atoms %s  load %5.1f"
                  "  DISCARDED" % (label, w, ws, n, nb, na, la), flush=True)
        except Exception as e:
            print("  warmup %s FAILED: %s" % (label, str(e)[:240]), flush=True)
            rows.append(dict(arm=label, stage="warmup", exc=str(e)[:600]))
            per[label] = None
            plan_arms.remove(a)

    # Interleave the reps round-robin instead of running each arm as a block. The R4 rung was
    # blocked and its b=1 reps ran at loadavg 19.0/19.0/16.5 while b=2 ran at 19.65/19.65/24.11,
    # so a drift in the box load aliased straight into the arm delta. Round-robin puts every arm
    # under the same drift, and it makes the per-round ratio a paired statistic.
    for r in range(args.reps):
        for b, label, eff, exp_batches in plan_arms:
            rfd3_design._BATCH_SPEED_CAP = b
            try:
                w, dg, n, nb, na, la, cs = fold(specs, "/tmp/rfd3_p116_%s_%d" % (label, r), N, b,
                                                args.steps)
                ok = (nb == exp_batches and n == N)
                sd = w / max(1, n)
                per_round.setdefault(label, {})[r] = sd
                per_robust.setdefault(label, []).append(cs["robust_s_per_design"])
                digests.setdefault(label, []).append(dg)
                print("  round%d %-4s %9.3f s sampler wall  %d design  %d batch(es) %s  "
                      "%8.3f s/design  robust %8.3f  (%d calls, q10 %.4f s, %.1f %% covered)  "
                      "load %5.1f  %s"
                      % (r, label, w, n, nb, "OK" if ok else "ARM WRONG", sd,
                         cs["robust_s_per_design"], cs["n"], cs["q10"], 100 * cs["covered"], la,
                         "|".join(dg[k] for k in sorted(dg))), flush=True)
                rows.append(dict(arm=label, batch=b, rep=r, round=r, sampler_wall_s=round(w, 3),
                                 n_designs=n, batches=nb, batches_expected=exp_batches,
                                 arm_verified=ok, s_per_design=round(sd, 3), digests=dg,
                                 effective_batch=eff, loadavg=round(la, 2), L_atoms=L,
                                 n_atoms_seen=na, calls=cs["n"],
                                 call_q10_s=round(cs["q10"], 5),
                                 call_median_s=round(cs["med"], 5),
                                 calls_covered_frac=round(cs["covered"], 4),
                                 robust_s_per_design=round(cs["robust_s_per_design"], 3)))
                assert ok, ("ARM WRONG: %d batches for %d designs at effective %d"
                            % (nb, n, eff))
            except Exception as e:
                print("  round%d %s FAILED: %s" % (r, label, str(e)[:240]), flush=True)
                rows.append(dict(arm=label, rep=r, round=r, exc=str(e)[:600]))
        OUT.write_text(json.dumps({"rows": rows, "partial": True}, indent=2) + "\n")

    for b, label, eff, exp_batches in plan_arms:
        got = sorted(per_round.get(label, {}).values())
        per[label] = ((statistics.median(got), min(got), max(got), len(got)) if got else None)
    rfd3_design._BATCH_SPEED_CAP = 1

    print("\n%-8s %14s %10s %10s %5s %9s" % ("arm", "s/design med", "min", "max", "n", "vs b=1"),
          flush=True)
    s1 = per["b1"][0] if per.get("b1") else None
    for b in arms:
        label = "b%d" % b
        if per.get(label):
            rel = ("%8.4fx" % (s1 / per[label][0])) if s1 else "-"
            print("%-8s %14.3f %10.3f %10.3f %5d %9s" % (label, *per[label], rel), flush=True)
        else:
            print("%-8s %14s" % (label, "no result"), flush=True)

    # the pre-registered decision rule (state/rfd3-b8-to-4x-p4.md 3.3)
    verdict, aa_frac, best, paired = None, None, None, {}
    rob, r_aa, robust_verdict = {}, None, None
    if s1:
        aa_frac = (per["b1"][2] - per["b1"][1]) / s1
        thresh = max(0.005, 3.0 * aa_frac)
        cand = [(per["b%d" % b][0], b) for b in arms if b != 1 and per.get("b%d" % b)]
        print("\nb=1 own-rep spread (A/A floor): %.3f s (%.3f %%) -> raise threshold %.3f %%"
              % (per["b1"][2] - per["b1"][1], 100 * aa_frac, 100 * thresh), flush=True)
        ref = digests.get("b1", [{}])[0]
        for _, b in sorted(cand):
            lbl = "b%d" % b
            same = all(d == ref for d in digests.get(lbl, []))
            print("  %s digests identical to b=1 at every design index: %s" % (lbl, same),
                  flush=True)
        if aa_frac > 0.01:
            verdict = "VOID: b=1 A/A spread %.3f %% > 1 %%, re-run the rung" % (100 * aa_frac)
        elif cand:
            bs, b = min(cand)
            gain = (s1 - bs) / s1
            lbl = "b%d" % b
            same = all(d == ref for d in digests.get(lbl, []))
            best = dict(batch=b, s_per_design=round(bs, 3), gain_frac=round(gain, 5),
                        digest_identical=same, threshold_frac=round(thresh, 5))
            if gain > thresh and same:
                verdict = ("RAISE to b=%d: %.3f vs %.3f s/design, %+.3f %% > %.3f %% threshold, "
                           "bit-exact" % (b, bs, s1, 100 * gain, 100 * thresh))
            elif gain > thresh and not same:
                verdict = ("NO: b=%d is %.3f %% faster but the digests differ -- not bit-exact"
                           % (b, 100 * gain))
            else:
                verdict = ("CAP STANDS at this size: best batched arm b=%d is %+.3f %% against a "
                           "%.3f %% threshold" % (b, 100 * gain, 100 * thresh))
        else:
            verdict = "no batched arm produced a result"
        # Paired diagnostic ONLY. The verdict above is the rule pre-registered in
        # state/rfd3-b8-to-4x-p4.md 3.3, computed from the arm medians, unchanged. This block
        # reports the per-round ratio, which cancels the load drift common to a round, so a rung
        # the unpaired rule voids can still be read for direction.
        for _, b in sorted(cand):
            lbl = "b%d" % b
            rs = [per_round[lbl][r] / per_round["b1"][r]
                  for r in sorted(per_round.get(lbl, {}))
                  if r in per_round.get("b1", {})]
            if rs:
                paired[lbl] = dict(n_rounds=len(rs), ratios=[round(x, 5) for x in rs],
                                   median_ratio=round(statistics.median(rs), 5),
                                   worst=round(max(rs), 5), best=round(min(rs), 5))
                print("  paired %s/b1 per round: %s -> median %.4fx (diagnostic, not the rule)"
                      % (lbl, " ".join("%.4f" % x for x in rs), statistics.median(rs)),
                      flush=True)
        # The load-robust readout. Same folds, same arms; the only difference is that the cost
        # comes from the q10 of the fold's module calls instead of the sum of its walls. Reported
        # with its own A/A floor so it cannot be believed on faith: if the b=1 spread on THIS
        # statistic is not far tighter than on sum(WALLS), the noise was never contention and
        # this readout is worth nothing either. That check is the point.
        rob = {k: dict(med=statistics.median(v), lo=min(v), hi=max(v), n=len(v))
               for k, v in per_robust.items() if v and all(x is not None for x in v)}
        if rob.get("b1"):
            r1 = rob["b1"]["med"]
            r_aa = (rob["b1"]["hi"] - rob["b1"]["lo"]) / r1
            print("\n%-8s %14s %10s %10s %9s" % ("arm", "robust s/des", "min", "max", "vs b=1"),
                  flush=True)
            for b in arms:
                lbl = "b%d" % b
                if rob.get(lbl):
                    print("%-8s %14.3f %10.3f %10.3f %8.4fx"
                          % (lbl, rob[lbl]["med"], rob[lbl]["lo"], rob[lbl]["hi"],
                             r1 / rob[lbl]["med"]), flush=True)
            print("  robust b=1 A/A floor: %.3f %% (sum-of-walls floor was %.3f %%) -> %s"
                  % (100 * r_aa, 100 * aa_frac,
                     "usable" if r_aa <= 0.01 else "STILL VOID, the noise is not contention"),
                  flush=True)
            rob_best = sorted((rob[l]["med"], b) for b in arms if b != 1
                              for l in ["b%d" % b] if rob.get(l))
            if rob_best and r_aa <= 0.01:
                bs, bb = rob_best[0]
                g = (r1 - bs) / r1
                th = max(0.005, 3.0 * r_aa)
                same = all(d == digests["b1"][0] for d in digests.get("b%d" % bb, []))
                robust_verdict = ("%s b=%d: robust %+.3f %% against a %.3f %% threshold%s"
                                  % ("RAISE" if (g > th and same) else "CAP STANDS", bb,
                                     100 * g, 100 * th,
                                     "" if same else ", AND THE DIGESTS DIFFER"))
                print("  robust readout: %s" % robust_verdict, flush=True)
            else:
                robust_verdict = None
        else:
            rob, r_aa, robust_verdict = {}, None, None

        print("\nverdict: %s" % verdict, flush=True)
        print("design.py's table at this size said: %s" % (
            {2299: "b=8 wins 1.141x", 2952: "b=8 wins 1.074x", 3844: "b=1 wins 1.082x",
             6051: "b=1 wins 1.161x"}.get(L, "not measured at L=%d" % L)), flush=True)

    OUT.write_text(json.dumps({
        "spec": args.spec, "rows": rows, "num_timesteps": args.steps, "reps": args.reps,
        "L_atoms": L, "atoms_per_spec": Ls, "arms": arms, "designs_per_fold": N,
        "atom_pair_budget_admits": budget,
        "shipped_speed_cap": 1, "shipped_speed_cap_above_atoms":
            rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS,
        "shipped_effective_batch_at_8": clamp(8, 1, L),
        "per_arm_s_per_design": {("b%d" % b): (per["b%d" % b][0] if per.get("b%d" % b) else None)
                                 for b in arms},
        "per_arm_min": {("b%d" % b): (per["b%d" % b][1] if per.get("b%d" % b) else None)
                        for b in arms},
        "per_arm_max": {("b%d" % b): (per["b%d" % b][2] if per.get("b%d" % b) else None)
                        for b in arms},
        "b1_aa_spread_frac": (round(aa_frac, 5) if aa_frac is not None else None),
        "best_batched_arm": best, "verdict": verdict,
        "paired_round_ratios": paired,
        "robust_per_arm": {k: {kk: round(vv, 5) for kk, vv in v.items()}
                           for k, v in sorted(rob.items())},
        "robust_aa_spread_frac": (round(r_aa, 5) if r_aa is not None else None),
        "robust_verdict": robust_verdict,
        "per_round_s_per_design": {k: {str(r): round(v, 3)
                                       for r, v in sorted(d.items())}
                                   for k, d in sorted(per_round.items())},
        "host": os.uname().nodename, "card": CARD, "partial": False,
    }, indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
