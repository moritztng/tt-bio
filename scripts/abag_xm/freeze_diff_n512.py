#!/usr/bin/env python3
"""Freeze gate: adding the 512 rung must not move a published pre-512 number.

    python3 freeze_diff_n512.py ~/abag_xm/deepn/analysis_curves.pre-n512.json \
                                /tmp/curves_n512.json

The runbook's earlier wording -- "diff the new curves against the snapshot, the rung-256
rows must be unchanged" -- is not runnable as stated, because a whole-file diff always
fires. Two report blocks move for structural reasons that have nothing to do with pooling,
and a gate that cannot tell those from a real regression gets rubber-stamped.

Measured on the real main() over identical 64/128/256 pools, adding a 512 rung on a subset
of targets (probe: 60 targets at 64/128/256, 45 also at 512):

  INVARIANT, and enforced here
    <model>[N]                  per-rung ladder rows. curve_points() groups strictly by
                                rung, so a row depends only on its own (target, N) keys.
                                Confirmed SAME at N=64/128/256.
    <model>__pairwise_gain_ci   each pair is paired on its own lo/hi intersection, so a
                                new rung cannot disturb an existing pair. Confirmed SAME
                                for 64->128 and 128->256. This is the stop-rule
                                comparator, so it is the one that most needs freezing.
    n16_ark_models              and with it the whole N=16 rung. The ark is opt-in on
                                DEEPN_N16_ARK=1, so a re-run that forgets the variable
                                silently reassembles N=16 from native folds only and this
                                gate fails on the [N=16] row, the 16->50 pair and this key
                                together. Three failures, one missing env var. Export it
                                for any run whose output is compared against a snapshot
                                that had it.

  EXPECTED TO MOVE, reported but never failed on
    <model>__paired_ci          common set is the intersection over ALL rungs
                                (all((t, n) in pools for n in ns)), so a 512 rung narrows
                                it and every rung's CI restates on the smaller set.
                                Probe: 60 -> 45 targets, N=256 oracle_mean +0.0021.
    <model>__deep               cap = max powered rung, so top_rung 256 -> 512 and
                                within_fold_common re-derives at depth D=512 over the
                                targets that reach it. Probe: depth 256 -> 512, common
                                curve at m=256 +0.0080.

Both movers are correct restatements on their new target sets, not regressions. The old
values stay valid for the old set; cite the set size with the number.

Exit 1 on any invariant mismatch.
"""
import json
import pathlib
import sys

MOVER_SUFFIXES = ("__paired_ci", "__deep")
GAIN_SUFFIX = "__pairwise_gain_ci"      # frozen per pair, in its own pass below
TARGET_RUNG = "512"
# Cost annotations on a gain pair. Pass 36 repriced this metric on purpose (it mixed an
# intensive numerator with an extensive denominator, so two pairs on different panels were
# not comparable) and priced the step by the samples it adds rather than the chunks the
# window happened to re-fold. Both are deliberate restatements of a published number, so
# they are reported and never failed. The scientific payload beside them -- gain_ci,
# common_targets, degenerate, doublings -- stays hard-frozen. Freezing the whole pair
# blindly makes every pre-512 pair fail on a value the campaign meant to correct, and a
# gate that always fails gets rubber-stamped, which is worse than no gate.
COST_KEYS = ("marginal_oracle_per_1000cs", "cost_h_per_target", "step_samples",
             "s_per_sample_hi", "s_per_sample_lo_same_panel", "cost_basis_targets")
# cost_basis_targets is the denominator of cost_h_per_target, so it belongs in the list
# above and not in the frozen payload. Leaving it out made every pre-512 pair fail on
# `null -> 160`, which is the pass-36 repricing arriving where the snapshot had no cost
# basis at all. That fired on all four models at once and is exactly the always-red gate
# the comment above warns about.
# The snapshot already carries a sparse 512 row for boltz2 (6 targets) and opendde-abag
# (5), the p28 remnant that seeded this rung's chunks 4-7. Filling it out is the whole
# point of the campaign, so 512 is the one rung that must NOT be frozen. Its old
# oracle_mean is a 5-6 target number and is not comparable to the 256 row -- cite the
# target count whenever the pre-campaign 512 value is quoted.


def load(p):
    return json.loads(pathlib.Path(p).expanduser().read_text())


def fmt(v):
    s = json.dumps(v, sort_keys=True)
    return s if len(s) <= 160 else s[:157] + "..."


def _model_of(key):
    for suf in MOVER_SUFFIXES + (GAIN_SUFFIX,):
        if key.endswith(suf):
            return key[:-len(suf)]
    return key


def _models_in(report):
    return {k for k, v in report.items()
            if isinstance(v, dict) and not k.endswith(MOVER_SUFFIXES)
            and not k.endswith(GAIN_SUFFIX) and k != "n16_ark_models"}


def _frozen_part(pair):
    """A gain pair minus its cost annotations, with gain_ci reduced to nothing here.

    gain_ci is compared metric by metric by the caller so a newly added metric (`gap`)
    is an addition rather than a mismatch.
    """
    return {k: v for k, v in pair.items() if k not in COST_KEYS and k != "gain_ci"}


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0])
        print("usage: freeze_diff_n512.py <pre-n512.json> <curves_n512.json>")
        return 2
    old, new = load(argv[1]), load(argv[2])

    fails, moved, grew, gone, checked = [], [], [], [], 0
    repriced, added = [], []

    # Step 8 runs the analysis one model at a time, so a per-model report legitimately
    # carries no blocks for the other three. Scope the freeze to the models the report
    # actually covers instead of failing on three quarters of the snapshot; without this
    # the gate can only ever pass on an all-model run, which the runbook never asked for.
    present = _models_in(new)
    skipped = sorted(_models_in(old) - present)
    if not present:
        print("FREEZE GATE FAIL: the new report carries no model blocks at all.")
        return 1

    for key in sorted(old):
        if _model_of(key) in skipped:
            continue
        if key.endswith(MOVER_SUFFIXES):
            if key not in new:
                # e.g. protenix-v2__paired_ci: its all-rung intersection is already
                # only 3 targets (the sparse 500 rung), so a 512 rung can empty it
                # and the block stops being emitted at all.
                gone.append(key)
            elif old[key] != new[key]:
                moved.append(key)
            continue
        if key.endswith(GAIN_SUFFIX):
            continue
        if key not in new:
            fails.append((key, "present in the snapshot, absent from the new report", ""))
            continue
        if key == "n16_ark_models":
            # A list of the models whose N=16 rung was restated with ARK DockQ. On a
            # per-model run it is the snapshot's list intersected with that one model.
            checked += 1
            want = [m for m in old[key] if m in present]
            if want != new[key]:
                fails.append((key, fmt(want), fmt(new[key])))
            continue
        if not isinstance(old[key], dict):
            checked += 1
            if old[key] != new[key]:
                fails.append((key, fmt(old[key]), fmt(new[key])))
            continue
        # per-rung ladder table: every frozen rung row must survive byte-identical
        for rung in sorted(old[key], key=int):
            if rung == TARGET_RUNG:
                a, b = old[key][rung], new[key].get(rung)
                if b is not None and a != b:
                    grew.append("%s[N=512] n_targets %d -> %d, oracle_mean %.4f -> %.4f"
                                % (key, a["n_targets"], b["n_targets"],
                                   a["oracle_mean"], b["oracle_mean"]))
                continue
            checked += 1
            if rung not in new[key]:
                fails.append(("%s[N=%s]" % (key, rung), "frozen row", "MISSING"))
            elif old[key][rung] != new[key][rung]:
                a, b = old[key][rung], new[key][rung]
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                fails.append(("%s[N=%s]" % (key, rung),
                              fmt({k: a.get(k) for k in diff}),
                              fmt({k: b.get(k) for k in diff})))

    # pairwise gain CIs: freeze the pairs that already existed, allow new ones
    for key in sorted(k for k in old if k.endswith(GAIN_SUFFIX)
                      and _model_of(k) not in skipped):
        for pair in sorted(old[key]):
            checked += 1
            tag = "%s[%s]" % (key, pair)
            if pair not in new.get(key, {}):
                fails.append((tag, "frozen pair", "MISSING"))
                continue
            a, b = old[key][pair], new[key][pair]
            if _frozen_part(a) != _frozen_part(b):
                d = sorted(k for k in set(_frozen_part(a)) | set(_frozen_part(b))
                           if a.get(k) != b.get(k))
                fails.append((tag, fmt({k: a.get(k) for k in d}),
                              fmt({k: b.get(k) for k in d})))
            # gain_ci metric by metric: a frozen metric must not move, a new one
            # (`gap`, added pass 37) is an addition. A dropped metric still fails,
            # because .get returns None and None never equals a CI triple.
            ga, gb = a.get("gain_ci") or {}, b.get("gain_ci") or {}
            for m in sorted(ga):
                if ga[m] != gb.get(m):
                    fails.append(("%s.gain_ci[%s]" % (tag, m),
                                  fmt(ga[m]), fmt(gb.get(m, "MISSING"))))
            for m in sorted(set(gb) - set(ga)):
                added.append("%s.gain_ci[%s] = %s" % (tag, m, fmt(gb[m])))
            for k in COST_KEYS:
                if a.get(k) != b.get(k):
                    repriced.append("%s %s: %s -> %s"
                                    % (tag, k, a.get(k, "absent"), b.get(k, "absent")))

    new_rungs = sorted({r for k, v in new.items()
                        if not k.endswith(MOVER_SUFFIXES) and not k.endswith(GAIN_SUFFIX)
                        and isinstance(v, dict) and isinstance(old.get(k), dict)
                        for r in set(v) - set(old[k])}, key=int)
    new_pairs = sorted({p for k in new if k.endswith("__pairwise_gain_ci")
                        for p in set(new[k]) - set(old.get(k, {}))})

    print("invariant assertions checked: %d" % checked)
    print("models in this report: %s" % ", ".join(sorted(present)))
    if skipped:
        print("models NOT in this report, so not frozen here: %s" % ", ".join(skipped))
        print("  a per-model run freezes only its own model. The all-model run is what"
              " covers the panel.")
    print("new rungs: %s" % (", ".join(new_rungs) or "none"))
    print("new gain pairs: %s" % (", ".join(new_pairs) or "none"))
    # POWER_MIN guard, per model: 512 is not a "new" rung for boltz2/opendde-abag (the
    # sparse p28 remnant is already in the snapshot), so keying this off new_rungs would
    # never fire for exactly the two models that most need it.
    for model in sorted(k for k in new if not k.endswith(MOVER_SUFFIXES)
                        and not k.endswith(GAIN_SUFFIX) and isinstance(new[k], dict)):
        row = new[model].get(TARGET_RUNG)
        if not isinstance(row, dict):
            continue
        if "256->512" not in new.get(model + GAIN_SUFFIX, {}):
            print("  WARNING: %s has a 512 rung over %d targets but no 256->512 gain"
                  " pair. Under POWER_MIN=50 complete targets the rung is dropped from"
                  " the stop-rule chain silently -- run n512_power_census.py."
                  % (model, row.get("n_targets", -1)))

    if added:
        print("\nmetrics added since the snapshot (additions, not movements):")
        for line in added:
            print("  %s" % line)
    if repriced:
        print("\ncost annotations restated by pass 36's repricing (reported, never failed;"
              "\nthe scientific payload beside them is frozen and was checked):")
        for line in repriced:
            print("  %s" % line)
    if grew:
        print("\nthe 512 rung filling out (this is the campaign's own output):")
        for line in grew:
            print("  %s" % line)
    if gone:
        print("\nblocks no longer emitted (all-rung intersection went empty, not a regression):")
        for key in gone:
            print("  %s (was over %s targets)"
                  % (key, old[key].get("common_targets", "?")))
    if moved:
        print("\nexpected-to-move blocks (restated on a narrower target set, not a regression):")
        for key in moved:
            extra = ""
            if key.endswith("__paired_ci"):
                extra = " common_targets %s -> %s" % (old[key].get("common_targets"),
                                                      new[key].get("common_targets"))
            elif key.endswith("__deep"):
                ow = (old[key].get("within_fold_common") or {})
                nw = (new[key].get("within_fold_common") or {})
                extra = " top_rung %s -> %s, common depth %s -> %s over %s -> %s targets" % (
                    old[key].get("top_rung"), new[key].get("top_rung"),
                    ow.get("depth"), nw.get("depth"),
                    ow.get("n_targets"), nw.get("n_targets"))
            print("  %s%s" % (key, extra))

    if fails:
        print("\nFREEZE GATE FAIL: %d frozen value(s) moved." % len(fails))
        for key, a, b in fails:
            print("  %s\n    was: %s\n    now: %s" % (key, a, b))
        print("\nA mismatch here is a pooling or assembly regression, not a restatement."
              " Do not report any 512 number until it is explained.")
        return 1
    print("\nFREEZE GATE PASS: every pre-512 ladder row and gain pair is unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
