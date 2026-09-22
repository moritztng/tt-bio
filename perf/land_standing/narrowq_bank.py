#!/usr/bin/env python3
"""Assemble the narrow-q rf3 896 aa cell from interleaved pairs banked across quiet windows.

WHY THE CELL IS BANKED. The four-rep single-session cell needs ~19 minutes under the benchlock
ceiling. qb2 handed this row 6 minutes on 2026-09-22 before an of3t campaign took the box, and
two of eight legs survived the ceiling cut. A pair is ~5.5 minutes. See narrowq_pair.sh.

WHAT THIS DOES NOT RELAX. Three bars, all applied per leg before anything is pooled:

  1. CONTENTION, scored as FOREIGN cpu. A leg is cut if any sample inside its own
     [t_start, t_end] shows more than 200 % cpu belonging to processes that are neither this
     harness nor known always-on infra. This bar replaced a loadavg1 <= 2.00 ceiling on
     2026-09-22, and the reason is worth keeping: loadavg1 INCLUDES the measured fold. The
     pre-flight tested the same 2.00 against a box with no fold running, so it admitted windows
     the scorer was arithmetically certain to refuse. On the one clean cell the fold's own
     contribution is ~1.2 (pre-flight 0.13, during max 1.33), and loadavg1 is a 60 s EWMA that
     keeps climbing while the process runs, so the second leg of a pair inherits the first's
     tail and clears 2.00 on an empty box. The 23:03:50Z pair was cut at peaks of 2.65 and 3.68
     whose entire content, read off the trace, was this worktree's own `tt_bio.main predict`
     and its multiprocessing children at 276 % and 229 %. A contention bar that counts the
     measurement as contention cuts the legs it exists to protect.

     THE REPLACEMENT IS CONTROLLED, because a looser-looking bar proposed by the row it
     unblocks is worth nothing on its own. The retake's reps 2-4 had a real co-tenant (of3t
     stacking ref_grad.py, c64_score.py and model_scope.py from 22:11:40Z) and they carry
     484-1846 % foreign cpu; the four clean legs carry 75-100 %. The bar sits at 200 %, inside
     a gap of a factor of five, and still cuts every leg it cut before for a real reason.
     narrowq_foreign_score.py prints that control.
  2. CLOCK. A leg is cut unless its during-sampled AICLK carries an unbroken run at or above
     1200 MHz long enough to CONTAIN its timed fold -- clock_during.py's contract, imported from
     it rather than restated, and with the same sample-evidence bar.
  3. PAIR INTEGRITY. A rep counts only if BOTH its arms survive 1 and 2. Half a pair is not a
     measurement, it is one number.

Pooling pairs from separate windows can only ADD between-window variation to the A/A floor, so
the assembled cell faces a floor at least as wide as a single-session cell's. This is a stricter
test than the one it replaces, not a workaround for it.

ORDER BALANCE IS REQUIRED, NOT REPORTED. fold_ab_flip's own 768 aa negative control read +2.542 %
at 3.80x its A/A floor with a fixed arm order at a length where the lever provably cannot fire.
Every banked pair is rep 1, so a bank left to itself is all off-first and reproduces that
confound. This REFUSES a bank that carries only one order, and reports the two order classes'
deltas separately so a surviving position effect is visible rather than averaged away.
"""
import argparse, importlib.util, json, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FOREIGN_CEILING = 200.0  # percent cpu held by anything that is not this harness
MIN_MHZ = 1200.0
MIN_SPAN_FRAC = 0.5

spec = importlib.util.spec_from_file_location(
    "clock_during", ROOT / "perf/pvx_gate_land/clock_during.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

# one definition of "mine vs foreign", shared with the control script rather than restated here
_fs_spec = importlib.util.spec_from_file_location(
    "narrowq_foreign_score", ROOT / "perf/land_standing/narrowq_foreign_score.py")
fs = importlib.util.module_from_spec(_fs_spec)
_fs_spec.loader.exec_module(fs)


def read_trace(path, card):
    clocks, load = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "t" not in r:
            continue
        v = cd.mhz((r.get("aiclk") or {}).get(card))
        if v is not None:
            clocks.append((r["t"], v))
        load.append((r["t"], r["loadavg"][0]))
    return clocks, load


def judge(fold, clocks, load):
    """(kept, reason). Applies the ceiling then the clock contract, in that order."""
    t0, t1 = fold["t_start"], fold["t_end"]
    lo = [l for t, l in load if t0 <= t <= t1]
    if not lo:
        return False, "no loadavg samples inside the leg"
    foreign = fold.get("_foreign", 0.0)
    if foreign > FOREIGN_CEILING:
        return False, "foreign cpu peaked %.0f %% over the %.0f %% bar (%s)" % (
            foreign, FOREIGN_CEILING, fold.get("_who") or "unnamed")
    inside = [(t, v) for t, v in clocks if t0 <= t <= t1]
    sub = t1 - t0
    span = (inside[-1][0] - inside[0][0]) if len(inside) > 1 else 0.0
    if len(inside) < 2 or span < MIN_SPAN_FRAC * sub:
        return False, "%d clock samples spanning %.0fs of a %.0fs subprocess" % (
            len(inside), span, sub)
    ok_span, _ = cd.longest_ok_run(inside, MIN_MHZ)
    if ok_span < fold["runtime_s"]:
        return False, "longest run at or above %.0f MHz is %.0fs, cannot contain a %.1fs fold" % (
            MIN_MHZ, ok_span, fold["runtime_s"])
    return True, "foreign %.0f %%, loadavg1 max %.2f, %.0fs at or above %.0f MHz" % (
        fold.get("_foreign", 0.0), max(lo), ok_span, MIN_MHZ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "perf/land_standing/narrowq_bank_sources.json"))
    ap.add_argument("--card", default="0")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    print("narrow-q rf3 896 aa, banked cell. contention bar %.0f %% foreign cpu, clock "
          "contract at or above %.0f MHz.\n" % (FOREIGN_CEILING, MIN_MHZ))

    pairs = []
    for src in man["sources"]:
        art_p, tr_p = ROOT / src["artifact"], ROOT / src["trace"]
        if not art_p.exists() or not tr_p.exists():
            print("  MISSING %s -- skipped" % src["artifact"])
            continue
        art = json.load(open(art_p))
        clocks, load = read_trace(tr_p, a.card)
        raw = [json.loads(l) for l in open(tr_p) if l.strip()]
        raw = [r for r in raw if "t" in r]
        for cell in art["cells"]:
            first = cell.get("first_arm", src.get("first_arm", "off"))
            by_rep = {}
            for f in cell.get("folds", []):
                _s = fs.score_leg(f, raw)
                f["_foreign"], f["_who"] = _s["foreign"], _s["who"]
                kept, why = judge(f, clocks, load)
                print("  %-34s %-3s rep%d %7.1fs  %-4s %s" % (
                    src["artifact"].split("/")[-1], f["arm"], f["rep"], f["runtime_s"],
                    "KEEP" if kept else "CUT", why))
                if kept:
                    by_rep.setdefault(f["rep"], {})[f["arm"]] = f
            for rep, d in sorted(by_rep.items()):
                if "off" in d and "on" in d:
                    # the order this pair actually ran, read off the artifact's slots
                    order = "off" if d["off"]["slot"] < d["on"]["slot"] else "on"
                    pairs.append({"src": src["artifact"].split("/")[-1], "rep": rep,
                                  "first": order, "off": d["off"]["runtime_s"],
                                  "on": d["on"]["runtime_s"]})

    print("\ncomplete pairs surviving both bars: %d" % len(pairs))
    for p in pairs:
        print("  %-34s rep%d  %s-first  lever-on %.1f  shipped %.1f  delta %+.2f s" % (
            p["src"], p["rep"], p["first"], p["off"], p["on"], p["on"] - p["off"]))

    if len(pairs) < 2:
        print("\nREFUSED: fewer than two complete pairs, so there is no A/A floor.")
        return 3
    classes = {p["first"] for p in pairs}
    if classes != {"off", "on"}:
        only = classes.pop()
        print("\nREFUSED: every surviving pair is %s-first. A single-order bank reproduces the "
              "position effect the 768 aa control measured (+2.542 %% at 3.80x its floor at a "
              "length where the lever cannot fire). Bank a pair with --first-arm %s."
              % (only, "on" if only == "off" else "off"))
        return 3

    off = [p["off"] for p in pairs]
    on = [p["on"] for p in pairs]
    omed, nmed = statistics.median(off), statistics.median(on)
    aa_off = 100.0 * (max(off) - min(off)) / omed
    aa_on = 100.0 * (max(on) - min(on)) / nmed
    aa = max(aa_off, aa_on)
    ab = 100.0 * (nmed - omed) / omed

    print("\n  lever ON  (TT_BIO_TRIATT_NARROW_Q_FALLBACK=1): %s  median %.2f s" % (off, omed))
    print("  shipped   (flag off on main)                 : %s  median %.2f s" % (on, nmed))
    print("  A/A floor %+.3f %%  (worst arm; lever-on %+.3f %%, shipped %+.3f %%)"
          % (aa, aa_off, aa_on))
    print("  A/B       %+.3f %%   speedup %.4fx   delta %+.4f s" % (ab, nmed / omed, nmed - omed))
    print("  effect / floor  %.2fx" % (ab / aa) if aa else "  effect / floor  inf")
    for cl in ("off", "on"):
        sel = [p for p in pairs if p["first"] == cl]
        if sel:
            print("  %s-first pairs (n=%d): delta median %+.3f s"
                  % (cl, len(sel), statistics.median([p["on"] - p["off"] for p in sel])))
    sep = max(off) < min(on)
    print("  separation: max lever-on %.1f vs min shipped %.1f -> %s"
          % (max(off), min(on), "NO OVERLAP" if sep else "OVERLAP"))

    ok = ab > aa and sep
    print("\nVERDICT: " + ("PASS -- outside its own A/A floor, arms do not overlap, both orders "
                           "present. Reportable." if ok else
                           "REFUSED -- inside the floor or the arms overlap."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
