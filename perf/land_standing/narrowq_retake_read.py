#!/usr/bin/env python3
"""The narrow-q cell, re-taken at the cadence its own instrument demanded.

Pass 17 measured +9.50 s for TT_BIO_TRIATT_NARROW_Q_FALLBACK on rf3 at 896 aa and then refused to
report it: clock_during.py proves the longest unthrottled run only from its first in-fold sample
to its last, so a 3 s cadence understated a 104.6 s fold's high-clock window by 0.6 s and the cell
could not be placed inside it. Nothing about the card was wrong; the sampler was too slow to prove
it. "tt-smi -s" costs 0.21 s on this box, so a 1 s cadence is affordable and that is what this
cell uses.

Two arms and a control, all on qb2 card 0, sibling card 1 idle:

  rf3 896 aa, four reps   the lever fires. The run-time census reads 0 of 1088 served on the
                          shipped path, fill_preconditions declining every call.
  rf3 768 aa, two reps    NEGATIVE CONTROL. 768 is a multiple of the production q_chunk, so the
                          policy function never reaches the fallback list and the arm must read
                          1.00x. An A/B that moves here is measuring something other than this
                          lever, and it is the only leg that can falsify the instrument rather
                          than the lever.

ARM NAMING, because it is inverted and silent errors live here. fold_ab_flip.py defines arm 'off'
as "the flag is SET" and arm 'on' as "shipped defaults". The shipped default for this flag is
False, so 'off' is the LEVER ON and 'on' is the SHIPPED path. This script prints the arms under
their meaning, not under the harness's label.
"""
import hashlib, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perf/land_standing/out"
CEILING = 2.00


def load_rows(artifact, jl_recs):
    cell = json.loads(Path(artifact).read_text())["cells"][0]
    if "folds" not in cell:
        print("  cell FAILED: %s" % cell.get("error"))
        return None, None
    rows = []
    for f in cell["folds"]:
        ins = [r for r in jl_recs if f["t_start"] <= r["t"] <= f["t_end"]]
        la = [r["loadavg"][0] for r in ins]
        clk = [float(str(c).strip()) for c in (r["aiclk"].get("0") for r in ins
               if isinstance(r["aiclk"], dict)) if c not in (None, "")]
        rows.append({"lever": "ON" if f["arm"] == "off" else "shipped",
                     "rep": f["rep"], "t": f["runtime_s"],
                     "load_max": max(la) if la else None,
                     "clk_min": min(clk) if clk else None,
                     "clk_max": max(clk) if clk else None,
                     "quiet": (max(la) <= CEILING) if la else False})
    return cell, rows


def report(name, artifact, jl_recs):
    print("\n=== %s ===" % name)
    if not Path(artifact).exists():
        print("  no artifact at %s" % artifact)
        return None
    cell, rows = load_rows(artifact, jl_recs)
    if rows is None:
        return None
    print("  lever     rep  runtime_s   load1 max   AICLK in-fold")
    for r in rows:
        print("  %-8s  %d   %8.2f   %9.2f   %s-%s %s"
              % (r["lever"], r["rep"], r["t"], r["load_max"] or -1,
                 r["clk_min"], r["clk_max"], "" if r["quiet"] else "  LOUD"))
    lev = [r["t"] for r in rows if r["lever"] == "ON"]
    shp = [r["t"] for r in rows if r["lever"] == "shipped"]
    aa_l = 100.0 * (max(lev) - min(lev)) / st.median(lev)
    aa_s = 100.0 * (max(shp) - min(shp)) / st.median(shp)
    aa = max(aa_l, aa_s)
    ab = 100.0 * (st.median(shp) - st.median(lev)) / st.median(lev)
    print("  paired, lever-on minus shipped:  %s"
          % "  ".join("rep%d %+.2f s" % (rep,
              next(r["t"] for r in rows if r["rep"] == rep and r["lever"] == "ON")
              - next(r["t"] for r in rows if r["rep"] == rep and r["lever"] == "shipped"))
              for rep in sorted({r["rep"] for r in rows})))
    print("  lever ON  median %.2f s  %s" % (st.median(lev), [round(x, 2) for x in lev]))
    print("  shipped   median %.2f s  %s" % (st.median(shp), [round(x, 2) for x in shp]))
    print("  A/A floor %.3f%% (worst arm)   A/B %+.3f%%   %.4fx   %+.2f s   effect/floor %.1fx"
          % (aa, ab, st.median(shp) / st.median(lev), st.median(shp) - st.median(lev),
             abs(ab) / aa if aa else float("inf")))
    sep = max(lev) < min(shp)
    print("  separation: max lever-on %.2f %s min shipped %.2f  -- %s"
          % (max(lev), "<" if sep else ">=", min(shp),
             "complete" if sep else "arms overlap"))
    return {"aa_pct": aa, "ab_pct": ab, "ratio": st.median(shp) / st.median(lev),
            "delta_s": st.median(shp) - st.median(lev), "separated": sep,
            "lever_med": st.median(lev), "shipped_med": st.median(shp)}


def digests(workdir, rung):
    """The accuracy leg, free: the A/B folds already wrote their structures. Same seed, same
    fixture, one flag apart, so any difference between the arms IS the lever's."""
    out = {}
    for d in sorted(Path(workdir).glob("out_rf3-%d-*" % rung)):
        cifs = sorted(d.rglob("*.cif"))
        if not cifs:
            continue
        h = hashlib.sha256()
        for c in cifs:
            h.update(c.read_bytes())
        out[d.name.replace("out_rf3-%d-" % rung, "")] = h.hexdigest()[:16]
    return out


def main() -> int:
    jl = OUT / "narrowq_retake_contention.jsonl"
    recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    print("contention samples: %d over %.0f s (cadence %.2f s)"
          % (len(recs), recs[-1]["t"] - recs[0]["t"],
             (recs[-1]["t"] - recs[0]["t"]) / max(1, len(recs) - 1)))
    r896 = report("rf3 896 aa -- the lever fires", OUT / "narrowq_retake_896_qb2c0.json", recs)
    r768 = report("rf3 768 aa -- NEGATIVE CONTROL, must read 1.00x",
                  OUT / "narrowq_retake_768_qb2c0.json", recs)

    print("\n=== accuracy: CIF sha256 per arm, same seed, one flag apart ===")
    for rung in (896, 768):
        d = digests(OUT / "narrowq_retake_work", rung)
        if not d:
            continue
        lev = {k: v for k, v in d.items() if k.startswith("off-")}
        shp = {k: v for k, v in d.items() if k.startswith("on-")}
        uniq_l, uniq_s = set(lev.values()), set(shp.values())
        print("  %d aa  lever-on %d legs -> %d distinct digest(s); shipped %d legs -> %d"
              % (rung, len(lev), len(uniq_l), len(shp), len(uniq_s)))
        if len(uniq_l) == 1 and len(uniq_s) == 1:
            same = uniq_l == uniq_s
            print("     arms are %s  (%s vs %s)"
                  % ("BIT-EXACT" if same else "NOT bit-exact",
                     list(uniq_l)[0], list(uniq_s)[0]))
        else:
            print("     an arm is not self-consistent across reps; digests: %s" % d)

    if r768 and r896:
        print("\n=== what the control says about the instrument ===")
        print("  896 aa (fires):   %+.3f%%   768 aa (cannot fire): %+.3f%%   control floor %.3f%%"
              % (r896["ab_pct"], r768["ab_pct"], r768["aa_pct"]))
        if abs(r768["ab_pct"]) <= r768["aa_pct"]:
            print("  The control is inside its own A/A floor, so the harness attributes nothing "
                  "to a lever\n  that cannot fire. The 896 aa reading is the lever.")
        else:
            print("  THE CONTROL MOVED OUTSIDE ITS OWN FLOOR. The harness attributes a delta to a "
                  "lever that\n  provably does not fire at 768 aa, so the 896 aa reading is not "
                  "safe to call the lever's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
