"""Diff this branch's parity gate against the committed control run (`gate_asg.json`).

Two modes, because the 44-leg gate takes five hours and a leg that finished in hour one is
scoreable in hour one:

  cmp_legs.py                 rebuild each FINISHED leg's detail from the leg's own report under
                              the workdir and diff it against the control, while the gate runs.
  cmp_legs.py --final <json>  diff the finished gate's own per-leg verdict and detail strings
                              against the control's, all 44 legs, including the report shapes
                              the rebuild cannot reconstruct.

Strict by construction: a leg whose detail this script cannot rebuild is reported UNSCORED, never
OK. A comparison that matches nothing must not read as a pass
(memory: gate-fixture-existence-vs-content-inversion).
"""
import json, os, glob, sys

CTL = "perf/b2z2_gate/gate_asg.json"
WD = "perf/b2z2_gate/pwaship_work"


def within(m):
    return m["kabsch_rmsd"]["within_noise_floor"]


def structures(r):
    # X/R/D are cross, REF floor and DEV floor -- the three the gate's own _structures_verdict
    # prints. Rebuilding R from `floor_mean` reads the same on every leg where the reference
    # floor is the wider of the two, and differently on the two opendde legs where it is not.
    out = []
    for name, t in r["targets"].items():
        k = t["kabsch_rmsd"]
        out.append("%s: X=%.3f R=%.3f D=%.3f within=%s" % (
            name, k["cross"]["mean"], k["ref_floor"]["mean"], k["dev_floor"]["mean"],
            k["within_noise_floor"]))
    return "; ".join(out)


def envelope(r):
    key, worst = max(r["metrics"].items(), key=lambda kv: kv[1]["ratio"])
    return "%s: num=%.4f env=%.4f ratio=%.2f" % (
        key, worst["numerator"], worst["envelope"], worst["ratio"]), r.get("verdict")


def detail_of(r):
    """Return (detail string, verdict or None). Raise if the shape is unknown."""
    if isinstance(r, list):                      # esmfold2 legs
        n = len(r)
        ok = sum(1 for e in r if within(e))
        return "%d proteins scored (%d within floor)" % (n, ok), None
    mode = r.get("mode")
    if mode == "integration_envelope":
        return envelope(r)
    if mode == "structures":
        return structures(r), None
    if "dev_vs_ref_pcc_min" in r:                # esmc
        return "min per-res PCC=%.5f" % r["dev_vs_ref_pcc_min"], None
    if "X_emb" in r:                             # saprot
        return "X_emb=%.5f" % r["X_emb"], None
    raise ValueError("unknown report shape: keys=%s" % sorted(r)[:8])


def final(path):
    """Every leg of a finished gate against the control, on the gate's own strings."""
    ctl = {l["leg"]: l for l in json.load(open(CTL))["legs"]}
    new = {l["leg"]: l for l in json.load(open(path))["legs"]}
    print("control %d legs, %s %d legs" % (len(ctl), os.path.basename(path), len(new)))
    same, diff = 0, 0
    for leg in sorted(set(ctl) | set(new)):
        c, n = ctl.get(leg), new.get(leg)
        if c is None or n is None:
            diff += 1
            print("  MISSING %-26s %s" % (leg, "not in control" if c is None else "not in run"))
        elif (c["verdict"], c["detail"]) == (n["verdict"], n["detail"]):
            same += 1
        else:
            diff += 1
            print("  DIFF %s\n       control: %s | %s\n       now    : %s | %s"
                  % (leg, c["verdict"], c["detail"], n["verdict"], n["detail"]))
    print("identical verdict+detail: %d of %d, differing %d" % (same, len(ctl), diff))
    return 1 if diff else 0


def inflight():
    ctl = {l["leg"]: l for l in json.load(open(CTL))["legs"]}
    done = sorted(os.path.basename(p)[:-5] for p in glob.glob(WD + "/*.json")
                  if os.path.basename(p) != "GATE_CODE.json")
    print("finished %d of %d legs" % (len(done), len(ctl)))
    ok = diff = unscored = 0
    for leg in done:
        c = ctl.get(leg)
        r = json.load(open(os.path.join(WD, leg + ".json")))
        try:
            detail, verdict = detail_of(r)
        except Exception as exc:
            unscored += 1
            print("  UNSCORED %-26s %s" % (leg, exc))
            continue
        if c is None:
            unscored += 1
            print("  UNSCORED %-26s not in control" % leg)
            continue
        cd = str(c["detail"])
        same = all(part in cd for part in detail.split("; "))
        same = same and (verdict is None or verdict == c["verdict"])
        if same:
            ok += 1
            print("  OK   %-26s %s | %s" % (leg, c["verdict"], detail))
        else:
            diff += 1
            print("  DIFF %-26s\n       control: %s | %s\n       now    : %s"
                  % (leg, c["verdict"], cd, detail))
    print("reproduced %d, differing %d, unscored %d" % (ok, diff, unscored))
    return 1 if (diff or unscored) else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--final":
        sys.exit(final(sys.argv[2]))
    sys.exit(inflight())
