#!/usr/bin/env python3
"""Diff this gate report against the post-b2z2 verdict, leg by leg.

The gate already answers "does each leg reproduce its COMMITTED record". This answers the
different question the K10 wave actually raises: "is any leg worse than it was at the last
GO verdict" -- i.e. new since 0cd6c415, not merely non-PASS. A GAP that was a GAP then is
fine; a GAP that was a PASS then is the thing that blocks a GO.

The baseline is read from the post-b2z2 run's own report.json, not transcribed from the
state doc's prose, so a typo in a hand-copied table cannot invent or hide a regression.
If that artifact is missing the transcribed table below is used instead, and the header
says which source answered.

--numeric goes one level deeper, into the per-leg JSONs both workdirs wrote. A verdict is a
threshold applied to a number, so two runs can agree on every verdict while the numbers under
them have moved (memory parity-gate-reproduces-committed-checks-verdict-not-value). The three
K10 levers were each merged on a bit-exact claim, so the accuracy leaves here should not merely
pass, they should be IDENTICAL to post-b2z2. Anything that moved is either a lever that is not
bit-exact after all or a leg that is not deterministic run to run; either way it has to be named
rather than absorbed by a PASS.
"""
import json, math, os, re, sys

B2Z2_REPORT = "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-b2z2/gate-0cd6c415/report.json"
K10_WORKDIR = "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f"
B2Z2_WORKDIR = "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-b2z2/gate-0cd6c415"

# Fallback: post-b2z2 section 4, the six legs that were not a bare PASS. Everything else PASS.
B2Z2_NON_PASS_FALLBACK = {
    "boltz2-prot-nomsa": "GAP",
    "boltz2-9ncy-nomsa": "GAP",
    "openfold3-7xi5-notmpl": "GAP",
    "af2ig-trunk-device": "GAP",
    "af2ig-trunk-monomer": "PASS-caveated",
    "protenix-9ncy-msa": "BLOCKED-REF-REGEN-NEEDED",
}

# Leaves that legitimately differ between two runs of the same code: wall clocks, pids, hosts,
# paths, dates. Comparing them would bury the accuracy drift this mode exists to find.
NOISE = re.compile(r"(wall|elapsed|sec|seconds|_s$|time|stamp|date|pid|host|path|dir|file|"
                   r"cwd|commit|version|workdir|loadavg|rate|throughput|per_s|s_per)", re.I)


def baseline():
    if os.path.exists(B2Z2_REPORT):
        r = json.load(open(B2Z2_REPORT))
        return {l["leg"]: l["verdict"] for l in r["legs"]}, B2Z2_REPORT
    return dict(B2Z2_NON_PASS_FALLBACK), "transcribed table (post-b2z2 report.json absent)"


def flatten(obj, prefix=""):
    """path -> scalar, for every leaf in a nested JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, "%s.%s" % (prefix, k) if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, "%s[%d]" % (prefix, i))
    else:
        yield prefix, obj


def _both_nan(a, b):
    return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)


def numeric_diff(workdir, base_workdir):
    """Compare every non-noise leaf of every per-leg JSON present in both workdirs."""
    legs = sorted(f[:-5] for f in os.listdir(workdir)
                  if f.endswith(".json") and f not in ("report.json", "GATE_CODE.json"))
    print("\n## numeric leaf diff, per-leg JSON vs post-b2z2")
    print("   new  = %s" % workdir)
    print("   base = %s" % base_workdir)
    moved_total = same_total = 0
    missing = []
    for leg in legs:
        bp = os.path.join(base_workdir, leg + ".json")
        if not os.path.exists(bp):
            missing.append(leg)
            continue
        try:
            new = dict(flatten(json.load(open(os.path.join(workdir, leg + ".json")))))
            old = dict(flatten(json.load(open(bp))))
        except Exception as e:
            print("  %-34s UNREADABLE %s" % (leg, e))
            continue
        moved, same = [], 0
        for k, v in new.items():
            if NOISE.search(k) or k not in old:
                continue
            o = old[k]
            if isinstance(v, bool) or isinstance(o, bool) or v is None or o is None:
                continue
            if isinstance(v, (int, float)) and isinstance(o, (int, float)):
                if v == o or _both_nan(v, o):
                    same += 1
                else:
                    moved.append((abs(v - o) / max(abs(o), 1e-12), k, o, v))
            elif isinstance(v, str) and isinstance(o, str):
                if v == o:
                    same += 1
                elif re.fullmatch(r"[0-9a-f]{8,}", v) and re.fullmatch(r"[0-9a-f]{8,}", o):
                    moved.append((float("inf"), k, o, v))
        same_total += same
        moved_total += len(moved)
        if moved:
            moved.sort(reverse=True, key=lambda t: t[0])
            print("  %-34s %d/%d compared leaves MOVED" % (leg, len(moved), len(moved) + same))
            for rel, k, o, v in moved[:6]:
                tag = "DIGEST" if rel == float("inf") else "%.3g%%" % (100 * rel)
                print("        %-50s %s -> %s   (%s)" % (k[:50], o, v, tag))
    print("  --- %d leaves identical, %d moved; %d legs with no post-b2z2 JSON %s"
          % (same_total, moved_total, len(missing), missing or ""))
    return moved_total


def main(path):
    r = json.load(open(path))
    legs = r["legs"]
    base, src = baseline()
    print("legs=%d  tally=%s  scored=%s  wall=%ss  workers=%s"
          % (len(legs), r.get("tally"), r.get("scored"), r.get("total_wall_s"), r.get("workers")))
    print("baseline source: %s" % src)

    regressed, reproduced, improved, unknown = [], [], [], []
    for leg in legs:
        name, v = leg["leg"], leg["verdict"]
        if name not in base:
            unknown.append((name, "(absent at post-b2z2)", v, leg.get("detail", "")))
            continue
        was = base[name]
        if v.startswith("PASS") and was.startswith("PASS"):
            continue
        if v == was:
            reproduced.append((name, was, v, leg.get("detail", "")))
        elif v.startswith("PASS"):
            improved.append((name, was, v, leg.get("detail", "")))
        else:
            regressed.append((name, was, v, leg.get("detail", "")))

    for title, rows in (("REGRESSED vs post-b2z2 (blocks GO)", regressed),
                        ("NEW leg, no post-b2z2 record (judge by hand)", unknown),
                        ("reproduces a post-b2z2 non-PASS record", reproduced),
                        ("improved since post-b2z2", improved)):
        print("\n## %s: %d" % (title, len(rows)))
        for n, was, now, d in rows:
            print("  %-34s was=%-26s now=%-26s %s" % (n, was, now, d[:70]))
    return 1 if regressed else 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--numeric"]
    rc = main(argv[0] if argv else os.path.join(K10_WORKDIR, "report.json"))
    if "--numeric" in sys.argv:
        numeric_diff(K10_WORKDIR, B2Z2_WORKDIR)
    sys.exit(rc)
