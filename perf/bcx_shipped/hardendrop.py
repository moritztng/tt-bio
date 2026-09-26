#!/usr/bin/env python3
"""Is the anneal->harden i_pTM collapse the PORT, or is it BindCraft 2's own method?

Three of the device's harden rejections looked alike: i_pTM falls off a cliff out of an
anneal that was still rising (STAGE-PROFILE.md, orchestrator 2026-09-25 18:3xZ). harden is
where BindCraft 2 converts the soft sequence to a hard one-hot, so SOME drop is the method.
Whether a drop of that SIZE is the method is a question only a reference harden answers, and
until ref_s1's trajectory 1 terminated at harden there was not one.

This reads the stage lines the run log prints, device and reference, and puts the reference's
drop against the range the device's drops span.

ON THE STATISTIC. A stage line is that stage's BEST round, so anneal's is a max over 45 draws
and harden's a max over 5: the printed step is biased DOWNWARD whether or not anything
changed (converge.py). That bias is the reason hardenstep.py re-reads both stages at a
matched round count. It is kept here on purpose -- this is the statistic the campaign's
device table was built from, and it is biased identically on both arms, so the device-against-
reference comparison it supports is sound even though the absolute value is not.

usage: hardendrop.py <class>=<log> [...]
"""
import re
import sys

RE_TRAJ = re.compile(r"=== trajectory \d+ \| \S*?_l(\d+)_([0-9a-f]+)")
RE_STAGE = re.compile(
    r"(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")


def trajectories(path):
    cur, out = None, []
    for line in open(path, errors="replace"):
        m = RE_TRAJ.search(line)
        if m:
            cur = {"draw": "l" + m.group(1), "hash": m.group(2), "stages": {}, "term": None}
            out.append(cur)
            continue
        m = RE_STAGE.search(line)
        if cur and m:
            verdict, stage, iptm, plddt, filt = m.groups()
            cur["stages"][stage] = (float(iptm), float(plddt))
            if verdict == "rejected at":
                cur["term"] = (stage, filt or "")
    return out


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    if not specs:
        sys.exit(__doc__)
    rows = []
    for label, log in specs:
        for t in trajectories(log):
            if "anneal" not in t["stages"] or "harden" not in t["stages"]:
                continue
            a_i, a_p = t["stages"]["anneal"]
            h_i, h_p = t["stages"]["harden"]
            rej = t["term"] and t["term"][0] == "harden"
            rows.append((label, t["draw"], a_i, a_p, h_i, h_p, h_i - a_i, h_p - a_p,
                         ("REJECTED [%s]" % t["term"][1]) if rej else "passed"))

    print("  %-10s %5s %12s %12s %8s %8s  harden verdict"
          % ("class", "draw", "anneal i/p", "harden i/p", "d i_pTM", "d pLDDT"))
    for r in sorted(rows, key=lambda r: (r[0], r[6])):
        print("  %-10s %5s  %.2f / %.2f  %.2f / %.2f %+8.2f %+8.2f  %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))

    for label in dict.fromkeys(r[0] for r in rows):
        grp = [r for r in rows if r[0] == label]
        rej = [r for r in grp if r[8].startswith("REJECTED")]
        print("\n  %s: %d trajectories reached harden, %d rejected there."
              % (label, len(grp), len(rej)))
        if rej:
            print("    d i_pTM over the rejections: %s"
                  % ", ".join("%+.2f" % r[6] for r in sorted(rej, key=lambda r: r[6])))
        pas = [r for r in grp if not r[8].startswith("REJECTED")]
        if pas:
            print("    d i_pTM over the passes:     %s"
                  % ", ".join("%+.2f" % r[6] for r in sorted(pas, key=lambda r: r[6])))

    dev = [r[6] for r in rows if r[0] == "device" and r[8].startswith("REJECTED")]
    ref = [r[6] for r in rows if r[0] == "reference" and r[8].startswith("REJECTED")]
    if dev and ref:
        print("\n  VERDICT")
        for v in sorted(ref):
            inside = min(dev) <= v <= max(dev)
            print("    reference drop %+.2f is %s the device's rejected range [%+.2f, %+.2f]"
                  % (v, "INSIDE" if inside else "OUTSIDE", min(dev), max(dev)))


if __name__ == "__main__":
    main()
