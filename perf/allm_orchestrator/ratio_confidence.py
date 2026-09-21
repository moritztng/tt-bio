#!/usr/bin/env python3
"""Which of the campaign's five transfer ratios were measured on a contended arm?

Pass 36 established something that changes how this table should be read: on OpenDDE, contention
did not merely widen the A/A floor from 0.150 s to 4.782 s, it **overstated the effect itself by
~3.5x** (0.941 s measured quiet, 3.259 s measured loud). So `cotenanted` is not a footnote about
error bars, it is a warning about the POINT ESTIMATE, and the campaign has been carrying five
ratios without asking which of them carry it.

Reads `perf/allm_audit/out/RATIOS.txt` on `wk/allm-audit` -- the committed artifact, with its own
per-cell `clean` and `cotenanted` flags -- rather than any row's prose. No device.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = ("origin/wk/allm-audit", "perf/allm_audit/out/RATIOS.txt")


def rows():
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{SRC[0]}:{SRC[1]}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {SRC[1]} on {SRC[0]} -- the audit artifact moved")
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    got = []
    for i, l in enumerate(lines):
        m = re.match(r"^(\S+)\s+s/(fold|design)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", l)
        if not m:
            continue
        detail = lines[i + 1] if i + 1 < len(lines) else ""
        eff = re.search(r"effect ([\d.]+) s, ([\d.]+)x the larger floor", detail)
        clean = re.search(r"clean (\w+)/(\w+)", detail)
        cot = re.search(r"cotenanted (\d+)/(\d+)", detail)
        if not (eff and clean and cot):
            raise SystemExit(f"detail line for {m.group(1)} is not in the expected shape; re-read it")
        got.append(dict(model=m.group(1), ratio=float(m.group(5)),
                        effect=float(eff.group(1)), margin=float(eff.group(2)),
                        clean=(clean.group(1) == "True", clean.group(2) == "True"),
                        cot=(int(cot.group(1)), int(cot.group(2)))))
    if len(got) != 5:
        raise SystemExit(f"expected 5 model rows, parsed {len(got)} -- the artifact's shape changed")
    return got


def main() -> int:
    print(f"{'model':14} {'ratio':>9} {'effect':>9} {'x floor':>8}  arms clean   cotenanted  verdict")
    worst = []
    for r in sorted(rows(), key=lambda r: -r["ratio"]):
        old_ok, new_ok = r["clean"]
        c_old, c_new = r["cot"]
        contended = (not old_ok) or (not new_ok) or c_old or c_new
        if not contended and r["margin"] >= 5:
            v = "SOLID"
        elif not contended:
            v = "clean arms, thin margin"
        elif r["margin"] >= 5:
            v = "contended arm, wide margin -- direction safe, MAGNITUDE suspect"
        else:
            v = "CONTENDED AND THIN -- treat as a bound, not a value"
            worst.append(r["model"])
        print(f"  {r['model']:12} {r['ratio']:8.4f}x {r['effect']:8.3f}s {r['margin']:7.1f}x"
              f"  {str(old_ok)[0]}/{str(new_ok)[0]}         {c_old}/{c_new}        {v}")

    print(f"""
HOW TO READ THIS, given pass 36

  OpenDDE's contended cell read 3.259 s where the quiet re-run reads 0.941 s -- the effect was
  overstated 3.5x, not merely noisy. So a cotenanted arm is a hazard to the NUMBER, and a wide
  margin over the A/A floor does NOT rescue it: that same contended OpenDDE cell had a floor of
  its own and still moved 3.5x when the load came off.

  Direction survives contention better than magnitude does. Every cell here is positive and the
  campaign's qualitative finding -- Boltz-2 transferred, the rest banded 1.03-1.29x -- does not
  depend on any single magnitude. What does depend on it is any arithmetic that SUBTRACTS these
  ratios, e.g. "OpenFold3 is 1.3266x short of Boltz-2". Those subtractions inherit the error.""")
    if worst:
        print(f"\n  Weakest cell(s): {', '.join(worst)} -- quote as a bound and say so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
