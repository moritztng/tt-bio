#!/usr/bin/env python3
"""Cut EGFR domain III out of the whole ectodomain, which is what we tell a user to do today.

Arm A of the acceptance pair is the cropped case and arm B is the same target uncropped, so the
crop has to be the one a customer would actually make: domain III (L2), residues 310-480 of the
mature chain, the subdomain cetuximab binds. All four hotspots the full target carries -- 384,
412, 465, 467 -- sit inside it, and deposited numbering is kept, so the two arms point the design
at the same four residues of the same protein. That is the whole point of the pair: the only
thing that differs between them is how much of the receptor is in the box.
"""
import pathlib
import sys

LO, HI = 310, 480
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1] / "perf/bgx_size/targets/hEGFR.pdb"
DST = HERE.parents[1] / "perf/bgx_size/targets/hEGFR_d3.pdb"


def main() -> int:
    out, n, res = [], 0, []
    for line in SRC.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if not (LO <= int(line[22:26]) <= HI):
            continue
        n += 1
        out.append(f"{line[:6]}{n:5d}{line[11:]}")
        if line[12:16].strip() == "CA":
            res.append(int(line[22:26]))
    DST.write_text("\n".join(out) + "\nTER\nEND\n")
    gaps = [(a, b) for a, b in zip(res, res[1:]) if b != a + 1]
    print(f"{DST}: {len(res)} aa, {res[0]}-{res[-1]}, {n} atoms, gaps {gaps}")
    missing = [h for h in (384, 412, 465, 467) if h not in res]
    if missing:
        print(f"hotspots missing from the crop: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
