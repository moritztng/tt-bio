"""The AF2-IG fixtures in dl_binder_design's chain order (binder = A, target = B), six files for one cold + five warm.

usage: python af2ig_inputs.py FIXTURE_DIR OUT_DIR   (reads af2ig_pdl1_l80_s{0,1,2}.pdb: target A, binder B)
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for s in range(3):
    lines = [l for l in (src / f"af2ig_pdl1_l80_s{s}.pdb").read_text().splitlines() if l.startswith("ATOM")]
    # binder first, renamed A; target second, renamed B; atoms renumbered
    swapped = [l[:21] + "A" + l[22:] for l in lines if l[21] == "B"] + [l[:21] + "B" + l[22:] for l in lines if l[21] == "A"]
    text = "".join(f"{l[:6]}{i:5d}{l[11:]}\n" for i, l in enumerate(swapped, 1)) + "END\n"
    for rep in ("a", "b"):
        (out / f"pdl1_l80_s{s}{rep}.pdb").write_text(text)
