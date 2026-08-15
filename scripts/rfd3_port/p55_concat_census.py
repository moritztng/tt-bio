"""Every ttnn.concat in a real RFD3 step, with its piece widths and what the cliff costs it.

p52 established a general defect: `ttnn.concat` runs 15-20x below its bandwidth floor if ANY input
piece is narrower than a 32-wide tile, independent of the output width and of the piece offsets. The
token encoder's 258-wide pair concat was one instance and is fixed (p53/p54). This finds the rest.

Wraps `ttnn.concat` for a whole fold, keys each call by caller line, and records the width of every
input along the concat axis. A call with any input width not divisible by 32 is on the slow path.
Times are sync-bracketed, so read the ranking rather than the absolute -- the point is which sites
are misaligned and how much they cost.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p55_concat_census.py
"""
import collections
import json
import os
import pathlib
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                          # noqa: E402
from tt_bio.rfd3 import model as M                                     # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p55/concat_census.json")
STEPS, WARM = 6, 2

STEP = [0]
ACC = collections.defaultdict(lambda: [0.0, 0])
_concat = ttnn.concat


def w(tensors, dim=0, **k):
    fr = sys._getframe(1)
    site = "%s:%d" % (fr.f_code.co_filename.rsplit("/", 1)[-1], fr.f_lineno)
    try:
        widths = tuple(int(list(t.padded_shape)[dim]) for t in tensors)
        logical = tuple(int(list(t.shape)[dim]) for t in tensors)
    except Exception:
        widths = logical = ()
    ttnn.synchronize_device(M.get_device())
    t0 = time.perf_counter()
    try:
        return _concat(tensors, dim=dim, **k)
    finally:
        ttnn.synchronize_device(M.get_device())
        dt = time.perf_counter() - t0
        if STEP[0] >= WARM:
            e = ACC[(site, dim, logical, widths)]
            e[0] += dt
            e[1] += 1


ttnn.concat = w


def main():
    call = M.RFD3DiffusionModule.__call__

    def stepped(self, *a, **k):
        try:
            return call(self, *a, **k)
        finally:
            STEP[0] += 1
    M.RFD3DiffusionModule.__call__ = stepped

    specs = json.loads(FIXTURE.read_text())
    os.system("rm -rf /tmp/rfd3_p55")
    rfd3_design.run_design(specs, "/tmp/rfd3_p55", checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=False)
    counted = STEP[0] - WARM

    rows = []
    for (site, dim, logical, widths), (tot, n) in ACC.items():
        misaligned = [x for x in widths if x % 32]
        rows.append({"site": site, "dim": dim, "logical_widths": list(logical),
                     "padded_widths": list(widths), "aligned": not misaligned,
                     "calls_per_step": round(n / counted, 1),
                     "ms_per_step": round(1000 * tot / counted, 3)})
    rows.sort(key=lambda r: -r["ms_per_step"])

    print("%-22s %5s %-26s %9s %11s %8s" %
          ("site", "dim", "widths (logical)", "ms/step", "calls/step", "aligned"))
    for r in rows:
        print("%-22s %5d %-26s %9.3f %11.1f %8s" %
              (r["site"], r["dim"], str(r["logical_widths"])[:26], r["ms_per_step"],
               r["calls_per_step"], "yes" if r["aligned"] else "NO"))
    slow = sum(r["ms_per_step"] for r in rows if not r["aligned"])
    print("\nmisaligned concats cost %.3f ms/step of %.3f total"
          % (slow, sum(r["ms_per_step"] for r in rows)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "counted_steps": counted, "atoms": 6051,
                               "misaligned_ms_per_step": round(slow, 3),
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
