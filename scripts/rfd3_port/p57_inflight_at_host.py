"""How much device work is actually in flight when the host work runs? The overlap lever, priced.

P3.7 measured what PERFECT host/device overlap would be worth -- at most 130.1 ms/step, since exposed
device time is a hard floor no host change can go under. That is a ceiling, not a plan. This asks the
question that decides whether any of it is reachable: **at the moment each host block starts, how
much device work is already enqueued and unfinished?** Whatever that is, is the overlap available
without reordering anything. Whatever is missing has to come from a restructure.

The probe: at the entry of every wrapped host function, call `ttnn.synchronize_device` and time it.
That wait IS the in-flight device work. It also serialises the step, so the arm's wall is not a
performance number and is not reported as one -- only the in-flight times are.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p57_inflight_at_host.py
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
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p57/inflight_at_host.json")
STEPS, WARM = 6, 2

STEP = [0]
INFLIGHT = collections.defaultdict(lambda: [0.0, 0])
BODY = collections.defaultdict(lambda: [0.0, 0])

HOST_FNS = ("_scatter_mean", "_scaled_distogram_bins", "_extend_with_neighbours",
            "_sparse_pair_gather", "_dense_attention_mask", "_create_attention_indices")


def wrap(name):
    fn = getattr(M, name, None)
    if fn is None or not callable(fn):
        return

    def w(*a, **k):
        t0 = time.perf_counter()
        ttnn.synchronize_device(M.get_device())
        wait = time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            body = time.perf_counter() - t1
            if STEP[0] >= WARM:
                e = INFLIGHT[name]
                e[0] += wait
                e[1] += 1
                BODY[name][0] += body
    setattr(M, name, w)


def main():
    for n in HOST_FNS:
        wrap(n)
    call = M.RFD3DiffusionModule.__call__

    def stepped(self, *a, **k):
        try:
            return call(self, *a, **k)
        finally:
            STEP[0] += 1
    M.RFD3DiffusionModule.__call__ = stepped

    specs = json.loads(FIXTURE.read_text())
    os.system("rm -rf /tmp/rfd3_p57")
    rfd3_design.run_design(specs, "/tmp/rfd3_p57", checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=False)
    counted = STEP[0] - WARM

    rows = []
    print("%-28s %11s %11s %11s %8s" %
          ("host fn", "body ms/st", "inflight ms", "calls/step", "overlap"))
    for name in HOST_FNS:
        if name not in INFLIGHT:
            continue
        wait, n = INFLIGHT[name]
        body = BODY[name][0]
        b_ms, w_ms = 1000 * body / counted, 1000 * wait / counted
        rows.append({"fn": name, "body_ms_per_step": round(b_ms, 3),
                     "inflight_ms_per_step": round(w_ms, 3),
                     "calls_per_step": round(n / counted, 1),
                     "overlappable_ms_per_step": round(min(b_ms, w_ms), 3)})
        print("%-28s %11.3f %11.3f %11.1f %8.3f" %
              (name, b_ms, w_ms, n / counted, min(b_ms, w_ms)))
    tb = sum(r["body_ms_per_step"] for r in rows)
    tw = sum(r["inflight_ms_per_step"] for r in rows)
    to = sum(r["overlappable_ms_per_step"] for r in rows)
    print("\nhost body %.2f ms/step, device in flight when it starts %.2f, so at most %.2f "
          "is overlappable where it stands" % (tb, tw, to))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "counted_steps": counted, "atoms": 6051,
                               "host_body_ms_per_step": round(tb, 3),
                               "inflight_ms_per_step": round(tw, 3),
                               "overlappable_ms_per_step": round(to, 3),
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
