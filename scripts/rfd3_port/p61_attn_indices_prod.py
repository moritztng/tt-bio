"""p61: the attn_indices row-blocking screen, on the production mask and production coordinates.

p59 screened the chain on a synthetic mask and got 1.31x at 8 threads with `torch.equal` at
every block size, but its mask was a uniform ~9-atoms-per-token band and its equality check
sorted both sides. Neither is the shipped thing. This captures the real `mask`, `seq_idx`,
coordinates and k out of a live design at the page fixture, then screens offline against them
with unsorted `torch.equal`.

    capture   run 4 timesteps of the page fixture and dump the first _create_attention_indices
              call's inputs to perf/p61/attn_inputs.pt. Needs the device; not a timed run.
    screen    load that dump, time the shipped chain against _neighbours_row_blocked at several
              block sizes and thread counts, and check every arm bit-exact. CPU only.

    env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rfd3-matched-batch-denominator-reopen \
      PYTHONPATH=$PWD /home/ttuser/tt-bio-dev/env/bin/python3 -u \
      scripts/rfd3_port/p61_attn_indices_prod.py capture
    ~/.coworker/scripts/benchlock.sh rfd3-matched-batch-denominator-reopen -- env PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u \
      scripts/rfd3_port/p61_attn_indices_prod.py screen
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())

DUMP = pathlib.Path("perf/p61/attn_inputs.pt")
OUT = pathlib.Path("perf/p61/attn_indices_prod.json")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
BLOCKS = (256, 512, 1024, 2048)
THREADS = (1, 8, 16)
REPS = 5


def capture():
    from tt_bio.rfd3 import design as rfd3_design
    from tt_bio.rfd3 import model as M

    orig = M._create_attention_indices
    done = []

    def spy(f, X_L, tok_idx, n_keys, n_seq_neighbours):
        if not done:
            parts = M._attention_index_prefix(f, tok_idx, n_keys, n_seq_neighbours)
            if "single" in parts:
                mask, seq_idx = parts["single"]
                x = X_L if X_L.ndim == 3 else X_L.unsqueeze(0)
                DUMP.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"mask": mask.clone(), "seq_idx": seq_idx.clone(),
                            "x": x[:1].clone().float(), "k": int(parts["k"]),
                            "n_keys": int(n_keys), "n_seq": int(n_seq_neighbours)}, DUMP)
                done.append(True)
                print("captured L=%d k=%d mask density %.4f -> %s"
                      % (x.shape[1], parts["k"], mask.float().mean().item(), DUMP), flush=True)
        return orig(f, X_L, tok_idx, n_keys, n_seq_neighbours)

    M._create_attention_indices = spy
    specs = json.loads(FIXTURE.read_text())
    os.system("rm -rf /tmp/rfd3_p61")
    rfd3_design.run_design(specs, "/tmp/rfd3_p61", checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=4, seed=42, num_designs=1, batch_size=1, verbose=False)
    assert done, "never reached the single-chain attn_indices path"


def shipped(mask, seq_idx, x, k):
    from tt_bio.rfd3 import model as M
    return M._extend_with_neighbours(mask, seq_idx, torch.cdist(x, x, p=2), k, inplace=True)


def blocked(mask, seq_idx, x, k, R):
    from tt_bio.rfd3 import model as M
    return M._neighbours_row_blocked(mask, seq_idx, x, k, block=R)


def timeit(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def screen():
    d = torch.load(DUMP)
    mask, seq_idx, x, k = d["mask"], d["seq_idx"], d["x"], d["k"]
    L = x.shape[1]
    ref = shipped(mask, seq_idx, x.clone(), k)
    print("L=%d k=%d mask density %.4f  torch %s  cpus %d"
          % (L, k, mask.float().mean().item(), torch.__version__, os.cpu_count()), flush=True)

    rows = []
    for R in BLOCKS:
        got = blocked(mask, seq_idx, x.clone(), k, R)
        exact = bool(torch.equal(got, ref))
        rows.append({"arm": "exactness R=%d" % R, "exact": exact})
        print("R=%-5d exact=%s" % (R, exact), flush=True)
    if not all(r["exact"] for r in rows):
        print("NOT BIT-EXACT -- stop, do not ship", flush=True)

    timings = []
    for nthr in THREADS:
        torch.set_num_threads(nthr)
        med, lo, hi = timeit(lambda: shipped(mask, seq_idx, x.clone(), k))
        timings.append({"arm": "shipped", "threads": nthr, "ms": round(med, 3),
                        "min": round(lo, 3), "max": round(hi, 3)})
        print("shipped        thr=%2d  %8.2f ms  [%.2f, %.2f]" % (nthr, med, lo, hi), flush=True)
        for R in BLOCKS:
            med, lo, hi = timeit(lambda: blocked(mask, seq_idx, x.clone(), k, R))
            timings.append({"arm": "blocked R=%d" % R, "threads": nthr, "ms": round(med, 3),
                            "min": round(lo, 3), "max": round(hi, 3)})
            print("blocked R=%-5d thr=%2d  %8.2f ms  [%.2f, %.2f]" % (R, nthr, med, lo, hi),
                  flush=True)

    base8 = next(t["ms"] for t in timings if t["arm"] == "shipped" and t["threads"] == 8)
    best = min((t for t in timings if t["arm"] != "shipped"), key=lambda t: t["ms"])
    print("\nshipped @8 %.2f ms  best %s @%d %.2f ms  -> %.3fx, %.2f ms/step saved on host"
          % (base8, best["arm"], best["threads"], best["ms"], base8 / best["ms"],
             base8 - best["ms"]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"L": L, "k": k, "exactness": rows, "timings": timings,
                               "mask_density": round(mask.float().mean().item(), 6),
                               "host": "qb2", "torch": torch.__version__,
                               "cpu_count": os.cpu_count(), "reps": REPS}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    {"capture": capture, "screen": screen}[sys.argv[1]]()
