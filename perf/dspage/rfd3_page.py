"""RFD3 seconds per design on one Blackhole p150a, pinned fixture rfd3_R4, shipped defaults.

Runs the shipped entry point (`tt_bio.rfd3.design.run_design`), so featurisation, the
runtime's own batch clamp, and CIF writing are all the released code path. The timed region
is `RFD3Sampler.sample`, which is the same region the H200/B200 arm timed: foundry's
"Finished inference batch in X seconds" wraps the sampler loop only, with featurisation and
file IO outside it.

Two arms:

  ceiling  --num_designs 8 --batch_size 8, the same request the GPU arm ran. At R4's 6051
           atoms `run_design`'s atom-pair budget shrinks the device forward to 2, so this is
           four chunks of two. Each side runs the batch it admits, on the same fixture, for
           the same eight designs.
  b1       --num_designs 4 --batch_size 1, the same-batch control against the GPU's b=1
           point (H200 27.490 s/design, B200 32.580).

Per arm: the first chunk is discarded as cold (it carries kernel compile), the rest are the
warm sample, median and min-max spread reported. Every written CIF is validated before the
timing counts. Records append to perf/dspage/results/rfd3_page.jsonl; a rerun skips an arm
already present, so a relaunch never redoes a finished arm.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:perf-page-design-models \
    PYTHONPATH=$(pwd) ~/.coworker/scripts/benchlock.sh perf-page-design-models -- \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/dspage/rfd3_page.py [ceiling|b1]
"""
import json
import os
import pathlib
import statistics
import sys
import time

import gemmi
import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler              # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path("perf/dspage/results/rfd3_page.jsonl")
STEPS = 200                    # upstream production default, what the GPU arm ran
SEED = 42                      # the CLI default
EXP_ATOMS = 6051               # featurised L at R4, MEASURED in the fixture ladder
EXP_RES = 685                  # 585 target + 100 designed binder
ARMS = {"ceiling": (8, 8), "b1": (4, 1)}

HOST, CARD, TTNN = "qb2", 0, "0.68.0"

# Every sampler.sample call, as (wall seconds, designs in that forward).
WALLS: list[tuple[float, int]] = []
_sample = RFD3Sampler.sample


def _timed_sample(self, dm, n, *a, **kw):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **kw)
    WALLS.append((time.perf_counter() - t0, int(n)))
    return out


RFD3Sampler.sample = _timed_sample


def validate(out_dir, n_expected):
    """A timing counts only if the designs are real. Design invariants, not folding ones:
    residue topology and finiteness, never atom equality between siblings."""
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    bad = []
    if len(cifs) != n_expected:
        bad.append("%d CIFs written, expected %d" % (len(cifs), n_expected))
    atoms = []
    for c in cifs:
        st = gemmi.read_structure(str(c))
        st.setup_entities()
        na = sum(1 for ch in st[0] for r in ch for _ in r)
        nr = sum(len(ch) for ch in st[0])
        nf = sum(1 for ch in st[0] for r in ch for a in r
                 if not all(abs(v) < 1e6 and v == v for v in (a.pos.x, a.pos.y, a.pos.z)))
        atoms.append(na)
        if na != EXP_ATOMS:
            bad.append("%s: %d atoms != %d" % (c.name, na, EXP_ATOMS))
        if nr != EXP_RES:
            bad.append("%s: %d residues != %d" % (c.name, nr, EXP_RES))
        if nf:
            bad.append("%s: %d non-finite coords" % (c.name, nf))
    return (not bad), bad, atoms


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else None
    arms = [which] if which else list(ARMS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["arm"])
    specs = json.loads(FIXTURE.read_text())

    for arm in arms:
        if arm in done:
            print("[rfd3] %s cached" % arm, flush=True)
            continue
        nd, bs = ARMS[arm]
        out_dir = "/tmp/rfd3_page_%s" % arm
        os.system("rm -rf %s" % out_dir)
        WALLS.clear()
        t0 = time.perf_counter()
        res = rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                                     num_timesteps=STEPS, seed=SEED, num_designs=nd,
                                     batch_size=bs, verbose=True)
        total = time.perf_counter() - t0
        chunk_b = sorted({n for _, n in WALLS})
        walls = [w for w, _ in WALLS]
        if len(chunk_b) != 1:
            print("[rfd3] %s ABORT: mixed chunk sizes %s" % (arm, chunk_b), flush=True)
            continue
        eff = chunk_b[0]
        warm = walls[1:]                       # chunk 0 carries kernel compile
        med = statistics.median(warm)
        ok, bad, atoms = validate(out_dir, nd)
        rec = {
            "arm": arm, "rung": "R4", "fixture": str(FIXTURE), "num_designs": nd,
            "batch_size_requested": bs, "effective_batch": eff, "num_timesteps": STEPS,
            "seed": SEED, "n_chunks": len(walls), "cold_chunk_s": round(walls[0], 3),
            "warm_chunks_s": [round(w, 3) for w in warm], "n_warm": len(warm),
            "chunk_s_median": round(med, 3), "chunk_s_min": round(min(warm), 3),
            "chunk_s_max": round(max(warm), 3),
            "spread_pct": round(100 * (max(warm) - min(warm)) / med, 2),
            "s_per_design": round(med / eff, 3),
            "designs_per_hour": round(3600 * eff / med, 1),
            "ms_per_step": round(1000 * med / STEPS, 3),
            "total_wall_s": round(total, 1),
            "n_results": len(res), "atoms": atoms,
            "output_ok": ok, "output_fail": bad,
            "host": HOST, "card": CARD, "ttnn": TTNN,
            "torch": torch.__version__,
        }
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[rfd3] %s: %.3f s/design (chunk %.3f s of %d, n=%d warm, spread %.2f%%), "
              "output_ok=%s %s" % (arm, rec["s_per_design"], med, eff, len(warm),
                                   rec["spread_pct"], ok, bad), flush=True)


if __name__ == "__main__":
    main()
