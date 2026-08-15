"""RFD3 end-to-end seconds per design against batch, one rung at a time, on one p150a.

Leg 1 of `rfd3-close-the-page-gap`. The shipped `effective_batch` clamp admits the largest
batch that fits an atom-pair budget; nothing in it has ever been about speed. At R4 the batch
it admits (2) is 1.161x SLOWER per design than the batch below it (144.044 vs 167.189 s,
perf/dspage/results/rfd3_page.jsonl). This sweep measures the same quantity at the other rungs
so a speed cap can be fitted to measured crossovers instead of guessed ones.

The quantity is end-to-end s/design at the shipped 200 timesteps, and only that. The marginal
per-step differential in perf/dsfix/results/rfd3_tt.jsonl measures a different thing -- it
excludes every per-forward fixed cost by construction, and its own records imply a NEGATIVE
fixed cost (R0 b=1: t_N1 1.1639 against 8 x 0.16761 = 1.341), so the per-step wall is not flat
along the noise schedule and no marginal number can predict an s/design. Batch defaults are
decided here.

Same instrument as perf/dspage/rfd3_page.py, which produced the two R4 points: the timed region
is `RFD3Sampler.sample`, the first chunk is dropped as cold, every written CIF is validated
before its timing counts.

    ~/.coworker/scripts/benchlock.sh rfd3-close-the-page-gap -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-close-the-page-gap PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/dsfix/rfd3_batch_e2e.py R0 1,4,8
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

CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path("perf/dsfix/results/rfd3_batch_e2e.jsonl")
STEPS = 200                    # upstream production default, what every page arm ran
SEED = 42                      # the CLI default
N_WARM = 2                     # chunks kept after the cold one; R4 spread was 0.06-0.23 %
HOST, CARD, TTNN = "qb2", 0, "0.68.0"

# atoms and target residues per rung, from perf/dsfix/results/rfd3_tt.jsonl. The binder is
# 100 residues at every rung, so the CIF must carry target_res + 100.
RUNGS = {
    "R0": (2299, 117),
    "R1": (2952, 196),
    "R2": (3844, 318),
    "R3": (4558, 414),
    "R4": (6051, 585),
}

WALLS: list[tuple[float, int]] = []
_sample = RFD3Sampler.sample


def _timed_sample(self, dm, n, *a, **kw):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **kw)
    WALLS.append((time.perf_counter() - t0, int(n)))
    return out


RFD3Sampler.sample = _timed_sample


def validate(out_dir, n_expected, exp_atoms, exp_res):
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
        if na != exp_atoms:
            bad.append("%s: %d atoms != %d" % (c.name, na, exp_atoms))
        if nr != exp_res:
            bad.append("%s: %d residues != %d" % (c.name, nr, exp_res))
        if nf:
            bad.append("%s: %d non-finite coords" % (c.name, nf))
    return (not bad), bad, atoms


def main():
    rung = sys.argv[1]
    batches = [int(b) for b in sys.argv[2].split(",")]
    # An optional arm tag keys an env-flag arm separately, so RFD3_TUNE_MATMUL=1 at R4 does
    # not collide with the shipped-default row for the same (rung, batch).
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    exp_atoms, target_res = RUNGS[rung]
    exp_res = target_res + 100
    fixture = pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % rung)
    specs = json.loads(fixture.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["rung"], r["batch_size_requested"]))

    for bs in batches:
        if (rung + tag, bs) in done:
            print("[e2e] %s b=%d cached" % (rung, bs), flush=True)
            continue
        nd = bs * (N_WARM + 1)
        out_dir = "/tmp/rfd3_e2e_%s%s_b%d" % (rung, tag.replace("+", "_"), bs)
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
            print("[e2e] %s b=%d ABORT: mixed chunk sizes %s" % (rung, bs, chunk_b), flush=True)
            continue
        eff = chunk_b[0]
        warm = walls[1:]
        med = statistics.median(warm)
        ok, bad, atoms = validate(out_dir, nd, exp_atoms, exp_res)
        rec = {
            "rung": rung + tag, "atoms": exp_atoms,
            "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("RFD3_")}, "fixture": str(fixture), "num_designs": nd,
            "batch_size_requested": bs, "effective_batch": eff, "num_timesteps": STEPS,
            "seed": SEED, "n_chunks": len(walls), "cold_chunk_s": round(walls[0], 3),
            "warm_chunks_s": [round(w, 3) for w in warm], "n_warm": len(warm),
            "chunk_s_median": round(med, 3), "chunk_s_min": round(min(warm), 3),
            "chunk_s_max": round(max(warm), 3),
            "spread_pct": round(100 * (max(warm) - min(warm)) / med, 2),
            "s_per_design": round(med / eff, 3),
            "designs_per_hour": round(3600 * eff / med, 1),
            "ms_per_step_per_design": round(1000 * med / eff / STEPS, 3),
            "total_wall_s": round(total, 1),
            "n_results": len(res), "atoms_written": atoms,
            "output_ok": ok, "output_fail": bad,
            "host": HOST, "card": CARD, "ttnn": TTNN, "torch": torch.__version__,
        }
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[e2e] %s b=%d (eff %d): %.3f s/design (chunk %.3f, n=%d warm, spread %.2f%%), "
              "output_ok=%s %s" % (rung, bs, eff, rec["s_per_design"], med, len(warm),
                                   rec["spread_pct"], ok, bad), flush=True)


if __name__ == "__main__":
    main()
