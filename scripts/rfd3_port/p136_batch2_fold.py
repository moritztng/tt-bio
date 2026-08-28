#!/usr/bin/env python3
"""p136 -- X2's owed b=2 fold: does a batched design now produce the b=1 structure?

§15.4 is the claim this settles. Before X2, `_pair_transition_chunk_h` divided the L1 budget by
the batch, so at b=2 the pair Transition chunked at h=61 where b=1 chunks at 64 (514 tokens,
hidden=512), and §15.2 measured that a moved height moves the CIF. Over the 689 chunked sizes and
both hidden widths the height differed from b=1 at 62.7 % of (size, hidden) pairs at b=2. X2 took
the batch out of the numerics instead of out of the budget: `Transition.__call__` walks
`_pair_transition_slices(batch, H, h)`, every slice has batch extent 1, and `h` is a pure function
of `(w_pad, hidden, height)`. At b=1 that is a no-op with one loop iteration, which is why the
whole fix could be accepted on pc against an unchanged 3-step digest -- and why the b=2 fold
itself stayed owed: pc holds one R3 fixture at 14.53 GB against 30 GB and no swap.

Two folds, same seeds, same fixture, same everything except the batch:

    A  num_designs=D, batch_size=1     D independent b=1 designs
    B  num_designs=D, batch_size=D     the same D designs in one batched forward

and the verdict is `digest_A[i] == digest_B[i]` for every i, plus three structural facts read off
the run rather than argued: b=2 completed at all (the 92 274 688 B L1 request that closed batching
for three passes is never made), every chunk height at b=2 equals b=1's at the same
`(w_pad, hidden)`, and every slice the loop cut has batch extent 1.

**`--lift-speed-cap` is not optional above 2952 atoms, and it is a test instrument, not a fix.**
`effective_design_batch` clamps the batch to 1 above `_BATCH_SPEED_CAP_ABOVE_ATOMS` = 2952 atoms
(design.py, pinned 2026-08-24 on measured end-to-end seconds), and every chunked fixture is above
it: R3 is 4558 atoms, R4 is 6051, and the chunked pair Transition needs >= 512 tokens, which at
the 8.8-9.4 atoms/token these targets run is >= ~4500 atoms. So without the lift, arm B silently
runs at b=1 and the whole run is an A/A wearing an A/B's label. The lift raises the threshold for
this process only, prints that it did, and records `effective_design_batch` on both sides.

    ~/.coworker/scripts/benchlock.sh rfd3-fusion-programme-p6 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:rfd3-fusion-programme-p6 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio/env/bin/python3 -u scripts/rfd3_port/p136_batch2_fold.py \
      perf/p136/batch2_R3.json 200 R3 --batch=2 --lift-speed-cap

benchlock is a courtesy here rather than a requirement: the verdict is a digest comparison, not a
timing, so this arm may run while another task holds the box.
"""
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402
from fold_ab import WEIGHTS                                              # noqa: E402

FIXTURES = pathlib.Path("perf/dsfix/fixtures")

HEIGHTS = {}          # (w_pad, hidden, rows, residents) -> h, per arm
SLICES = {}           # (batch, height, h) -> set of slice batch extents
BATCHES = []          # the n each sampler call actually got


def _install_probes():
    """Record what the chunk formula was asked and what the loop actually cut."""
    _h = M._pair_transition_chunk_h

    def chunk_h(w_pad, hidden, height, residents=2):
        v = _h(w_pad, hidden, height, residents)
        HEIGHTS[(int(w_pad), int(hidden), int(height), int(residents))] = int(v)
        return v
    M._pair_transition_chunk_h = chunk_h

    _sl = M._pair_transition_slices

    def slices(batch, height, h):
        out = list(_sl(batch, height, h))
        # Does the slice list tile the tensor exactly once, one batch element per chunk? That is
        # X2's whole invariant, and it is read off the live call rather than asserted in a test.
        SLICES[(int(batch), int(height), int(h))] = {
            "n_slices": len(out), "expected": int(batch) * -(-int(height) // int(h)),
            "b_values": sorted({int(b) for b, _s, _e in out}),
            "row_extents": sorted({int(e) - int(s) for _b, s, e in out})}
        return out
    M._pair_transition_slices = slices

    _sample = RFD3Sampler.sample

    def sample(self, dm, n, *a, **k):
        BATCHES.append(int(n))
        return _sample(self, dm, n, *a, **k)
    RFD3Sampler.sample = sample


def _fold(specs, tag, steps, designs, batch, seed=42):
    out_dir = "/tmp/%s" % tag
    shutil.rmtree(out_dir, ignore_errors=True)
    HEIGHTS.clear(); SLICES.clear(); BATCHES.clear()
    t0 = time.perf_counter()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=steps, seed=seed, num_designs=designs,
                           batch_size=batch, verbose=False)
    wall = time.perf_counter() - t0
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    digests = {c.name: hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs}
    rec = {"tag": tag, "batch_arg": batch, "designs": designs, "wall_s": round(wall, 1),
           "digests": digests, "n_cifs": len(cifs),
           "sampler_batches": sorted(set(BATCHES)),
           "chunk_heights": {"%d/%d/%d/r%d" % k: v for k, v in sorted(HEIGHTS.items())},
           "slice_tiling": {"%d/%d/%d" % k: v for k, v in sorted(SLICES.items())}}
    print("[p136] %-10s batch_arg=%d  sampler batches %s  %.1f s  %d cif  %s"
          % (tag, batch, rec["sampler_batches"], wall, len(cifs),
             list(digests.values())), flush=True)
    return rec


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = argv[0] if argv else "perf/p136/batch2_R3.json"
    steps = int(argv[1]) if len(argv) > 1 else 200
    rung = argv[2] if len(argv) > 2 else "R3"
    batch = 2
    designs = None
    for a in sys.argv[1:]:
        if a.startswith("--batch="):
            batch = int(a.split("=", 1)[1])
        if a.startswith("--designs="):
            designs = int(a.split("=", 1)[1])
    designs = designs or batch
    lift = "--lift-speed-cap" in sys.argv

    specs = json.loads((FIXTURES / ("rfd3_%s.json" % rung)).read_text())
    _install_probes()

    res = {"rung": rung, "steps": steps, "batch": batch, "designs": designs,
           "lifted_speed_cap": lift, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES", "?"),
           "speed_cap_above_atoms_shipped": rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS,
           "fc1_split_silu": bool(M._FC1_SPLIT_SILU)}
    if lift:
        # For THIS process only, and it is recorded in the artifact next to the shipped value so
        # nobody reads the b=2 row as a production configuration.
        rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS = 10 ** 9
        print("[p136] _BATCH_SPEED_CAP_ABOVE_ATOMS 2952 -> 1e9 for this process. This is a TEST "
              "INSTRUMENT: production still clamps b=1 above 2952 atoms.", flush=True)

    arm_a = _fold(specs, "p136_%s_b1" % rung, steps, designs, 1)
    arm_b = _fold(specs, "p136_%s_b%d" % (rung, batch), steps, designs, batch)
    res["arm_b1"] = arm_a
    res["arm_bN"] = arm_b

    names = sorted(set(arm_a["digests"]) | set(arm_b["digests"]))
    same = {n: arm_a["digests"].get(n) == arm_b["digests"].get(n) for n in names}
    res["per_design_digest_equal"] = same
    res["batched_actually_batched"] = arm_b["sampler_batches"] == [batch]
    def tiles_once(rec):
        for k, v in rec["slice_tiling"].items():
            b = int(k.split("/")[0])
            if v["n_slices"] != v["expected"] or v["b_values"] != list(range(b)):
                return False
        return bool(rec["slice_tiling"])
    res["slices_tile_exactly_once"] = tiles_once(arm_b)
    res["heights_match_b1"] = arm_a["chunk_heights"] == arm_b["chunk_heights"]
    res["n_cifs_match"] = arm_a["n_cifs"] == arm_b["n_cifs"] == designs

    ok = (all(same.values()) and same and res["batched_actually_batched"]
          and res["slices_tile_exactly_once"] and res["heights_match_b1"]
          and res["n_cifs_match"])
    res["verdict"] = ("b=%d reproduces b=1 per design, the chunk height did not read the batch, "
                      "and the slices tiled the tensor exactly once" % batch) if ok else (
                     "FAILED -- read per_design_digest_equal, batched_actually_batched and "
                     "heights_match_b1; a False in the second means the run was an A/A")
    print("\nper-design digest equal: %s" % same)
    print("batched arm really ran at b=%d: %s   slices tile exactly once: %s   "
          "heights match b=1: %s" % (batch, res["batched_actually_batched"],
                                     res["slices_tile_exactly_once"], res["heights_match_b1"]))
    print("VERDICT: " + res["verdict"])

    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", p)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
