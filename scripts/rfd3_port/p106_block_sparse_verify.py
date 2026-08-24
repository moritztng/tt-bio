#!/usr/bin/env python3
"""p106 -- does the block-sparse atom attention arm produce the same structure?

E9.9 item 1 is a build and a fold A/B. This is the correctness half, which does not care what
the box load is: two arms in one process, 200 timesteps, one design each, same seed and same
fixture. It checks four things and none of them is a timing:

1. **The shipped arm still reproduces `5295e526ebd0b757`**, the batch-1 digest of record. If it
   does not, the harness is wrong and nothing else here means anything.
2. **The arms are two-sided.** The off arm must show `blocked == 0 and fallback == 0`, the on arm
   `shipped == 0`, at the same call total. A silent arm is what made an earlier batching result
   wrong, so the arm is verified rather than assumed.
3. **The block-sparse digest differs.** It has to: the softmax row sum reduces U terms instead of
   the full key axis. A matching digest would mean the arm never ran.
4. **The structures agree.** Same seed, same target, same trajectory, so the two coordinate sets
   are directly comparable with no alignment -- an RMSD here is a real per-atom disagreement and
   not a superposition artifact. This is the number that says whether the reassociation cost
   anything, and it is what the accuracy envelope would be scaled up from.

Also reports how many steps took each bucket and how many fell back to dense, which is the
in-production check on p103's cost model: if the fallback fraction is far from the 20 % the model
assumed, the predicted +3.787 s/design is priced against the wrong mix.
"""
import collections
import hashlib
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import block_sparse as BS                               # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p106/verify.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
DIGEST_OF_RECORD = "5295e526ebd0b757"

# Which bucket each plan picked, and how many steps found none.
PICKS = collections.Counter()
_plan = BS.plan


def _spy(indices, n_key, q_block=None, buckets=None):
    out = _plan(indices, n_key, q_block, buckets)
    if indices.shape[1] > 2000:                 # the atom site, not the DiT's 685
        PICKS["dense" if out is None else "U%d" % out[2]] += 1
    return out


BS.plan = _spy
import tt_bio.rfd3.model as M                                            # noqa: E402
M._BS.plan = _spy


def coords(cif):
    """Cartesian coordinates out of an mmCIF _atom_site loop, in file order."""
    lines = cif.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            tags, j = [], i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                tags.append(lines[j].strip())
                j += 1
            if any(t.startswith("_atom_site.") for t in tags):
                cx, cy, cz = (tags.index("_atom_site.Cartn_" + a) for a in "xyz")
                rows = []
                while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
                    f = lines[j].split()
                    if len(f) >= len(tags):
                        rows.append([float(f[cx]), float(f[cy]), float(f[cz])])
                    j += 1
                return torch.tensor(rows)
            i = j
            continue
        i += 1
    raise SystemExit("no _atom_site loop in %s" % cif)


def fold(label, on):
    out_dir = "/tmp/rfd3_p106_%s" % label
    os.system("rm -rf %s" % out_dir)
    PICKS.clear()
    BS.STATS[0] = BS.STATS[1] = BS.STATS[2] = 0
    was = BS.set_enabled(on)
    try:
        rfd3_design.run_design(json.loads(FIXTURE.read_text()), out_dir, checkpoint_dir=CKPT,
                               from_pdb=True, num_timesteps=STEPS, seed=SEED, num_designs=1,
                               batch_size=1, verbose=False)
    finally:
        BS.set_enabled(was)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    if not cifs:
        raise SystemExit("[p106] %s produced no CIF" % label)
    blocked, fallback, shipped = BS.STATS
    row = dict(arm=label, enabled=on, n_cifs=len(cifs),
               digest=hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16],
               blocked_calls=blocked, fallback_calls=fallback, shipped_calls=shipped,
               picks=dict(PICKS))
    print("[p106] %-6s digest %s  blocked=%-5d fallback=%-5d shipped=%-5d  picks=%s"
          % (label, row["digest"], blocked, fallback, shipped, dict(PICKS)), flush=True)
    return row, coords(cifs[0])


def main():
    print("[p106] steps=%d seed=%d card=%s  Q=%d buckets=%s"
          % (STEPS, SEED, os.environ.get("TT_VISIBLE_DEVICES"), BS.config()[0], BS.config()[1]),
          flush=True)
    off, xyz_off = fold("off", False)
    on, xyz_on = fold("on", True)

    checks = {}
    checks["shipped_reproduces_record"] = off["digest"] == DIGEST_OF_RECORD
    checks["off_arm_silent"] = off["blocked_calls"] == 0 and off["fallback_calls"] == 0
    checks["on_arm_took_over"] = on["shipped_calls"] == 0 and on["blocked_calls"] > 0
    checks["same_call_total"] = (off["shipped_calls"]
                                == on["blocked_calls"] + on["fallback_calls"])
    checks["digest_changed"] = on["digest"] != off["digest"]
    checks["same_atom_count"] = xyz_off.shape == xyz_on.shape

    delta = None
    if checks["same_atom_count"]:
        d = (xyz_on - xyz_off)
        per_atom = d.norm(dim=-1)
        delta = dict(n_atoms=int(xyz_off.shape[0]),
                     rmsd=round(float((per_atom ** 2).mean().sqrt()), 4),
                     max_atom_shift=round(float(per_atom.max()), 4),
                     median_atom_shift=round(float(per_atom.median()), 4),
                     max_coord_abs=round(float(d.abs().max()), 4))
        print("\n[p106] no-alignment structural delta over %d atoms: RMSD %.4f A, "
              "median shift %.4f, max shift %.4f"
              % (delta["n_atoms"], delta["rmsd"], delta["median_atom_shift"],
                 delta["max_atom_shift"]), flush=True)

    n_atom_calls = on["blocked_calls"] + on["fallback_calls"]
    if n_atom_calls:
        frac = on["fallback_calls"] / n_atom_calls
        print("[p106] dense fallback %d/%d atom calls (%.1f %%); p103's model assumed ~20 %%"
              % (on["fallback_calls"], n_atom_calls, 100 * frac), flush=True)

    print("\n" + "=" * 74, flush=True)
    for k, v in checks.items():
        print("  %-28s %s" % (k, "OK" if v else "FAIL"), flush=True)
    ok = all(checks.values())
    print("[p106] %s" % ("ALL CHECKS PASS" if ok else "CHECKS FAILED"), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(steps=STEPS, seed=SEED, fixture=str(FIXTURE),
                                   q_block=BS.config()[0], buckets=list(BS.config()[1]),
                                   digest_of_record=DIGEST_OF_RECORD,
                                   card=os.environ.get("TT_VISIBLE_DEVICES"),
                                   host=os.uname().nodename, arms=[off, on],
                                   checks=checks, structural_delta=delta,
                                   all_pass=ok), indent=2) + "\n")
    print("wrote", OUT, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
