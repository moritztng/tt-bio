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
4. **How far the structure moves**, on the backbone and on the sequence.

On (4): the first version of this compared all atoms and failed its own shape check. That was the
instrument's fault. RFD3 designs a sequence, so the two arms can give a residue different
identities and therefore different sidechain atom counts (5126 against 5124), and no all-atom
array can be subtracted across that. The backbone is what both arms always have, and since both
ran the same seed on the same target down the same trajectory it needs no superposition -- an
RMSD here is a real per-atom disagreement, not an alignment artifact.

What (4) does NOT do is decide whether the arm is acceptable. The pre-registered bar is on design
quality -- success rate and ipTM/pLDDT over many designs -- and "differs from the shipped chain"
is a different question from "is worse than it". A design model that produces a different, equally
good design has not regressed. This number sizes the envelope; it does not replace it.

Also reports how many steps took each bucket and how many fell back to dense, which is the
in-production check on p103's cost model: if the fallback fraction is far from the 20 % the model
assumed, the predicted prize is priced against the wrong mix.
"""
import collections
import hashlib
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p106b_cif_compare import parse as _parse_cif                        # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import block_sparse as BS                               # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p106/verify.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
DIGEST_OF_RECORD = "5295e526ebd0b757"
BACKBONE = ("N", "CA", "C", "O")
DIRS = {"off": "/tmp/rfd3_p106_off", "on": "/tmp/rfd3_p106_on"}

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


def compare(dir_a, dir_b):
    """Backbone RMSD and sequence identity between two output directories."""
    at_a, seq_a = _parse_cif(sorted(pathlib.Path(dir_a).glob("*.cif"))[0])
    at_b, seq_b = _parse_cif(sorted(pathlib.Path(dir_b).glob("*.cif"))[0])
    res = sorted(set(seq_a) & set(seq_b))
    same = sum(1 for k in res if seq_a[k] == seq_b[k])
    keys = [k + (nm,) for k in res for nm in BACKBONE
            if k + (nm,) in at_a and k + (nm,) in at_b]
    d = (torch.tensor([at_a[k] for k in keys])
         - torch.tensor([at_b[k] for k in keys])).norm(dim=-1)
    return dict(n_atoms_a=len(at_a), n_atoms_b=len(at_b),
                n_residues_a=len(seq_a), n_residues_b=len(seq_b),
                n_residues_shared=len(res), n_residues_differing=len(res) - same,
                sequence_identity=round(same / len(res), 5) if res else None,
                n_backbone_compared=len(keys),
                backbone_rmsd=round(float((d ** 2).mean().sqrt()), 4),
                backbone_median_shift=round(float(d.median()), 4),
                backbone_p99_shift=round(float(d.quantile(0.99)), 4),
                backbone_max_shift=round(float(d.max()), 4))


def fold(label, on):
    out_dir = DIRS[label]
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
    return row


def main():
    print("[p106] steps=%d seed=%d card=%s  Q=%d buckets=%s"
          % (STEPS, SEED, os.environ.get("TT_VISIBLE_DEVICES"), BS.config()[0], BS.config()[1]),
          flush=True)
    off = fold("off", False)
    on = fold("on", True)

    checks = {}
    checks["shipped_reproduces_record"] = off["digest"] == DIGEST_OF_RECORD
    checks["off_arm_silent"] = off["blocked_calls"] == 0 and off["fallback_calls"] == 0
    checks["on_arm_took_over"] = on["shipped_calls"] == 0 and on["blocked_calls"] > 0
    checks["same_call_total"] = (off["shipped_calls"]
                                == on["blocked_calls"] + on["fallback_calls"])
    checks["digest_changed"] = on["digest"] != off["digest"]

    delta = compare(DIRS["off"], DIRS["on"])
    checks["residue_count_unchanged"] = delta["n_residues_a"] == delta["n_residues_b"]
    print("\n[p106] sequence identity %.4f (%d of %d residues differ)"
          % (delta["sequence_identity"], delta["n_residues_differing"],
             delta["n_residues_shared"]), flush=True)
    print("[p106] backbone over %d atoms: RMSD %.4f A, median %.4f, p99 %.4f, max %.4f"
          % (delta["n_backbone_compared"], delta["backbone_rmsd"],
             delta["backbone_median_shift"], delta["backbone_p99_shift"],
             delta["backbone_max_shift"]), flush=True)
    print("[p106] all-atom counts %d vs %d -- the difference is the sequence, not a lost residue"
          % (delta["n_atoms_a"], delta["n_atoms_b"]), flush=True)

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
