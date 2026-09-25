"""Is `EvoformerOnDevice`'s MSA-mask cache sound across BindCraft 2's draws? Card-free.

BindCraft 2 samples a binder length per trajectory and `run_arm.py` builds one
`EvoformerOnDevice` for the whole campaign, so its mask caches outlive every trajectory.
The cache key used to be `tuple(mask.shape)` alone. That shape is the token axis AFTER
`_pad_inputs` rounds it up to 32, so every binder length in a bucket shares it: against
the 115-residue PD-L1 target, binders 142, 144 and 151 all land on `(1, 288)` carrying
257, 259 and 266 real residues. The first draw in a bucket populated the entry and every
later draw was served ITS mask, so the difference ran as padding -- 9 real residues for
binder 151.

`profile_s1` (qb2 card 0, seed 1) drew 111, 142, 109, 151, 144 in that order. Its
trajectories on a fresh mask are healthy and the first one served a stale mask is the one
that reported iptm = ptm = 1.00 and plddt_loss 0.07 frozen for all 50 screen rounds.

`--legacy` re-keys on shape alone to show the defect this test now guards.
"""
import hashlib
import sys

import numpy as np
import torch

sys.path.insert(0, "perf/bcx_predictor")
from splice import BUCKET, EvoformerOnDevice, _pad32  # noqa: E402

TARGET = 115  # target_hPDL1, read off pool_ckpt2/sequences.jsonl
DRAWS = [("traj1", 111), ("traj2", 142), ("traj3", 109), ("traj4", 151), ("traj5", 144)]


class StubDev:
    """`up` stamps each upload, so a served tensor names the draw that uploaded it."""

    def __init__(self):
        self.uploads = 0

    def up(self, t):
        self.uploads += 1
        return {"upload": self.uploads, "ones": int(t.sum().item()), "n": t.shape[-1]}


def msa_mask_for(binder_len):
    """What BindCraft 2 hands the splice, padded the way `_pad_inputs` pads it."""
    n = TARGET + binder_len
    n32 = _pad32(n)
    mask = torch.zeros(1, n32)
    mask[0, :n] = 1.0
    return n, n32, mask


def pair_mask_for(n, n32):
    pm = torch.zeros(n32, n32)
    pm[:n, :n] = 1.0
    return pm


def main():
    legacy = "--legacy" in sys.argv
    evo = EvoformerOnDevice.__new__(EvoformerOnDevice)
    evo.dev = StubDev()
    evo._mask_dev = {}
    evo._pair_mask_dev = {}
    if legacy:
        def _legacy_mask(mask_np, _cache={}, _dev=evo.dev):
            """The key this test guards against: the padded shape and nothing else."""
            key = tuple(np.asarray(mask_np).shape)
            if key not in _cache:
                _cache[key] = _dev.up(
                    torch.from_numpy(np.asarray(mask_np, dtype=np.float32).copy()))
            return _cache[key]

        evo._mask = _legacy_mask

    print("key = %s   BUCKET = %d   target = %d"
          % ("shape only (legacy)" if legacy else "shape + content hash", BUCKET, TARGET))
    print()
    print("%-7s %-7s %-5s %-5s %-6s %-6s %s"
          % ("draw", "binder", "n", "n32", "want", "served", "verdict"))
    stale = []
    for name, blen in DRAWS:
        n, n32, mask = msa_mask_for(blen)
        want = int(mask.sum())
        got = evo._mask(mask.numpy())
        ok = got["ones"] == want and got["n"] == n32
        if not ok:
            stale.append((name, blen, want, got))
        print("%-7s %-7d %-5d %-5d %-6d %-6d %s"
              % (name, blen, n, n32, want, got["ones"],
                 "own mask" if ok else "STALE: %d real residues run as padding"
                 % (want - got["ones"])))

    print()
    print("device uploads: %d for %d draws" % (evo.dev.uploads, len(DRAWS)))
    if stale:
        print()
        print("FAIL: %d of %d draws were served another draw's MSA mask" % (len(stale), len(DRAWS)))
        return 1
    print("PASS: every draw was served its own MSA mask")

    # The pair-mask cache keyed on `(shape, sum)`, which separates these draws but not two
    # masks of equal weight and different placement. Content keying covers both.
    seen = {}
    for name, blen in DRAWS:
        n, n32, _ = msa_mask_for(blen)
        pm = np.ascontiguousarray(pair_mask_for(n, n32).numpy())
        k = (pm.shape, hashlib.blake2b(pm.tobytes(), digest_size=16).digest())
        if k in seen and seen[k] != (n, n32):
            print("FAIL: pair-mask key collides %s with %s" % (name, seen[k]))
            return 1
        seen[k] = (n, n32)
    print("PASS: %d distinct pair masks, %d distinct content keys" % (len(DRAWS), len(seen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
