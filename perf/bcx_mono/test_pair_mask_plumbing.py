#!/usr/bin/env python3
"""The pair mask reaches the 48 blocks. No card, and it fails on the unfixed splice.

The defect this row root-caused was plumbing, not arithmetic: `splice.py` handed the device
Evoformer `masks["msa"]` and nothing else, so the pair track ran unmasked on every fold
BindCraft 2 pads. It cost the positive control pLDDT 0.534 against BindCraft 2's own 0.950 on a
natural 115-residue protein (`masked_fold.json`). A defect that lives in an argument list is
testable without a device, which is the point of this file: it stubs the device side and asserts
that a padded fold's pair mask arrives at `Dev.stack`, and that an unpadded fold's does not
(`af2_pair_masks` returns `(None, None)` on an all-ones mask, which is what keeps every fold
PXDesign runs today bit-exact).
"""
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "perf" / "bcx_predictor"), str(ROOT)):
    sys.path.insert(0, p)

from splice import EvoformerOnDevice, _pad_inputs


class StubDev:
    """Just enough `afgrad.Dev` to record what the stack was asked to run."""

    device = None

    def __init__(self):
        self.seen = []

    def up(self, t):
        return t

    def down(self, t, shape):
        return torch.zeros(shape)

    def sync(self):
        pass

    def stack(self, m, z, k_extra, k_evo, ckpt=False, msa_mask=None, pair_masks=(None, None)):
        self.seen.append({"k_evo": k_evo, "msa_mask": msa_mask, "pair_masks": pair_masks})
        return m, z


class StubEvo(EvoformerOnDevice):
    """`_pair_mask` without ttnn: the same `(None, None)`-on-all-ones contract as
    `af2.af2_pair_masks`, so the plumbing is what is under test and not the upload."""

    def _mask(self, mask_np):
        return ("msa_mask", tuple(np.shape(mask_np)))

    def _pair_mask(self, pm):
        if bool((pm == 1).all()):
            return (None, None)
        return (("multiply", tuple(pm.shape)), ("key_bias", tuple(pm.shape)))


def one(n, pad):
    """A fold of `n` real residues with `pad` masked, as BindCraft 2 hands it over."""
    total = n + pad
    msa = np.zeros((2, total, 256), np.float32)
    pair = np.zeros((total, total, 128), np.float32)
    seq = np.concatenate([np.ones(n), np.zeros(pad)]).astype(np.float32)
    return msa, pair, np.tile(seq, (2, 1)), seq[:, None] * seq[None, :]


fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}")
    if not ok:
        fails.append(f"{name}: got {got!r}, want {want!r}")


# 1. `as_jax` takes the pair mask. On the unfixed splice `stack` has three parameters and the
#    fold runs with the pair track unmasked; this is the signature that broke.
dev = StubDev()
evo = StubEvo(dev, k_evo=48)
check("as_jax arity", evo.as_jax().__wrapped__.__code__.co_argcount
      if hasattr(evo.as_jax(), "__wrapped__") else evo.as_jax().__code__.co_argcount, 4)

# 2. a padded fold: the pair mask reaches the stack.
msa, pair, mm, pm = one(115, 13)
evo._primal(msa, pair, mm, pm)
check("padded fold pair_masks", dev.seen[-1]["pair_masks"],
      (("multiply", (128, 128)), ("key_bias", (128, 128))))

# 3. an unpadded fold: nothing is passed, so the arithmetic is the one this port shipped.
dev.seen.clear()
msa, pair, mm, pm = one(192, 0)
evo._primal(msa, pair, mm, pm)
check("unpadded fold pair_masks", dev.seen[-1]["pair_masks"], (None, None))

# 4. `_pad_inputs` masks what it adds. tt-bio's own 32-bucket padding has to be masked out on
#    the pair axis too, or the fix is undone by the pad the splice itself appends.
m, z, mk, pmk, n, n32 = _pad_inputs(torch.zeros(2, 100, 256), torch.zeros(100, 100, 128),
                                    torch.ones(2, 100), torch.ones(100, 100))
check("_pad_inputs n32", (n, n32), (100, 128))
check("_pad_inputs pair_mask shape", tuple(pmk.shape), (128, 128))
check("_pad_inputs pair_mask real block", float(pmk[:100, :100].min()), 1.0)
check("_pad_inputs pair_mask pad row", float(pmk[100:, :].max()), 0.0)
check("_pad_inputs pair_mask pad col", float(pmk[:, 100:].max()), 0.0)
check("_pad_inputs msa_mask pad", float(mk[:, 100:].max()), 0.0)

# 5. the stack replacement hands BindCraft 2's own pair mask over, not a constructed one.
src = (ROOT / "perf" / "bcx_predictor" / "splice.py").read_text()
check('on_device passes masks["pair"]', 'masks["pair"]' in src, True)

if fails:
    raise SystemExit("FAILED:\n" + "\n".join(fails))
print(f"\nall {5 + 6} checks pass")
