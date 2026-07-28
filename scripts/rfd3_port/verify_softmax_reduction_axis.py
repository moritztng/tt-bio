"""Gate: no attention softmax may reduce over an axis that carries tile padding.

p23 root-caused the sequenced-ligand mis-fold to exactly that: `ttnn.softmax(dim=-1)` reads the
tile padding of its last axis, and not every producer writes it (`ttnn.scatter` does not), so a
non-tile-multiple key axis makes the logical answer depend on whatever DRAM the previous fold
left behind. The fix is structural -- the key axis is padded out to a tile multiple -- so the
regression test is structural too: fold a design whose atom count is deliberately NOT a multiple
of 32 and assert every softmax inside RFD3AtomBlock has shape[-1] == padded_shape[-1].

Three softmax calls outside RFD3AtomBlock still reduce over a tile-padded axis. p24 showed
they cannot pick up heap garbage -- every op in their producer chains writes its own output
padding, and a real fold on a DRAM heap deliberately primed with +/-inf leaves their pad
region exactly 0 (scripts/rfd3_port/p24_pad_origination.py, p24_dirty_heap_taint.py). They
are listed in KNOWN_UNALIGNED rather than fixed, so this gate still FAILS on a fourth one:
the argument for leaving them alone is evidence about today's producer chains, and a refactor
that changes those chains has to come back through here.

  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> \
    python3 scripts/rfd3_port/verify_softmax_reduction_axis.py

Exits non-zero on the first violation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

ap = argparse.ArgumentParser()
ap.add_argument("--contig", default="A1-10,20,A31-40", help="419 atoms: 419 % 32 == 3")
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
args = ap.parse_args()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

# The three sites p24 measured as unable to reach heap garbage, keyed by the innermost scope
# and the ratio of the two axes that identifies them independently of design size.
KNOWN_UNALIGNED = {
    "gca:query-major",    # GatedCrossAttention upcast, (I,4,1,14)
    "gca:key-major",      # GatedCrossAttention decoder downcast, (I,4,14,3)
    "pairformer:square",  # PairformerAttention in the token initializer, (1,16,I,I)
}

SCOPE = []
VIOLATIONS = []
ALLOWED = {}
CHECKED = [0]
_softmax = ttnn.softmax


def classify(scope, shape):
    """Name a softmax site by the two axes that identify it, not by the call scope.

    Scope is unreliable -- the decoder reaches GatedCrossAttention through a traced path that
    does not go through the wrapped `run_device` -- and the (query, key) pair is both stable
    across design sizes and specific. 14 and 3 are the model's token-group sizes; if either
    ever changes, this gate stops recognising the site and fails, which is the intent.
    """
    q, k = shape[-2], shape[-1]
    if (q, k) == (1, 14):
        return "gca:query-major"
    if (q, k) == (14, 3):
        return "gca:key-major"
    if q == k:
        return "pairformer:square"
    return scope or "(unscoped)"


def patched(x, *a, **kw):
    where = "/".join(SCOPE)
    shape, padded = tuple(x.shape), tuple(x.padded_shape)
    CHECKED[0] += 1
    if shape[-1] == padded[-1]:
        return _softmax(x, *a, **kw)
    site = classify(where, shape)
    if site in KNOWN_UNALIGNED:
        ALLOWED[(site, where, shape, padded)] = ALLOWED.get((site, where, shape, padded), 0) + 1
    else:
        VIOLATIONS.append((site, where, shape, padded))
    return _softmax(x, *a, **kw)


ttnn.softmax = patched


def wrap(cls, meth, label):
    orig = getattr(cls, meth)

    def p(self, *a, **kw):
        SCOPE.append(label)
        try:
            return orig(self, *a, **kw)
        finally:
            SCOPE.pop()
    setattr(cls, meth, p)


wrap(R.RFD3AtomBlock, "__call__", "atomblock")
wrap(R.GatedCrossAttention, "run_device", "gca")
wrap(R.PairformerAttention, "__call__", "pairformer")
wrap(R.CompactStreamingDecoder, "run_device", "decoder")
wrap(R.LocalAtomTransformer, "run_device", "encoder")
wrap(R.LocalTokenTransformer, "run_device", "dit")

ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu",
                weights_only=True)
dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu",
                 weights_only=True)
dev_ti = R.build_token_initializer(ti)
dm = R.build_diffusion_module(dmw)

spec = InputSpecification.from_dict({"input": str(PDB), "contig": args.contig})
spec.validate()
f = featurize(str(PDB), spec)
f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
L = f["ref_pos"].shape[0]
with torch.no_grad():
    init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
torch.manual_seed(0)
X1 = torch.randn(1, L, 3) * 16.0
for D in args.batches:
    with torch.no_grad():
        dm(X_noisy_L=X1.expand(D, -1, -1).contiguous(), t=torch.full((D,), 8.0), f=f, **init)
    print(f"folded L={L} (L % 32 == {L % 32}) D={D}", flush=True)

print(f"checked {CHECKED[0]} softmax calls")
for (site, where, s, p), n in ALLOWED.items():
    print(f"  known-unaligned (p24: cannot reach heap garbage): {site} at {where} "
          f"shape={s} padded={p} x{n}")
missing = KNOWN_UNALIGNED - {k[0] for k in ALLOWED}
if missing:
    print(f"  note: allowlisted site(s) not exercised by this fold: {sorted(missing)}")
if VIOLATIONS:
    for site, w, s, p in VIOLATIONS:
        print(f"FAIL {w} [{site}]: shape {s} padded {p} -- reduction axis carries tile "
              f"padding and is not in KNOWN_UNALIGNED")
    raise SystemExit(1)
print(f"PASS: every softmax reduces over a tile-multiple axis, except the "
      f"{len(ALLOWED)} allowlisted shape(s)")
