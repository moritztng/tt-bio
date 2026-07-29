"""Does each tile-padded softmax call's LOGICAL output actually depend on its pad columns?

p24's taint probe found the PairformerAttention softmax in the token initializer reduces over
pad columns that are not zero -- absmax ~12-20, i.e. score-like values, every element
non-zero. That is a different failure mode from p23's: not heap garbage, but real data that
the producer chain replicated into the padding. Whether it matters depends entirely on
whether `ttnn.softmax` folds those columns into the logical result, which p23 established for
one shape and one dtype and cannot be assumed for all.

The test is direct and does not care about the mechanism: at every softmax call whose
reduction axis is tile-padded, run the op twice --

  a) on the tensor as it is, and
  b) on a host round-trip of the SAME logical values (an upload zero-fills the padding)

-- and diff the logical outputs. Nonzero means that call's answer depends on its padding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--contigs", nargs="*", default=[])
ap.add_argument("--specs", type=Path, nargs="*", default=[])
ap.add_argument("--batches", type=int, nargs="+", default=[1])
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

SCOPE = []
HITS = {}
_softmax = ttnn.softmax


def patched(x, *a, **kw):
    out = _softmax(x, *a, **kw)
    shape, padded = tuple(x.shape), tuple(x.padded_shape)
    if shape[-1] == padded[-1]:
        return out
    where = "/".join(SCOPE) or "(top)"
    logical = ttnn.to_torch(x)
    clean = ttnn.from_torch(logical, dtype=x.dtype, layout=ttnn.TILE_LAYOUT,
                            device=x.device())
    ref = _softmax(clean, *a, **kw)
    d = (ttnn.to_torch(out).float() - ttnn.to_torch(ref).float()).abs().max().item()
    # what the pad columns hold, for context
    full = x.cpu().to_torch_with_padded_shape().float()
    cols = full[..., : shape[-2], shape[-1]:]
    fin = cols[torch.isfinite(cols)]
    key = (where, shape, padded)
    prev = HITS.get(key, (0, 0.0, 0.0))
    HITS[key] = (prev[0] + 1, max(prev[1], d),
                 max(prev[2], float(fin.abs().max()) if fin.numel() else 0.0))
    ttnn.deallocate(ref)
    ttnn.deallocate(clean)
    return out


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


wrap(R.CompactStreamingDecoder, "run_device", "decoder")
wrap(R.LocalAtomTransformer, "run_device", "encoder")
wrap(R.LocalTokenTransformer, "run_device", "dit")
wrap(R.RFD3AtomBlock, "__call__", "atomblock")
wrap(R.GatedCrossAttention, "run_device", "gca")
wrap(R.PairformerAttention, "__call__", "pairformer")

ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu",
                weights_only=True)
dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu",
                 weights_only=True)
dev_ti = R.build_token_initializer(ti)
dm = R.build_diffusion_module(dmw)

cases = [(c, str(PDB), {"input": str(PDB), "contig": c}) for c in args.contigs]
for spec_path in args.specs:
    data = json.loads(Path(spec_path).read_text())
    src = Path(data["input"])
    if not src.is_absolute():
        src = Path(spec_path).parent / src
    cases.append((Path(spec_path).parent.name, str(src), dict(data, input=str(src))))

for contig, pdb_path, spec_data in cases:
    s = InputSpecification.from_dict(spec_data)
    s.validate()
    f = featurize(pdb_path, s)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    for D in args.batches:
        with torch.no_grad():
            out = dm(X_noisy_L=X1.expand(D, -1, -1).contiguous(),
                     t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        print("%s L=%d D=%d: sum=%.12e" % (contig, L, D, out.double().sum().item()),
              flush=True)

print("\n%-14s %-22s %-22s %6s %14s %12s" % (
    "scope", "shape", "padded", "calls", "max|softmax d|", "pad absmax"), flush=True)
worst = 0.0
for (where, shape, padded), (n, d, pa) in sorted(HITS.items(), key=lambda kv: -kv[1][1]):
    worst = max(worst, d)
    print("%-14s %-22s %-22s %6d %14.6g %12.6g   %s" % (
        where, shape, padded, n, d, pa,
        "PAD-SENSITIVE" if d else "pad-independent"), flush=True)
print("\nworst logical difference caused by tile padding: %.6g" % worst, flush=True)
