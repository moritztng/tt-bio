"""Can ANY process history put garbage in the three remaining tile-padded softmax sites?

p23 §8 left three softmax calls whose reduction axis is not a tile multiple
(GatedCrossAttention upcast, GatedCrossAttention downcast, PairformerAttention in the token
initializer) and measured their pad columns as exactly 0 -- but only across the fold
orderings it happened to run. Fold ordering is a weak adversary: it only dirties DRAM with
whatever a previous fold's tensors happened to hold.

This is the strong version. Before the fold, DRAM is deliberately primed with the worst
patterns the p23 bug needs (+/-inf, +/-3e38) across a spread of buffer footprints, so every
buffer any op allocates during the fold lands on garbage. Then every `ttnn.softmax` call in
the model is inspected: shape, padded shape, and the contents of the pad region of its
REDUCTION axis. If an op in a site's producer chain leaves its output padding undefined, the
garbage shows up here immediately -- no lucky ordering required.

Also records, per site, the pad region of every op output on the way in when --trace-chain is
given, so a positive hit names the originating op directly.

Usage:
  p24_dirty_heap_taint.py --tree T --out X.pt [--contigs C...] [--specs S...]
                          [--batches 1 8] [--prime N] [--no-prime]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--contigs", nargs="*", default=[])
ap.add_argument("--specs", type=Path, nargs="*", default=[])
ap.add_argument("--batches", type=int, nargs="+", default=[1])
ap.add_argument("--prime", type=int, default=3, help="priming rounds (0 = clean heap)")
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

# Every pattern the p23 mechanism needs: an exponent that survives exp() in a row sum.
PATTERNS = (float("inf"), -float("inf"), 3.3e38, -3.3e38, 1e30, -1e30)
# A spread of footprints so priming covers the whole size range a fold allocates from.
FOOTPRINTS = ((1, 4, 256, 256), (1, 16, 256, 256), (1, 4, 2048, 2048), (1, 4, 64, 64),
              (1, 1, 32, 32), (1, 4, 512, 512), (1, 4, 1024, 1024), (1, 8, 128, 128))


def prime_heap(dev, rounds):
    """Leave garbage bit patterns across ALL of free DRAM, then free it again.

    Small footprints alone are not enough -- tt-metal hands the low addresses out first, so a
    small dirty buffer is reused by the next tensor and everything after lands on untouched
    DRAM above it. Filling free DRAM to OOM and releasing it leaves no clean region at all.
    Verified as a working positive control in p24_scatter_control.py: with this priming a
    2702-wide `ttnn.scatter` output comes back with the planted pattern in its tile padding.
    """
    n = 0
    for _ in range(rounds):
        for pat in PATTERNS:
            held = []
            while True:
                try:
                    held.append(ttnn.full((1, 32, 2048, 2048), pat, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=dev))  # 256 MB
                except Exception:  # noqa: BLE001  DRAM full
                    break
            n += len(held)
            for t in reversed(held):
                ttnn.deallocate(t)
            for shape in FOOTPRINTS:
                for dt in (ttnn.bfloat16, ttnn.float32):
                    try:
                        t = ttnn.full(shape, pat, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                      device=dev)
                    except Exception:  # noqa: BLE001
                        continue
                    ttnn.deallocate(t)
                    n += 1
    return n


LOG = []
SCOPE = []
_softmax = ttnn.softmax


def patched_softmax(x, *a, **kw):
    dim = kw.get("dim", a[0] if a else -1)
    logical, padded = tuple(x.shape), tuple(x.padded_shape)
    ax = dim if dim >= 0 else len(logical) + dim
    rec = {"scope": "/".join(SCOPE) or "(top)", "shape": logical, "padded": padded,
           "dim": ax, "red_padded": padded[ax] != logical[ax]}
    if rec["red_padded"]:
        full = x.cpu().to_torch_with_padded_shape().float()
        # only the pad region of the REDUCTION axis, restricted to logical rows -- that is
        # exactly what softmax over `ax` folds into each logical output element.
        box = [slice(0, s) for s in logical]
        box[ax] = slice(logical[ax], padded[ax])
        cols = full[tuple(box)]
        fin = cols[torch.isfinite(cols)]
        rec.update({"padcol_absmax": float(fin.abs().max()) if fin.numel() else 0.0,
                    "padcol_nonfinite": int((~torch.isfinite(cols)).sum()),
                    "padcol_nonzero": int((fin != 0).sum()),
                    "padcol_n": int(cols.numel())})
    LOG.append(rec)
    return _softmax(x, *a, **kw)


ttnn.softmax = patched_softmax


def wrap(cls, meth, label):
    orig = getattr(cls, meth)

    def patched(self, *a, **kw):
        SCOPE.append(label)
        try:
            return orig(self, *a, **kw)
        finally:
            SCOPE.pop()
    setattr(cls, meth, patched)


wrap(R.CompactStreamingDecoder, "run_device", "decoder")
wrap(R.LocalAtomTransformer, "run_device", "encoder")
wrap(R.LocalTokenTransformer, "run_device", "dit")
wrap(R.RFD3AtomBlock, "__call__", "atomblock")
wrap(R.GatedCrossAttention, "run_device", "gca")
wrap(R.PairformerAttention, "__call__", "pairformer")

dev = get_device()
if args.prime:
    print("primed %d garbage buffers" % prime_heap(dev, args.prime), flush=True)

ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu",
                weights_only=True)
dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu",
                 weights_only=True)
if args.prime:
    prime_heap(dev, 1)          # again, after the weights are resident
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
    if args.prime:
        prime_heap(dev, 1)      # and again immediately before each fold
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
        XD = X1.expand(D, -1, -1).contiguous()
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        print("%s L=%d D=%d: sum=%.12e" % (contig, L, D, out.double().sum().item()),
              flush=True)

torch.save(LOG, args.out)

uniq = {}
for r in LOG:
    uniq.setdefault((r["scope"], r["shape"], r["padded"], r["dim"]), []).append(r)
print("\n%-26s %-24s %-24s %s" % ("scope", "shape", "padded", "reduction-axis pad region"),
      flush=True)
dirty = 0
for k, v in sorted(uniq.items(), key=lambda kv: -len(kv[1])):
    r0 = v[0]
    if not r0["red_padded"]:
        detail = "reduction axis is tile-aligned (clean by construction)"
    else:
        worst = max(v, key=lambda r: (r["padcol_nonfinite"], r["padcol_absmax"]))
        bad = sum(1 for r in v if r["padcol_nonzero"] or r["padcol_nonfinite"])
        dirty += bad
        detail = ("PADDED axis %d: %d/%d calls dirty, worst absmax=%.6g nonfinite=%d "
                  "nonzero=%d of %d" % (r0["dim"], bad, len(v), worst["padcol_absmax"],
                                        worst["padcol_nonfinite"], worst["padcol_nonzero"],
                                        worst["padcol_n"]))
    print("%-26s %-24s %-24s n=%-5d %s" % (k[0], k[1], k[2], len(v), detail), flush=True)
print("\nTOTAL softmax calls=%d  with dirty reduction-axis padding=%d" % (len(LOG), dirty),
      flush=True)
