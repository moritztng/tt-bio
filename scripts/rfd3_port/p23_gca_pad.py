"""Is GatedCrossAttention's softmax input tile-padded on its reduction axis, and is that
padding process-dependent? Same question p23 answered for RFD3AtomBlock, for the other
softmax in the model. Records every softmax call's pad characterisation, tagged by scope.
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
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

LOG = []
LAST = [False]
SCOPE = []
_softmax = ttnn.softmax


def patched_softmax(x, *a, **kw):
    if not LAST[0]:
        return _softmax(x, *a, **kw)
    logical, padded = tuple(x.shape), tuple(x.padded_shape)
    rec = {"scope": "/".join(SCOPE), "shape": logical, "padded": padded}
    if padded[-1] != logical[-1]:
        full = x.cpu().to_torch_with_padded_shape().float()
        cols = full[..., :logical[-2], logical[-1]:]
        fin = cols[torch.isfinite(cols)]
        rec.update({"padcol_absmax": float(fin.abs().max()) if fin.numel() else 0.0,
                    "padcol_inf": int(torch.isinf(cols).sum()),
                    "padcol_sum": round(fin.double().sum().item(), 6) if fin.numel() else 0.0})
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
    last = contig == cases[-1][0]
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
        LAST[0] = last and D == args.batches[0]
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        LAST[0] = False
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save(LOG, args.out)
uniq = {}
for r in LOG:
    uniq.setdefault((r["scope"], r["shape"], r["padded"]), []).append(r)
for k, v in uniq.items():
    r = v[0]
    extra = ""
    if "padcol_absmax" in r:
        extra = " PADCOL absmax=%.4g inf=%d sum=%.6g" % (
            r["padcol_absmax"], r["padcol_inf"], r["padcol_sum"])
    print("scope=%-22s shape=%s padded=%s n=%d%s" % (k[0], k[1], k[2], len(v), extra),
          flush=True)
