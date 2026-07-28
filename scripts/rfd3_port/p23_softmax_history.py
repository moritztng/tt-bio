"""Log EVERY ttnn.softmax call in a fold, process-wide, with program-cache accounting.

p23 named `ttnn.softmax` inside the decoder's RFD3AtomBlock as the first op whose output
differs between the isolated and the sequenced mpro fold. Neither run compiles a program at
that call (cache count does not move), so the entry it hits was created earlier. This says
by WHOM: every softmax call's shape/dtype/memcfg plus whether it compiled or hit.
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
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

LOG = []
TAG = ["init"]
IN_DECODER = [0]
IN_BLOCK = [0]
_softmax = ttnn.softmax


def patched_softmax(x, *a, **kw):
    dev = get_device()
    n0 = dev.num_program_cache_entries()
    out = _softmax(x, *a, **kw)
    n1 = dev.num_program_cache_entries()
    rec = {
        "tag": TAG[0],
        "where": ("decoder_block" if (IN_DECODER[0] and IN_BLOCK[0])
                  else "block" if IN_BLOCK[0] else "other"),
        "shape": tuple(x.shape),
        "padded": tuple(x.padded_shape),
        "dtype": str(x.dtype),
        "layout": str(x.layout),
        "mem": str(x.memory_config()),
        "kw": {k: str(v) for k, v in sorted(kw.items())},
        "cache": (n0, n1),
        "compiled": n1 > n0,
    }
    if IN_DECODER[0] and IN_BLOCK[0]:
        y = ttnn.to_torch(out).double()
        rec["sum"] = round(y.sum().item(), 9)
    LOG.append(rec)
    return out


ttnn.softmax = patched_softmax


def wrap_scope(cls, meth, counter):
    orig = getattr(cls, meth)

    def patched(self, *a, **kw):
        counter[0] += 1
        try:
            return orig(self, *a, **kw)
        finally:
            counter[0] -= 1
    setattr(cls, meth, patched)


wrap_scope(R.CompactStreamingDecoder, "run_device", IN_DECODER)
wrap_scope(R.RFD3AtomBlock, "__call__", IN_BLOCK)

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

final = {}
for contig, pdb_path, spec_data in cases:
    s = InputSpecification.from_dict(spec_data)
    s.validate()
    f = featurize(pdb_path, s)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    TAG[0] = f"{contig}|init"
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    for D in args.batches:
        XD = X1.expand(D, -1, -1).contiguous()
        TAG[0] = f"{contig}|D{D}"
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        final[f"{contig}|D{D}"] = out.clone()
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save({"final": final, "log": LOG}, args.out)
print(f"wrote {args.out} with {len(LOG)} softmax calls", flush=True)
