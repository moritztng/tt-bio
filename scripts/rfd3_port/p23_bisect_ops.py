"""Name the op behind the sequenced-ligand divergence (p23, one level under p22).

p22 bisected the contamination to `RFD3AtomBlock.__call__` inside the decoder: bit-identical
inputs, different output. This wraps every ttnn op the block calls and records, per call,
the argument fingerprint, the output checksum, and the device's program-cache entry count
before/after -- so the first differing checksum names the op, and the cache counter says
whether that op compiled a fresh program (isolated) or reused one (sequenced).

Run once isolated (--specs only) and once sequenced (--contigs ... --specs), then diff.
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

TRACE = []
RECORD = [False]      # last design, D=1 only
IN_DECODER = [0]
IN_BLOCK = [0]


def active():
    return RECORD[0] and IN_DECODER[0] and IN_BLOCK[0]


def ck(x):
    """(shape, float64 sum, absmax) of a device or host tensor."""
    if isinstance(x, ttnn.Tensor):
        try:
            x = ttnn.to_torch(x)
        except Exception as e:  # noqa: BLE001
            return f"<to_torch failed {e}>"
    if torch.is_tensor(x):
        y = x.float().double()
        return (tuple(x.shape), round(y.sum().item(), 9), round(y.abs().max().item(), 9))
    return None


def spec(x):
    """Cheap structural fingerprint -- no device readback."""
    if isinstance(x, ttnn.Tensor):
        return ("T", tuple(x.shape), tuple(x.padded_shape), str(x.dtype), str(x.layout),
                str(x.memory_config()))
    if torch.is_tensor(x):
        return ("t", tuple(x.shape), str(x.dtype))
    if isinstance(x, (int, float, bool, str, type(None))):
        return x
    if isinstance(x, (tuple, list)):
        return [spec(v) for v in x]
    return type(x).__name__


OPS = ("rms_norm", "linear", "matmul", "permute", "reshape", "add", "multiply", "subtract",
       "sigmoid", "typecast", "softmax", "scatter", "layer_norm", "embedding", "to_layout",
       "pad", "transpose", "concat")


def wrap_op(name):
    orig = getattr(ttnn, name)

    def patched(*a, **kw):
        if not active():
            return orig(*a, **kw)
        dev = get_device()
        before = dev.num_program_cache_entries()
        out = orig(*a, **kw)
        after = dev.num_program_cache_entries()
        TRACE.append({
            "op": name,
            "args": [spec(v) for v in a],
            "kwargs": {k: spec(v) for k, v in sorted(kw.items())},
            "cache": (before, after),
            "out": ck(out),
        })
        return out
    patched.__name__ = name
    setattr(ttnn, name, patched)


for _n in OPS:
    if hasattr(ttnn, _n):
        wrap_op(_n)


def wrap_scope(cls, meth, counter, label=None):
    orig = getattr(cls, meth)

    def patched(self, *a, **kw):
        counter[0] += 1
        if label and RECORD[0] and IN_DECODER[0]:
            TRACE.append({"op": f"<{label} entry>",
                          "inputs": [ck(v) if isinstance(v, ttnn.Tensor) else spec(v)
                                     for v in a],
                          "kwinputs": {k: (ck(v) if isinstance(v, ttnn.Tensor) else spec(v))
                                       for k, v in sorted(kw.items())}})
        try:
            return orig(self, *a, **kw)
        finally:
            counter[0] -= 1
    setattr(cls, meth, patched)


wrap_scope(R.CompactStreamingDecoder, "run_device", IN_DECODER)
wrap_scope(R.RFD3AtomBlock, "__call__", IN_BLOCK, label="atomblock")

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
        RECORD[0] = last and D == 1
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        RECORD[0] = False
        final[f"{contig}|D{D}"] = out.clone()
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save({"final": final, "trace": TRACE,
            "cache_entries": get_device().num_program_cache_entries()}, args.out)
print(f"wrote {args.out} with {len(TRACE)} trace entries", flush=True)
