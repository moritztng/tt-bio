"""Which op first writes garbage into the tile padding the softmax then reads?

Same wrapping as p23_bisect_ops.py, but every recorded output is read with its TILE PADDING
(`Tensor.cpu().to_torch_with_padded_shape()`), so the trace carries both the logical checksum
and what the op left in the pad region. Run isolated and sequenced and diff.
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
ap.add_argument("--max-ops", type=int, default=40)
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

TRACE = []
LAST = [False]
IN_DECODER = [0]
IN_BLOCK = [0]
NOPS = [0]


def active():
    return LAST[0] and IN_DECODER[0] and IN_BLOCK[0] and NOPS[0] < args.max_ops


def stats(t):
    """logical checksum + tile-padding characterisation of a device tensor."""
    if not isinstance(t, ttnn.Tensor):
        return None
    logical = tuple(t.shape)
    padded = tuple(t.padded_shape)
    try:
        full = t.cpu().to_torch_with_padded_shape().float()
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)}
    box = tuple(slice(0, s) for s in logical)
    log = full[box]
    rec = {
        "shape": logical, "padded": padded,
        "log_sum": round(log.double().sum().item(), 9),
        "log_absmax": round(log.abs().max().item(), 9),
    }
    if padded != logical:
        rest = full.clone()
        rest[box] = 0.0
        finite = rest[torch.isfinite(rest)]
        rec.update({
            "pad_absmax": (float(finite.abs().max()) if finite.numel() else 0.0),
            "pad_inf": int(torch.isinf(rest).sum()),
            "pad_nan": int(torch.isnan(rest).sum()),
            "pad_sum": (round(finite.double().sum().item(), 6) if finite.numel() else 0.0),
        })
    return rec


OPS = ("rms_norm", "linear", "matmul", "permute", "reshape", "add", "multiply", "subtract",
       "sigmoid", "typecast", "softmax", "scatter", "embedding", "to_layout", "pad")


def wrap_op(name):
    orig = getattr(ttnn, name)

    def patched(*a, **kw):
        out = orig(*a, **kw)
        if active():
            NOPS[0] += 1
            TRACE.append({"i": NOPS[0], "op": name, "out": stats(out)})
        return out
    patched.__name__ = name
    setattr(ttnn, name, patched)


for _n in OPS:
    if hasattr(ttnn, _n):
        wrap_op(_n)


def wrap_scope(cls, meth, counter, inputs=False):
    orig = getattr(cls, meth)

    def patched(self, *a, **kw):
        counter[0] += 1
        if inputs and LAST[0] and IN_DECODER[0] and NOPS[0] == 0:
            TRACE.append({"i": 0, "op": "<block inputs>",
                          "args": [stats(v) for v in a],
                          "kw": {k: stats(v) for k, v in sorted(kw.items())}})
        try:
            return orig(self, *a, **kw)
        finally:
            counter[0] -= 1
    setattr(cls, meth, patched)


wrap_scope(R.CompactStreamingDecoder, "run_device", IN_DECODER)
wrap_scope(R.RFD3AtomBlock, "__call__", IN_BLOCK, inputs=True)

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
        LAST[0] = last and D == 1
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        LAST[0] = False
        final[f"{contig}|D{D}"] = out.clone()
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save({"final": final, "trace": TRACE}, args.out)
print(f"wrote {args.out} with {len(TRACE)} entries", flush=True)
