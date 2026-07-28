"""Localise the sequenced-ligand contamination to a submodule.

Runs the same forwards as p22_repro.py but wraps a handful of rfd3 submodules and prints
a checksum of every output.  Run once isolated and once sequenced; the first checksum
that differs names the module that first sees the contamination.
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

TRACE = []
ACTIVE = [False]


def ck(x):
    if isinstance(x, (tuple, list)):
        return [ck(v) for v in x]
    if isinstance(x, dict):
        return {k: ck(v) for k, v in x.items()}
    if isinstance(x, ttnn.Tensor):
        try:
            x = ttnn.to_torch(x)
        except Exception:  # noqa: BLE001
            return "<ttnn?>"
    if torch.is_tensor(x):
        y = x.float()
        return (tuple(y.shape), round(y.double().sum().item(), 9),
                round(y.abs().max().item(), 9))
    return None


def wrap(cls, name, label):
    orig = getattr(cls, name)

    def patched(self, *a, **kw):
        out = orig(self, *a, **kw)
        if ACTIVE[0]:
            TRACE.append((label, ck(out)))
        return out
    setattr(cls, name, patched)


for cls_name, meth, label in (
    ("DiffusionTokenEncoder", "run_device", "tokenenc"),
    ("LocalAtomTransformer", "run_device", "encoder"),
    ("CompactStreamingDecoder", "run_device", "decoder"),
    ("RFD3DiffusionModule", "_process_", "process"),
    ("RFD3AtomBlock", "__call__", "atomblock"),
    ("GatedCrossAttention", "run_device", "gca"),
    ("CompactStreamingDecoder", "_pack_atoms_device", "pack"),
    ("CompactStreamingDecoder", "_unpack_atoms_device", "unpack"),
):
    cls = getattr(R, cls_name, None)
    if cls is None or not hasattr(cls, meth):
        print(f"skip {cls_name}.{meth}")
        continue
    wrap(cls, meth, label)

# also every module-level helper that produces a device tensor from host data
for fn_name in ("_sparse_qk_inputs",):
    orig = getattr(R, fn_name)

    def mk(orig=orig, label=fn_name):
        def patched(*a, **kw):
            out = orig(*a, **kw)
            if ACTIVE[0]:
                TRACE.append((label, ck(out[0])))
            return out
        return patched
    setattr(R, fn_name, mk())

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
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    f = featurize(pdb_path, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    for D in args.batches:
        XD = X1.expand(D, -1, -1).contiguous()
        ACTIVE[0] = last and D == 1
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        ACTIVE[0] = False
        final[f"{contig}|D{D}"] = out.clone()
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save({"final": final, "trace": TRACE}, args.out)
print(f"wrote {args.out} with {len(TRACE)} trace entries", flush=True)
