"""p22 root-cause harness for the sequenced-ligand mis-fold.

Same forwards as dump_forward_for_crosstree_parity.py, plus:
  --fresh-module   rebuild the RFD3 module (and so every python-side cache) per design
  --clear-program-cache  clear the ttnn program cache between designs
Either one, if it makes the sequenced mpro answer agree with the isolated answer,
localises the contamination to that layer.
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
ap.add_argument("--fresh-module", action="store_true")
ap.add_argument("--clear-program-cache", action="store_true")
ap.add_argument("--churn", type=int, default=0)
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

from tt_bio.rfd3 import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                map_location="cpu", weights_only=True)
dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                 map_location="cpu", weights_only=True)
dev_ti = build_token_initializer(ti)
dm = build_diffusion_module(dmw)

cases = [(c, str(PDB), {"input": str(PDB), "contig": c}) for c in args.contigs]
for spec_path in args.specs:
    data = json.loads(Path(spec_path).read_text())
    src = Path(data["input"])
    if not src.is_absolute():
        src = Path(spec_path).parent / src
    cases.append((Path(spec_path).parent.name, str(src), dict(data, input=str(src))))

dump = {}
for contig, pdb_path, spec_data in cases:
    if args.fresh_module and dump:
        del dm
        import gc
        gc.collect()
        dm = build_diffusion_module(dmw)
        print(f"-- rebuilt module before {contig}", flush=True)
    if args.churn and dump:
        import ttnn as _t, tt_bio.tenstorrent as _T
        side = int((args.churn * 1024 * 1024 // 2) ** 0.5) // 32 * 32
        z = _t.full((1, 1, side, side), 1.0, dtype=_t.bfloat16,
                    layout=_t.TILE_LAYOUT, device=_T.get_device())
        _t.deallocate(z)
        print(f"-- churned {side}x{side} bf16 before {contig}", flush=True)
    if args.clear_program_cache and dump:
        import tt_bio.tenstorrent as T
        d = T.get_device()
        d.disable_and_clear_program_cache()
        d.enable_program_cache()
        print(f"-- cleared program cache before {contig}", flush=True)
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
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        dump[f"{contig}|D{D}"] = out.clone()
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

torch.save(dump, args.out)
print("wrote", args.out, flush=True)
