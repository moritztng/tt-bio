"""Hash everything RFD4's host path produces, so two trees can be compared byte for byte.

    PYTHONPATH=<tree> python perf/rfd4/hostio/identity.py --out DIR spec.yaml [spec.yaml ...]

For each spec and seed: sha256 of every feature tensor (dtype, shape and bytes) and of every CIF
written from two stand-in engine outputs (tests/test_rfd4_write_designs.py), with the CIF's
clock pinned so the file is a function of its inputs. Prints one JSON manifest.
"""
import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()


class _Clock(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


def _tensor_sha(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype} {tuple(t.shape)}".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="1234,7")
    p.add_argument("specs", nargs="+")
    a = p.parse_args()
    sys.path.insert(0, str(HERE.parents[3] / "tests"))
    from test_rfd4_write_designs import _stand_in

    import tt_bio._vendor.atomworks.io.utils.io_utils as io_utils
    from tt_bio import rfd4

    io_utils.datetime = _Clock
    out, manifest = Path(a.out), {}
    for spec in a.specs:
        for seed in map(int, a.seeds.split(",")):
            d = out / f"{Path(spec).stem}-s{seed}"
            ex = rfd4.prepare_examples(spec, d / "inputs")[0]
            features, atom_array = rfd4.featurize(ex.cif, seed=seed)
            rec = {"input_cif": hashlib.sha256(ex.cif.read_bytes()).hexdigest(),
                   "features": {k: _tensor_sha(v) for k, v in sorted(features.items())}}
            torch.manual_seed(seed)
            for g in range(2):
                output = _stand_in(features, 2)
                if g:  # a second output: other residue types and noise
                    gen = torch.Generator().manual_seed(seed + g)
                    output = type(output)(coords_nm=output.coords_nm + 0.03 * torch.randn(output.coords_nm.shape, generator=gen) * output.atom_mask[..., None],
                                          residue_type=output.residue_type, atom_mask=output.atom_mask,
                                          insertion_logits=torch.randn(output.insertion_logits.shape, generator=gen))
                for k in range(2):
                    path = d / f"g{g}d{k}.cif"
                    try:  # the stand-in has no ligand residue types, so a ligand spec's motif check refuses it
                        rfd4.write_design(output, features, atom_array, path, row=k)
                        rec[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except ValueError as e:
                        rec[path.name] = f"ValueError: {e}"
            manifest[d.name] = rec
    print(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
