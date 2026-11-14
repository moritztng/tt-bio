"""The design writer, tt-bio's against upstream's, on the same engine output.

`write_cif` needs a model output (coordinates, residue types, atom37 occupancy, insertion
logits), and the model is another row's work. So the output is synthesized, and made
CONSISTENT, because the writer checks it: designed (`<M>`) positions become glycine, the
atom mask is each residue type's atom37 template, and coordinates are the input's plus
seeded noise. Then the same tensors go through upstream `rfd4_proteina_prep.write_cif`
and through `tt_bio.rfd4.write_design`, each in its own process, and the CIFs must match
line for line.

Excluded from the comparison, and only these: the data block name and `_entry.id`, which
atomworks takes from the output FILE NAME, and `_entry.date/time/author`, which it takes
from the clock and the login name.

    rfd4-ref-env/venv/bin/python scripts/rfd4_port/writer_parity.py <annotated.cif> [--seed 0]

Upstream runs from the prep clone (PREP_SRC); ours from this checkout.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREP_SRC = Path("/home/moritz/rfd4-prep-src/repo/src")
STAMP = re.compile(r"^(data_|_entry\.(id|date|time|author)\b)")


def write_one(impl: str, cif: str, out: str, seed: int) -> None:
    import types

    import torch

    if impl == "vendored":
        from tt_bio._vendor.rfd4_prep._cif import encoding as enc
        from tt_bio.rfd4 import featurize, write_design

        feats, arr = featurize(cif, seed=seed)
        write = write_design
    else:
        from rfd4_proteina_prep._cif import encoding as enc
        from rfd4_proteina_prep.processing import prepare_cif, write_cif

        feats, arr = prepare_cif(cif, seed=seed, device=torch.device("cpu"))
        write = lambda o, f, a, p, row=0: write_cif(o, f, a, p, row=row)  # noqa: E731

    E = enc.UNIFIED_ATOM37_ENCODING
    n = feats["padding_mask"].shape[-1]
    rt = feats["residue_type"].clone()
    rt[rt == E.token_to_idx["<M>"]] = E.token_to_idx["GLY"]
    slots = torch.as_tensor(E.idx_to_atom != "")
    g = torch.Generator().manual_seed(seed)
    output = types.SimpleNamespace(
        coords_nm=feats["coords_nm"] + 0.05 * torch.randn(feats["coords_nm"].shape, generator=g),
        residue_type=rt,
        atom_mask=slots[rt],
        insertion_logits=torch.zeros((1, n, n)),
    )
    write(output, feats, arr, Path(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--_worker", nargs=2, metavar=("IMPL", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._worker:
        write_one(a._worker[0], a.cif, a._worker[1], a.seed)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        outs = {}
        for impl, path in (("upstream", PREP_SRC), ("vendored", REPO)):
            outs[impl] = Path(tmp) / f"{impl}.cif"
            r = subprocess.run([sys.executable, __file__, a.cif, "--seed", str(a.seed),
                                "--_worker", impl, str(outs[impl])],
                               env={**__import__("os").environ, "PYTHONPATH": f"{path}:" +
                                    __import__("os").environ.get("PYTHONPATH", "")},
                               capture_output=True, text=True)
            if r.returncode:
                print(f"{impl} failed:\n{r.stderr[-1500:]}")
                return 1
        lines = {k: [l for l in v.read_text().splitlines() if not STAMP.match(l)]
                 for k, v in outs.items()}
        if lines["upstream"] != lines["vendored"]:
            for i, (x, y) in enumerate(zip(lines["upstream"], lines["vendored"])):
                if x != y:
                    print(f"line {i}: upstream {x!r}\n         ours     {y!r}")
                    break
            print(f"DIFFER ({len(lines['upstream'])} vs {len(lines['vendored'])} lines)")
            return 1
        print(f"IDENTICAL: {len(lines['vendored'])} lines, excluding the file-name id and the clock stamp")
        return 0


if __name__ == "__main__":
    sys.exit(main())
