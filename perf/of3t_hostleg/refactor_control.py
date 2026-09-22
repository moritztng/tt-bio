#!/usr/bin/env python3
"""A/A control on the host leg's refactor: the shipped host `ref_atom_embed` must be
BIT-IDENTICAL after the block construction and the five single-feature blocks were factored out
for the device leg to share. CPU only, no card, no checkpoint.

The original is reconstructed here from `git show origin/wk/of3t:tt_bio/openfold3_host_prep.py`
rather than retyped, so the control cannot drift from what the tree actually shipped.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load_base():
    src = subprocess.run(["git", "-C", str(REPO), "show",
                          "origin/wk/of3t:tt_bio/openfold3_host_prep.py"],
                         capture_output=True, text=True, check=True).stdout
    # Import it as a member of the real package so its relative imports resolve.
    import tt_bio
    d = Path(tempfile.mkdtemp())
    f = d / "openfold3_host_prep_base.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(
        "tt_bio.openfold3_host_prep_base", f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def fake_features(n_atom, n_res, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {
        "ref_pos": r(n_atom, 3),
        "atom_mask": torch.ones(n_atom),
        "ref_charge": r(n_atom),
        "ref_mask": torch.ones(n_atom),
        "ref_element": torch.nn.functional.one_hot(
            torch.randint(0, 119, (n_atom,), generator=g), 119).float(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.randint(0, 64, (n_atom, 4), generator=g), 64).float(),
        "ref_space_uid": torch.randint(0, n_res, (n_atom,), generator=g).float(),
    }


def fake_weights(seed):
    g = torch.Generator().manual_seed(seed + 7)
    shp = {"linear_ref_pos": (128, 3), "linear_ref_charge": (128, 1),
           "linear_ref_mask": (128, 1), "linear_ref_element": (128, 119),
           "linear_ref_atom_chars": (128, 256), "linear_ref_offset": (16, 3),
           "linear_inv_sq_dists": (16, 1), "linear_valid_mask": (16, 1)}
    return {f"{k}.weight": torch.randn(*v, generator=g) * 0.05 for k, v in shp.items()}


def main():
    base = load_base()
    from tt_bio import openfold3_host_prep as new

    rows = []
    for n_atom, n_res, seed in ((602, 76, 0), (1000, 130, 1), (31, 4, 2)):
        f = fake_features(n_atom, n_res, seed)
        w = fake_weights(seed)
        cl_a, plm_a = base.ref_atom_embed(w, f)
        cl_b, plm_b = new.ref_atom_embed(w, f)
        row = {
            "n_atom": n_atom, "n_res": n_res, "seed": seed,
            "cl_bit_identical": bool(torch.equal(cl_a, cl_b)),
            "plm_bit_identical": bool(torch.equal(plm_a, plm_b)),
            "cl_max_abs_diff": float((cl_a - cl_b).abs().max()),
            "plm_max_abs_diff": float((plm_a - plm_b).abs().max()),
            "cl_shape": list(cl_a.shape), "plm_shape": list(plm_a.shape),
        }
        # the factored-out pieces are the SAME function as the ones inlined before
        dlm, vlm, isd = new.ref_atom_block_inputs(f, f["atom_mask"].float())
        row["block_shapes"] = [list(dlm.shape), list(vlm.shape), list(isd.shape)]
        singles = new.ref_atom_single_inputs(f)
        row["single_shapes"] = [list(t.shape) for t in singles]
        # the pad contract the device leg relies on: bias-free linears take zero rows to zero
        NP = ((n_atom + 31) // 32) * 32
        padded = new.ref_atom_single_inputs(f, NP)
        row["pad_rows_are_zero"] = all(
            bool((t[n_atom:] == 0).all()) for t in padded)
        row["pad_keeps_real_rows"] = all(
            bool(torch.equal(t[:n_atom], u)) for t, u in zip(padded, singles))
        row["NP"] = NP
        rows.append(row)
        print(json.dumps(row), flush=True)

    ok = all(r["cl_bit_identical"] and r["plm_bit_identical"]
             and r["pad_rows_are_zero"] and r["pad_keeps_real_rows"] for r in rows)
    out = {
        "instrument": "of3t-hostleg refactor_control.py -- the shipped HOST leg after the "
                      "refactor, against origin/wk/of3t's own bytes, on CPU",
        "base": "origin/wk/of3t:tt_bio/openfold3_host_prep.py",
        "rows": rows,
        "verdict": "BIT-IDENTICAL" if ok else "MOVED",
    }
    p = Path(__file__).with_name("REFACTOR_CONTROL.json")
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n{out['verdict']}  -> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
