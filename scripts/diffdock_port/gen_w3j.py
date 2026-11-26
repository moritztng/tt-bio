"""Dump the Wigner 3j tensors DiffDock's tensor products contract over.

Runs in the upstream reference environment (e3nn 0.5.0), writes
``tt_bio/diffdock/w3j_tables.json``. Not part of the package's runtime: tt-bio ships the
table, not the generator, the same way it ships ``tt_bio/rfd4/ae_tables.json``.

These are constants of the irreps, not learned values, but they are not free to choose: a
3j tensor is unique only up to sign, and the checkpoint's weights were fitted against
e3nn's sign. Taking them from e3nn is the only way to be sure the sign matches; the test
then re-derives the same table here and checks equivariance without e3nn.

    /tmp/dfdvenv/bin/python scripts/diffdock_port/gen_w3j.py <out.json>
"""
import json
import sys

import torch
from e3nn import o3

# The triples reachable in the released DiffDock-L, as two separate families.
#
# 1. The weighted convolutions. Feature irreps never exceed l=1 and conv outputs never
#    exceed l=1, so only the harmonics side runs high -- the torsion conv's harmonics come
#    out of FullTensorProduct(0e+1o+2e, 2e) and reach l=4, paired with l_out=0.
# 2. That FullTensorProduct itself. It is unweighted, but it still contracts 3j tensors, and
#    its *first* argument is the harmonics, so it reaches l1=2 -- outside family 1 entirely.
#    Missing these is not a silent wrong answer: wigner_3j raises and names this script.
CONV = {
    (l1, l2, l3)
    for l1 in range(2)
    for l2 in range(5)
    for l3 in range(2)
    if abs(l1 - l2) <= l3 <= l1 + l2
}
FULL_TP_TOR = {
    (l1, 2, l3)
    for l1 in (0, 1, 2)
    for l3 in range(5)
    if abs(l1 - 2) <= l3 <= l1 + 2
}
NEEDED = sorted(CONV | FULL_TP_TOR)


def main(out_path):
    table = {}
    for l1, l2, l3 in NEEDED:
        # dtype=float64 matters: e3nn's default is float32, and a table rounded to
        # float32 carries a 2e-8 error that shows up directly as an equivariance
        # residual. These are exact constants; ship them at full precision.
        w = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
        assert w.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
        table[f"{l1},{l2},{l3}"] = w.flatten().tolist()
    payload = {
        "_source": "e3nn o3.wigner_3j",
        "_e3nn_version": __import__("e3nn").__version__,
        "_note": "row-major (2l1+1, 2l2+1, 2l3+1); Frobenius norm 1 to float64; sign is e3nn's. Generated at float64: e3nn's own runtime table is float32 and is 2e-8 off, which is visible as an equivariance residual.",
        "tables": table,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(table)} tables to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tt_bio/diffdock/w3j_tables.json")
