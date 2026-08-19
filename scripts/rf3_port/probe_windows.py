#!/usr/bin/env python3
"""Do tt-bio's window gather and RF3's window indices actually agree?

The state file asserts they span the same keys ([32k-48, 32k+80)), derived by hand
from `get_indexing_matrix`'s half-block arithmetic. This measures it instead:
it builds RF3's indicesQ/indicesK directly from `atom_attention`, builds tt-bio's
gather from `get_indexing_matrix` + `single_to_keys`, and compares the actual
gathered values on a real captured tensor.

Also reports what tt-bio puts where RF3 masks, since that is a known difference.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

W, H = 32, 128


def rf3_indices(L, qbatch=32, kbatch=128):
    nq = (L + qbatch - 1) // qbatch
    Cs = torch.arange(nq) * qbatch + qbatch // 2
    iq = Cs[:, None] + (torch.arange(qbatch) - qbatch // 2)[None, :]
    ik = Cs[:, None] + (torch.arange(kbatch) - kbatch // 2)[None, :]
    return iq, (iq < 0) | (iq > L - 1), ik, (ik < 0) | (ik > L - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_enc.pt")
    args = ap.parse_args()

    from tt_bio.boltz2 import get_indexing_matrix, single_to_keys

    cap = torch.load(args.capture, weights_only=False)
    q_l = cap["out"][1]                      # [1, L, c_atom], a real operating point
    L = q_l.shape[1]

    iq, maskQ, ik, maskK = rf3_indices(L)
    K_rf3 = iq.shape[0]

    # tt-bio pads L up to a multiple of W before windowing
    Lpad = ((L + W - 1) // W) * W
    x = torch.zeros(1, Lpad, q_l.shape[-1]); x[:, :L] = q_l[0]
    K = Lpad // W
    idx = get_indexing_matrix(K, W, H, x.device)
    tt_keys = single_to_keys(x, indexing_matrix=idx, W=W, H=H)   # [1, K, H, D]

    # RF3's own gather, with out-of-range clamped exactly as the reference does
    rf3_keys = x[:, ik.clamp(0, Lpad - 1)]                        # [1, K, H, D]

    rep = {"L": L, "Lpad": Lpad, "K_ttbio": K, "K_rf3": K_rf3,
           "shape_tt": list(tt_keys.shape), "shape_rf3": list(rf3_keys.shape)}

    if K == K_rf3 and tt_keys.shape == rf3_keys.shape:
        inrange = ~maskK                                          # [K, H]
        m = inrange[None, :, :, None].expand_as(tt_keys)
        rep["in_range_identical"] = bool(torch.equal(tt_keys[m], rf3_keys[m]))
        rep["in_range_maxabs"] = round(float((tt_keys - rf3_keys)[m].abs().max()), 8)
        # out-of-range: RF3 masks these out of the softmax; tt-bio gathers something
        out = (~inrange)[None, :, :, None].expand_as(tt_keys)
        if out.any():
            rep["out_of_range_slots"] = int(out.sum() // tt_keys.shape[-1])
            rep["ttbio_puts_zero_there"] = bool(
                float(tt_keys[out].abs().max()) == 0.0)
            rep["rf3_clamped_value_nonzero"] = bool(
                float(rf3_keys[out].abs().max()) > 0.0)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
