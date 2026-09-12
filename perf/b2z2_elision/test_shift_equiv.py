#!/usr/bin/env python3
"""Host-only proof that the shift gather reproduces Boltz-2's one-hot gather exactly.

No device. Runs the reference `get_indexing_matrix` / `single_to_keys` against a torch
transcription of `_atom_shift_gather`, over the padded and unpadded cases the fold produces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tt_bio.boltz2 import get_indexing_matrix, single_to_keys
from tt_bio.tenstorrent import ATOM_WINDOW, ATOM_DIM, ATOM_KEY_SHIFT, _atom_gather_shift_windows

W, H = ATOM_WINDOW, ATOM_DIM
PIECES = H // W


def shift_gather(s, windows):
    """Torch transcription of tt_bio.tenstorrent._atom_shift_gather."""
    b, k, w, d = s.shape
    flat = s.reshape(b, k * w, d)[:, : windows * w, :]
    tail = (k + PIECES) * w - ATOM_KEY_SHIFT - windows * w
    shifted = torch.nn.functional.pad(flat, (0, 0, ATOM_KEY_SHIFT, tail))
    src = shifted.reshape(b, k + PIECES, w, d)
    out = torch.cat([src[:, j:j + windows] for j in range(PIECES)], dim=2)
    return torch.nn.functional.pad(out, (0, 0, 0, 0, 0, k - windows))


fails = 0
for k_real in (4, 17, 125, 139, 140):
    for k_bucket in (140, 224):
        if k_real > k_bucket:
            continue
        ki = get_indexing_matrix(k_real, W, H, "cpu")
        ki = torch.nn.functional.pad(
            ki, (0, PIECES * 2 * k_bucket - ki.shape[1], 0, 2 * k_bucket - ki.shape[0]))
        got_w = _atom_gather_shift_windows(ki)
        torch.manual_seed(0)
        s = torch.randn(1, k_bucket, W, 8)
        ref = single_to_keys(s.reshape(1, k_bucket * W, 8), ki, W, H)
        ok_w = got_w == k_real
        ok_v = ok_w and torch.equal(shift_gather(s, got_w), ref)
        fails += not (ok_w and ok_v)
        print(f"K_real={k_real:4d} bucket={k_bucket:4d} windows={got_w} "
              f"{'OK' if ok_w and ok_v else 'FAIL'}")

# A gather that is NOT the centred window must be refused, or the check proves nothing.
ki = get_indexing_matrix(20, W, H, "cpu")
zeros = (ki == 0).nonzero()
assert len(zeros), "no zero entry to flip: the control would prove nothing"
for spot in (zeros[0], zeros[len(zeros) // 2]):
    neg = ki.clone()
    neg[spot[0], spot[1]] = 1.0
    assert not torch.equal(neg, ki), "the control did not change the matrix the check reads"
    bad = _atom_gather_shift_windows(neg) is not None
    print(f"negative control (extra 1 at {int(spot[0])},{int(spot[1])}):",
          "FAIL accepted" if bad else "OK refused")
    fails += bad
for drop in (0, 7):
    neg = ki.clone()
    col = (neg[:, drop] != 0).nonzero()
    if not len(col):
        continue
    neg[col[0, 0], drop] = 0.0
    bad = _atom_gather_shift_windows(neg) is not None
    print(f"negative control (dropped the 1 in column {drop}):",
          "FAIL accepted" if bad else "OK refused")
    fails += bad
ki2 = get_indexing_matrix(20, W, ATOM_DIM // 2, "cpu") if ATOM_DIM // 2 % (W // 2) == 0 else None
if ki2 is not None:
    print("negative control (64-atom key window):",
          "OK refused" if _atom_gather_shift_windows(ki2) is None else "FAIL accepted")
    fails += _atom_gather_shift_windows(ki2) is not None
sys.exit(1 if fails else 0)
