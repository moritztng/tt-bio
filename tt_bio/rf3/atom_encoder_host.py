"""Host-side inputs for RF3's atom attention encoder.

Everything here is index and geometry work that ttnn is poor at and that does not
depend on learned weights: the 393-column single-feature concat, the pair inputs
(D, inverse squared distance, same-reference-space mask), the window gather matrix,
and the atom->token averaging matrix.

The three pair terms are concatenated into ONE input block rather than three, because
the reference multiplies all three by V_LL:

    P = process_d(D)*V + process_inverse_dist(inv)*V + process_valid_mask(V)*V
      = [D | inv | V] @ [W_d | W_inv | W_valid]^T * V

which is the same function of the same weights, with the 5 input columns padded to a
tile in one go instead of three separate sub-tile matmuls. It re-associates the bf16
rounding, so it is a device-precision deviation and not a formula change; the host
oracle in probe_pair_formula.py keeps the unfused form.
"""

from __future__ import annotations

import torch

ATOM_WINDOW = 32
ATOM_KEYS = 128

ATOM_1D_FEATURES = [
    "ref_pos", "ref_charge", "ref_mask", "ref_element",
    "ref_atom_name_chars", "ref_pos_ground_truth", "has_atom_level_embedding",
]


def _collapse(t: torch.Tensor, n_atom: int) -> torch.Tensor:
    t = t.float()
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return t.reshape(n_atom, -1)


def single_features(f: dict, n_atom: int) -> torch.Tensor:
    """The [L, 393] concat that `process_input_features` consumes."""
    return torch.cat([_collapse(f[n], n_atom) for n in ATOM_1D_FEATURES], dim=-1)


def pair_inputs(f: dict, n_atom: int) -> tuple[torch.Tensor, torch.Tensor]:
    """([L, L, 5] fused pair input, [L, L, 1] validity mask).

    Columns are [D(3) | 1/(1 + sum(D*D))(1) | V(1)] -- the squared form, because this
    checkpoint sets use_inv_dist_squared=True.
    """
    ref_pos = f["ref_pos"].float()[:n_atom]
    suid = f["ref_space_uid"][:n_atom]
    d = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)                 # [L, L, 3]
    inv = 1.0 / (1.0 + (d * d).sum(-1, keepdim=True))                 # [L, L, 1]
    v = (suid.unsqueeze(-1) == suid.unsqueeze(-2)).unsqueeze(-1).float()
    return torch.cat([d, inv, v], dim=-1), v


def fused_pair_weight(w: dict) -> torch.Tensor:
    """[16, 5]: process_d | process_inverse_dist | process_valid_mask, in column order."""
    return torch.cat(
        [w["process_d.weight"], w["process_inverse_dist.weight"],
         w["process_valid_mask.weight"]], dim=1
    )


def atom_to_token_mean(f: dict, n_atom: int, n_token: int) -> torch.Tensor:
    """[I, L] row-normalised map, so `M @ Q` is the reference's scatter_mean.

    Rows with no atoms would divide by zero; there are none in a well-formed input,
    but the clamp keeps a malformed one from producing NaN silently.
    """
    a2t = f["atom_to_token_map"].long()[:n_atom]
    m = torch.zeros(n_token, n_atom)
    m[a2t, torch.arange(n_atom)] = 1.0
    return m / m.sum(-1, keepdim=True).clamp(min=1.0)


def pad_to_window(x: torch.Tensor, n_atom: int, dims: tuple[int, ...],
                  value: float = 0.0) -> torch.Tensor:
    """Pad the named atom dims up to a multiple of ATOM_WINDOW."""
    n_pad = (-n_atom) % ATOM_WINDOW
    if not n_pad:
        return x
    for d in dims:
        pad = [0, 0] * (x.dim() - 1 - d) + [0, n_pad]
        x = torch.nn.functional.pad(x, pad, value=value)
    return x


def key_window_slices(n_atom_padded: int) -> list[tuple[int, int]]:
    """Per query block, the [start, stop) key span in the LEFT-PADDED bias tensor.

    RF3 centres a 128-key window on each 32-query block: block k covers atoms
    [32k, 32k+32) and attends to [32k-48, 32k+80). Shifting the whole bias right by
    48 makes every such window a contiguous slice [32k, 32k+128), so the gather is a
    slice rather than an index op -- and the shifted-in columns are exactly the
    out-of-range slots, which get the -1e9 fill.
    """
    k = n_atom_padded // ATOM_WINDOW
    return [(i * ATOM_WINDOW, i * ATOM_WINDOW + ATOM_KEYS) for i in range(k)]


PAD_LEFT = 48   # (ATOM_KEYS - ATOM_WINDOW) // 2
PAD_RIGHT = ATOM_KEYS - ATOM_WINDOW - PAD_LEFT
