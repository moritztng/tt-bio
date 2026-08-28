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

The pair inputs are built WINDOWED, not dense. The atom transformer they feed is
32-query / 128-key local attention, so of the L_atom^2 pairs a dense build produces it
reads L_atom x 128 -- 32.5x fewer at 512 aa, 64.8x at 1024, and the dense tensor is
4.40 GB at 1024 aa. `pair_inputs_windowed` evaluates the same three terms at exactly the
pairs that are read, with the out-of-window slots zero, which is what the dense tensor
carried there. `pair_inputs` and `window_pair` keep the dense form for the parity
harnesses and for the bit-exactness check in perf/rf3/pair_track_ab.py.
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


PAD_LEFT = 48   # (ATOM_KEYS - ATOM_WINDOW) // 2
PAD_RIGHT = ATOM_KEYS - ATOM_WINDOW - PAD_LEFT


def window_index(n_atom_padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """([K, ATOM_WINDOW] query atom index, [K, ATOM_KEYS] key atom index).

    The key index starts at -PAD_LEFT, so it runs off both ends of the atom axis; those
    slots are the ones `window_mask` fills with -1e9.
    """
    k = n_atom_padded // ATOM_WINDOW
    base = torch.arange(k).unsqueeze(-1) * ATOM_WINDOW
    return (base + torch.arange(ATOM_WINDOW),
            base + torch.arange(ATOM_KEYS) - PAD_LEFT)


def window_valid(n_atom: int, n_atom_padded: int) -> torch.Tensor:
    """[K, ATOM_WINDOW, ATOM_KEYS, 1]: 1.0 where the dense pair entry exists."""
    q, j = window_index(n_atom_padded)
    ok = (q < n_atom).unsqueeze(-1) & ((j >= 0) & (j < n_atom)).unsqueeze(-2)
    return ok.unsqueeze(-1).float()


def pair_inputs_windowed(f: dict, n_atom: int, n_atom_padded: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """([K, ATOM_WINDOW, ATOM_KEYS, 5] pair input, [..., 1] validity mask).

    The same columns as `pair_inputs`, evaluated only at the pairs the 32-query/128-key
    window reads. Out-of-window slots are zero, which is what the dense tensor's padding
    held, so this is that tensor restricted rather than a different one.
    """
    q, j = window_index(n_atom_padded)
    ok = window_valid(n_atom, n_atom_padded)
    ref_pos = f["ref_pos"].float()[:n_atom]
    suid = f["ref_space_uid"][:n_atom]
    qi, ji = q.clamp(max=n_atom - 1), j.clamp(0, n_atom - 1)
    d = ref_pos[qi].unsqueeze(-2) - ref_pos[ji].unsqueeze(-3)     # [K, W, KEYS, 3]
    inv = 1.0 / (1.0 + (d * d).sum(-1, keepdim=True))
    v = (suid[qi].unsqueeze(-1) == suid[ji].unsqueeze(-2)).unsqueeze(-1).float() * ok
    return torch.cat([d, inv, v], dim=-1) * ok, v


def window_pair(x: torch.Tensor) -> torch.Tensor:
    """[1, Lp, Lp, C] -> [K, ATOM_WINDOW, ATOM_KEYS, C], the dense pair tensor gathered
    into the windows. The reference form of `pair_inputs_windowed`, used by the parity
    harnesses and to check the cheap build against the dense one."""
    _, lp, _, _ = x.shape
    q, j = window_index(lp)
    ok = ((j >= 0) & (j < lp)).to(x.dtype)
    g = x[0][q.unsqueeze(-1), j.clamp(0, lp - 1).unsqueeze(-2)]    # [K, W, KEYS, C]
    return g * ok.unsqueeze(-2).unsqueeze(-1)


def token_to_atom_windowed(a2t: torch.Tensor, n_atom_padded: int) -> torch.Tensor:
    """[1, K, I, ATOM_KEYS] from the [1, Lp, I] atom->token one-hot.

    Turns `_trunk_pair`'s second gather from a dense [Lp*c, I] @ [I, Lp] into a batched
    [K, 32c, I] @ [K, I, 128]: the Lp/128 saving on the one matmul that carried the
    atom-pair track's cost.
    """
    t = torch.nn.functional.pad(a2t[0].t(), (PAD_LEFT, PAD_RIGHT))   # [I, Lp + 96]
    k = n_atom_padded // ATOM_WINDOW
    return torch.stack([t[:, i * ATOM_WINDOW:i * ATOM_WINDOW + ATOM_KEYS]
                        for i in range(k)]).unsqueeze(0)



def pad_pair(x: torch.Tensor, n_atom_padded: int) -> torch.Tensor:
    """[L, L, c] -> [1, Lp, Lp, c]: the padding the dense pair path used to take."""
    l, _, c = x.shape
    y = torch.zeros(1, n_atom_padded, n_atom_padded, c, dtype=x.dtype)
    y[0, :l, :l] = x
    return y


def window_pair_valid(x_win: torch.Tensor, n_atom: int) -> torch.Tensor:
    """The in-range entries of a [K, ATOM_WINDOW, ATOM_KEYS, c] track, flattened.

    Out-of-window slots hold whatever the arithmetic produced there and the -1e9 mask
    means nothing reads them, so a parity score has to drop them rather than compare
    them.
    """
    ok = window_valid(n_atom, x_win.shape[0] * ATOM_WINDOW)[..., 0].bool()
    return x_win[ok]
