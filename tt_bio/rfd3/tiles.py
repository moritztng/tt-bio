"""Tile geometry for the RFD3 attention path.

The Tensix tile face is 32x32, so a key axis that is not a multiple of 32 carries tile
padding that ttnn reductions can read. `model.py` and `block_sparse.py` both pad that axis
out before reducing over it; `pad_axis` says why in detail.

This module imports only ttnn on purpose. `model.py` already imports `block_sparse`, so
neither of them can host the helpers without a cycle.
"""
import ttnn

TILE = 32


def align_tile(n):
    return -(-n // TILE) * TILE


def pad_axis(x, width, axis, value):
    """Extend `axis` of a TILE tensor out to `width`, filling the new region with `value`.

    Attention reduces over the key axis, and a ttnn reduction over a last dim that is not a
    tile multiple reads the tile padding along with the data (p23: measured end to end -- the
    same logical scores give two different softmax answers when the 18 pad columns differ).
    Tile padding is not written by every op -- `ttnn.scatter` leaves whatever the freshly
    allocated buffer held -- so the fix is to leave no tile padding on that axis: pad it out
    logically, with -1e4 on the mask (weight exactly 0 after exp) and 0 on the values.
    `ttnn.pad` writes the value, so the result is defined by construction rather than by luck.
    """
    if x.shape[axis] == width:
        return x
    pad = [(0, 0)] * len(x.shape)
    pad[axis] = (0, width - x.shape[axis])
    return ttnn.pad(x, pad, value)
