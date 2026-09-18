"""A stand-in module whose frames are the ones a census is supposed to name.

It has to live in its own file: `_site_name` skips every frame belonging to `ops.py`,
`dispatch.py`, `lora.py` and `autograd.py`, so a "model" defined inside the checker would
be named correctly for the wrong reason.
"""

from tt_bio import ops


def block(x, wq, wv, gamma):
    q = ops.linear(x, wq)                      # site 1
    v = ops.linear(x, wv, None, dtype="bf16")  # site 2, bias positional
    n = ops.layer_norm(x, gamma)               # not adaptable
    return q, v, n


def forward(x, wq, wv, gamma, blocks=3):
    for _ in range(blocks):
        block(x, wq, wv, gamma)
