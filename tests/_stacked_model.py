"""A stand-in whose blocks run ONE call site with a DIFFERENT weight each.

This is the shape that separates full-weight training from LoRA and the reason
`tt_bio.train.lora._Substitute` keys on the weight and not on the site alone. `_site_name` is
`file:line:qualname`, so the 48 blocks of a real trunk collapse to one name; one adapter that
every block reads is a legitimate modelling choice, and one WEIGHT that every block reads is
not a choice, it is a different model from the one on disk.

Its own file because `_site_name` skips frames belonging to `ops.py`, `dispatch.py`, `lora.py`
and `autograd.py`. A model defined inside the checker would be named correctly for the wrong
reason.
"""

from tt_bio import ops


def block(x, w):
    return ops.linear(x, w)            # one site, reached once per block


def forward(x, weights):
    for w in weights:                  # every block brings its own weight
        x = block(x, w)
    return x


def shared(x, w, blocks=3):
    for _ in range(blocks):            # one site, one weight, reached `blocks` times
        x = block(x, w)
    return x


def reuploading(x, make_weight, blocks=3):
    """A forward that mints a fresh weight object on every call. Trainable by nothing."""
    for _ in range(blocks):
        x = block(x, make_weight())
    return x
