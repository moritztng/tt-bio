"""Training for tt-bio: the AF3 loss set, LoRA adapters, the optimizer and the loop.

OPT-IN AND INERT WHEN OFF. Nothing on the inference path imports this package, and the
tape it drives (`tt_bio.autograd`) prunes every node when `_GRAD_ENABLED` is false, so
importing `tt_bio` does not change one byte of inference behaviour. The adapters attach
inside `_KeyedWeights._lin`/`_ln`, so the training path calls the SAME forward the
inference path does rather than a copy of it.
"""

from . import losses  # noqa: F401

__all__ = ["losses"]
