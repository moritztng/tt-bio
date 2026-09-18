"""Training for tt-bio: the AF3 loss set, LoRA adapters, the optimizer and the loop.

OPT-IN AND INERT WHEN OFF. Nothing on the inference path imports this package, and the
tape it drives (`tt_bio.autograd`) prunes every node when `_GRAD_ENABLED` is false, so
importing `tt_bio` does not change one byte of inference behaviour. Both halves of that
are checked statically: `tt_bio/__init__.py` pulls in neither this package nor
`tt_bio.autograd`, and no module under `tt_bio/` outside `autograd.py`/`finetune.py`
imports either.

WHAT IS NOT TRUE YET, and it is the one thing that matters most: the training path does
NOT call the forward the inference path calls. There is no attach point in the shipped
forward at all -- `_KeyedWeights._lin`/`_ln` (`protenix.py:419`, `:435`) hand raw ttnn
tensors to `ttnn.linear`/`ttnn.layer_norm` with no tape and no adapter hook, and
`protenix.py` is byte-identical on `main`, `wk/ptxft-build` and `wk/hallgrad-build`.
Everything measured so far was measured on `perf/ptxft/tape_block.py`, which
re-implements the pairformer block, the DiT block and the confidence and distogram heads
as separate taped classes loaded from the same checkpoint.

That file is honest about it: it calls itself a TWIN in its own docstring and gives the
reason, which is that production "cannot be differentiated and must not be slowed down to
make it differentiable", with `perf/ptxft/block_parity.py` holding twin and production
together by PCC. So the twin was a deliberate, documented decision and not undisclosed
drift. The docstring you are reading is the one that was wrong, by claiming an attach
point that does not exist.

The twin still has to go, for reasons its own rationale does not cover. Its measured
parity against production is 2.56e-02 on the denoiser and 1.6e-2 on the atom block, so
the two have already diverged. It is Protenix-only by construction, whereas the
`PairformerLayer` it copies (`tenstorrent.py:8494`) is instantiated by six device modules
(protenix, opendde and four openfold3 legs). And the training path OOMs at 384 aa while
production folds 512 aa on one chip, which says the twin does not carry production's
chunking and residency levers whatever its intent.

The rationale is also answerable now: an attach point that DISPATCHES -- grad off calls
the production op unchanged, grad on calls the differentiable one -- makes the shipped
forward differentiable without slowing it down, which is the only thing the twin was
protecting. That is the first build step, not a later cleanup. See
`~/.coworker/state/train/INVENTORY.md`.
"""

from . import losses  # noqa: F401

__all__ = ["losses"]
