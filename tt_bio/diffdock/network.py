"""The seam between the rollout and whatever computes the score.

:func:`load` is the one place `tt-bio dock` asks for a score network. What it returns has a
config and one method, which is all :func:`tt_bio.diffdock.sampling.sample` ever calls:

    network.config  -> tt_bio.diffdock.config.DiffDockConfig  (sigma bounds, no_torsion,
                       crop_beyond, all_atoms: everything the rollout reads off the weights)
    network.score_fn(graph) -> Callable[[StepInput], tuple[Tensor, Tensor, Tensor]]

`graph` is a :class:`tt_bio.diffdock.input.ComplexGraph`; the returned callable gets a
:class:`~tt_bio.diffdock.sampling.StepInput` per step (poses `[B, N, 3]` in the receptor's
centred frame, the three t values, the three sigmas and this step's receptor crop radius) and
returns `(tr_score [B,3], rot_score [B,3], tor_score [B*m])`. Anything the network needs to
keep between steps -- the receptor graph, the ESM-2 embeddings, a captured ttnn trace -- lives
in that closure, which is why the host reference and the device path are interchangeable here
and why neither reaches into the rollout.

An implementation also provides, optionally:

    network.confidence(graph, poses) -> Tensor[B] | None

for the ranking pass. Returning `None` ranks poses in sample order, which is what
`--no_confidence` does.

**Nothing implements this yet.** The port's score network is `dfd-parity`'s `score_model`
feature; until it lands, `load` raises and `tt-bio dock` says so in one line instead of
failing somewhere inside a loader. Wiring it up is replacing the body of `load` -- the dock
path above it (graph, ESM-2 embeddings, randomised start, rollout, ranking, upstream's output
layout) is written and runs against any object with these two members.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


class ScoreNetworkUnavailable(RuntimeError):
    """Raised when no score network is installed for the requested weights."""


@runtime_checkable
class ScoreNetwork(Protocol):
    #: the checkpoint's :class:`tt_bio.diffdock.config.DiffDockConfig`
    config: object

    def score_fn(self, graph): ...

    def confidence(self, graph, poses: Tensor): ...


def load(options, *, cache=None, download: bool = False) -> ScoreNetwork:
    """The score network for ``options.weights``, on the host or on a card."""
    raise ScoreNetworkUnavailable(
        "the DiffDock-L score network is not implemented yet (dfd-parity owns `score_model`); "
        "the reverse-diffusion sampler it plugs into is complete and graded, and "
        "`--no_model` runs the seed-conformer baseline end to end today")
