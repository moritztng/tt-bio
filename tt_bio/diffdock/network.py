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

A device implementation provides one more, and this is the seam the traced rollout runs
through:

    network.trace_body(graph, start, cfg, bounds, noise) -> (body, read_poses)

`body(t)` issues step `t`'s device work into the same buffers every time -- the fixed shape
this row is accountable for -- and `read_poses()` brings the final poses back as `[B, N, 3]`.
Split that way, `tt_bio.diffdock.capture.capture_loop` can warm the body up, capture all 19
steps as one trace and replay it without the rollout knowing a card is involved. A host
network does not implement it and is run step by step;
`tests/test_diffdock_trace.py::test_traced_rollout_matches_untraced` is what says the two
agree, and it skips until a device network offers this member.

`network.rollout(graph, start, cfg, bounds, noise) -> sampling.Rollout` runs that body eagerly
and reads once. `tt-bio dock` does not call it yet: on a 287k-edge complex its poses came back
12 A from the stepped rollout at the same seed, so the dock path steps `score_fn` until the
resident step is graded against `static_forward` on a card.

`load` returns the host implementation, `score_model.HostScoreNetwork`, around the released
v1.1 weights, or with ``DockOptions(device="tt")`` `device.DeviceScoreNetwork`, the same
weights with every convolution on the card. The dock path above it (graph, ESM-2 embeddings,
randomised start, rollout, ranking, upstream's output layout) runs against either.
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
    """The score network for ``options.checkpoint``, with its confidence model.

    Host torch today: :class:`tt_bio.diffdock.score_model.HostScoreNetwork` around the
    released v1.1 weights, which this port loads strictly (30,222,796 parameters, no
    missing and no unexpected key). The card path replaces this return value and nothing
    else -- that is what the two-member interface above is for.

    The confidence checkpoint is loaded here too, once per screen rather than once per
    complex, unless ``--no_confidence`` says the poses keep their sampling order.

    ``--weights`` carries upstream's two architecture booleans (D-dfd-confidence-3). The
    confidence side is the old architecture and always has been: upstream's own recipe sets
    ``old_filtering_model: true`` and the released 19 MB checkpoint is that network. The
    score side is DiffDock-L, and ``--weights diffdock`` asks for the pre-DiffDock-L score
    architecture, which this port does not implement and has no released weights for -- the
    v1.1 asset carries DiffDock-L only. It is refused here by name rather than silently
    running DiffDock-L under a flag that says otherwise.
    """
    import torch

    from tt_bio.diffdock import checkpoints
    from tt_bio.diffdock import config as dfd_config
    from tt_bio.diffdock import confidence_model as cm
    from tt_bio.diffdock import score_model

    if options.old_score_model:
        raise ScoreNetworkUnavailable(
            "--weights diffdock asks for the original DiffDock score architecture "
            "(upstream --old_score_model). This port implements DiffDock-L's score network, "
            "and the released v1.1 asset carries no original-DiffDock score weights. Use "
            "--weights diffdock-l, which is the default.")
    ckpt, params = checkpoints.resolve("score", options.checkpoint, cache,
                                       download=download)
    cfg = dfd_config.load(params.parent, old_model=options.old_score_model, name="diffdock")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = score_model.load_released(state, cfg)

    confidence = None
    if not options.no_confidence:
        c_ckpt, c_params = checkpoints.resolve("confidence", options.confidence_checkpoint,
                                               cache, download=download)
        confidence = cm.load_checkpoint(c_ckpt, c_params.parent,
                                        old_model=options.old_filtering_model)
    host = score_model.HostScoreNetwork(model, confidence_model=confidence)
    if options.device == "cpu":
        return host
    from tt_bio.diffdock.device import DeviceScoreNetwork
    from tt_bio.main import ensure_p300_mesh_descriptor
    from tt_bio.tenstorrent import get_device

    ensure_p300_mesh_descriptor()
    # The fp32 GEMM form is the default. The fused edge-tile kernel (options.edge_tp="tile") is
    # 3.9x faster per step and passes the SE(3) check; it becomes the default once it clears
    # QUALITY.md's PoseBusters top-1 bar, not on a same-seed comparison alone.
    return DeviceScoreNetwork(host, get_device(), edge_tp=options.edge_tp)
