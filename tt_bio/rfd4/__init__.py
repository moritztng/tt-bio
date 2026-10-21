"""RFD4-Proteina all-atom design on Tenstorrent: the host side.

``tt-bio design --model rfd4`` takes a design spec (the same YAML/JSON RFD4 itself takes)
or an annotated CIF, featurizes it, runs the flow-matching model, and writes an annotated
CIF per design.

Everything in this package is the host path:

- :mod:`tt_bio.rfd4.io` — spec -> features (``featurize``) and output -> annotated CIF
  (``write_design``), both over the vendored ``rfd4_prep``, which is upstream's own
  host code rather than a transcription of it.
- :mod:`tt_bio.rfd4.run` — :func:`tt_bio.rfd4.design`, spec to CIFs in one call, the same
  path the CLI takes.
- :mod:`tt_bio.rfd4.engine` — the engine: the sampler drives the trunk and the autoencoder
  on the card and returns the four tensors the writer needs.

The model itself is :mod:`tt_bio.rfd4.network` (the host glue around the trunk),
:mod:`tt_bio.rfd4.trunk` (14 attention blocks, 7 pair updates, recycles),
:mod:`tt_bio.rfd4.autoencoder` (the decoder that turns latents into atoms) and
:mod:`tt_bio.rfd4.sampler` (flow matching, schedules, per-design RNG).
"""
from tt_bio.rfd4.engine import DesignEngine, Engine, get_engine
from tt_bio.rfd4.io import (DesignInput, featurize, load_spec, prepare_examples,
                            write_design)
from tt_bio.rfd4.run import design

__all__ = ["DesignEngine", "DesignInput", "Engine", "design", "featurize", "get_engine",
           "load_spec", "prepare_examples", "write_design"]
