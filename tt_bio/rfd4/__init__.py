"""RFD4-Proteina all-atom design on Tenstorrent: the host side.

``tt-bio design --model rfd4`` takes a design spec (the same YAML/JSON RFD4 itself takes)
or an annotated CIF, featurizes it, runs the flow-matching model, and writes an annotated
CIF per design.

Everything in this package is the host path:

- :mod:`tt_bio.rfd4.io` — spec -> features (``featurize``) and output -> annotated CIF
  (``write_design``), both over the vendored ``rfd4_prep``, which is upstream's own
  host code rather than a transcription of it.
- :mod:`tt_bio.rfd4.engine` — the interface the device implementation plugs into, and the
  stub that raises until it lands.

The model is not ported yet: ``tt-bio design --model rfd4`` prepares its inputs, reports
what it would run and stops at the engine with a clear message. The spec surface, the
features and the writer are exact against upstream today (scripts/rfd4_port/).
"""
from tt_bio.rfd4.engine import Engine, NotPortedEngine, get_engine
from tt_bio.rfd4.io import DesignInput, featurize, load_spec, write_design

__all__ = ["DesignInput", "Engine", "NotPortedEngine", "featurize", "get_engine",
           "load_spec", "write_design"]
