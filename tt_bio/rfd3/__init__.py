"""RFdiffusion3 (RFD3) all-atom structure design on Tenstorrent.

Self-contained engine package behind ``tt-bio design --model rfd3``:

- :mod:`tt_bio.rfd3.input` — the InputSpecification / contig-string grammar.
- :mod:`tt_bio.rfd3.featurize` — host featurizer (PDB + contig -> model features).
- :mod:`tt_bio.rfd3.model` — ttnn TokenInitializer + DiffusionModule ports.
- :mod:`tt_bio.rfd3.sampler` — the EDM diffusion sampler.
- :mod:`tt_bio.rfd3.design` — the end-to-end design runner (featurize ->
  on-device forward -> sample -> one CIF per design) and weight extraction.
"""
