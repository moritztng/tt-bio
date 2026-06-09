"""ai& Bio — a minimal, dependency-free web platform on top of tt-bio.

The platform exposes everything tt-bio can do — Boltz-2 / ESMFold2 structure
prediction, Boltz-2 binding-affinity prediction, and BoltzGen drug design —
through a small HTTP server (Python stdlib only) and a single-page frontend.

It drives the real `tt-bio` CLI as a subprocess per job, so the served results
are identical to running tt-bio by hand; there is no second code path to drift.

Run it with `tt-bio serve` (see tt_bio.main) or `python -m tt_bio.platform`.
"""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
