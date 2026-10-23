"""Annotated-CIF preprocessing and export for the Diffusers `RFD4ProteinaPipeline`."""

__all__ = ["prepare_cif", "write_cif"]


# Resolved on first use (PEP 562), because importing this package is not the same as
# using it. `processing` reaches torch and atomworks, and the contig grammar under
# `_cif/contig_grammar.py` deliberately reaches neither: a caller that only reads a
# contig -- size_limits' spec guard, JapanFold's web process at submit -- paid 1.2 s and
# ~376 MB of RSS for a string parse purely because it had to walk through this line.
def __getattr__(name):
    if name in __all__:
        from . import processing
        return getattr(processing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
