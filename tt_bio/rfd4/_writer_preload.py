"""Imported once by the CIF writer's forkserver (`tt_bio.rfd4.io.start_writer`).

A write looks up CCD entries, and the first lookup decodes biotite's whole Chemical Component
Dictionary, ~1.25 GB and ~3 s. Doing it here means every writer process forks with it already in
memory and shares those pages with the server instead of decoding its own copy.

The server forks, so it must stay single-threaded: importing numpy, torch and pyarrow would
otherwise start BLAS, OpenMP and jemalloc threads, and a fork taken while one of them holds a
lock can deadlock the child. Nothing imports those before this module (`tt_bio.rfd4` does not).
A writer is single-threaded Python anyway.
"""
import gc
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  JE_ARROW_MALLOC_CONF="background_thread:false")

import torch  # noqa: E402
from biotite.structure import info  # noqa: E402

from tt_bio._vendor.atomworks.io.template import get_empty_ccd_template  # noqa: E402
from tt_bio._vendor.rfd4_prep._cif import transforms_unindexing  # noqa: E402,F401  (builds stereo templates)
from tt_bio._vendor.rfd4_prep.processing import write_cif  # noqa: E402,F401
from tt_bio.rfd4.io import write_design  # noqa: E402,F401

torch.set_num_threads(1)
for _code in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
              "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"):
    get_empty_ccd_template(_code)
    info.standardize_order(info.residue(_code)), info.bonds_in_residue(_code), info.link_type(_code)  # biotite index

# A writer's garbage collector would otherwise write to every object this module left behind and
# so copy the pages it shares with the server.
gc.freeze()
