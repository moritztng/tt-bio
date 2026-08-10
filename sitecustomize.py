"""THROWAWAY (branch wk/protenix-trunk--z-narrowbw-512, Phase 2): set the narrow-projection
in0_block_w cap from the environment, in every interpreter, without touching tt_bio/.

`scripts/full_parity_gate.py` runs each device fold in a SUBPROCESS whose command line is built by
`Worker.wrap`, which pins `PYTHONPATH=<repo>` and offers no hook for extra environment beyond
`TT_BIO_SHARED_DRAW_SEED`. So an in-process assignment to `tt_bio.tenstorrent._NARROW_PROJ_BW`
cannot reach the fold, and the production default must stay at 1 (Phase 2 forbids a production
change). `site` imports `sitecustomize` from the first importable sys.path entry at interpreter
startup, and <repo> is always on PYTHONPATH for both the gate and its folds, so a repo-root
sitecustomize reaches every process in the tree.

Inert unless TTBIO_NARROW_PROJ_BW is set: no env var, no meta_path hook, no behaviour change.

  TTBIO_NARROW_PROJ_BW=8            -> _NARROW_PROJ_BW = 8   (in0_block_w = 8 at k_tiles = 8)
  TTBIO_NARROW_PROJ_BW=none         -> _NARROW_PROJ_BW = None (the core_grid= fallback)
  TTBIO_NARROW_PROJ_BW_MARK=<path>  -> append one line per process that actually took the patch,
                                       so "the cap reached the fold" is read off the branch and
                                       never assumed.
"""
import os
import sys

_CAP = os.environ.get("TTBIO_NARROW_PROJ_BW")

if _CAP:
    from importlib.machinery import PathFinder

    _VALUE = None if _CAP.lower() in ("none", "null") else int(_CAP)

    class _NarrowBWPatch(PathFinder):
        """Patch the module the instant its own body has run -- not earlier, not later."""

        @classmethod
        def find_spec(cls, name, path=None, target=None):
            if name != "tt_bio.tenstorrent":
                return None
            spec = super().find_spec(name, path, target)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return None
            inner = spec.loader.exec_module

            def exec_module(module, _inner=inner):
                _inner(module)
                module._NARROW_PROJ_BW = _VALUE
                module._pair_proj_program_config.cache_clear()
                mark = os.environ.get("TTBIO_NARROW_PROJ_BW_MARK")
                if mark:
                    with open(mark, "a") as fh:
                        fh.write(f"{os.getpid()} {sys.argv[0]} _NARROW_PROJ_BW={_VALUE}\n")

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _NarrowBWPatch)
