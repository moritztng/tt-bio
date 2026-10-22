"""Run RFD4's OWN condition-spec and contig-spec tests against tt-bio's vendored copy.

The row's bar for the condition spec is that parse errors match upstream's behaviour, not
that they merely raise. Upstream pins that behaviour in
``tests/inference_utils/test_condition_spec.py`` and ``test_contig_spec.py`` with
``pytest.raises(match=...)``. Instead of transcribing those cases, this points the upstream
test files at the vendored modules: every ``rfproteina.*`` module the tests import is aliased
in ``sys.modules`` to its flattened sibling under ``tt_bio._vendor.rfd4_prep._cif``, so the
assertions run unchanged against our code.

A transcribed test can drift from upstream silently. This one cannot: it IS upstream's test.

The fixtures parse three real entries (6lyz, 1mso, 1ehz) out of a PDB mirror, so it needs
one and an atomworks CCD cache. That is why it runs against the reference env rather than
in CI:

    PDB_MIRROR_PATH=<mirror> PYTHONPATH=<pytest> \
        rfd4-ref-env/venv/bin/python scripts/rfd4_port/run_upstream_spec_tests.py [-k ...]

A mirror is <base>/<id[1:3]>/<id>.cif.gz; files.rcsb.org/download/<id>.cif.gz fills it.
39 passed at pin 2db07bb4 on 2026-10-21, the 23 pinned error-message cases among them.
"""
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_TESTS = Path("/home/moritz/rfd4-upstream/RFD4-Proteina/tests")
VENDOR = "tt_bio._vendor.rfd4_prep._cif"

#: rfproteina module -> the flattened module that replaces it. `vendor_from_main.py`'s rule:
#: `rfproteina.transforms.atom_names` -> `transforms_atom_names`.
def flat(module: str) -> str:
    return f"{VENDOR}.{module.removeprefix('rfproteina.').replace('.', '_')}"


class Finder:
    """Import hook mapping any rfproteina module onto the vendored flat one."""

    def find_module(self, name, path=None):
        return self if name == "rfproteina" or name.startswith("rfproteina.") else None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        if name == "rfproteina":
            mod = types.ModuleType(name)
            mod.__path__ = []
        else:
            import importlib

            try:
                mod = importlib.import_module(flat(name))
            except ModuleNotFoundError:
                # A module outside the vendored set: fail here, naming it, rather than
                # letting the test import a stub and assert against nothing.
                raise ModuleNotFoundError(
                    f"{name} is not vendored (expected {flat(name)}); it is part of the "
                    "surface this row must port") from None
        sys.modules[name] = mod
        return mod


def main() -> int:
    sys.path.insert(0, str(REPO))
    sys.meta_path.insert(0, Finder())
    import pytest

    args = [str(UPSTREAM_TESTS / "inference_utils" / "test_condition_spec.py"),
            str(UPSTREAM_TESTS / "inference_utils" / "test_contig_spec.py"),
            "-p", "no:cacheprovider", "-q", *sys.argv[1:]]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
