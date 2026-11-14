"""A test that opens the card itself closes it with `tenstorrent._close_device_locked`.

`ttnn.close_device()` alone keeps the process's claim on the chip (MetalContext, two driver fds,
the CHIP_IN_USE mutex) until exit, so every later test in the same pytest process starts on a
held card. Four device tests did this, and test_cleanup_drops_the_driver_fds failed in whichever
pytest_device shard ran one before it. Opens no device.
"""
import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _closes_bare(path):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "close_device" and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "ttnn"
               for n in ast.walk(ast.parse(path.read_text())))


def test_no_test_closes_the_card_without_releasing_it():
    bare = [p.name for p in sorted(TESTS.glob("test_*.py")) if _closes_bare(p)]
    assert not bare, (f"{bare} call ttnn.close_device() directly, which leaves the chip held for "
                      "the rest of the pytest process. Use tenstorrent._close_device_locked(dev).")
