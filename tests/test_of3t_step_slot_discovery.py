"""The parameter a training step declares must be reachable from where the FORWARD reads it.

`_w_tt` (`openfold3_diffusion_transformer.py`, `openfold3_diffusion_module.py`,
`openfold3_atom_transformer.py`) uploads a weight once and keeps it in two places: the module
attribute the forward reads, and `self._wc[key]`. `fullstep.declare_all` walks every device
tensor a module holds and dedupes by tensor IDENTITY, so one weight arrives under two names and
only one survives.

WHICH one survives is not cosmetic. It is where `Parameters.rebind` writes the new weight after
`AdamW.step`, and `_wc` is populated in `__init__` and never read again. With the cache path
winning the tie -- and `_` sorts before every letter, so it always did -- the optimizer's update
went into a dead dict while the module kept the pre-step handle, which the value setter has
already de-registered as a tape leaf. The next forward ran untaped through it.

Measured on pc card 0 at crop 384, `origin/main` + the fix, two matched first arms: 270 of 3,152
weights, every one `diffusion.dm.*._wc.*`, took a gradient on the cold rep and none after it.
Restoring them moves the steady trained count 2,674 -> 2,944 and costs 1.018x on the step. The
failure is silent in every other signal -- the tape node count, the rebind count and the leaf
count all hold -- so it needs a test rather than a reader.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def fullstep():
    return pytest.importorskip("perf.of3t_stepfloor.fullstep")


class _Module:
    """A module holding one uploaded weight the way `_w_tt` leaves it: in the cache AND in the
    attribute the forward reads."""

    def __init__(self, weight):
        self._wc = {("linear_a.weight", True): weight}
        self.w_la = weight


def test_a_cache_path_never_wins_the_identity_tie(fullstep):
    """The attribute the forward reads is the slot; the `_wc` alias is the one dropped."""
    weight = object()
    m = _Module(weight)
    found = {
        "diffusion.dm.dit.blocks.0._wc.('linear_a.weight', True)":
            (m._wc, ("linear_a.weight", True), weight),
        "diffusion.dm.dit.blocks.0.w_la": (m, "w_la", weight),
    }
    kept, unwritable = fullstep._dedupe_slots(found)

    assert list(kept) == ["diffusion.dm.dit.blocks.0.w_la"]
    owner, key, _ = kept["diffusion.dm.dit.blocks.0.w_la"]
    assert owner is m and key == "w_la", (
        "the slot must be the module attribute -- writing AdamW's weight into `_wc` freezes it"
    )
    assert not unwritable


def test_the_ordering_is_the_mechanism_not_the_name(fullstep):
    """`_slot_rank` demotes a cache path whatever it is called, including a trailing `._wc`."""
    names = [
        "d.blocks.0._wc.('x', True)",   # the shape `_DiTBlock` and `OF3AtomTransformer` produce
        "d.blocks.0._wc",               # the dict itself, if a walk ever yields it
        "d.blocks.0.w_la",
        "d.blocks.0.a_weight",
    ]
    ordered = [n for n, _ in sorted(((n, None) for n in names), key=fullstep._slot_rank)]
    assert ordered[:2] == ["d.blocks.0.a_weight", "d.blocks.0.w_la"]
    assert all("_wc" in n for n in ordered[2:])


def test_a_tuple_owner_is_counted_and_gets_no_slot(fullstep):
    """`rebind` cannot write into a tuple, so such a name is declared but left slotless."""
    weight = object()
    found = {"trunk.pair.0": ((weight,), 0, weight)}
    kept, unwritable = fullstep._dedupe_slots(found)
    assert list(kept) == ["trunk.pair.0"]
    assert unwritable == ["trunk.pair.0"]


def test_declare_all_uses_the_ranked_dedupe(fullstep):
    """A future edit that drops the ordering and re-inlines a plain sort fails here."""
    import ast
    src = (ROOT / "perf" / "of3t_stepfloor" / "fullstep.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "declare_all")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_dedupe_slots" in called, (
        "declare_all must go through _dedupe_slots -- a bare sorted(found.items()) puts the "
        "`_wc` alias back in the slot and re-freezes 270 diffusion weights"
    )
