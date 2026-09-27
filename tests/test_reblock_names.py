"""Every global a function in `reblock_permute.py` reads is bound at module level.

The three builders assemble near-identical kernel descriptors, and an edit meant for two of them
reached the third: `_build_gated` read `genq_ct`, a local of `_build` and `_build_back` that it
never assigns, and raised NameError on every gated channel move. `except RuntimeError` at the call
site does not catch that, and no host test builds a descriptor, so only a device run noticed.

Host only, no card, no import of ttnn: the check reads the symbol table.
"""

import builtins
import symtable
from pathlib import Path

import tt_bio

SRC = Path(tt_bio.__file__).resolve().parent / "reblock_permute.py"


def _functions(table):
    for child in table.get_children():
        if child.get_type() == "function":
            yield child
        yield from _functions(child)


def test_every_global_read_is_bound():
    top = symtable.symtable(SRC.read_text(), str(SRC), "exec")
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    bound |= set(dir(builtins))
    unbound = sorted(
        (fn.get_name(), s.get_name())
        for fn in _functions(top)
        for s in fn.get_symbols()
        if s.is_global() and s.is_referenced() and s.get_name() not in bound
    )
    assert not unbound, unbound
