"""ProtenixDesign skips Protenix.__init__, so state added there must be mirrored or not read.

`ProtenixDesign(Protenix)` deliberately does not call `Protenix.__init__` -- the pxdesign
checkpoint has no trunk and no confidence head, and building either from an empty state dict
would silently make a wrong module. The cost of that choice is an invariant nobody was checking:
any attribute `Protenix.__init__` assigns is invisible to ProtenixDesign, so an INHERITED method
that reads it raises AttributeError.

It happened. `1c393dda8` added `self._paircond_rows_refused = {}` to `Protenix.__init__` and read
it from `_diffusion_pair_cond`, which ProtenixDesign inherits and reaches on every design call
(its diffusion is fp32 by default, and fp32 is the branch that goes to row_block_after_refusal,
whose first statement is `memo.get(key)`). Every pxdesign run on that commit died on
AttributeError at any target size.

Host-only and AST-only: no device, no weights, no import of the engine.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTENIX = ROOT / "tt_bio" / "protenix.py"
PXMODEL = ROOT / "tt_bio" / "pxdesign" / "model.py"


def _class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _init_attrs(cls):
    """Names assigned as `self.X = ...` (or annotated) directly in the class's __init__."""
    out = set()
    for fn in cls.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
            continue
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    out.add(t.attr)
    return out


def _attrs_read_outside_init(cls):
    """Names read as `self.X` anywhere in the class EXCEPT its own __init__.

    Those are the ones a subclass inherits and can therefore trip over."""
    out = set()
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name) and node.value.id == "self"):
                out.add(node.attr)
    return out


def test_pxdesign_sets_every_protenix_init_attr_that_an_inherited_method_reads():
    parent = _class(ast.parse(PROTENIX.read_text()), "Protenix")
    child = _class(ast.parse(PXMODEL.read_text()), "ProtenixDesign")

    set_by_parent_init = _init_attrs(parent)
    read_by_inherited = _attrs_read_outside_init(parent)
    set_by_child_init = _init_attrs(child)
    # Anything the child overrides with its own method is not inherited, so not at risk.
    overridden = {fn.name for fn in child.body if isinstance(fn, ast.FunctionDef)}
    still_inherited = set()
    for fn in parent.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name in overridden or fn.name == "__init__":
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name) and node.value.id == "self"):
                still_inherited.add(node.attr)

    at_risk = sorted((set_by_parent_init & still_inherited) - set_by_child_init)
    assert not at_risk, (
        "ProtenixDesign does not call Protenix.__init__, so these attributes are set by the "
        "parent's __init__, read by a method ProtenixDesign INHERITS, and never set by "
        "ProtenixDesign.__init__ -- each one is an AttributeError waiting for the call that "
        f"reaches it: {at_risk}. Either set it in ProtenixDesign.__init__ "
        "(tt_bio/pxdesign/model.py) or make the reader tolerate its absence.")
    assert read_by_inherited, "sanity: the parent reads no self attributes outside __init__"
