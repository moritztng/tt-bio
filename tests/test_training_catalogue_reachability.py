"""A featuriser in the tree must be reachable from Tier 0, not only from an import.

Done-definition items 1 and 5 both depend on one fact: the top of the disclosure ladder has to
reach the model you can actually train. `tt_bio/train/catalogue.py` ships an EMPTY registry and
says so honestly -- Tier 0 supplies the forward, the featuriser is per model on purpose, and
until a model registers one `tt-bio finetune --model X` refuses with the name of what is
missing. That is a correct design and this gate does not argue with it.

What this gate catches is the next step going wrong quietly. A row that builds a training run
loop inside the package also builds the featuriser to feed it, and it has no reason to visit
`catalogue.py` while doing so. The result is a capability that exists, is measured, is written
up -- and is unreachable from the command the README tells a user to run. That is the same
defect the data-parallelism gate next door was written for, and it has now happened three times
on this campaign, each time surviving a clean merge with no conflict.

So: if any class under `tt_bio/train/` satisfies the catalogue's OWN dataset contract, some
module under `tt_bio/` must call `catalogue.register`. The contract is read out of
`catalogue.py` by AST rather than copied here, so it tracks the contract instead of freezing a
stale copy of it -- if `REQUIRED_DATASET_MEMBERS` grows a fifth member, this gate asks about the
fifth member too.

Green on `main` today, where the registry is empty and no class satisfies the contract. It is
meant to go red at the merge that would introduce the gap, which is the only moment fixing it is
cheap.

Static and importless: no wheel, no card.
"""

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "tt_bio" / "train"
CATALOGUE = TRAIN / "catalogue.py"

pytestmark = pytest.mark.skipif(
    not CATALOGUE.is_file(),
    reason="tt_bio/train/catalogue.py is not on this tree yet, so there is no contract to hold "
           "a featuriser to")


def _label(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise -- the controls scan a tmp tree."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return os.fspath(path)


def _contract(source: str) -> tuple:
    """REQUIRED_DATASET_MEMBERS as catalogue.py currently defines it."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "REQUIRED_DATASET_MEMBERS" for t in node.targets):
            return tuple(e.value for e in node.value.elts)
    raise AssertionError(
        "catalogue.py no longer defines REQUIRED_DATASET_MEMBERS. The contract moved; re-cut "
        "this gate against wherever it lives rather than deleting it")


def _self_attributes(node: ast.AST, out: set) -> None:
    """Every ``self.X = ...`` under ``node``, including tuple targets.

    The tuple form matters and is not a hypothetical: the first draft of this detector handled
    only ``self.x = v`` and silently under-reported a real dataset class that assigns
    ``self.n, self.cfg, self.micro, self.tokens, self.device = ...`` on one line. The runtime
    check in ``catalogue.load`` accepts instance attributes (``m in vars(dataset)``), so a
    static detector that ignores them is not checking the same contract the code enforces.
    """
    def target(t: ast.AST) -> None:
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
            out.add(t.attr)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for element in t.elts:
                target(element)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for t in sub.targets:
                target(t)
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            target(sub.target)


def _members(cls: ast.ClassDef) -> set:
    """Names a reader of ``catalogue.load`` would find on an instance of ``cls``."""
    out = set()
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(stmt.name)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    _self_attributes(cls, out)
    return out


def _featurisers(train_dir: Path, contract: tuple) -> list:
    found = []
    for path in sorted(train_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and set(contract) <= _members(node):
                found.append(f"{_label(path)}:{node.lineno} {node.name}")
    return found


def _registers(package: Path) -> list:
    """Call sites of the CATALOGUE's ``register``, excluding catalogue.py's own definition.

    Matching any call named ``register`` is not good enough and this is not a theoretical
    tightening: the first draft did exactly that, and ``tt_bio/`` contains nine ``atexit.register``
    calls plus an unrelated ``register`` in ``objectives.py``. The gate went GREEN on a tree
    carrying the very defect it exists to catch, and only an end-to-end run against the real
    injected change exposed it. So the match is either ``<...>.catalogue.register(...)`` or a bare
    ``register(...)`` whose name was imported from the catalogue module.
    """
    sites = []
    for path in sorted(package.rglob("*.py")):
        if path.resolve() == CATALOGUE.resolve():
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue
        # Bare `register` is only the catalogue's if it was imported from there.
        from_catalogue = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("catalogue"):
                for alias in node.names:
                    if alias.name == "register":
                        from_catalogue.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "register":
                owner = func.value
                base = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
                if base == "catalogue":
                    sites.append(f"{_label(path)}:{node.lineno}")
            elif isinstance(func, ast.Name) and func.id in from_catalogue:
                sites.append(f"{_label(path)}:{node.lineno}")
    return sites


def test_a_featuriser_in_the_tree_is_registered_so_tier_0_can_reach_it():
    contract = _contract(CATALOGUE.read_text(encoding="utf-8"))
    featurisers = _featurisers(TRAIN, contract)
    if not featurisers:
        pytest.skip(
            f"no class under tt_bio/train/ satisfies the dataset contract {list(contract)} yet, "
            f"so there is no featuriser to reach. This is the honest state the registry itself "
            f"documents, not a hole in the gate")
    assert _registers(ROOT / "tt_bio"), (
        f"these classes satisfy the catalogue's dataset contract {list(contract)}, so the "
        f"featuriser the empty registry says is missing now EXISTS: {featurisers}. Nothing "
        f"calls catalogue.register, so `tt-bio finetune --model <it>` still refuses and the "
        f"model is trainable only by importing the run loop directly. That leaves "
        f"done-definition items 1 and 5 blocked by one missing call. Register it next to its "
        f"featuriser -- and check the two contract details that are easy to miss: the batch "
        f"method is named `batch`, not `micro_batch`, and `device` must resolve LAZILY rather "
        f"than arriving in the constructor, because a data-parallel driver has to reach the "
        f"launcher holding no card.")


def test_the_control_the_detector_sees_what_catalogue_load_would_see():
    """Four cases, because each is a way this gate could pass on a tree that has the defect."""
    contract = ("__len__", "tokens", "device", "batch")

    satisfies_via_methods = (
        "class D:\n"
        "    def __len__(self): return 1\n"
        "    def batch(self, i): return {}\n"
        "    @property\n"
        "    def tokens(self): return 256\n"
        "    @property\n"
        "    def device(self): return None\n")
    satisfies_via_tuple_attrs = (
        "class D:\n"
        "    def __init__(self, tokens, device):\n"
        "        self.n, self.tokens, self.device = 1, tokens, device\n"
        "    def __len__(self): return self.n\n"
        "    def batch(self, i): return {}\n")
    one_short = (
        "class D:\n"
        "    def __init__(self, tokens, device):\n"
        "        self.n, self.tokens, self.device = 1, tokens, device\n"
        "    def __len__(self): return self.n\n"
        "    def micro_batch(self, i): return {}\n")

    def satisfied(src: str) -> bool:
        cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef))
        return set(contract) <= _members(cls)

    assert satisfied(satisfies_via_methods), "methods and properties must count"
    assert satisfied(satisfies_via_tuple_attrs), (
        "a tuple-target `self.a, self.b = ...` must count -- catalogue.load accepts instance "
        "attributes, and missing this form is exactly how the first draft under-reported")
    assert not satisfied(one_short), (
        "a class missing `batch` is not a featuriser and must not trip the gate; `micro_batch` "
        "is a different name and the contract is exact")


def test_the_control_the_register_detector_does_not_count_an_unrelated_register(tmp_path):
    """``atexit.register`` must not look like registering a featuriser.

    This control replaces a tautology. The first version of it asserted
    ``_registers(x) == _registers(x)``, which is true of any implementation including a broken
    one -- and the implementation WAS broken: it matched every call named ``register``, of which
    ``tt_bio/`` has ten that have nothing to do with the catalogue. A control that cannot fail is
    worse than no control, because it reads as coverage.
    """
    pkg = tmp_path / "tt_bio"
    (pkg / "train").mkdir(parents=True)
    (pkg / "train" / "__init__.py").write_text("")
    decoys = (
        "import atexit\n"
        "atexit.register(lambda: None)\n"
        "from .objectives import register\n"
        "register('not-a-featuriser', None)\n")
    (pkg / "decoy.py").write_text(decoys)
    assert _registers(pkg) == [], f"decoys were counted as a registration: {_registers(pkg)}"

    (pkg / "real.py").write_text(
        "from tt_bio.train import catalogue\n"
        "catalogue.register('abodybuilder3', lambda p, tokens=None: (None, None))\n")
    assert _registers(pkg), "an attribute call on `catalogue` must count"

    (pkg / "real2.py").write_text(
        "from .catalogue import register as reg\n"
        "reg('x', lambda p, tokens=None: (None, None))\n")
    sites = _registers(pkg)
    assert any("real2.py" in s for s in sites), (
        f"a bare call to a name imported from the catalogue must count; got {sites}")
