"""Training must stay opt-in and inert on import, and it must not fork the forward.

Two invariants the training effort rests on, both of which were asserted in a docstring and
neither of which was checked by anything until this file existed. The first protects the
release gate: if importing `tt_bio` ever pulls in the tape, every inference path pays for it
and an accuracy or perf regression arrives from a module nobody thought was running. The
second is the rule Moritz put above the others -- the training path calls the SAME forward the
inference path calls, because a forked forward trains a model we do not serve and the drift is
invisible until it matters.

The fork check is deliberately a census, not a pass/fail on zero: `perf/ptxft/tape_block.py`
re-implements four shipped modules today and de-forking it is scheduled build work, not a bug
to fail CI on. What this test pins is that the fork does not GROW and does not move into
`tt_bio/`. Shipping a taped copy of a production module under `tt_bio/` is the failure this
catches.

Host-only: pure AST, no ttnn import, no device.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = REPO_ROOT / "tt_bio"

# The training modules. Anything on the inference path importing one of these breaks inertness.
TRAINING = {"autograd", "finetune", "train"}
# The only modules allowed to import them: the training stack itself.
ALLOWED = {"autograd.py", "finetune.py"}


def _imports(path: Path) -> set[str]:
    """Top-level-ish names this file imports from inside tt_bio, by module basename."""
    tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "tt_bio" and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            # `from tt_bio.autograd import x` / `from tt_bio import autograd`
            if parts[0] == "tt_bio":
                if len(parts) > 1:
                    found.add(parts[1])
                found |= {a.name for a in node.names}
            # `from . import autograd` / `from .autograd import x`, node.level >= 1
            elif node.level and mod:
                found.add(parts[0])
            elif node.level:
                found |= {a.name for a in node.names}
    return found


def test_tt_bio_init_does_not_import_training():
    """`import tt_bio` must not reach the tape, the optimizer or the loss set."""
    leaked = _imports(PKG / "__init__.py") & TRAINING
    assert not leaked, (
        f"tt_bio/__init__.py imports {sorted(leaked)} -- training is opt-in, so importing tt_bio "
        f"must not pull it in. Every inference path and the release gate pay for whatever lands here."
    )


def test_no_inference_module_imports_training():
    """Nothing under tt_bio/ outside the training stack itself may import it."""
    offenders = {}
    for path in sorted(PKG.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        rel = path.relative_to(PKG)
        if rel.parts[0] == "train" or rel.name in ALLOWED:
            continue
        leaked = _imports(path) & TRAINING
        if leaked:
            offenders[str(rel)] = sorted(leaked)
    assert not offenders, (
        f"inference modules import the training stack: {offenders}. Training stays opt-in; route "
        f"the dependency the other way so the inference path never reaches it."
    )


# ---------------------------------------------------------------- the no-fork rule

# Shipped forwards, and the taped re-implementation of each that exists today under perf/.
# `PairformerLayer` is the one that matters most: six device modules instantiate it, so taping
# the shipped class makes all six differentiable while taping a copy makes none of them.
KNOWN_FORK = {
    "PairTrackBlock": "tenstorrent.py:PairformerLayer",
    "DiTBlock": "protenix.py:DiffusionModule._denoise_device",
    "ConfidenceHeads": "protenix.py:ConfidenceHead",
    "DistogramHead": "protenix.py:ConfidenceHead (distogram head)",
}
FORK_FILE = REPO_ROOT / "perf" / "ptxft" / "tape_block.py"


def _classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


@pytest.mark.skipif(not FORK_FILE.is_file(), reason="the fork is gone, which is the goal")
def test_the_known_fork_does_not_grow():
    """The taped copies under perf/ are a census. New ones mean the fork is spreading."""
    extra = _classes(FORK_FILE) - set(KNOWN_FORK)
    assert not extra, (
        f"perf/ptxft/tape_block.py grew new taped classes {sorted(extra)}. Every class here is a "
        f"copy of a shipped forward; adding one widens the drift instead of closing it. Tape the "
        f"shipped module in place. Known copies and their originals: {KNOWN_FORK}"
    )


def test_the_training_package_defines_no_forward():
    """`tt_bio/train/` may hold losses, optimizers and the loop. It may not hold a forward.

    Matching on a bare class name across the whole package does not work -- `ConfidenceHeads`
    and `DistogramHead` are legitimate shipped inference classes in `boltz2.py`,
    `esmfold2.py`, `rf3/distogram_head.py` and `boltzgen/model/modules/confidence.py`, and a
    name census flags all four. The real risk is narrower and this is it: the training package
    growing its own copy of a module the inference path already ships. Any class name defined
    under `tt_bio/train/` that collides with one defined elsewhere under `tt_bio/` is either
    that copy or a shadow confusing enough to be worth renaming.

    A taped copy anywhere ELSE under `tt_bio/` would have to import the tape to be taped at
    all, so `test_no_inference_module_imports_training` already covers that case.
    """
    train_dir = PKG / "train"
    shipped: dict[str, str] = {}
    for path in sorted(PKG.rglob("*.py")):
        if "_vendor" in path.parts or train_dir in path.parents or path == train_dir:
            continue
        for name in _classes(path):
            shipped.setdefault(name, str(path.relative_to(PKG)))

    collisions = {}
    for path in sorted(train_dir.rglob("*.py")):
        for name in _classes(path) & set(shipped):
            collisions[f"{path.relative_to(PKG)}:{name}"] = shipped[name]
    assert not collisions, (
        f"tt_bio/train/ defines classes that already exist on the inference path: {collisions}. "
        f"The training path calls the shipped forward; it does not carry its own. If this is a "
        f"genuine unrelated name clash, rename it -- the ambiguity is the problem."
    )
