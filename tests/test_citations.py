"""Every ``file.py::symbol`` citation into this repo's own source must resolve.

Comments cite the code they were reasoned against. Written as ``file.py:412`` that
pointer is correct for exactly as long as nobody inserts a line above it, and
nothing checks it, so the rot is silent. All nine citations in ``token_axis.py``'s
rfd3 row went stale in one merge that added 413 lines to ``rfd3/model.py``; the
census's boltz2 row pointed ``MSAModule`` at ``tenstorrent.py:7786``, 789 lines
short of the class; ``rfd3/design.py`` cited ``featurize.py:490`` for the atom14
pad names and line 490 was a bare ``)``.

A line number carries nothing a symbol name does not, and the symbol survives the
merge. So citations into our own tree are written ``path/to/file.py::symbol`` and
checked here: the file must exist and ``ast`` must find the symbol in it. Bare
upstream citations (``modules.py:312`` in AlphaFold, ``af/loss.py:188-259`` in
ColabDesign) name files that are not in this tree, cannot be resolved against it,
and are left alone.

Shaped after tests/test_perf_citations.py, which does this for ``perf/`` artifacts.
"""

import ast
import re
from pathlib import Path

import pytest

from conftest import git_tracked

REPO = Path(__file__).resolve().parents[1]

# `pkg/mod.py::symbol`. The symbol may be nested, by attribute (`Class.method`) or
# as a pytest node id (`test_x.py::Class::test_y`). A trailing `[param]` on a node
# id is not part of the symbol, and neither is a closing quote or backtick.
CITATION = re.compile(
    r"(?<![\w./-])([\w.\-]+(?:/[\w.\-]+)*\.py)::"
    r"([A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*)"
)

# `pkg/mod.py:412`, the form this test exists to keep out of files we own.
LINE_CITATION = re.compile(r"(?<![\w./-])([\w.\-]+(?:/[\w.\-]+)*\.py):(\d+)")

# Read for citations, but never parsed as Python by the resolver.
SKIP_SUFFIXES = {".json", ".cif", ".pdb", ".a3m", ".sto", ".npz", ".npy", ".pt", ".safetensors"}


def _sources():
    """Files that may carry a citation: shipped source, docs and the test suite."""
    tracked = git_tracked(REPO, "tt_bio", "docs", "tests", "scripts")
    if tracked is not None:
        names = tracked
    else:
        names = []
        for prefix in ("tt_bio", "docs", "tests", "scripts"):
            for path in sorted((REPO / prefix).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    names.append(str(path.relative_to(REPO)))
    return [n for n in names if Path(n).suffix not in SKIP_SUFFIXES
            and "_vendor" not in Path(n).parts]


def _scan(pattern):
    for name in _sources():
        path = REPO / name
        if not path.is_file():
            continue
        for match in pattern.finditer(path.read_text(errors="ignore")):
            yield (name, *match.groups())


def _symbols(path):
    """Every name a citation may point at in *path*: qualified and bare.

    Module-level constants count. ``esmfold2.py::PAD_MULTIPLE`` is exactly the kind
    of thing the census cites, and it is an assignment, not a def.
    """
    qualified, bare = set(), set()

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified.add(prefix + child.name)
                bare.add(child.name)
                visit(child, prefix + child.name + ".")
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        qualified.add(prefix + target.id)
                        bare.add(target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                qualified.add(prefix + child.target.id)
                bare.add(child.target.id)

    visit(ast.parse(path.read_text()), "")
    return qualified, bare


def _defines(path, symbol):
    qualified, bare = _symbols(path)
    parts = re.split(r"::|\.", symbol)
    if len(parts) == 1:
        # A method may be cited without its class; the file is the unit that matters.
        return parts[0] in bare
    chain = ".".join(parts)
    return any(q == chain or q.endswith("." + chain) for q in qualified)


@pytest.mark.parametrize("source,cited,symbol", sorted(set(_scan(CITATION))))
def test_cited_symbol_is_defined(source, cited, symbol):
    target = REPO / cited
    if not target.is_file():
        # An upstream path (AlphaFold, ColabDesign, the RFD3 reference). Not ours
        # to resolve, and a repo-relative path is what makes a citation checkable.
        pytest.skip(f"{cited} is not a file in this repo")
    assert _defines(target, symbol), (
        f"{source} cites {cited}::{symbol}, which {cited} does not define. Either "
        f"the symbol was renamed or the citation was wrong when it was written."
    )


@pytest.mark.parametrize("source,cited,line", sorted(set(_scan(LINE_CITATION))))
def test_no_line_citation_into_our_own_source(source, cited, line):
    if not (REPO / cited).is_file():
        return  # upstream, and its line numbers are not ours to keep true
    assert False, (
        f"{source} cites {cited}:{line}. Line numbers into this repo go stale on the "
        f"next merge that grows {cited}, and nothing outside this test would notice. "
        f"Cite the symbol instead: {cited}::<symbol>."
    )
