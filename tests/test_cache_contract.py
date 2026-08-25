"""Regression: an artifact cache publishes by rename and only counts a non-empty file.

The weight registry fixed exactly this bug class at seven download sites. The MSA
path had it too: two of five producers wrote straight to the final name, and six of
seven readers gated on bare ``Path.exists()``. A killed search then left a truncated
or zero-byte ``{hash}.a3m`` that every later fold of that sequence accepted forever.

These tests pin the contract in ``tt_bio.cache`` and scan every non-vendored module
under ``tt_bio/`` for a call site that bypasses it. The scan used to name five modules
and match two suffix literals, which is how Nesso-1's host pipeline shipped three
publish sites straight to their final names and kept the check green.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

from tt_bio import cache as artifact_cache


def test_seq_hash_is_the_documented_key():
    assert artifact_cache.seq_hash("MKTVR") == hashlib.sha256(b"MKTVR").hexdigest()[:16]


def test_empty_file_is_not_a_cache_hit(tmp_path: Path):
    p = tmp_path / "abc.a3m"
    assert not artifact_cache.cached(p)          # missing
    p.write_text("")
    assert p.exists() and not artifact_cache.cached(p)   # present but empty
    p.write_text(">query\nMKT\n")
    assert artifact_cache.cached(p)


def test_publish_text_is_atomic_and_leaves_no_tmp(tmp_path: Path):
    dst = tmp_path / "sub" / "abc.a3m"
    artifact_cache.publish_text(dst, ">query\nMKT\n")
    assert dst.read_text() == ">query\nMKT\n"
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_publish_text_failure_leaves_no_partial_under_the_final_name(tmp_path: Path,
                                                                    monkeypatch):
    """A search that dies mid-write must not publish. The tmp file is the casualty."""
    dst = tmp_path / "abc.a3m"

    real = Path.write_text

    def boom(self, *a, **k):
        real(self, ">query\nPARTI")     # a short write
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        artifact_cache.publish_text(dst, ">query\nPARTIAL\n")
    monkeypatch.undo()
    assert not dst.exists()


def test_staged_leaves_nothing_under_the_final_name_when_the_producer_raises(tmp_path: Path):
    """The OpenFold3 template fetch shape: a download that dies mid-transfer."""
    dst = tmp_path / "1abc.cif"
    with pytest.raises(ConnectionError):
        with artifact_cache.staged(dst) as tmp:
            tmp.write_text("data_1ABC\n_partial")
            raise ConnectionError("connection reset")
    assert not dst.exists()
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_staged_tmp_keeps_the_destination_suffix(tmp_path: Path):
    """A suffix-sensitive producer must see the suffix it expects.

    ``np.savez_compressed`` appends ``.npz`` unless the path already ends in it, so a
    tmp named ``.x.npz.<pid>.tmp`` would be written as ``...tmp.npz`` and the publish
    rename would fail on a path that does not exist. Nesso-1's structure cache
    (``nesso1_input.py:_parse_one``) is that producer.
    """
    import numpy as np

    dst = tmp_path / "tyr48.npz"
    with artifact_cache.staged(dst) as tmp:
        assert tmp.suffix == ".npz", tmp
        np.savez_compressed(str(tmp), a=np.arange(3))
    assert dst.exists() and not list(tmp_path.rglob(".*.tmp*"))
    assert list(np.load(dst)["a"]) == [0, 1, 2]


def test_publish_file_is_atomic(tmp_path: Path):
    src = tmp_path / "src.a3m"
    src.write_text(">query\nMKT\n")
    artifact_cache.publish_file(src, tmp_path / "out" / "dst.a3m")
    assert (tmp_path / "out" / "dst.a3m").read_text() == ">query\nMKT\n"
    assert not list(tmp_path.rglob(".*.tmp*"))


def test_template_fetch_publishes_by_rename():
    """tt_bio/worker.py fetched RCSB template CIFs straight to the final name, gated on
    bare exists() -- the 8th site of the bug class the weight registry fixed at seven."""
    import tt_bio.worker
    src = Path(tt_bio.worker.__file__).read_text()
    assert 'urlretrieve(url, struct_dir' not in src
    assert 'with staged(struct_dir / f"{p}.cif") as tmp:' in src


def _tt_bio_sources():
    """Every non-vendored module under tt_bio/, discovered.

    The scoped version of this scan named five modules and matched two suffix literals.
    Nesso-1's host pipeline was neither, so its three publish sites stayed green -- the
    same hardcoded-scope failure the weights registry and token-axis guards had.
    """
    import tt_bio

    root = Path(tt_bio.__file__).parent
    return [f for f in sorted(root.rglob("*.py")) if "_vendor" not in f.parts]


#: ``.a3m`` is an alignment wherever it sits, so the contract follows it everywhere.
#: ``.csv`` is a cache entry only under the MSA directory -- BoltzGen's result merge reads
#: plain ``aggregate_metrics_analyze.csv`` files out of run directories, and those are
#: outputs a later run overwrites, not entries a later run skips work because of.
_CACHE_SUFFIXES = (".a3m",)
_MSA_ONLY_SUFFIXES = (".csv",)

#: Attribute calls that bypass the contract, and what to use instead.
_BYPASS = {"exists": "gate it with tt_bio.cache.cached()",
           "write_text": "publish it with tt_bio.cache.publish_text()",
           "write_bytes": "publish it with tt_bio.cache.publish_text()"}

#: A nested scope owns its own names, so a `src` in one function is not the `src` in another.
_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _basename(node: ast.AST) -> str | None:
    """The literal tail of ``<dir> / "name.a3m"`` or ``<dir> / f"{x}.a3m"``, else None."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return None
    right = node.right
    if isinstance(right, ast.Constant) and isinstance(right.value, str):
        return right.value
    if isinstance(right, ast.JoinedStr) and right.values:
        last = right.values[-1]
        return last.value if isinstance(last, ast.Constant) else ""
    return None


def _rooted_in_msa_dir(node: ast.AST) -> bool:
    """True when the leftmost operand of the ``/`` chain names an MSA directory."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        node = node.left
    name = node.id if isinstance(node, ast.Name) else (
        node.attr if isinstance(node, ast.Attribute) else "")
    return "msa" in name.lower()


def _is_cache_path(node: ast.AST) -> bool:
    base = _basename(node)
    if base is None:
        return False
    return (base.endswith(_CACHE_SUFFIXES)
            or (base.endswith(_MSA_ONLY_SUFFIXES) and _rooted_in_msa_dir(node)))


def _own_nodes(scope: ast.AST):
    """Every node in `scope` that is not inside a nested function scope."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPE):
            stack.extend(ast.iter_child_nodes(node))


def _scan_scope(scope: ast.AST, outer: frozenset, name: str, bad: list) -> None:
    own = list(_own_nodes(scope))
    names = set(outer)
    for node in own:
        if isinstance(node, ast.Assign) and _is_cache_path(node.value):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.NamedExpr) and _is_cache_path(node.value):
            names.add(node.target.id)
    for node in own:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        fix = _BYPASS.get(node.func.attr)
        if fix is None:
            continue
        recv = node.func.value
        if _is_cache_path(recv) or (isinstance(recv, ast.Name) and recv.id in names):
            bad.append(f"{name}:{node.lineno}: .{node.func.attr}() -- {fix}")
    frozen = frozenset(names)
    for node in own:
        if isinstance(node, _SCOPE):
            _scan_scope(node, frozen, name, bad)


def test_no_module_gates_the_msa_cache_on_bare_exists():
    """No MSA cache gate reads bare ``Path.exists()`` and no publish writes to the final
    name, so a zero-byte a3m from a failed search is redone instead of accepted forever.

    Parsed, not grepped, and resolved through one local binding. The literal version
    matched ``.a3m").exists()`` on the line itself, so ``src = a3m_out / f"{name}.a3m"``
    on one line and ``if src.exists():`` on the next was invisible to it -- which is how
    ``main.py``'s colabfold publish shipped (b03dc65f) and how ``openfold3_data.py`` kept
    a ``shutil.copyfile`` to a final name green.

    One binding is the whole widening, and it is deliberately where this stops. A path
    that reaches its ``.exists()`` through a helper, a comprehension variable or a
    container is still invisible: ``openfold3_data.py``'s ``{i: p for i, p in
    paths.items() if p.exists()}`` is exactly that shape and this does not see it. A
    narrow guard that says what it misses beats a broad one carrying exemptions.
    """
    bad = []
    for f in _tt_bio_sources():
        _scan_scope(ast.parse(f.read_text(), filename=str(f)), frozenset(), f.name, bad)
    assert not bad, "bypasses the MSA cache contract at:\n" + "\n".join(bad)


_FSTR = re.compile(r'f"([^"]*)"')
_GATE = re.compile(r'\.exists\(\)|(?<![\w.])cached\(')
_WRITE = re.compile(r'(?<![\w.])(?:save_file|np\.savez\w*|torch\.save|json\.dump)\('
                    r'|\.(?:write_text|write_bytes|dump)\(')


def _basename_patterns(line: str) -> set[str]:
    """f-string basenames on a line, placeholders normalised: f"{mid}.safetensors" -> "{}.safetensors"."""
    out = set()
    for lit in _FSTR.findall(line):
        norm = re.sub(r"\{[^}]*\}", "{}", lit)
        if "." in norm and "/" not in norm:
            out.add(norm)
    return out


def test_nothing_a_later_run_skips_on_is_published_to_its_final_name():
    """The poisoning shape, stated exactly: a file a later run SKIPS work because of,
    written straight to its final name.

    A truncated prediction output is visible to the user and overwritten next run. A
    truncated file something GATES on is accepted forever -- that is the whole bug class,
    and it is what makes this scan precise where a scan over cache-looking paths is not.
    On db56e207 it found one site (nesso1_input.py:178, the ESM-2 650M embedding, gated at
    :170) and no false positives anywhere in the tree.

    No module list, no directory list, no suffix list: for each file, the basename patterns
    used in a skip gate must not also appear on a write call outside ``staged``.
    """
    bad = []
    for f in _tt_bio_sources():
        lines = f.read_text().splitlines()
        gated: set[str] = set()
        for ln in lines:
            if _GATE.search(ln):
                gated |= _basename_patterns(ln)
        if not gated:
            continue
        for i, ln in enumerate(lines, 1):
            if not _WRITE.search(ln) or "staged(" in ln:
                continue
            if hit := _basename_patterns(ln) & gated:
                bad.append(f"{f.name}:{i}: gated on {sorted(hit)} -- {ln.strip()}")
    assert not bad, ("published to a final name that something else skips on; wrap it in "
                     "tt_bio.cache.staged:\n" + "\n".join(bad))
