#!/usr/bin/env python3
"""D56: prove `TT_BIO_SOFTMAX_BW_RENORM` cannot reach a fold, before shipping it on.

Ask 9633 told the campaign the lever "lives in `taped_ttnn.py`". On the d116 tree it lives in
`autograd.py`, it is read from three places, and one of its consumers is `host_f64_softmax` --
the path D137 is about. So the premise is not the thing to inherit, and the claim that survives
is structural:

    the flag is read only inside `softmax_bw_inner` and inside `bw(g)` closures, and every
    caller of `softmax_bw_inner` is itself inside a `bw(g)` closure that only `_tape` builds.

d116 moved the branch into a helper, which is the right factoring and also the reason the old
version of this check no longer means anything: a read inside a module-level helper is not by
itself a read inside a backward. The helper's CALLERS carry the property now, so they are what
gets checked.

Five properties, each with a negative control that breaks exactly what the check reads (A17):

  SITES      every load of `SOFTMAX_BW_RENORM` / `_SOFTMAX_BW_RENORM` in `tt_bio/` that sits
             in a BRANCH CONDITION is inside `softmax_bw_inner` or inside a `bw` nested in a
             factory handed to `_tape`. Branch condition rather than any mention, because the
             property is that nothing DECIDES on the flag outside a backward -- an alias and a
             counter dump mention it and compute nothing, and a check that banned the word
             would be satisfied by hiding it behind one more name.
  NOOPS      a read that is not a branch condition may not appear in a statement that also
             calls `ttnn`.
  CALLERS    every call to `softmax_bw_inner` is inside such a `bw`.
  ONEPARSE   `TT_BIO_SOFTMAX_BW_RENORM` is parsed from the environment EXACTLY ONCE in the
             package. Two parses is two defaults, and shipping one on would leave the other
             off -- the half-fix d116 unified the inline expressions to prevent.
  TAPEONLY   `host_f64_softmax` refuses without an installed tape, and `_swap(True)`, which
             installs the taped proxy at all, is called only from `tape()`/`recompute_scope()`.
             `host_f64_softmax` does not exist on main, where this copy runs, so its clause is
             vacuous rather than passing; the `_swap` half still scores.

Copied from `origin/wk/of3t:perf/of3t_d56renorm/renorm_reach.py` with that one adaptation. The
tree it scores here routes THREE call sites, not two: `tt_bio.autograd.softmax` is exported in
`__all__` and the first repair of this expression left it inline.

Static only. The runtime half -- counters at 0 over a real fold, non-zero under a tape -- is
`renorm_blast_radius.py` and `renorm_tape_control.py`. Neither half is sufficient alone: AST
cannot see a `getattr`, and a counter reading 0 proves nothing until something makes it read
non-zero.
"""
import ast
import json
import pathlib
import re
import sys

NAMES = {"SOFTMAX_BW_RENORM", "_SOFTMAX_BW_RENORM"}
HELPER = "softmax_bw_inner"
ENVVAR = "TT_BIO_SOFTMAX_BW_RENORM"


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for kid in ast.iter_child_nodes(node):
            out[kid] = node
    return out


def _chain(node, parents):
    out = []
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(cur.name)
        cur = parents.get(cur)
    return out


def tape_factories(text):
    """Names handed to `_tape(value, inputs, <factory>)`, however `_tape` is spelled."""
    out = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fn = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if fn != "_tape" or len(node.args) < 3:
            continue
        if isinstance(node.args[2], ast.Name):
            out.add(node.args[2].id)
    return out


def _in_bw(chain, factories):
    return len(chain) >= 2 and chain[0] == "bw" and chain[1] in factories


def _gating(node, parents):
    """Whether this read sits in a BRANCH CONDITION, i.e. can steer what gets computed.

    The safety property is not "the flag is never mentioned outside a backward" -- an alias
    and a counter dump mention it and compute nothing. It is that nothing DECIDES on it
    outside a backward. A read in an `if` test or a conditional expression decides; a read
    bound to a name or stored in a dict does not.
    """
    cur, prev = parents.get(node), node
    while cur is not None:
        if isinstance(cur, (ast.If, ast.IfExp)) and prev is cur.test:
            return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return False
        prev, cur = cur, parents.get(cur)
    return False


def _touches_ttnn(node, parents):
    """Whether the statement holding this read also names `ttnn`."""
    cur = parents.get(node)
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = parents.get(cur)
    if cur is None:
        return False
    return any(isinstance(n, ast.Name) and n.id == "ttnn" for n in ast.walk(cur))


def scan(name, text):
    """Flag reads and helper calls, each with the function chain it sits in."""
    tree = ast.parse(text)
    parents = _parents(tree)
    reads, calls = [], []
    for node in ast.walk(tree):
        bare = isinstance(node, ast.Name) and node.id in NAMES and isinstance(node.ctx, ast.Load)
        attr = (isinstance(node, ast.Attribute) and node.attr in NAMES
                and isinstance(node.ctx, ast.Load))
        if bare or attr:
            reads.append({"file": name, "line": node.lineno, "funcs": _chain(node, parents),
                          "gating": _gating(node, parents),
                          "touches_ttnn": _touches_ttnn(node, parents)})
        if isinstance(node, ast.Call):
            f = node.func
            fn = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if fn == HELPER:
                calls.append({"file": name, "line": node.lineno,
                              "funcs": _chain(node, parents)})
    return reads, calls


def check_sites(srcs):
    bad, seen_r, seen_c = [], [], []
    for name, text in srcs:
        fac = tape_factories(text)
        reads, calls = scan(name, text)
        for s in reads:
            seen_r.append(s)
            c = s["funcs"]
            ok_place = c[:1] == [HELPER] or _in_bw(c, fac)
            if s["gating"] and not ok_place:
                bad.append("%s:%d BRANCHES on the flag outside %s and outside a bw closure "
                           "(enclosing: %s)" % (name, s["line"], HELPER, c or "<module>"))
            if not s["gating"] and s["touches_ttnn"] and not ok_place:
                bad.append("%s:%d reads the flag in a statement that also calls ttnn, outside "
                           "%s and outside a bw closure (enclosing: %s)"
                           % (name, s["line"], HELPER, c or "<module>"))
        for s in calls:
            seen_c.append(s)
            if not _in_bw(s["funcs"], fac):
                bad.append("%s:%d calls %s outside a bw closure handed to _tape "
                           "(enclosing: %s)" % (name, s["line"], HELPER,
                                                s["funcs"] or "<module>"))
    if not seen_r:
        bad.append("no read of the flag found at all -- the check is scanning the wrong tree")
    if not seen_c:
        bad.append("no call to %s found at all -- the check is scanning the wrong tree" % HELPER)
    return bad, seen_r, seen_c


def check_one_parse(srcs):
    """Exactly one place turns the environment variable into a bool."""
    hits = []
    for name, text in srcs:
        for m in re.finditer(r"(env_flag|os\.environ\.get|os\.getenv)\s*\(\s*[\"']%s[\"']"
                             % re.escape(ENVVAR), text):
            hits.append("%s:%d" % (name, text[:m.start()].count("\n") + 1))
    if len(hits) != 1:
        return ("%s is parsed from the environment %d times (%s); two parses is two defaults "
                "and shipping one on leaves the other off" % (ENVVAR, len(hits), ", ".join(hits)))
    return None


def check_host_needs_tape(text):
    """`host_f64_softmax` raises unless a tape is installed."""
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.FunctionDef) and node.name == "host_f64_softmax"):
            continue
        for n in node.body:
            if not isinstance(n, ast.If):
                continue
            src = ast.dump(n.test)
            if "installed" in src and isinstance(n.test, ast.UnaryOp):
                if any(isinstance(r, ast.Raise) for r in ast.walk(n)):
                    return None
        return "host_f64_softmax does not refuse when no tape is installed"
    # Absent on this tree. The float64 host backward is a `wk/of3t` path (D137) and main has no
    # such function, so the property is vacuous here rather than violated. Vacuous, NOT skipped:
    # the moment the function appears the clause above scores it, so this cannot be the hole a
    # later merge slips through.
    return None


def check_shim(text):
    tree = ast.parse(text)
    parents = _parents(tree)
    callers = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_swap"):
            continue
        a = node.args[0] if node.args else None
        if not (isinstance(a, ast.Constant) and a.value is True):
            continue
        c = _chain(node, parents)
        callers.add(c[-1] if c else "<module>")
    extra = callers - {"tape", "recompute_scope"}
    if extra:
        return "_swap(True) is called from %s, not only tape/recompute_scope" % sorted(extra)
    if not callers:
        return "_swap(True) is never called -- the check is reading the wrong file"
    return None


# --- negative controls: each must break exactly what its check reads -----------------------
_CTRL_READ = """
def _tape(v, i, m): pass
def softmax_bw_inner(y, g):
    return SOFTMAX_BW_RENORM
def forward(x):
    if SOFTMAX_BW_RENORM:
        return ttnn.divide(x, x)
    return x
"""
_CTRL_TTNN = """
def _tape(v, i, m): pass
def softmax_bw_inner(y, g):
    return SOFTMAX_BW_RENORM
SCALE = ttnn.full(SOFTMAX_BW_RENORM)
def v(a):
    def make():
        def bw(g): softmax_bw_inner(1, 2)
        return bw
    return _tape(1, [2], make)
"""
_CTRL_ALIAS = """
def _tape(v, i, m): pass
ALIAS = SOFTMAX_BW_RENORM
def softmax_bw_inner(y, g):
    if SOFTMAX_BW_RENORM: pass
    return 1
def v(a):
    def make():
        def bw(g): softmax_bw_inner(1, 2)
        return bw
    return _tape(1, [2], make)
"""
_CTRL_CALL = """
def _tape(v, i, m): pass
def softmax_bw_inner(y, g):
    return SOFTMAX_BW_RENORM
def forward(x):
    return softmax_bw_inner(x, x)
"""
_CTRL_HOST = """
def host_f64_softmax(x, dim=-1):
    if not installed():
        pass
    return y
"""
_CTRL_SHIM = """
def _swap(b): pass
def tape(): _swap(True)
def some_inference_entry(): _swap(True)
"""
_CTRL_PARSE = [("a.py", 'X = env_flag("%s", True)' % ENVVAR),
               ("b.py", 'Y = os.environ.get("%s", "0")' % ENVVAR)]


def controls():
    bad_read, _, _ = check_sites([("ctrl.py", _CTRL_READ)])
    bad_call, _, _ = check_sites([("ctrl.py", _CTRL_CALL)])
    bad_ttnn, _, _ = check_sites([("ctrl.py", _CTRL_TTNN)])
    bad_alias, _, _ = check_sites([("ctrl.py", _CTRL_ALIAS)])
    return {"forward_branch_on_flag_is_caught": bool(bad_read),
            "flag_into_a_ttnn_call_is_caught": bool(bad_ttnn),
            # And the other direction: a bare alias computes nothing and must stay QUIET,
            # or the check is just banning the word and would be satisfied by hiding it.
            "bare_alias_stays_quiet": not bad_alias,
            "forward_caller_of_helper_is_caught": bool(bad_call),
            "two_parses_is_caught": bool(check_one_parse(_CTRL_PARSE)),
            "host_without_raise_is_caught": bool(check_host_needs_tape(_CTRL_HOST)),
            "extra_swap_caller_is_caught": bool(check_shim(_CTRL_SHIM))}


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    ctl = controls()
    if not all(ctl.values()):
        print("REFUSING: a negative control did not fire -- " + json.dumps(ctl))
        return 2

    srcs = [(str(p.relative_to(root)), p.read_text())
            for p in sorted((root / "tt_bio").rglob("*.py"))]
    fail, reads, calls = check_sites(srcs)
    for e in (check_one_parse(srcs),
              check_host_needs_tape((root / "tt_bio/autograd.py").read_text()),
              check_shim((root / "tt_bio/taped_ttnn.py").read_text())):
        if e:
            fail.append(e)

    rep = {"root": str(root), "controls": ctl, "read_sites": reads,
           "helper_call_sites": calls, "read_count": len(reads),
           "call_count": len(calls), "failures": fail,
           "verdict": "PASS" if not fail else "FAIL"}
    print(json.dumps(rep, indent=1))
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(json.dumps(rep, indent=1, sort_keys=True))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
