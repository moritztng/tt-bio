#!/usr/bin/env python3
"""Which taped call sites a real training step actually runs, and whose gradient they carry.

PROTOCOL A2 says a coverage claim has to come from a census taken DURING a taped forward,
because the one failure mode that matters is silent: `tape()` rebinds the module-global name
`ttnn`, so a module that holds no such global runs its ttnn calls untaped, receives raw handles
and never raises. A static walk of the source cannot see that, and neither can a per-verb count
taken inside the proxy -- the calls that never reach the proxy are exactly the ones in question.

So this measures three things at once, in one run:

  EXECUTED      a line-coverage map, from `sys.monitoring` LINE events armed only while a tape
                is open. A static call site whose line never fires inside a tape did not run.
  TAPED         a per-call-site count, from the caller's frame inside the proxy's own verb
                wrapper. This is what the tape actually followed.
  REACH         the parameters upstream of each site's output, from the tape itself. `_Node`
                gains one integer -- a bitmask of the parameter leaves its subgraph contains --
                computed at node construction from its parents, which costs one OR per edge and
                nothing at all when no tape is open.

EXECUTED minus TAPED is the A2 hole, stated as a number rather than as a source reading: a site
that ran inside a tape and reached no tape entry ran untaped.

REACH is what makes the census compose with the gradient equivalence. A site's share of the
squared gradient norm is the mass of the parameters its output reaches, scored against the same
float64 reference denominator the 92.1568 % is taken in. A parameter no executed site reaches
has no gradient path in this step, whatever its per-parameter agreement says.

Nothing here edits a shipped file. The four patches are installed on module objects at runtime
and every one of them degrades to the shipped behaviour if the thing it patches has moved.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
import time

TOOL_ID = 3
TOOL_NAME = "of3t-pathcov"

# --------------------------------------------------------------------------------------
# The static inventory
# --------------------------------------------------------------------------------------


def _ttnn_names(tree):
    """Every local name bound to the `ttnn` module in this file, module- or function-scope.

    `import ttnn as _tn` inside a function is the A2 defect wearing an alias, so the alias has
    to be in the root set or the census would count the file as having no call sites at all.
    """
    roots = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.name == "ttnn":
                    roots[al.asname or "ttnn"] = node.lineno
    return roots


def _attr_chain(node):
    """`ttnn.experimental.foo` -> ("ttnn", "experimental.foo"), or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or not parts:
        return None
    return node.id, ".".join(reversed(parts))


def inventory(root: pathlib.Path, verbs=None):
    """Every `ttnn.<verb>(...)` call site in tt-bio's own sources.

    `verbs` is `taped_ttnn.VERBS`; a site whose verb has an entry is one the tape MUST follow,
    and a site whose verb has none is one that raises if it is ever handed a taped tensor. Both
    are in the inventory, flagged, because the second set is where the next A2 comes from.
    """
    sites = []
    for path in sorted(root.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        try:
            src = path.read_text()
            tree = ast.parse(src, str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        roots = _ttnn_names(tree)
        if not roots:
            continue
        # enclosing function, by line span, so a site can be attributed to a body
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((node.lineno, getattr(node, "end_lineno", node.lineno), node.name))
        funcs.sort(key=lambda f: (f[1] - f[0]))

        def enclosing(line):
            for a, b, name in funcs:          # narrowest first
                if a <= line <= b:
                    return name, a, b
            return None, None, None

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _attr_chain(node.func)
            if chain is None or chain[0] not in roots:
                continue
            qual = chain[1]
            fn, fa, fb = enclosing(node.lineno)
            sites.append({
                "file": str(path.relative_to(root.parent)),
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "qual": qual,
                "func": fn,
                "func_span": [fa, fb],
                "root": chain[0],
                "taped_verb": (verbs is None) or (qual in verbs),
            })
    return sites


# --------------------------------------------------------------------------------------
# The runtime census
# --------------------------------------------------------------------------------------


class Census:
    def __init__(self, repo_root):
        self.root = pathlib.Path(repo_root)
        self.sites = []
        self.by_pos = {}              # (abs file, line) -> [site index]
        self.taped_calls = {}         # site index -> count
        self.untracked_calls = {}     # (file, line, qual) -> count, no static match
        self.touched = {}             # (abs file, line) -> 1, inside a tape only
        self.reach = {}               # site index -> param bitmask
        self.leaf_bit = {}            # id(leaf) -> bit index
        self.leaves = []              # bit index -> leaf Tensor
        self.name_map = None          # the instrument's own id(handle) -> checkpoint name
        self.in_tape = 0
        self.n_tape_opens = 0
        self.hook_calls = 0
        self.shimmed = []
        self.skipped = []
        self.armed_at = None
        self.line_events = 0

    # -- site bookkeeping ---------------------------------------------------------------

    def load_inventory(self, verbs):
        self.sites = inventory(self.root / "tt_bio", verbs)
        for i, s in enumerate(self.sites):
            f = str((self.root / s["file"]).resolve())
            s["_abs"] = f
            for ln in range(s["line"], s["end_line"] + 1):
                self.by_pos.setdefault((f, ln), []).append(i)
        return len(self.sites)

    def _site_for(self, filename, lineno, qual):
        cands = self.by_pos.get((filename, lineno))
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        for i in cands:                        # disambiguate by verb when a line holds two
            if self.sites[i]["qual"] == qual:
                return i
        return cands[0]

    # -- parameter reach ----------------------------------------------------------------

    def _bit(self, leaf):
        b = self.leaf_bit.get(id(leaf))
        if b is None:
            b = len(self.leaves)
            self.leaf_bit[id(leaf)] = b
            self.leaves.append(leaf)
        return 1 << b

    def mask_of(self, t):
        node = getattr(t, "node", None)
        if node is not None:
            return getattr(node, "pmask", 0)
        try:
            import tt_bio.autograd as ag
            v = t._value
            leaf = ag._PARAMS.get(id(v))
        except Exception:
            return 0
        return self._bit(t) if leaf is t else 0


_C: Census | None = None


def census() -> Census:
    return _C


def install(repo_root, want_lines=True):
    """Arm the census. Call before the instrument builds anything."""
    global _C
    import tt_bio.autograd as ag
    import tt_bio.taped_ttnn as tp

    c = _C = Census(repo_root)
    n = c.load_inventory(tp.VERBS)

    # 1. every tape node carries the parameter set of its own subgraph.
    base_node = ag._Node

    class _CNode(base_node):
        __slots__ = ("pmask",)

        def __init__(self, fn, parents):
            base_node.__init__(self, fn, parents)
            m = 0
            for p in parents:
                m |= _C.mask_of(p)
            self.pmask = m

    ag._Node = _CNode

    # 2. the proxy's verb wrapper, with the caller's position recorded. `_Ttnn` caches a
    #    resolved verb into its instance dict on first use, so the cache is purged too --
    #    a verb resolved before this point would keep the unrecorded wrapper forever.
    base_verb = tp._taped_verb

    def _taped_verb(qual, shipped):
        inner = base_verb(qual, shipped)

        def call(*args, **kwargs):
            fr = sys._getframe(1)
            out = inner(*args, **kwargs)
            i = _C._site_for(fr.f_code.co_filename, fr.f_lineno, qual)
            if i is None:
                k = (fr.f_code.co_filename, fr.f_lineno, qual)
                _C.untracked_calls[k] = _C.untracked_calls.get(k, 0) + 1
            else:
                _C.taped_calls[i] = _C.taped_calls.get(i, 0) + 1
                node = getattr(out, "node", None)
                if node is not None:
                    _C.reach[i] = _C.reach.get(i, 0) | getattr(node, "pmask", 0)
            return out

        return call

    tp._taped_verb = _taped_verb
    _purge(tp._SHIM)

    # 3. the two-verb hook seam (`ops.linear`, `ops.layer_norm`) is counted but not attributed:
    #    it is a different seam from the proxy and its own coverage is `of3t-tape`'s census.
    base_hook = ag._hook

    def _hook(name, shipped, args, kwargs):
        _C.hook_calls += 1
        return base_hook(name, shipped, args, kwargs)

    ag._hook = _hook

    # 4. line coverage, armed only while a tape is open, so a call site that runs during weight
    #    loading is not counted as having run inside the forward.
    base_swap = tp._swap

    def _swap(to_shim):
        base_swap(to_shim)
        if to_shim:
            _C.in_tape += 1
            _C.n_tape_opens += 1
            _C.shimmed = sorted(m.__name__ for m in tp._SHIMMED)
            _C.skipped = sorted(
                nm for nm, mod in sys.modules.items()
                if nm.startswith("tt_bio") and mod is not None
                and nm not in tp._NEVER_SHIM and mod not in tp._SHIMMED)
            if want_lines:
                _arm_lines()
        else:
            _C.in_tape = max(0, _C.in_tape - 1)
            if want_lines:
                _disarm_lines()

    tp._swap = _swap

    # 5. the instrument's own id -> checkpoint-name map, read once out of the frame that
    #    registers the parameters. Optional: without it the reach is reported by leaf index.
    base_param = ag.parameter

    def parameter(raw, requires_grad=True):
        if _C.name_map is None:
            _C.name_map = _sniff_names(sys._getframe(1))
        return base_param(raw, requires_grad)

    ag.parameter = parameter
    return n


def _purge(shim):
    """Drop `_Ttnn`'s resolved-verb cache, recursively, keeping the two real slots."""
    import tt_bio.taped_ttnn as tp
    for k, v in list(vars(shim).items()):
        if k in ("_real", "_prefix"):
            continue
        if isinstance(v, tp._Ttnn):
            _purge(v)
        object.__delattr__(shim, k)


def _sniff_names(frame):
    """The largest int->str dict in the caller's locals, which is the instrument's name map.

    `perf/of3t_diffusion/device_gradient.py` builds `reg` (id(device handle) -> checkpoint name)
    at load time and registers its parameters from the same frame. Read rather than required:
    an instrument without one still gets a full census, with the reach reported by leaf index.
    """
    best = None
    for v in frame.f_locals.values():
        if not isinstance(v, dict) or not v:
            continue
        k0 = next(iter(v))
        if isinstance(k0, int) and isinstance(v[k0], str):
            if best is None or len(v) > len(best):
                best = v
    return best


_LINE_ARMED = False


def _line_cb(code, lineno):
    c = _C
    if c is None or not c.in_tape:
        return sys.monitoring.DISABLE
    key = (code.co_filename, lineno)
    c.line_events += 1
    if key in c.by_pos:
        c.touched[key] = c.touched.get(key, 0) + 1
        return None                      # keep counting a site line
    c.touched[key] = 1
    return sys.monitoring.DISABLE        # every other line costs exactly one callback


def _arm_lines():
    global _LINE_ARMED
    mon = sys.monitoring
    if not _LINE_ARMED:
        mon.use_tool_id(TOOL_ID, TOOL_NAME)
        mon.register_callback(TOOL_ID, mon.events.LINE, _line_cb)
        _LINE_ARMED = True
        _C.armed_at = time.time()
    mon.set_events(TOOL_ID, mon.events.LINE)
    mon.restart_events()


def _disarm_lines():
    if _LINE_ARMED:
        sys.monitoring.set_events(TOOL_ID, 0)


def release():
    global _LINE_ARMED
    if _LINE_ARMED:
        sys.monitoring.set_events(TOOL_ID, 0)
        sys.monitoring.free_tool_id(TOOL_ID)
        _LINE_ARMED = False


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def _leaf_names(c):
    out = []
    for leaf in c.leaves:
        nm = None
        if c.name_map is not None:
            try:
                nm = c.name_map.get(id(leaf._value))
            except Exception:
                nm = None
        out.append(nm)
    return out


def report(c=None):
    """The census, as plain data. Mass is scored separately, on host, by `census.py`."""
    c = c or _C
    names = _leaf_names(c)
    rows = []
    for i, s in enumerate(c.sites):
        ran = any((s["_abs"], ln) in c.touched
                  for ln in range(s["line"], s["end_line"] + 1))
        hits = sum(c.touched.get((s["_abs"], ln), 0)
                   for ln in range(s["line"], s["end_line"] + 1))
        taped = c.taped_calls.get(i, 0)
        mask = c.reach.get(i, 0)
        idx = [b for b in range(len(c.leaves)) if mask >> b & 1] if mask else []
        rows.append({
            "file": s["file"], "line": s["line"], "end_line": s["end_line"],
            "qual": s["qual"], "func": s["func"], "root": s["root"],
            "taped_verb": s["taped_verb"],
            "executed_in_tape": bool(ran), "line_hits": hits, "taped_calls": taped,
            "n_params_reached": len(idx),
            "params_reached": [names[b] or f"<leaf {b}>" for b in idx] if len(idx) <= 8 else None,
            "params_idx": idx,
        })
    # which functions were entered at all, for the reachable/unreachable split
    entered = set()
    for (f, ln) in c.touched:
        entered.add(f)
    func_entered = {}
    for s in c.sites:
        key = (s["file"], s["func"])
        a, b = s["func_span"]
        if a is None:
            func_entered[key] = None
            continue
        func_entered[key] = any((s["_abs"], ln) in c.touched for ln in range(a, b + 1))
    return {
        "n_sites": len(c.sites),
        "n_tape_opens": c.n_tape_opens,
        "hook_calls": c.hook_calls,
        "line_events": c.line_events,
        "files_touched_in_tape": sorted(
            os.path.basename(f) for f in entered if "/tt_bio/" in f),
        "shimmed_modules": c.shimmed,
        "skipped_modules": c.skipped,
        "n_leaves": len(c.leaves),
        "leaf_names": names,
        "untracked_taped_calls": [
            {"file": k[0], "line": k[1], "qual": k[2], "count": v}
            for k, v in sorted(c.untracked_calls.items(), key=lambda kv: -kv[1])],
        "func_entered": {f"{k[0]}::{k[1]}": v for k, v in func_entered.items()},
        "sites": rows,
    }
