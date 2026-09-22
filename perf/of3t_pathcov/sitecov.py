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
import types
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
                "col": node.col_offset,
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
        self.by_exact = {}            # (abs file, line, col) -> site index
        self.taped_calls = {}         # site index -> count
        self.untracked_calls = {}     # (file, line, qual) -> count, no static match
        self.raw_calls = {}           # site index -> count of RAW ttnn calls inside a tape
        self.raw_unattributed = {}    # (file, line, col) -> count, raw call with no site
        self.wrapper_ids = set()      # id() of every recording verb wrapper
        self.callee_kind = {}         # id(callable) -> "raw" | "wrapper" | "other"
        self.raw_ids = set()          # id() of every callable reachable as ttnn.<verb>
        self.raw_names = {}           # (file, line, col) -> callee name, for the unattributed
        self.call_events = 0
        self.touched = {}             # (abs file, line) -> 1, inside a tape only
        self.reach = {}               # site index -> param bitmask
        self.leaf_bit = {}            # id(leaf) -> bit index
        self.leaves = []              # bit index -> leaf Tensor
        self.name_map = None          # candidate id(handle) -> checkpoint name maps
        self.name_map_hits = 0
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
            self.by_exact[(f, s["line"], s.get("col"))] = i
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
    c.raw_ids = _raw_verb_ids()

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

        _C.wrapper_ids.add(id(call))
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
    """Every int->str dict in the caller's locals, as CANDIDATE name maps.

    `perf/of3t_diffusion/device_gradient.py` builds `reg` (id(device handle) -> checkpoint name)
    at load time and registers its parameters from the same frame. It also holds `fp_name`, a
    fingerprint -> name map with eight times as many entries, so "the biggest one" picks the
    wrong dict and every leaf comes back unnamed. The choice is made at report time instead, by
    which candidate actually resolves the leaves: a map that names nothing is not the map.
    """
    out = []
    # Three frames out, not one. `aux_instrument.py` registers its parameters inside
    # `grad_device.tape_parameters()`, so the map lives in the instrument's frame and not in
    # the one that calls `ag.parameter`; one frame reads nothing there and the arm's whole
    # mass goes unattributed while its site census looks fine.
    for up in range(3):
        if frame is None:
            break
        out.extend(_cands(frame))
        frame = frame.f_back
    return out or None


def _cands(frame):
    """Every dict local, by reference, with no shape test at all.

    The shape test used to live here and it cost the aux arm its whole mass.
    `aux_instrument.py` reaches the tape through `grad_device.tape_parameters`, whose
    bijection is `params`: checkpoint name -> (leaf, ...), str-keyed, and EMPTY at the moment
    the first `ag.parameter()` call triggers the sniff. A filter that demands eight int keys
    rejects it twice over, and the arm reports its leaves with no names and 0.0000 % mass
    while its site census looks perfectly healthy.

    Holding the dict OBJECT fixes both halves: `params` is filled in place by the rest of
    `tape_parameters`, so by report time the reference is the finished map. Which candidate is
    a name map, and in which direction it points, is decided by `_leaf_names` -- where the
    docstring always said the choice belonged.
    """
    return [v for v in frame.f_locals.values() if isinstance(v, dict)]


def _names_of(rec):
    """The checkpoint name(s) an instrument's registry entry carries.

    Two shapes in use: `device_gradient.py` stores the name, `msa_instrument.py` stores
    `(name_a, name_b)` because one device weight can be a FUSED pair, and both halves own the
    gradient that flows through it. Both are read; a fused leaf reports both names, so its
    mass is not silently halved.
    """
    if isinstance(rec, str):
        return [rec]
    if isinstance(rec, (tuple, list)) and rec and isinstance(rec[0], str):
        return [x for x in rec if isinstance(x, str)]
    return []


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


def _positions(code):
    ps = _POS.get(code)
    if ps is None:
        ps = _POS[code] = list(code.co_positions())
    return ps


def _raw_verb_ids():
    """Every callable reachable as `ttnn.<name>` or `ttnn.<ns>.<name>`, by identity.

    The membership test has to be identity against the real module, not `__module__`. A pybind
    METHOD on a tensor (`mc.is_sharded()`, `t.memory_config()`) also answers "ttnn" there, and
    counting those as raw verb calls turns 5,760 memory-config reads into an A2 hole. Classes
    are excluded for the reason the proxy itself excludes them: `_Ttnn.__getattr__` hands a
    type straight back, so `ttnn.MatmulMultiCoreReuseProgramConfig(...)` is raw by design and
    carries no gradient.
    """
    import ttnn as real
    out, seen = set(), set()

    def walk(ns, depth):
        if id(ns) in seen or depth > 2:
            return
        seen.add(id(ns))
        for name in dir(ns):
            if name.startswith("__"):
                continue
            try:
                attr = getattr(ns, name)
            except Exception:
                continue
            if isinstance(attr, types.ModuleType):
                walk(attr, depth + 1)
            elif callable(attr) and not isinstance(attr, type):
                # Identity alone is not enough either: `ttnn` re-exports a few stdlib
                # callables into its namespace, and `math.sqrt` reached by that route turned
                # `c_out = sigma_data * t / math.sqrt(...)` into 48 raw ttnn calls a tape had
                # missed. Both tests, and only their intersection counts.
                mod = getattr(attr, "__module__", None) or getattr(type(attr), "__module__", "")
                if isinstance(mod, str) and mod.startswith("ttnn"):
                    out.add(id(attr))
    walk(real, 0)
    return out


def _classify(callee):
    """raw ttnn verb, this module's recording wrapper, or neither."""
    if id(callee) in _C.wrapper_ids:
        return "wrapper"
    return "raw" if id(callee) in _C.raw_ids else "other"


def _call_cb(code, offset, callee, arg0=None):
    """A ttnn verb called from a tt-bio frame, by exact bytecode position.

    The LINE map cannot separate `ttnn.deallocate(x)` from `if cond: ttnn.deallocate(x)` --
    the line fires either way -- so an A2 hole counted as "line ran and the proxy was never
    entered" reads 19 false positives on the diffusion arm alone. This counts the raw call
    itself: the callee is either the proxy's wrapper or `ttnn`'s own function, and only the
    second one is a call the tape could not have followed.
    """
    c = _C
    if c is None or not c.in_tape:
        return sys.monitoring.DISABLE
    if "/tt_bio/" not in code.co_filename:
        return sys.monitoring.DISABLE        # one callback per foreign location, then gone
    c.call_events += 1
    kind = c.callee_kind.get(id(callee))
    if kind is None:
        kind = c.callee_kind[id(callee)] = _classify(callee)
    if kind != "raw":
        return None
    try:
        ln, _end, col, _ec = _positions(code)[offset // 2]
    except Exception:
        ln = col = None
    i = c.by_exact.get((code.co_filename, ln, col))
    if i is None:
        cands = c.by_pos.get((code.co_filename, ln))
        i = cands[0] if cands and len(cands) == 1 else None
    if i is None:
        k = (code.co_filename, ln, col)
        c.raw_unattributed[k] = c.raw_unattributed.get(k, 0) + 1
        c.raw_names.setdefault(k, getattr(callee, "__name__", type(callee).__name__))
    else:
        c.raw_calls[i] = c.raw_calls.get(i, 0) + 1
    return None


_POS: dict = {}


def _arm_lines():
    global _LINE_ARMED
    mon = sys.monitoring
    if not _LINE_ARMED:
        mon.use_tool_id(TOOL_ID, TOOL_NAME)
        mon.register_callback(TOOL_ID, mon.events.LINE, _line_cb)
        mon.register_callback(TOOL_ID, mon.events.CALL, _call_cb)
        _LINE_ARMED = True
        _C.armed_at = time.time()
    mon.set_events(TOOL_ID, mon.events.LINE | mon.events.CALL)
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
    """EVERY candidate resolution, best-by-count first. The choice is not made here.

    Counting hits cannot pick the right map and it is worth being precise about why, because
    both wrong answers were measured on this instrument. `device_gradient.py` holds `reg`
    (id -> checkpoint name) and `orient` (id -> "T"/"N", the load orientation) over the SAME
    547 keys, so hits alone is a coin flip and distinctness breaks that tie. But it also holds
    the port's OWN device-side names, 870 of them against `reg`'s 547, and those win on both
    counts while resolving to `npe.ln_s_w` where the reference says
    `diffusion_module.diffusion_conditioning.layer_norm_s.weight`. The arm then reports every
    leaf named and 0.0000 % of the model's mass, which is a worse failure than naming nothing.

    A map that names nothing IN THE REFERENCE is not the map, and this process does not hold
    the reference -- `score.py` does. So all candidates are reported and the scorer picks the
    one whose names actually land in the float64 denominator.
    """
    ids = []
    for leaf in c.leaves:
        try:
            ids.append(id(leaf._value))
        except Exception:
            ids.append(None)
    leaf_ids = [id(leaf) for leaf in c.leaves]
    out, seen = [], set()
    for cand in (c.name_map or []):
        got = _resolve(cand, ids, leaf_ids)
        if not got:
            continue
        key = tuple(sorted((j, tuple(v)) for j, v in got.items()))
        if key in seen:
            continue
        seen.add(key)
        flat = [nm for row in got.values() for nm in row]
        out.append({"n_hits": len(flat), "n_distinct": len(set(flat)),
                    "names": [got.get(j, []) for j in range(len(ids))]})
    out.sort(key=lambda d: (-d["n_hits"], -d["n_distinct"]))
    c.name_map_hits = out[0]["n_hits"] if out else 0
    return out


def _resolve(cand, ids, leaf_ids):
    """One candidate map, resolved to leaf index -> checkpoint names, in either direction.

    FORWARD (`device_gradient.py`, `msa_instrument.py`): id(device handle) -> name, so a leaf
    is found by the handle it wraps.

    REVERSE (`grad_device.tape_parameters`): name -> (leaf, inverse, where, block, lookup), so
    a leaf is found by identity against the record. Both directions score the same way and the
    one naming more distinct leaves wins, which is what stops a dict that merely happens to be
    str-keyed from being mistaken for a bijection.
    """
    k0 = next(iter(cand), None)
    out = {}
    if isinstance(k0, int):
        pos = {i: j for j, i in enumerate(ids) if i is not None}
        for i, rec in cand.items():
            j = pos.get(i)
            if j is not None and (nms := _names_of(rec)):
                out.setdefault(j, []).extend(nms)
    elif isinstance(k0, str):
        pos = {i: j for j, i in enumerate(leaf_ids)}
        pos.update({i: j for j, i in enumerate(ids) if i is not None and i not in pos})
        for nm, rec in cand.items():
            for o in (rec if isinstance(rec, (tuple, list)) else (rec,)):
                j = pos.get(id(o))
                if j is not None:
                    out.setdefault(j, []).append(nm)
                    break
    return {j: sorted(set(v)) for j, v in out.items()}


def report(c=None):
    """The census, as plain data. Mass is scored separately, on host, by `census.py`."""
    c = c or _C
    cands = _leaf_names(c)
    allnames = cands[0]["names"] if cands else [[] for _ in c.leaves]
    names = [a[0] if a else None for a in allnames]
    rows = []
    for i, s in enumerate(c.sites):
        # The CALL instruction carries the line the call EXPRESSION starts on, so the start
        # line is the site's own execution. The span is kept beside it because an argument on
        # a later line of a multi-line call fires its own LINE event, and a site scored on the
        # span alone reads as executed whenever any of its arguments were evaluated, which is
        # not the same claim.
        hits = c.touched.get((s["_abs"], s["line"]), 0)
        ran = hits > 0
        span_hits = sum(c.touched.get((s["_abs"], ln), 0)
                        for ln in range(s["line"], s["end_line"] + 1))
        taped = c.taped_calls.get(i, 0)
        mask = c.reach.get(i, 0)
        idx = [b for b in range(len(c.leaves)) if mask >> b & 1] if mask else []
        rows.append({
            "file": s["file"], "line": s["line"], "end_line": s["end_line"],
            "qual": s["qual"], "func": s["func"], "root": s["root"],
            "taped_verb": s["taped_verb"],
            "executed_in_tape": bool(ran), "line_hits": hits,
            "executed_in_span": span_hits > 0, "span_line_hits": span_hits,
            "taped_calls": taped,
            "raw_calls_in_tape": c.raw_calls.get(i, 0),
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
        "call_events": c.call_events,
        "raw_unattributed": [{"file": k[0], "line": k[1], "col": k[2], "count": v,
                              "callee": c.raw_names.get(k)}
                             for k, v in sorted(c.raw_unattributed.items(),
                                                key=lambda kv: -kv[1])[:50]],
        "hook_calls": c.hook_calls,
        "line_events": c.line_events,
        "files_touched_in_tape": sorted(
            os.path.basename(f) for f in entered if "/tt_bio/" in f),
        "shimmed_modules": c.shimmed,
        "skipped_modules": c.skipped,
        "n_leaves": len(c.leaves),
        "leaf_names": names,
        "leaf_names_all": allnames,
        "leaf_name_candidates": cands,
        "n_leaves_named": sum(1 for n in names if n),
        "name_map_hits": c.name_map_hits,
        "touched_tt_bio": {f: sorted(l for (g, l) in c.touched if g == f)
                           for f in sorted({g for (g, _) in c.touched if "/tt_bio/" in g})},
        "untracked_taped_calls": [
            {"file": k[0], "line": k[1], "qual": k[2], "count": v}
            for k, v in sorted(c.untracked_calls.items(), key=lambda kv: -kv[1])],
        "func_entered": {f"{k[0]}::{k[1]}": v for k, v in func_entered.items()},
        "sites": rows,
    }
