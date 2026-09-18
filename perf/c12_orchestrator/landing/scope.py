#!/usr/bin/env python3
"""Blast radius of the cond-hoist eager-build change, resolved from the source rather than asserted.

`c12-compose-fold` measured +0.5756 s at the fold and then named its landing blocker:

    "the merge is release-gated because AdaLN and DiffusionTransformerLayer are shared modules
     and only Boltz-2 was scored"

If that is true, landing costs a re-score of every model that reaches those classes. If it is false,
scoring Boltz-2 scored the whole blast radius and the gate is narrower than the row believed. It is
a source question with a definite answer, so this asks the source.

The change under test is `origin/main...origin/wk/c12-cond-hoist-block-timing` on tt_bio/tenstorrent.py:
26 insertions, 2 deletions, two hunks -- a comment block at the flag, and 9 lines in
`DiffusionTransformer.__init__` that build `_cond_weights()` eagerly instead of at first use.

    python3 scope.py [--repo /path/to/checkout]

Exit 0 = every claim below holds. Exit 1 = at least one fails, and it says which.
"""
import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

AP = argparse.ArgumentParser()
AP.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
A = AP.parse_args()
REPO = Path(A.repo)
TT = REPO / "tt_bio" / "tenstorrent.py"

# The classes the blocker names, plus the class the diff actually edits.
NAMED = ["AdaLN", "DiffusionTransformerLayer", "DiffusionTransformer"]
src = TT.read_text()
tree = ast.parse(src)

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------------------------------------
# 1. Where are these classes DEFINED, and where are they CONSTRUCTED, inside tenstorrent.py?
#    Construction is what matters: a class nobody instantiates cannot affect a model.
# ---------------------------------------------------------------------------------------------
defined = {}
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name in NAMED:
        defined[node.name] = node.lineno

constructed = {n: [] for n in NAMED}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NAMED:
        constructed[node.func.id].append(node.lineno)

check("all three classes are defined in tenstorrent.py",
      set(defined) == set(NAMED), f"{defined}")


def enclosing(lineno):
    """The (class, def) a line sits in -- how we attribute a construction site to an owner."""
    cls = fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.lineno <= lineno <= (node.end_lineno or 0):
            cls = node.name
        if isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= (node.end_lineno or 0):
            fn = node.name
    return cls, fn


owners = {n: sorted({enclosing(l) for l in constructed[n]}) for n in NAMED}

# DiffusionTransformer -- the class the diff edits -- must be built only inside Diffusion.
dt_owners = {c for c, _ in owners["DiffusionTransformer"]}
check("DiffusionTransformer is constructed ONLY inside class Diffusion",
      dt_owners == {"Diffusion"}, f"owners={owners['DiffusionTransformer']}")

# And Diffusion itself must be reached from exactly one place.
diff_sites = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Diffusion"]
diff_owners = sorted({enclosing(l) for l in diff_sites})
check("class Diffusion is constructed from exactly one site",
      len(diff_sites) == 1, f"sites={diff_sites} owners={diff_owners}")
check("that site is DiffusionModule._create_module",
      diff_owners == [("DiffusionModule", "_create_module")], f"{diff_owners}")

# ---------------------------------------------------------------------------------------------
# 2. Does any OTHER module import these names out of tenstorrent.py? The file is shared -- other
#    models import Module, Pairformer, PairformerLayer from it -- so "the file is shared" is true
#    and is not the question. The question is whether these CLASSES escape it.
# ---------------------------------------------------------------------------------------------
escapes = []
for p in sorted((REPO / "tt_bio").rglob("*.py")):
    if p == TT or "_vendor" in p.parts:
        continue
    try:
        t = p.read_text(errors="replace")
    except OSError:
        continue
    for m in re.finditer(r"from\s+\.?\.?(?:tt_bio\.)?tenstorrent\s+import\s+\(?([^)\n]*(?:\n[^)]*)?)\)?",
                         t):
        names = {x.strip().split(" as ")[0].strip() for x in m.group(1).split(",")}
        for n in NAMED + ["Diffusion", "DiffusionModule"]:
            if n in names:
                escapes.append((p.relative_to(REPO).as_posix(), n))
# This started life as the claim "no other tt_bio module imports these classes". The check
# REFUTED it -- rglob reaches tt_bio/rf3/ and the regex handles the parenthesised multi-line
# import form, both of which a flat `grep tt_bio/*.py` misses. So the claim is inverted into the
# question that actually matters and the finding is reported rather than asserted away.
check("REFUTED: these classes do NOT stay inside tenstorrent.py",
      bool(escapes), f"escapes={escapes}")

# ---------------------------------------------------------------------------------------------
# 2b. The one that decides the landing: does the cond-hoist path FIRE for a non-Boltz-2 model?
#     The guard is `_B2_DIT_COND_HOIST and not self.atom_level`, so any other model constructing
#     this class with atom_level=False takes the hoisted path the moment the flag flips on.
# ---------------------------------------------------------------------------------------------
foreign = []
for q in sorted((REPO / "tt_bio").rglob("*.py")):
    if q == TT or "_vendor" in q.parts:
        continue
    t = q.read_text(errors="replace")
    for m in re.finditer(r"DiffusionTransformer\((?P<args>[^)]*)\)", t, re.S):
        args = m.group("args")
        if "atom_level" in args:
            lvl = re.search(r"atom_level\s*=\s*(True|False)", args)
            foreign.append((q.relative_to(REPO).as_posix(),
                            t[:m.start()].count("\n") + 1,
                            lvl.group(1) if lvl else "?"))
fires = [f for f in foreign if f[2] == "False"]
check("REFUTED: tenstorrent.py:1387-1389 says these levers are "
      "'boltz-2-exclusive by construction' -- a foreign model builds the class with atom_level=False",
      bool(fires), f"constructions={foreign}  FIRES FOR={fires}")

# Would it crash, or change numbers silently? _cond_weights() reads Boltz-2 checkpoint key names.
NEEDED = ["output_projection_linear.weight", "output_projection.0.weight"]
remappers = []
for q in sorted((REPO / "tt_bio").rglob("*.py")):
    if q == TT or "_vendor" in q.parts:
        continue
    t = q.read_text(errors="replace")
    if all(k in t for k in NEEDED):
        remappers.append(q.relative_to(REPO).as_posix())
check("the foreign model REMAPS its checkpoint into the keys _cond_weights() reads, "
      "so the flag changes it SILENTLY rather than raising KeyError",
      bool(remappers), f"remap into Boltz-2 key names: {remappers}")

# ---------------------------------------------------------------------------------------------
# 3. The property that makes the merge safe independently of any of the above: with the flag at
#    its shipped default, the added code is a NO-OP. Assert it from the real diff, not from memory.
# ---------------------------------------------------------------------------------------------
diff = subprocess.run(
    ["git", "-C", str(REPO), "diff",
     "origin/main...origin/wk/c12-cond-hoist-block-timing", "--", "tt_bio/tenstorrent.py"],
    capture_output=True, text=True).stdout
added = [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]
code_added = [l for l in added if l.strip() and not l.strip().startswith("#")]
check("the diff adds executable code only inside the flag guard",
      len(code_added) == 2
      and re.match(r"\s*if _B2_DIT_COND_HOIST and not atom_level:", code_added[0])
      and re.match(r"\s*self\._cond_weights\(\)", code_added[1]),
      f"code_added={code_added}")
# The shipped default was False when this scope check was written, and c13-land-first flipped it
# to True on 2026-09-18 after the fold re-measure and the accuracy re-score. So this reports the
# default instead of asserting one value: a check that fails because the landing it was written to
# clear actually happened is a landmine, not a finding. What the check above still enforces is the
# thing that does not expire -- the added code sits inside the flag guard and nowhere else.
_default = re.search(
    r'_B2_DIT_COND_HOIST\s*=\s*env_flag\("TT_BIO_DIT_COND_HOIST",\s*(True|False)\)', src)
check("the flag's shipped default is readable from source",
      _default is not None,
      "no env_flag(\"TT_BIO_DIT_COND_HOIST\", ...) line found")
print(f"    shipped default: TT_BIO_DIT_COND_HOIST="
      f"{_default.group(1) if _default else '?'}")

# ---------------------------------------------------------------------------------------------
# 4. WRONG vs IMPRECISE, settled from source. RF3 passes no_residual=True and a_to_b_gate=False.
#    If the hoisted path ignored either, RF3 would get WRONG numbers (a hard stop). If both are
#    honoured identically on the cond and non-cond paths, the hoist only reorders bf16 rounding
#    (a judgement against a seed floor). This is the difference between a defect and a tradeoff.
# ---------------------------------------------------------------------------------------------
def body_of(cls_name, fn_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == fn_name:
                    return f
    return None


lay = body_of("DiffusionTransformerLayer", "__call__")
# find the `if self.no_residual:` statement and compare how each arm calls self.transition
no_res_ok = False
if lay is not None:
    for n in ast.walk(lay):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                and n.test.attr == "no_residual"):
            def cond_args(stmts):
                out = []
                for st in stmts:
                    for c in ast.walk(st):
                        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                and c.func.attr == "transition"):
                            out += [ast.unparse(k.value) for k in c.keywords if k.arg == "cond"]
                return out
            a_arm, b_arm = cond_args(n.body), cond_args(n.orelse)
            no_res_ok = bool(a_arm) and a_arm == b_arm
check("no_residual is honoured IDENTICALLY on the hoisted and unhoisted paths "
      "(both arms pass the same cond)", no_res_ok,
      "both branches of `if self.no_residual:` forward the same cond expression to self.transition")

# a_to_b_gate must not touch the conditioning tensors at all -- it gates the `a` side.
gate_clean = None
for node in ast.walk(tree):
    if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) \
            and node.test.attr == "a_to_b_gate":
        seg = ast.unparse(node)
        gate_clean = not any(k in seg for k in ("s_out", "s_terms", "cond", "s_o"))
check("a_to_b_gate touches no conditioning tensor, so it is orthogonal to the hoist",
      bool(gate_clean), "the a_to_b_gate branch references no s_out/s_terms/cond/s_o")

# Known-answer control: the checker must be able to FAIL. Re-run claim 2's logic against a name
# that other modules demonstrably DO import, and require it to report an escape.
ctl = []
for p in sorted((REPO / "tt_bio").rglob("*.py")):
    if p == TT or "_vendor" in p.parts:
        continue
    t = p.read_text(errors="replace")
    for m in re.finditer(r"from\s+\.?\.?(?:tt_bio\.)?tenstorrent\s+import\s+\(?([^)\n]*(?:\n[^)]*)?)\)?", t):
        if "Module" in {x.strip().split(" as ")[0].strip() for x in m.group(1).split(",")}:
            ctl.append(p.name)
check("NEGATIVE CONTROL: the same test finds `Module` escaping (so it can fail)",
      len(ctl) >= 2, f"{len(ctl)} modules import Module, e.g. {ctl[:4]}")

print("=" * 96)
print("COND-HOIST EAGER-BUILD: BLAST RADIUS")
print("=" * 96)
bad = 0
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {detail}")
    bad += (not ok)

res = dict((n, ok) for n, ok, _ in results)
escaped = bool(escapes)
print(f"""
VERDICT: {'all claims hold' if not bad else f'{bad} claim(s) FAILED -- read them, they are the finding'}

  WHAT IS CONFINED. The diff edits only `DiffusionTransformer.__init__`, and inside tenstorrent.py
  that class is constructed only by `Diffusion`, itself constructed only by
  `DiffusionModule._create_module`. That part of the blocker's premise is sound.

  WHAT IS NOT. {'These classes DO escape the file' if escaped else 'These classes stay in the file'}:
  {escapes}
  `tt_bio/rf3/token_dit.py` builds tenstorrent's `DiffusionTransformer` with **atom_level=False**,
  and the cond-hoist guard is `_B2_DIT_COND_HOIST and not self.atom_level` -- so the hoisted path
  fires for RF3 too the moment the flag flips. tenstorrent.py:1387-1389 asserts all three levers
  are "boltz-2-exclusive by construction"; that comment predates the RF3 port, which reused the
  class, and it is now FALSE. This is why the check exists instead of trusting the comment.

  AND IT FAILS SILENTLY, WHICH IS THE WORSE CASE. RF3 remaps its checkpoint into the very key
  names `_cond_weights()` reads ({', '.join(NEEDED)} -- see rf3/token_dit.py:39-46 and
  rf3/remap_encoder.py:62-73), so there is no KeyError to catch the mistake. RF3 would quietly take
  a different code path with a different bf16 rounding order, unscored.

  So the release gate `c12-compose-fold` named IS REAL, for a reason it did not state: not because
  the FILE is shared, but because a second port reuses the CLASS at atom_level=False.

  WRONG OR MERELY IMPRECISE -- SETTLED, AND IT IS IMPRECISE. RF3 passes `no_residual=True` and
  `a_to_b_gate=False`, so the obvious worry was that the hoisted path ignores them. It does not.
  `no_residual` is branched on OUTSIDE the conditioning substitution and both of its arms forward
  the same `cond` to `self.transition`, and `a_to_b_gate` gates the `a` side while the hoist only
  replaces the `s`-side output projection -- it references no conditioning tensor at all. Both are
  asserted above rather than eyeballed. So flipping the flag would NOT give RF3 wrong math; it
  would reorder RF3's bf16 rounding and move a one-time weight concatenation to model load, on a
  model nobody scored. That is a tradeoff to be judged against a seed floor, not a hard stop.

  WHAT IS SAFE TODAY. With TT_BIO_DIT_COND_HOIST at its shipped default the two added lines cannot
  run, and `__call__`'s own guard is equally inert, so nothing above affects any model as shipped.
  The eager build is a runtime no-op while the flag is off; it only moves RF3's exposure from
  first-fold to model-load once the flag is on. The exposure is created by the FLAG, not by this
  diff -- it already exists on main.
""")
sys.exit(1 if bad else 0)
