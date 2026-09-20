"""Assert the hand-resolved openfold3_confidence.forward kept BOTH sides' parameters.

Two rows added keyword arguments to the same signature: one `s_path`/`dtype`, the other
`token_mask`/`single_mask` (of3t-auxfind's reference-mask fix for the aux_heads A18 failure).
A merge that silently dropped either side would still compile and still import -- the callers
that pass the dropped name would fail only at runtime, on a device, in whichever arm happens to
use it. So this checks the signature from the AST, not the file's syntax.

Every parameter must also keep a DEFAULT: both sides added theirs as optional, and a merge that
turned one into a positional would break every existing caller of a shipped function.
"""
import ast
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "tt_bio/openfold3_confidence.py"
WANT = {"s_path", "dtype", "token_mask", "single_mask"}

src = open(PATH).read()
if "<<<<<<<" in src or ">>>>>>>" in src:
    sys.exit(f"FAIL {PATH}: conflict markers survived the resolution")

try:
    tree = ast.parse(src)
except SyntaxError as e:
    # a resolution that reorders parameters can produce a non-default-after-default, which is a
    # SyntaxError rather than a bad signature -- report it as a failure, not a traceback.
    sys.exit(f"FAIL {PATH}: does not parse after the resolution -- {e}")
found = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "forward":
        names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        if "si_trunk" in names and "zij_trunk" in names:      # the confidence forward, not another
            found = (node, names)
            break

if found is None:
    sys.exit(f"FAIL {PATH}: no confidence forward(si_trunk, zij_trunk, ...) found at all")

node, names = found
missing = sorted(WANT - set(names))
if missing:
    sys.exit(f"FAIL {PATH}: confidence forward is missing {missing} -- the merge dropped one "
             f"side's parameters; signature is {names}")

# every wanted name must be optional
n_pos = len(node.args.args) - len(node.args.defaults)
required = set(node.args.args[i].arg for i in range(n_pos))
no_default = sorted(WANT & required)
if no_default:
    sys.exit(f"FAIL {PATH}: {no_default} lost their default and are now REQUIRED -- every "
             f"existing caller of this shipped function breaks")

# The two masks are a RELEASE-GATED fix (of3t-auxfind): the mechanism may ride in the
# composition, but only while it is off by default. Control 6 on that branch showed the default
# path bit-identical on 7 of 7 tensors -- that is a property of `token_mask is None`, so assert
# the default VALUE, not merely that a default exists. "Defaults off" is a claim about the
# composed branch and must be read from it, never from a write-up.
_defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):],
                     node.args.defaults)) if node.args.defaults else {}
_bad_default = []
for _n in ("token_mask", "single_mask"):
    _d = _defaults.get(_n)
    if not (isinstance(_d, ast.Constant) and _d.value is None):
        _bad_default.append(f"{_n}={ast.unparse(_d) if _d is not None else '<none>'}")
if _bad_default:
    sys.exit(f"FAIL {PATH}: the release-gated masks are no longer off by default "
             f"({', '.join(_bad_default)}) -- a gated fix has become a shipped default")

# and the shipped inference call site must still not pass them
import os as _os
_fold = _os.path.join(_os.path.dirname(PATH), "openfold3_fold.py")
if _os.path.exists(_fold):
    _fsrc = open(_fold).read()
    _passed = [n for n in ("token_mask=", "single_mask=") if n in _fsrc]
    if _passed:
        sys.exit(f"FAIL {_fold}: the shipped fold path now passes {_passed} into the confidence "
                 f"head -- that is the release-gated change becoming live inference")

print(f"OK {PATH}: confidence forward keeps all of {sorted(WANT)}; the release-gated masks "
      f"default to None and the shipped fold path does not pass them")
