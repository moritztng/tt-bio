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

print(f"OK {PATH}: confidence forward keeps all of {sorted(WANT)}, each with a default")
