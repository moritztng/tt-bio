"""The change to tt_bio/autograd.py is comment-only, PROVEN not asserted.

BCX's bar is the right instrument for a change that can execute. This one cannot: if the
parsed syntax tree is identical before and after, no input can reach a different instruction,
which is a stronger statement than any arm comparison and costs no card. Line numbers are
stripped because a comment block shifts them and that shift is the whole diff.
"""
import ast, subprocess, sys

REPO = "/home/moritz/.coworker/wt/of3t-innercfg"
PATH = "tt_bio/autograd.py"
before = subprocess.run(["git", "-C", REPO, "show", f"origin/main:{PATH}"],
                        capture_output=True, text=True).stdout
after = open(f"{REPO}/{PATH}").read()
a, b = ast.dump(ast.parse(before)), ast.dump(ast.parse(after))
same = a == b
print(f"origin/main bytes {len(before)}  worktree bytes {len(after)}  delta {len(after)-len(before)}")
print(f"AST dump equal: {same}")
# the control: the check must be able to say NO
mutated = after.replace("inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)",
                        "inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True, "
                        "compute_kernel_config=config or precise_config())", 1)
ctl = ast.dump(ast.parse(mutated)) == a
print(f"CONTROL -- the one-line change under test, same check: AST dump equal: {ctl} "
      f"(must be False, or the check cannot see a code change)")
sys.exit(0 if (same and not ctl) else 1)
