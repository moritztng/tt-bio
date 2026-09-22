"""Assert a hand-resolved tt_bio/openfold3_trunk.py kept BOTH sides and changed no default.

A syntax check passes on a merge that silently dropped one side, which is the failure mode this
exists for. So this asserts the two things the resolution has to preserve, from the AST:

  1. `tri_att_end_bias_follows_pair = not is_openbind(state_dict)` is still assigned
     UNCONDITIONALLY -- of3t-pairbias's side.
  2. the env override exists and every write to that name from the env sits INSIDE an
     `if <forced> is not None:` guard -- of3t-foldab's side, and the reason it is safe.

It is an AST assertion, not an import: importing tt_bio pulls in ttnn and this host holds no
card lease. That is a real limit and it is why (2) checks the guard structurally rather than by
running with the variable unset.
"""
import ast, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "tt_bio/openfold3_trunk.py"
NAME = "tri_att_end_bias_follows_pair"
ENV = "TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR"

src = open(PATH).read()
tree = ast.parse(src)                      # raises on a merge that left conflict markers

if "<<<<<<<" in src or ">>>>>>>" in src:
    sys.exit(f"FAIL {PATH}: conflict markers survived the resolution")

base, guarded, unguarded = [], [], []

class V(ast.NodeVisitor):
    def __init__(self):
        self.guard_depth = 0

    def visit_If(self, node):
        is_guard = ENV in ast.dump(node.test) or (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.ops[0], ast.IsNot)
        )
        self.guard_depth += 1 if is_guard else 0
        self.generic_visit(node)
        self.guard_depth -= 1 if is_guard else 0

    def visit_Assign(self, node):
        tgt = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if NAME in tgt:
            d = ast.dump(node.value)
            if "is_openbind" in d and self.guard_depth == 0:
                base.append(node.lineno)
            elif self.guard_depth > 0:
                guarded.append(node.lineno)
            else:
                unguarded.append(node.lineno)
        self.generic_visit(node)

V().visit(tree)

env_reads = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and n.value == ENV]

fail = []
if len(base) != 1:
    fail.append(f"expected exactly 1 unconditional `{NAME} = not is_openbind(...)`, found {base} "
                "-- of3t-pairbias's side was dropped or duplicated by the resolution")
if not env_reads:
    fail.append(f"`{ENV}` is not read -- of3t-foldab's measurement lever was dropped")
if unguarded:
    fail.append(f"`{NAME}` is written unguarded at {unguarded} -- a shipped default now moves "
                "with an environment variable")
if env_reads and not guarded:
    fail.append(f"`{ENV}` is read but never assigns `{NAME}` inside a not-None guard")

if fail:
    print(f"FAIL {PATH}:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print(f"OK {PATH}: base assignment line {base[0]} unconditional, "
      f"env override guarded at {guarded}, no unguarded default move")
