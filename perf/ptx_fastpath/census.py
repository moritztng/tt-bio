"""Census of the shipped fast pairformer chain in tt_bio/tenstorrent.py.

Walks the call graph from `Pairformer` over module-level functions and classes defined in
tenstorrent.py, and counts the three things that stand between the shipped forward and a
gradient: unrouted linear/matmul/layer_norm calls, in-place ops, and deallocates.

Routed means the call goes through `tt_bio.ops`, which carries the grad hook. Everything
else is a direct `ttnn.*` call the tape cannot see.
"""
import ast, collections, json, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "tt_bio/tenstorrent.py"
ROOT = sys.argv[2] if len(sys.argv) > 2 else "Pairformer"

tree = ast.parse(open(SRC).read(), SRC)

# module-level defs: name -> node ; class -> {method: node}
funcs, classes = {}, {}
for n in tree.body:
    if isinstance(n, ast.FunctionDef):
        funcs[n.name] = n
    elif isinstance(n, ast.ClassDef):
        classes[n.name] = n

MATMUL = {"linear", "matmul"}
NORM = {"layer_norm", "rms_norm"}
GRADOPS = MATMUL | NORM


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class Scan(ast.NodeVisitor):
    """Collect calls inside one def, and the names it constructs or invokes."""

    def __init__(self):
        self.gradops = []      # (qualname, lineno)
        self.inplace = []      # (qualname, lineno)
        self.dealloc = []      # lineno
        self.calls = set()     # bare names called/constructed (for graph walk)
        self.attr_types = {}   # self.<attr> = ClassName(...)

    def visit_Call(self, node):
        q = dotted(node.func)
        tail = q.rsplit(".", 1)[-1]
        head = q.split(".", 1)[0]
        if head in ("ttnn", "tt_bio", "ops") or q == tail:
            if tail in GRADOPS and head in ("ttnn", "ops"):
                self.gradops.append((q, node.lineno))
            if tail == "deallocate":
                self.dealloc.append(node.lineno)
            if head == "ttnn" and tail.endswith("_") and not tail.startswith("_"):
                self.inplace.append((q, node.lineno))
        if q == tail:
            self.calls.add(tail)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.value, ast.Call)):
            cls = dotted(node.value.func)
            if cls in classes:
                self.attr_types[node.targets[0].attr] = cls
        self.generic_visit(node)


def scan_def(node):
    s = Scan()
    for child in node.body:
        s.visit(child)
    return s


# Walk: a class contributes every method; an attribute assigned a known class pulls it in.
seen, queue = set(), [ROOT]
report = collections.OrderedDict()
while queue:
    name = queue.pop(0)
    if name in seen:
        continue
    seen.add(name)
    if name in classes:
        node = classes[name]
        defs = [m for m in node.body if isinstance(m, ast.FunctionDef)]
    elif name in funcs:
        defs = [funcs[name]]
    else:
        continue
    for d in defs:
        s = scan_def(d)
        key = f"{name}.{d.name}" if name in classes else name
        if s.gradops or s.inplace or s.dealloc:
            report[key] = {"gradops": s.gradops, "inplace": s.inplace,
                           "dealloc": s.dealloc}
        for c in s.calls:
            if c in classes or c in funcs:
                queue.append(c)
        for c in s.attr_types.values():
            queue.append(c)

tot = collections.Counter()
sites = {"gradops": [], "inplace": [], "dealloc": []}
for key, r in report.items():
    for q, ln in r["gradops"]:
        routed = q.startswith("ops.")
        tot["gradops"] += 1
        tot["routed" if routed else "unrouted"] += 1
        sites["gradops"].append({"where": key, "call": q, "line": ln, "routed": routed})
    for q, ln in r["inplace"]:
        tot["inplace"] += 1
        sites["inplace"].append({"where": key, "call": q, "line": ln})
    for ln in r["dealloc"]:
        tot["dealloc"] += 1
        sites["dealloc"].append({"where": key, "line": ln})

print(f"chain from {ROOT}: {len(seen)} names, {len(report)} defs with sites")
print(f"  linear/matmul/layer_norm : {tot['gradops']}  "
      f"({tot['routed']} routed through tt_bio.ops, {tot['unrouted']} raw ttnn)")
print(f"  in-place ttnn ops        : {tot['inplace']}")
print(f"  deallocates              : {tot['dealloc']}")
print()
by_op = collections.Counter(s["call"] for s in sites["gradops"])
for k, v in by_op.most_common():
    print(f"    {k:28s} {v}")
print()
by_ip = collections.Counter(s["call"] for s in sites["inplace"])
for k, v in by_ip.most_common():
    print(f"    {k:28s} {v}")
print()
print("members walked:", ", ".join(sorted(seen & (set(classes) | set(funcs)))))
json.dump({"root": ROOT, "totals": dict(tot), "sites": sites},
          open(sys.argv[3] if len(sys.argv) > 3 else "/tmp/census.json", "w"), indent=1)
