"""Which ttnn verbs the Pairformer chain calls, and which of them the tape can follow.

`census.py` counts the chain's calls. This maps them onto `tt_bio.autograd._VERBS`, so
the coverage number is read off the tape rather than asserted. A verb in the `pass` column
never meets a taped tensor -- a config class, a device query, a host transfer -- and needs
no entry.
"""
import ast, collections, json, sys, os
sys.path.insert(0, os.getcwd())

SRC, ROOT = "tt_bio/tenstorrent.py", "Pairformer"
tree = ast.parse(open(SRC).read(), SRC)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

# Verbs that cannot carry a gradient because they do not carry a tensor into the graph:
# device and grid queries, config factories, host transfers and the allocator.
NO_TENSOR = {
    "get_max_worker_l1_unreserved_size", "get_memory_view", "synchronize_device",
    "create_sharded_memory_config", "CoreCoord", "CoreGrid", "Shape", "DispatchCoreConfig",
    "UnaryWithParam", "SDPAProgramConfig", "MinimalMatmulConfig", "get_arch_name",
    "num_cores_to_corerangeset", "open_device", "close_device", "from_torch", "to_torch",
    "allocate_tensor_on_device", "zeros",
    "MatmulMultiCoreReuseMultiCast1DProgramConfig", "MatmulMultiCoreReuseProgramConfig",
    "MatmulMultiCoreReuseMultiCastProgramConfig",
}


def dotted(n):
    p = []
    while isinstance(n, ast.Attribute):
        p.append(n.attr); n = n.value
    if isinstance(n, ast.Name): p.append(n.id)
    return ".".join(reversed(p))


class S(ast.NodeVisitor):
    def __init__(self): self.t = collections.Counter(); self.c = set(); self.a = set()
    def visit_Call(self, n):
        q = dotted(n.func)
        if q.startswith("ttnn."): self.t[q[5:]] += 1
        if "." not in q and q: self.c.add(q)
        self.generic_visit(n)
    def visit_Assign(self, n):
        if (len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                and isinstance(n.value, ast.Call) and dotted(n.value.func) in classes):
            self.a.add(dotted(n.value.func))
        self.generic_visit(n)


seen, q, tot = set(), [ROOT], collections.Counter()
while q:
    n = q.pop(0)
    if n in seen: continue
    seen.add(n)
    ds = ([m for m in classes[n].body if isinstance(m, ast.FunctionDef)] if n in classes
          else [funcs[n]] if n in funcs else [])
    for d in ds:
        s = S()
        for c in d.body: s.visit(c)
        tot += s.t
        q += [c for c in s.c if c in classes or c in funcs] + list(s.a)

from tt_bio import taped_ttnn as tp
taped, missing, passthru = {}, {}, {}
for verb, n in tot.items():
    if verb in tp.VERBS: taped[verb] = n
    elif verb in NO_TENSOR: passthru[verb] = n
    else: missing[verb] = n

tcalls, mcalls = sum(taped.values()), sum(missing.values())
print(f"chain from {ROOT}: {sum(tot.values())} ttnn calls, {len(tot)} verbs\n")
print(f"  taped   {tcalls:4d} calls / {len(taped):2d} verbs")
print(f"  missing {mcalls:4d} calls / {len(missing):2d} verbs")
print(f"  no tensor {sum(passthru.values()):3d} calls / {len(passthru):2d} verbs\n")
print(f"  coverage {100.0*tcalls/(tcalls+mcalls):.1f}% of the calls that carry a tensor\n")
for label, d in (("TAPED", taped), ("MISSING", missing)):
    print(f"  {label}")
    for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}  ttnn.{k}")
    print()
json.dump({"taped": taped, "missing": missing, "passthru": passthru},
          open("perf/ptx_fastpath/tape_coverage.json", "w"), indent=1, sort_keys=True)
