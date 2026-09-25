"""Which ttnn verbs AF2's device trunk calls, and which the tape can follow. No device, no ttnn.

`perf/ptx_fastpath/coverage.py` answers this for the Pairformer and imports `taped_ttnn` to read
the verb table. This walks from AF2's own block classes across every tt_bio module (AF2 borrows
TriangleMultiplication, TriangleAttention and OuterProductMean from `tenstorrent.py`, which in
turn call helpers in half a dozen files) and reads the verb table out of `taped_ttnn.py`'s source,
so it runs on a host with no ttnn installed.

    python3 perf/bcx_plan/af2_tape_coverage.py [ROOT ...]
"""
import ast, collections, glob, json, re, sys

ROOTS = sys.argv[1:] or ["AF2EvoformerBlock", "AF2PairBlock"]
NO_TENSOR = {
    "get_max_worker_l1_unreserved_size", "get_memory_view", "synchronize_device",
    "create_sharded_memory_config", "CoreCoord", "CoreGrid", "CoreRange", "CoreRangeSet", "Shape",
    "DispatchCoreConfig", "UnaryWithParam", "SDPAProgramConfig", "MinimalMatmulConfig",
    "get_arch_name", "num_cores_to_corerangeset", "open_device", "close_device", "from_torch",
    "to_torch", "allocate_tensor_on_device", "zeros", "DeviceComputeKernelConfig",
    "MatmulMultiCoreReuseMultiCast1DProgramConfig", "MatmulMultiCoreReuseProgramConfig",
    "MatmulMultiCoreReuseMultiCastProgramConfig", "MemoryConfig", "ShardSpec", "load_tensor", "dump_tensor", "to_device",
    "types.BlackholeComputeKernelConfig", "types.WormholeComputeKernelConfig",
}

src = open("tt_bio/taped_ttnn.py").read()
verbs = set(re.findall(r'_VERBS\["([\w.]+)"\]', src))
for args in re.findall(r"@_verb\(([^)]*)\)", src):
    verbs |= set(re.findall(r'"([\w.]+)"', args))

funcs, classes = {}, {}
# The tape itself is excluded: `tape()` does not rebind `ttnn` inside it, so its calls are the
# backward's own and would otherwise alias the verbs it defines (`layer_norm`, `softmax`).
for path in sorted(set(glob.glob("tt_bio/*.py")) - {"tt_bio/autograd.py", "tt_bio/taped_ttnn.py"}):
    tree = ast.parse(open(path).read(), path)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            funcs.setdefault(n.name, n)
        elif isinstance(n, ast.ClassDef):
            classes.setdefault(n.name, n)


def dotted(n):
    p = []
    while isinstance(n, ast.Attribute):
        p.append(n.attr); n = n.value
    if isinstance(n, ast.Name):
        p.append(n.id)
    return ".".join(reversed(p))


class S(ast.NodeVisitor):
    def __init__(self):
        self.t, self.c = collections.Counter(), set()

    def visit_Call(self, n):
        q = dotted(n.func)
        if q.startswith("ttnn."):
            self.t[q[5:]] += 1
        head = q.split(".")[-1] if q.startswith("self.") else q
        if head and "." not in head:
            self.c.add(head)
        self.generic_visit(n)


seen, q, tot = set(), list(ROOTS), collections.Counter()
while q:
    n = q.pop(0)
    if n in seen:
        continue
    seen.add(n)
    if n in classes:
        cls = classes[n]
        ds = [m for m in cls.body if isinstance(m, ast.FunctionDef)]
        q += [dotted(b) for b in cls.bases if dotted(b) in classes]
    elif n in funcs:
        ds = [funcs[n]]
    else:
        continue
    for d in ds:
        s = S()
        s.visit(d)
        tot += s.t
        q += [c for c in s.c if c in classes or c in funcs]

taped, missing, passthru = {}, {}, {}
for v, n in tot.items():
    (taped if v in verbs else passthru if v in NO_TENSOR or v.startswith(("MathFidelity", "UnaryOpType", "Arch", "types."))
     else missing)[v] = n
print(f"roots {ROOTS}: walked {len(seen)} names, {sum(tot.values())} ttnn calls, {len(tot)} verbs")
print(f"  taped {sum(taped.values())} calls / {len(taped)} verbs; missing {sum(missing.values())} / {len(missing)}")
for k, v in sorted(missing.items(), key=lambda kv: -kv[1]):
    print(f"  MISSING {v:4d}  ttnn.{k}")
json.dump({"roots": ROOTS, "taped": taped, "missing": missing, "passthru": passthru},
          open("perf/bcx_plan/af2_tape_coverage.json", "w"), indent=1, sort_keys=True)
