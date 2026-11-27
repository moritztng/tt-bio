"""BOARD-DET (D66): per-VJP digests of one taped run, for diffing two boards op by op.

Wraps every taped op: forward output digest at tape time; in the backward, the incoming gradient's digest and the
digest of every contribution the closure deposits (add_grad / add_grad_slice), keyed by forward index. Reading a
tensor back does not change it, so the run's numbers are the production ones. Writes JSONL to $VJPLOG.

Usage: VJPLOG=out.jsonl [VJPBREAK=<op index>] [VJPRUN=perf/r4t_step/trunk_step.py] python3 perf/r4t_tape/vjp_digest.py <that script's args>
       python3 perf/r4t_tape/vjp_digest.py --cmp A.jsonl B.jsonl
The op to blame is the first backward entry whose incoming gradient agrees on both boards and whose contributions do
not: same inputs, different outputs. Forward entries are compared first (a forward diff explains everything after it)."""
import hashlib, json, os, runpy, sys


def _cmp(pa, pb):
    A = [json.loads(l) for l in open(pa)]
    B = [json.loads(l) for l in open(pb)]
    fa = [e for e in A if e["k"] == "f"]; fb = [e for e in B if e["k"] == "f"]
    ba = [e for e in A if e["k"] == "b"]; bb = [e for e in B if e["k"] == "b"]
    print(f"forward ops {len(fa)} vs {len(fb)}, backward closures {len(ba)} vs {len(bb)}")
    nf = sum(x["out"] != y["out"] for x, y in zip(fa, fb))
    for x, y in zip(fa, fb):
        if x["op"] != y["op"]:
            print("STRUCTURE DIFF forward", x["i"], x["op"], y["op"]); return 2
        if x["out"] != y["out"]:
            print("FIRST FORWARD DIFF", x["i"], x["op"], x["shape"], x["dtype"], x["out"], y["out"]); break
    print(f"forward outputs differing: {nf}")
    by = {e["i"]: e for e in bb}
    blame, nb, nin = [], 0, 0
    for x in ba:
        y = by.get(x["i"])
        if y is None or x["op"] != y["op"]:
            print("STRUCTURE DIFF backward", x["i"], x["op"], y and y["op"]); return 2
        if x["out"] != y["out"]:
            nb += 1
            if x["gin"] == y["gin"]:
                nin += 1
                blame.append((x, y))
    print(f"backward closures with differing contributions: {nb}; of them with identical incoming gradient: {nin}")
    for x, y in blame[:12]:
        print("SOURCE", x["i"], x["op"], "fwd shape", x.get("shape"), "dtype", x.get("dtype"), "gin", x["gin"],
              "out A", x["out"], "out B", y["out"])
    return 0 if not nb and not nf else 1


if len(sys.argv) > 1 and sys.argv[1] == "--cmp":
    sys.exit(_cmp(sys.argv[2], sys.argv[3]))

sys.path.insert(0, os.getcwd())
import ttnn  # noqa: E402
from tt_bio import autograd as AG  # noqa: E402

OUT = open(os.environ["VJPLOG"], "w")
CUR = [None]
N = [0]
META = {}
BREAK = int(os.environ.get("VJPBREAK", "-1"))   # broken arm: scale this closure's incoming gradient by 1+2^-6


def dg(v):
    try:
        if hasattr(v, "storage_type"):
            if v.storage_type() != ttnn.StorageType.DEVICE or not v.is_allocated():
                return None
            t = ttnn.to_torch(v)
        else:
            t = v
        return hashlib.sha1(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


orig_tape = AG._tape


def tape(out_value, parents, make_fn, reads=None):
    i = N[0]; N[0] += 1
    name = getattr(make_fn, "__qualname__", str(make_fn)).split(".<locals>")[0]
    out = orig_tape(out_value, parents, make_fn, reads)
    shape = list(getattr(out_value, "shape", []) or [])
    dt = str(getattr(out_value, "dtype", "")).split(".")[-1]
    OUT.write(json.dumps({"k": "f", "i": i, "op": name, "shape": shape, "dtype": dt, "out": dg(out_value)}) + "\n")
    node = getattr(out, "node", None)
    if node is not None:
        fn = node.fn
        META[i] = (name, shape, dt)

        def wrapped(g, _fn=fn, _i=i):
            rec = {"k": "b", "i": _i, "op": META[_i][0], "shape": META[_i][1], "dtype": META[_i][2], "gin": dg(g), "out": []}
            prev = CUR[0]; CUR[0] = rec
            if _i == BREAK:
                g = ttnn.multiply(g, 1.015625); rec["broken"] = True
            try:
                return _fn(g)
            finally:
                CUR[0] = prev
                OUT.write(json.dumps(rec) + "\n")
        node.fn = wrapped
    return out


import tt_bio.taped_ttnn  # noqa: E402,F401  binds _tape by name at import
AG._tape = tape
for mod in list(sys.modules.values()):  # verbs bound `_tape` by name at import
    if mod is not None and getattr(mod, "_tape", None) is orig_tape and mod is not AG:
        mod._tape = tape

orig_add, orig_slice = AG.Tensor.add_grad, AG.Tensor.add_grad_slice


def add_grad(self, grad):
    if CUR[0] is not None:
        CUR[0]["out"].append(dg(grad))
    return orig_add(self, grad)


def add_grad_slice(self, grad, starts, ends):
    if CUR[0] is not None:
        CUR[0]["out"].append(dg(grad))
    return orig_slice(self, grad, starts, ends)


AG.Tensor.add_grad = add_grad
AG.Tensor.add_grad_slice = add_grad_slice

run = os.environ.get("VJPRUN", "perf/r4t_step/trunk_step.py")
sys.argv = [run] + sys.argv[1:]
try:
    runpy.run_path(run, run_name="__main__")
finally:
    late = [m.__name__ for m in list(sys.modules.values()) if m is not None and getattr(m, "_tape", None) is orig_tape and m is not AG]
    OUT.write(json.dumps({"k": "meta", "ops": N[0], "unpatched": late}) + "\n")
    OUT.close()
    print(f"vjp_digest: {N[0]} taped ops logged; modules that bound _tape after the patch (missed): {late}")
