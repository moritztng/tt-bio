"""D49(2): dtype audit of a taped forward. Every taped op's output and parent dtypes; in the fp32 arm any non-float32
one is a site where the 'fp32' PairUpdate is not fp32. Exits after the first tape closes (no backward, no FD).
Usage: python3 perf/r4t_tape/dtlog.py <block_gradcheck args>"""
import collections, contextlib, os, runpy, sys
sys.path.insert(0, os.getcwd())
from tt_bio import autograd as AG
import tt_bio.taped_ttnn as TT

LOG = []
orig = AG._tape


def tape(out_value, parents, make_fn, reads=None):
    name = getattr(make_fn, "__qualname__", str(make_fn)).split(".<locals>")[0]
    LOG.append((name, str(out_value.dtype).split(".")[-1], tuple(str(p.value.dtype).split(".")[-1] for p in parents),
                tuple(out_value.shape)))
    return orig(out_value, parents, make_fn, reads)


AG._tape = tape
for m in list(sys.modules.values()):
    if m and getattr(m, "__name__", "").startswith("tt_bio") and getattr(m, "_tape", None) is orig:
        m._tape = tape
TT._tape = tape
otape = TT.tape


@contextlib.contextmanager
def tape_ctx(*a, **k):
    with otape(*a, **k) as t:
        yield t
    c = collections.Counter((n, o, p) for n, o, p, _ in LOG)
    print(f"DTLOG {len(LOG)} taped ops")
    for (n, o, p), k_ in sorted(c.items()):
        flag = "" if o == "FLOAT32" and all(x == "FLOAT32" for x in p) else "  <-- not fp32"
        print(f"  {k_:4d} x {n}: out {o} parents {p}{flag}")
    first = [(i, e) for i, e in enumerate(LOG) if e[1] != "FLOAT32" or any(x != "FLOAT32" for x in e[2])][:15]
    for i, e in first:
        print("  first non-fp32 #", i, e)
    sys.stdout.flush()
    os._exit(0)


TT.tape = tape_ctx
sys.argv = ["perf/r4t_tape/block_gradcheck.py"] + sys.argv[1:]
runpy.run_path("perf/r4t_tape/block_gradcheck.py", run_name="__main__")
