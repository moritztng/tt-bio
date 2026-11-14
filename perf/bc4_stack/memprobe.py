"""DRAM allocated before/after every card callback of a BindCraft 2 bench run: argv passes to perf/spd/bench.py."""
import functools, gc, os, runpy, sys, time
from tt_bio import bindcraft2 as B

def dram(dev):
    import ttnn
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks) / 2**30

def census(where):
    """Types in the cyclic garbage one collection finds, and the functions whose closures are in it."""
    import collections, ttnn
    gc.set_debug(gc.DEBUG_SAVEALL); gc.collect(); gc.set_debug(0)
    junk, gc.garbage[:] = list(gc.garbage), []
    kinds = collections.Counter(type(o).__module__ + "." + type(o).__qualname__ for o in junk)
    dev = sum(1 for o in junk if isinstance(o, ttnn.Tensor))
    fns = collections.Counter(f"{o.__code__.co_filename.rsplit('/', 1)[-1]}:{o.__code__.co_firstlineno}:{o.__qualname__}"
                              for o in junk if hasattr(o, "__code__"))
    print(f"GARBAGE after {where}: {len(junk)} objs, {dev} ttnn tensors; {kinds.most_common(12)}; fns {fns.most_common(15)}",
          file=sys.stderr, flush=True)


def wrap(cls, name):
    f = getattr(cls, name)
    @functools.wraps(f)
    def g(self, slot, *a, **k):
        pool = getattr(self, "pool", None) or getattr(getattr(self, "evo", None), "pool", None)
        dev = pool.trunk_for(slot).device
        before = dram(dev)
        try:
            return f(self, slot, *a, **k)
        finally:
            freed = gc.collect() if os.environ.get("MEMPROBE_GC") else -1
            if os.environ.get("MEMPROBE_SAVEALL"):
                census(name)
            print(f"MEM gc={freed} {time.time():.1f} {cls.__name__}.{name} {before:.2f} -> {dram(dev):.2f} GiB", file=sys.stderr, flush=True)
    setattr(cls, name, g)

for c in (B.EvoformerOnDevice, B.StructureOnDevice, B.ExtraMsaOnDevice, B.TemplateOnDevice):
    for n in ("_primal", "_taped", "_backward"):
        wrap(c, n)
sys.argv = ["perf/spd/bench.py"] + sys.argv[1:]
runpy.run_path("perf/spd/bench.py", run_name="__main__")
