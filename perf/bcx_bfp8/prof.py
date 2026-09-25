"""Where the bfp8 arm's backward time goes: cProfile one block step per arm, same process."""
import cProfile, pstats, io, sys, os, collections, json, time
sys.path.insert(0, "perf/bcx_stack")
import stack as S
from types import SimpleNamespace
args = SimpleNamespace(params=S.A.DEFAULT_PARAMS, card=3)
lv, dev, ref = S.open_all(args)
m0, z0, wm, wz = S.inputs(ref, 224, 0)
import ttnn
calls = collections.Counter()
for arm in ("stack", "stack+b8", "stack", "stack+b8"):
    lv.arm(arm)
    S.block_step(dev, lv, m0, z0, wm, wz, "evo", k=1)
    pr = cProfile.Profile(); pr.enable()
    r, _ = S.block_step(dev, lv, m0, z0, wm, wz, "evo", k=1)
    pr.disable()
    st = pstats.Stats(pr); st.sort_stats("tottime")
    s = io.StringIO(); st.stream = s; st.print_stats(18)
    print("=== ARM", arm, {k: round(v, 4) for k, v in r.items() if k in ("fwd", "bwd", "bwd_cpu")})
    print("\n".join(l[:200] for l in s.getvalue().splitlines()[6:32]))
