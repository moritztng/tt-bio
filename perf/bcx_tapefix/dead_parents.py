"""Which ops hand `_tape` a parent that is already freed, on one checkpointed AF2 block.

Run from a wk/bcx-ckpt tree (it needs perf/bcx_stack). For each taped op whose parent is no
longer allocated it records the function that called `_tape` and whether the output is live.
Then it checks the premise the fix rests on: a real view does not outlive its source's
deallocate, so a freed parent cannot be the storage of a live output.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import torch  # noqa: E402
import ttnn  # noqa: E402
from perf.bcx_stack import stack as S  # noqa: E402
from tt_bio import autograd as ag  # noqa: E402
import tt_bio.taped_ttnn as TT  # noqa: E402

seen = collections.Counter()
orig = ag._tape


def probe(out_value, parents, make_fn, *a, **k):
    seen["tapes"] += 1
    dead = [p for p in parents if not p.value.is_allocated()]
    if dead:
        f = sys._getframe(1)
        verb = getattr(f.f_locals.get("shipped"), "__name__", None) or f.f_code.co_name
        seen["dead_parent"] += len(dead)
        seen[f"{verb} out_live={out_value.is_allocated()}"] += len(dead)
    return orig(out_value, parents, make_fn, *a, **k)


ag._tape = probe
TT._tape = probe
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()


class NS:
    params = S.A.DEFAULT_PARAMS
    card = 0


lv, dev, ref = S.open_all(NS)
m0, z0, wm, wz = S.inputs(ref, a.n, 0)
lv.arm("stack@new")
res = {}
for st in ("evo", "extra"):
    seen.clear()
    S.block_step(dev, lv, m0, z0, wm, wz, st, k=1, ckpt=True)
    res[st] = dict(seen)
    print(st, json.dumps(res[st]), flush=True)

x = ttnn.from_torch(torch.randn(1, 4, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=dev)
y = ttnn.reshape(x, (1, 128, 64))
res["view"] = {"same_address": y.buffer_address() == x.buffer_address()}
ttnn.deallocate(x)
res["view"]["view_alive_after_source_deallocate"] = y.is_allocated()
print("view", json.dumps(res["view"]), flush=True)
pathlib.Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
