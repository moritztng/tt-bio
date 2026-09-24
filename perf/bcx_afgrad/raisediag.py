"""Which tt_bio call sites raise (and swallow) "not allocated" / L1 OOM in one AF2 block at n?

One device open. Untaped forward, taped forward, backward, each under a sys.monitoring RAISE
hook that records the innermost tt_bio frames of every such exception, deduplicated by site.
"""
import collections
import pathlib
import sys
import traceback

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.bcx_afgrad.afgrad import Dev, embed, load_models  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
kind = sys.argv[2] if len(sys.argv) > 2 else "evo"
dm, ref = load_models("/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz")
dev = Dev(dm)
ag = dev.ag
torch.manual_seed(0)
m0, z0 = embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
m0, z0 = m0.detach(), z0.detach()

M = sys.monitoring
TOOL = 3
M.use_tool_id(TOOL, "raisediag")
hits = collections.Counter()
phase = ["?"]


def on_raise(code, offset, exc):
    msg = str(exc)
    if "not allocated" in msg or "Out of Memory" in msg or "clash" in msg:
        fr = sys._getframe(1)
        sites = []
        while fr is not None and len(sites) < 6:
            fn = fr.f_code.co_filename
            if "tt_bio" in fn or "bcx_afgrad" in fn:
                sites.append(f"{pathlib.Path(fn).name}:{fr.f_lineno}:{fr.f_code.co_name}")
            fr = fr.f_back
        key = (phase[0], msg.splitlines()[0][:70], " <- ".join(sites))
        hits[key] += 1


M.register_callback(TOOL, M.events.RAISE, on_raise)
M.set_events(TOOL, M.events.RAISE)


def run_block(m, z):
    if kind == "extra":
        return None, dev.extra(0, z)
    return dev.evo(0, m, z)


phase[0] = "untaped"
mo, zo = run_block(dev.up(m0), dev.up(z0))
dev.sync()
phase[0] = "taped_fwd"
ml, zl = dev.leaf(m0), dev.leaf(z0)
with dev.tt.tape():
    mo, zo = run_block(ml, zl)
dev.sync()
phase[0] = "backward"
roots = [zo] if mo is None else [mo, zo]
seeds = [dev.seed(torch.randn(z0.shape), zo)] if mo is None else \
    [dev.seed(torch.randn(m0.shape), mo), dev.seed(torch.randn(z0.shape), zo)]
try:
    ag.backward(roots, seeds)
    dev.sync()
    print("BACKWARD OK")
except Exception:
    print("BACKWARD RAISED")
    traceback.print_exc(limit=6)
M.set_events(TOOL, 0)
for (ph, msg, sites), c in hits.items():
    print(f"HIT x{c} [{ph}] {msg}\n    {sites}")


def l1_used():
    mv = dev.ttnn.get_memory_view(dev.device, dev.ttnn.BufferType.L1)
    return int(mv.total_bytes_allocated_per_bank), int(mv.largest_contiguous_bytes_free_per_bank)


import gc  # noqa: E402
del roots, seeds, mo, zo, ml, zl
gc.collect()
print("after step0 L1 per bank (alloc, largest free):", l1_used(), flush=True)
phase[0] = "steps"
M.set_events(TOOL, M.events.RAISE)
for s in range(1, int(sys.argv[3]) if len(sys.argv) > 3 else 6):
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    with dev.tt.tape():
        mo, zo = run_block(ml, zl)
    dev.sync()
    print(f"step{s} after fwd L1:", l1_used(), flush=True)
    roots = [zo] if mo is None else [mo, zo]
    seeds = [dev.seed(torch.randn(z0.shape), zo)] if mo is None else \
        [dev.seed(torch.randn(m0.shape), mo), dev.seed(torch.randn(z0.shape), zo)]
    try:
        ag.backward(roots, seeds)
        dev.sync()
    except Exception as e:
        print(f"step{s} BACKWARD RAISED", str(e).splitlines()[0][:200], flush=True)
        break
    del roots, seeds, mo, zo, ml, zl
    gc.collect()
    print(f"step{s} after bwd+gc L1:", l1_used(), flush=True)
M.set_events(TOOL, 0)
for (ph, msg, sites), c in hits.items():
    if ph == "steps":
        print(f"HIT x{c} [{ph}] {msg}\n    {sites}")
