"""Anchor: real single-card compute time of ONE full trimul call and ONE pair
transition at N=1024, using the loaded ESMFold2 trunk. This is the compute a
sharded implementation would parallelize, against which comms must be compared.
"""
from __future__ import annotations
import gc, json, time
import torch, ttnn

N = 1024
C_Z = 128


def _bench(fn, dev, warmup=3, iters=8):
    for _ in range(warmup):
        o = fn(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = []
    for _ in range(iters):
        outs.append(fn())
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / iters
    for o in outs:
        ttnn.deallocate(o)
    return dt


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(20260712)
    from tt_bio import esmfold2 as E
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model

    ref = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False).eval()
    prefix = "folding_trunk."
    trunk_state = {k[len(prefix):]: v.float()
                   for k, v in ref.state_dict().items() if k.startswith(prefix)}
    n_layers = ref.config.folding_trunk.n_layers
    del ref; gc.collect()

    trunk = E.FoldingTrunk(n_layers=n_layers)
    trunk.load_state_dict(trunk_state, strict=False)
    del trunk_state; gc.collect()

    gen = torch.Generator().manual_seed(7)
    z = torch.randn((1, N, N, C_Z), generator=gen) * 0.1
    z_tt = trunk._from_torch(z)
    block = trunk.module.blocks[0]

    rec = {"N": N, "channels": C_Z}
    rec["full_trimul_s"] = _bench(lambda: block.tri_out(z_tt, None), trunk.tt_device)
    rec["pair_transition_s"] = _bench(lambda: block.transition(z_tt), trunk.tt_device)
    print("ANCHOR " + json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
